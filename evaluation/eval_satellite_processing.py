"""Default segmentation backend for the QASR agent (``eval_agent.py``).

``segment_image_from_path`` is the function the LLM-generated code calls. It
routes each request to one of three backends, transparently to the LLM:

* ``USE_GT_MASKS=1``: ground-truth masks (oracle mode, ``eval_gt_segmentation``).
* DINOv3 hybrid mode, enabled by ``init_dinov3_model``: topics in
  ``DINOV3_SUPPORTED_TOPICS`` go to the local DINOv3+Mask2Former model and the
  remaining topics to the HTTP server.
* Otherwise, every topic goes to the segmentation HTTP server at
  ``SEGMENTATION_API_BASE``.

The HTTP protocol is a three-step exchange: ``init-tiles`` returns an
accumulation token, ``process-image/`` segments the image for the requested
topics, and ``get-combined-results/`` returns one mask URL per topic. Any
segmenter that implements these endpoints can replace the ConvNeXt-UNet server
used in the paper.

Module state read by the benchmark scripts: ``LAST_MASKS`` (topic -> mask of
the last call) and ``VIZ_COUNTER`` (number of visualizations written).
"""

import json
import logging
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from .eval_shapes import mask_to_shapes
from .eval_visualization import create_visualization

USE_GT_MASKS = os.environ.get("USE_GT_MASKS", "0") == "1"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_segmentation")

SEGMENTATION_API_BASE = "http://localhost:8000/"
INIT_TILES_URL = SEGMENTATION_API_BASE + "init-tiles"
PROCESS_IMAGE_URL = SEGMENTATION_API_BASE + "process-image/"
GET_COMBINED_RESULTS_URL = SEGMENTATION_API_BASE + "get-combined-results/"

VIZ_DIR = None
VIZ_COUNTER = 0

LAST_MASKS = {}

_DINOV3_ENABLED = False
DINOV3_SUPPORTED_TOPICS = ['forest', 'agric', 'barren', 'water', 'roof']
API_ONLY_TOPICS = ['solar', 'grass', 'urban']


def init_dinov3_model(checkpoint_path, device='cuda'):
    """Load the DINOv3 model and enable hybrid DINOv3/HTTP routing.

    Args:
        checkpoint_path (str): Hugging Face repo id or path of the DINOv3+Mask2Former weights.
        device (str): Torch device for inference.
    """
    global _DINOV3_ENABLED
    from .eval_dinov3_segmentation import init_dinov3_model as init_dinov3

    init_dinov3(checkpoint_path, device)
    _DINOV3_ENABLED = True
    logger.info("DINOv3 hybrid mode enabled")
    logger.info(f"  DINOv3: {DINOV3_SUPPORTED_TOPICS}")
    logger.info(f"  API: {API_ONLY_TOPICS}")


def set_visualization_dir(path):
    """Enable or disable saving segmentation overlays for this backend.

    Args:
        path (str | Path | None): Output directory, created if missing. A falsy
            value disables visualization.
    """
    global VIZ_DIR
    if path:
        VIZ_DIR = Path(path)
        VIZ_DIR.mkdir(exist_ok=True, parents=True)
    else:
        VIZ_DIR = None


def segment_image_from_path(image_input, topics, min_area_pixels=0, gsd=1.0):
    """Load and segment an image from a file path or PIL Image using mask-based approach.
    
    Args:
        image_input (str | PIL.Image): Either a string file path or a PIL Image object.
        topics (list[str]): List of segmentation topics from AvailableTopics in system prompt
            (e.g., ["water", "forest", "SolarPanels", ...]).
        min_area_pixels (int, optional): Minimum area threshold in pixels. Defaults to 0.
        gsd (float, optional): Ground sample distance in meters per pixel. Defaults to 1.0.
        
    Returns:
        dict: Dictionary containing:
            - shapes (list[dict]): List of shape dictionaries, where each dict contains:
                - id (int): Unique identifier for the shape.
                - class_type (str): The topic/class this shape belongs to. 
                    NOTE: This will exactly match the input topic string (e.g., "SolarPanels" not "solarpanels").
                - area_pixels (float): Area of the shape in pixels.
                - area_m2 (float): Area in square meters.
                - area_hectares (float): Area in hectares.
                - center (list[float, float]): [x, y] pixel coordinates of centroid.
                - polygon (list[float]): Flattened list of coordinates.
                - holes (list[list[float]] | None): List of hole polygons.
            - image_width (int): Image width in pixels.
            - image_height (int): Image height in pixels.
            - total_pixels (int): Total image pixels (width * height).
    
    Example:
        >>> result = segment_image_from_path(IMAGE_PATH, ["water", "forest"], gsd=0.3)
        >>> shapes = result["shapes"]
        >>> total_pixels = result["total_pixels"]
        >>> 
        >>> forest_shapes = [s for s in shapes if s["class_type"] == "forest"]
        >>> forest_pixels = sum(s["area_pixels"] for s in forest_shapes)
        >>> percentage = (forest_pixels / total_pixels) * 100
    """
    if USE_GT_MASKS:
        from .eval_gt_segmentation import segment_image_with_gt, LAST_GT_MASKS
        result = segment_image_with_gt(image_input, topics, min_area_pixels, gsd)
        LAST_MASKS.clear()
        LAST_MASKS.update(LAST_GT_MASKS)
        return result

    dinov3_topics = [t for t in topics if t in DINOV3_SUPPORTED_TOPICS] if _DINOV3_ENABLED else []

    api_topics = [t for t in topics if t not in dinov3_topics]

    all_shapes = []
    image_width = 0
    image_height = 0
    total_pixels = 0

    if dinov3_topics:
        try:
            from .eval_dinov3_segmentation import segment_image_with_dinov3
            result = segment_image_with_dinov3(image_input, dinov3_topics, gsd)
            all_shapes.extend(result['shapes'])
            image_width = result['image_width']
            image_height = result['image_height']
            total_pixels = result['total_pixels']
        except Exception as e:
            logger.error(f"DINOv3 failed: {e}")
            api_topics.extend(dinov3_topics)

    if api_topics:
        result = _segment_image_from_path_api(image_input, api_topics, min_area_pixels, gsd)
        all_shapes.extend(result['shapes'])
        if image_width == 0:
            image_width = result['image_width']
            image_height = result['image_height']
            total_pixels = result['total_pixels']

    # In hybrid mode the overlay is drawn here so it shows shapes from both
    # backends. In pure HTTP mode _segment_image_from_path_api draws it.
    global VIZ_COUNTER
    if VIZ_DIR is not None and all_shapes and _DINOV3_ENABLED:
        if isinstance(image_input, str):
            img_for_viz = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            img_for_viz = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            img_for_viz = None

        if img_for_viz is not None:
            VIZ_COUNTER += 1
            viz_path = VIZ_DIR / f"seg_{VIZ_COUNTER:04d}.png"

            viz_result = {
                'shapes': all_shapes,
                'image_width': image_width,
                'image_height': image_height,
                'total_pixels': total_pixels
            }

            create_visualization(img_for_viz, viz_result, str(viz_path))

            if dinov3_topics and api_topics:
                logger.info(f"Saved hybrid viz (DINOv3 + API): {viz_path}")
            elif dinov3_topics:
                logger.info(f"Saved DINOv3 viz: {viz_path}")
            elif api_topics:
                logger.info(f"Saved API-only viz (hybrid mode): {viz_path}")

    return {
        'shapes': all_shapes,
        'image_width': image_width,
        'image_height': image_height,
        'total_pixels': total_pixels
    }


def _segment_image_from_path_api(image_input, topics, min_area_pixels=0, gsd=1.0):
    """Segment an image with the HTTP segmentation server.

    Same arguments and return format as ``segment_image_from_path``. Also
    replaces ``LAST_MASKS`` with this call's per-topic masks and, outside
    hybrid mode, saves an overlay when ``VIZ_DIR`` is set. Any failure (bad
    image, HTTP error, exception) is logged and yields an empty result with
    zero image size.
    """
    try:
        if isinstance(image_input, str):
            logger.info(f"Processing image: {image_input}")
            with open(image_input, 'rb') as f:
                img_bytes = f.read()
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                logger.error(f"Could not load image: {image_input}")
                return {"shapes": [], "image_width": 0, "image_height": 0, "total_pixels": 0}
        else:
            logger.info("Processing PIL image")
            image = np.array(image_input)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, img_encoded = cv2.imencode('.png', image)
            img_bytes = img_encoded.tobytes()

        height, width = image.shape[:2]

        response = requests.get(INIT_TILES_URL)
        if response.status_code != 200:
            logger.error(f"Failed to initialize tile accumulation: {response.status_code}")
            return {"shapes": [], "image_width": 0, "image_height": 0, "total_pixels": 0}

        token = int(response.json())
        logger.info(f"Got accumulation token: {token}")

        files = {'file': img_bytes}
        data = {
            'topics': ';'.join(topics),
            'gsd': gsd,
            'polyfmt': 'milde',
            'toff': f"{token};0;0;{width};{height}"  # token;x_offset;y_offset;tile_width;tile_height
        }

        response = requests.post(PROCESS_IMAGE_URL, files=files, data=data)
        if response.status_code != 200:
            logger.error(f"Segmentation error: {response.status_code}")
            return {"shapes": [], "image_width": 0, "image_height": 0, "total_pixels": 0}

        logger.info("Image processed, retrieving combined results...")

        data = {
            "token": token,
            "boundary": json.dumps([0, 0, width, 0, width, height, 0, height, 0, 0]),
            "polyfmt": "none",
            "get_masks": True
        }

        response = requests.post(GET_COMBINED_RESULTS_URL, data=data)
        if response.status_code != 200:
            logger.error(f"Results error: {response.status_code}")
            return {"shapes": [], "image_width": 0, "image_height": 0, "total_pixels": 0}

        results = response.json()

        all_shapes = []
        shape_id = 0

        global LAST_MASKS
        LAST_MASKS = {}

        for topic in topics:
            if topic not in results:
                continue

            topic_info = results[topic]
            mask_url = topic_info.get('mask_url')
            if not mask_url:
                continue

            response = requests.get(SEGMENTATION_API_BASE + mask_url)
            if response.status_code != 200:
                continue

            # The server may scale masks horizontally by mask_sx. Undo it.
            mask_img = Image.open(BytesIO(response.content))
            sx = 1 / topic_info.get('mask_sx', 1.0)
            new_width = int(mask_img.width * sx)
            mask_img = mask_img.resize((new_width, mask_img.height), Image.NEAREST)

            mask = np.array(mask_img)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]

            LAST_MASKS[topic] = (mask > 0).astype(int).tolist()

            mask = (mask > 0).astype(np.uint8) * 255

            shapes = mask_to_shapes(mask, topic, min_area_pixels, gsd, shape_id)
            all_shapes.extend(shapes)
            shape_id += len(shapes)

        result = {
            "shapes": all_shapes,
            "image_width": width,
            "image_height": height,
            "total_pixels": width * height
        }

        global VIZ_COUNTER
        if VIZ_DIR and all_shapes and not _DINOV3_ENABLED:
            viz_path = VIZ_DIR / f"seg_{VIZ_COUNTER:04d}.jpeg"
            VIZ_COUNTER += 1
            create_visualization(image, result, viz_path)
            logger.info(f"Saved visualization to {viz_path}")

        return result

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"shapes": [], "image_width": 0, "image_height": 0, "total_pixels": 0}
