"""SAM3 open-vocabulary segmentation backend (QASR-F).

Provides ``segment_image_from_path`` with the same signature and return format
as ``eval_satellite_processing``, but topics are free-form text prompts for
SAM3 instead of a fixed class list. The function's docstring is part of the
LLM prompt (see ``eval_api_module_sam3.build_api_docs``).

Usage:
    from .eval_sam3_segmentation import init_sam3_model, segment_image_from_path

    init_sam3_model(device='cuda')
    result = segment_image_from_path(IMAGE_PATH, ["building", "parking lot"], gsd=0.3)

Requirements:
    pip install transformers>=4.45.0 torch>=2.7.0
    huggingface-cli login
    Accept license at: https://huggingface.co/facebook/sam3
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .eval_shapes import mask_to_shapes
from .eval_visualization import create_visualization

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sam3_segmentation")

_SAM3_MODEL = None
_SAM3_PROCESSOR = None
_SAM3_DEVICE = None
_SAM3_INITIALIZED = False

# Same module state as eval_satellite_processing, read by the benchmark scripts.
VIZ_DIR = None
VIZ_COUNTER = 0
LAST_MASKS = {}


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


def init_sam3_model(checkpoint_path=None, device='cuda', model_id="facebook/sam3", use_bf16=True):
    """Load SAM3 for text-prompted segmentation.

    Errors are printed and reported through the return value rather than
    raised. The model is gated on Hugging Face: run ``huggingface-cli login``
    and accept the license at https://huggingface.co/facebook/sam3 first.

    Args:
        checkpoint_path: Ignored. Kept for signature compatibility with
            ``init_dinov3_model``.
        device (str): Torch device ('cuda' or 'cpu').
        model_id (str): Hugging Face model id.
        use_bf16 (bool): Load weights in bfloat16 when running on CUDA.

    Returns:
        bool: True if the model was loaded.
    """
    global _SAM3_MODEL, _SAM3_PROCESSOR, _SAM3_DEVICE, _SAM3_INITIALIZED

    try:
        from transformers import Sam3Processor, Sam3Model

        print(f"Loading SAM3 from {model_id}...")
        logger.info(f"Loading SAM3 from {model_id}...")

        dtype = torch.bfloat16 if use_bf16 and device == 'cuda' else torch.float32

        _SAM3_MODEL = Sam3Model.from_pretrained(model_id).to(device, dtype=dtype)
        _SAM3_PROCESSOR = Sam3Processor.from_pretrained(model_id)
        _SAM3_DEVICE = device
        _SAM3_INITIALIZED = True

        print(f"SAM3 initialized on {device}")
        logger.info(f"SAM3 initialized on {device} with dtype {dtype}")
        return True

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install transformers>=4.45.0 torch>=2.7.0")
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: SAM3 initialization failed: {e}")
        print("Make sure you have HuggingFace access. Run: huggingface-cli login")
        logger.error(f"SAM3 init failed: {e}")
        return False


def is_sam3_initialized():
    """Return whether ``init_sam3_model`` has loaded the model.

    Returns:
        bool: True if the model is ready for inference.
    """
    return _SAM3_INITIALIZED


def segment_image_from_path(image_input, topics, min_area_pixels=0, gsd=1.0):
    """Load and segment an image using SAM3 open-vocabulary text prompts.
    
    This function has the SAME interface and return format as the standard
    segment_image_from_path in eval_satellite_processing.py, but uses SAM3
    for open-vocabulary segmentation instead of the API/DINOv3 backends.
    
    IMPORTANT: Unlike the standard API which uses predefined topics, SAM3 accepts
    ANY descriptive noun phrase as a topic. You can segment concepts like:
    - "building", "parking lot", "swimming pool", "tennis court"
    - "agricultural field", "forest", "grassland"
    - "solar panel", "water tower", "runway"
    
    Args:
        image_input (str | PIL.Image): Either a string file path or a PIL Image object.
        topics (list[str]): List of text prompts describing what to segment.
            Unlike the standard API, these can be ANY descriptive noun phrase.
            Examples: ["building", "parking lot"], ["forest", "water"], ["solar panel"]
        min_area_pixels (int, optional): Minimum area threshold in pixels. Defaults to 0.
        gsd (float, optional): Ground sample distance in meters per pixel. Defaults to 1.0.
        
    Returns:
        dict: Dictionary containing:
            - shapes (list[dict]): List of shape dictionaries, where each dict contains:
                - id (int): Unique identifier for the shape.
                - class_type (str): The topic/class this shape belongs to.
                    NOTE: This will exactly match the input topic string.
                - area_pixels (float): Area of the shape in pixels.
                - area_m2 (float): Area in square meters (area_pixels * gsd^2).
                - area_hectares (float): Area in hectares (area_m2 / 10000).
                - center (list[float, float]): [x, y] pixel coordinates of centroid.
                - polygon (list[float]): Flattened list of coordinates [x1, y1, x2, y2, ...].
                - holes (list[list[float]] | None): List of hole polygons, or None.
                - confidence (float): SAM3 detection confidence score (0-1).
            - image_width (int): Image width in pixels.
            - image_height (int): Image height in pixels.
            - total_pixels (int): Total image pixels (width * height).
    
    Example:
        >>> result = segment_image_from_path(IMAGE_PATH, ["building", "parking lot"], gsd=0.3)
        >>> shapes = result["shapes"]
        >>> total_pixels = result["total_pixels"]
        >>> 
        >>> buildings = [s for s in shapes if s["class_type"] == "building"]
        >>> building_pixels = sum(s["area_pixels"] for s in buildings)
        >>> percentage = (building_pixels / total_pixels) * 100
        >>> print(f"Building coverage: {percentage:.1f}%")
    
    Raises:
        RuntimeError: If SAM3 model is not initialized. Call init_sam3_model() first.
    """
    global LAST_MASKS, VIZ_COUNTER

    if not _SAM3_INITIALIZED:
        raise RuntimeError("SAM3 not initialized. Call init_sam3_model() first.")

    if isinstance(image_input, str):
        logger.info(f"Processing image: {image_input}")
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        logger.info("Processing PIL image")
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    width, height = image.size
    total_pixels = width * height

    all_shapes = []
    shape_id = 0
    LAST_MASKS = {}

    for topic in topics:
        try:
            inputs = _SAM3_PROCESSOR(
                images=image,
                text=topic,
                return_tensors="pt"
            ).to(_SAM3_DEVICE)

            with torch.no_grad():
                outputs = _SAM3_MODEL(**inputs)

            results = _SAM3_PROCESSOR.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = results.get("masks", [])

            # Ends up in the generated code's captured stdout, which the
            # benchmark saves for error attribution.
            print(f"Initial {topic}: {len(masks)}")

            # LAST_MASKS holds the union of all instance masks for the topic.
            if len(masks) > 0:
                combined_mask = torch.zeros((height, width), dtype=torch.bool)
                for mask in masks:
                    # SAM3 may return integer rather than bool masks.
                    mask_bool = mask.cpu().bool() if mask.dtype != torch.bool else mask.cpu()
                    combined_mask |= mask_bool
                LAST_MASKS[topic] = combined_mask.numpy().astype(int).tolist()
            else:
                LAST_MASKS[topic] = [[0] * width for _ in range(height)]

            for mask in masks:
                mask_cpu = mask.cpu()
                if mask_cpu.dtype == torch.bool:
                    mask_np = mask_cpu.numpy().astype(np.uint8) * 255
                else:
                    mask_np = (mask_cpu > 0).numpy().astype(np.uint8) * 255

                shapes = mask_to_shapes(mask_np, topic, min_area_pixels, gsd, shape_id)
                all_shapes.extend(shapes)
                shape_id += len(shapes)

        except Exception as e:
            logger.error(f"SAM3 failed for topic '{topic}': {e}")
            print(f"SAM3 error for '{topic}': {e}")
            LAST_MASKS[topic] = [[0] * width for _ in range(height)]
            continue

    result = {
        'shapes': all_shapes,
        'image_width': width,
        'image_height': height,
        'total_pixels': total_pixels
    }

    if VIZ_DIR is not None and all_shapes:
        VIZ_COUNTER += 1
        viz_path = VIZ_DIR / f"seg_{VIZ_COUNTER:04d}.png"

        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        create_visualization(image_bgr, result, str(viz_path))
        logger.info(f"Saved visualization to {viz_path}")

    return result
