"""RemoteSAM segmentation backend (QASR-E).

RemoteSAM (Yao et al., "RemoteSAM: Towards Segment Anything for Earth
Observation", ACM MM 2025, arXiv:2505.18022) is a referring segmentation model
trained on RemoteSAM-270K, which does not overlap with the SQuID sources. It
uses a Swin-B image encoder with a BERT-base text encoder (LAVT variant) and
EPOC edge prediction to separate adjacent instances.
Code: https://github.com/1e12Leon/RemoteSAM

Per topic, ``segment_image_from_path`` runs RemoteSAM detection (instance
boxes, used only for the printed instance count and as a fallback) and
semantic segmentation (a binary mask). Shapes are the connected components of
the semantic mask, with pixel-count areas as in the SQuID ground truth. The
detection boxes become shapes only when the mask is missing or empty.

Usage:
    from .eval_remotesam_segmentation import (
        init_remotesam_model, segment_image_from_path
    )

    init_remotesam_model(
        remotesam_repo_path="/path/to/RemoteSAM",
        checkpoint_path=None,  # defaults to <repo>/pretrained_weights/checkpoint.pth
        device="cuda",
    )
    result = segment_image_from_path(IMAGE_PATH, ["building", "vehicle"], gsd=0.3)

Requirements:
    git clone https://github.com/1e12Leon/RemoteSAM.git
    huggingface-cli download 1e12Leon/RemoteSAM --local-dir ./RemoteSAM/pretrained_weights
    # Python 3.8, PyTorch 1.13, mmcv-full 1.7.1, timm 0.6.11
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .eval_shapes import mask_to_shapes as _mask_to_shapes

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("remotesam_segmentation")

_REMOTESAM_MODEL = None
_DEVICE = None
_INITIALIZED = False

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


def init_remotesam_model(
    remotesam_repo_path,
    checkpoint_path=None,
    device="cuda",
    use_EPOC=True,
    EPOC_threshold=0.25,
):
    """Load RemoteSAM from a clone of its repository.

    Errors are printed and reported through the return value rather than
    raised. Falls back to CPU when CUDA is unavailable.

    Args:
        remotesam_repo_path (str): Path to the cloned RemoteSAM repository.
            Must contain tasks/code/model.py and pretrained_weights/.
        checkpoint_path (str | None): Path to the checkpoint. Defaults to
            ``<remotesam_repo_path>/pretrained_weights/checkpoint.pth``.
        device (str): Torch device ('cuda' or 'cpu').
        use_EPOC (bool): Enable EPOC edge-based separation of dense or
            adjacent instances.
        EPOC_threshold (float): EPOC edge confidence threshold. Lower values
            split more aggressively.

    Returns:
        bool: True if the model was loaded.
    """
    global _REMOTESAM_MODEL, _DEVICE, _INITIALIZED
    import sys
    import torch

    _DEVICE = device if torch.cuda.is_available() else "cpu"

    repo = Path(remotesam_repo_path).resolve()
    if not repo.exists():
        print(f"ERROR: RemoteSAM repo not found at {repo}")
        return False

    if checkpoint_path is None:
        checkpoint_path = str(repo / "pretrained_weights" / "checkpoint.pth")

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("  Run: huggingface-cli download 1e12Leon/RemoteSAM "
              f"--local-dir {repo / 'pretrained_weights'}")
        return False

    try:
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from tasks.code.model import RemoteSAM, init_demo_model

        print(f"Loading RemoteSAM from {checkpoint_path}...")
        base_model = init_demo_model(checkpoint_path, _DEVICE)
        _REMOTESAM_MODEL = RemoteSAM(
            base_model,
            _DEVICE,
            use_EPOC=use_EPOC,
            EPOC_threshold=EPOC_threshold,
        )
        _INITIALIZED = True
        print(f"RemoteSAM initialized on {_DEVICE}")
        print(f"  EPOC: {'enabled' if use_EPOC else 'disabled'} "
              f"(threshold={EPOC_threshold})")
        return True

    except ImportError as e:
        print(f"ERROR: Cannot import RemoteSAM: {e}")
        print(f"  Make sure {repo}/tasks/code/model.py exists")
        return False
    except Exception as e:
        print(f"ERROR: RemoteSAM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def is_remotesam_initialized():
    """Return whether ``init_remotesam_model`` has loaded the model.

    Returns:
        bool: True if the model is ready for inference.
    """
    return _INITIALIZED


def _to_rgb_numpy(image_input):
    """Convert an image input to the RGB uint8 array RemoteSAM expects.

    Numpy inputs are assumed to be RGB (or BGRA when they have 4 channels).

    Args:
        image_input (str | Path | PIL.Image.Image | np.ndarray): Image file
            path, PIL image, or array.

    Returns:
        tuple[np.ndarray, int, int]: RGB array (H, W, 3), width, height.

    Raises:
        ValueError: If the file cannot be read.
        TypeError: If the input type is not supported.
    """
    if isinstance(image_input, str) or isinstance(image_input, Path):
        bgr = cv2.imread(str(image_input))
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
        else:
            rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    rgb = rgb.astype(np.uint8)
    h, w = rgb.shape[:2]
    return rgb, w, h


def segment_image_from_path(image_input, topics, min_area_pixels=0, gsd=1.0):
    """Detect and segment objects using RemoteSAM.

    This function has the SAME interface and return format as
    segment_image_from_path in eval_groundedsam_segmentation.py and
    eval_satellite_processing.py, making it a drop-in replacement.

    Pipeline per topic:
        1. RemoteSAM detection() → instance bounding boxes with EPOC separation
        2. RemoteSAM semantic_seg() → binary mask (for area + LAST_MASKS)
        3. Each box becomes one shape in the output (accurate counting)
        4. Area is computed from the semantic mask where possible

    Args:
        image_input (str | PIL.Image): File path or PIL Image object.
        topics (list[str]): Text prompts for what to detect/segment.
            RemoteSAM supports 1000+ remote sensing categories natively.
            Use simple, common nouns: ["building"], ["vehicle"], ["tree"].
        min_area_pixels (int, optional): Minimum area threshold. Defaults to 0.
        gsd (float, optional): Ground sample distance (m/px). Defaults to 1.0.

    Returns:
        dict: Dictionary containing:
            - shapes (list[dict]): List of shape dictionaries, each with:
                - id (int): Unique shape identifier.
                - class_type (str): The topic/class name.
                - area_pixels (float): Area in pixels.
                - area_m2 (float): Area in square meters (area_pixels * gsd^2).
                - area_hectares (float): Area in hectares.
                - center (list[float]): [x, y] centroid in pixel coords.
                - polygon (list[float]): Flattened [x1,y1, x2,y2, ...] coords.
                - holes (list | None): Always None for box-based shapes.
                - confidence (float): Detection confidence (0-1).
            - image_width (int): Image width in pixels.
            - image_height (int): Image height in pixels.
            - total_pixels (int): Total image area (width * height).

    Example:
        >>> result = segment_image_from_path(IMAGE_PATH, ["building"], gsd=0.3)
        >>> print(f"Found {len(result['shapes'])} buildings")
        >>> total_area = sum(s['area_pixels'] for s in result['shapes'])

    Raises:
        RuntimeError: If model not initialized.
    """
    global VIZ_COUNTER

    if not _INITIALIZED:
        raise RuntimeError(
            "RemoteSAM not initialized. Call init_remotesam_model() first."
        )

    image_rgb, width, height = _to_rgb_numpy(image_input)
    total_pixels = width * height

    all_shapes = []
    shape_id = 0

    for topic in topics:
        try:
            # detection() runs semantic_seg -> EPOC edges -> instance split ->
            # mask-to-box -> NMS and returns {topic: [[x1, y1, x2, y2, conf], ...]}.
            det_result = _REMOTESAM_MODEL.detection(
                image=image_rgb,
                classnames=[topic],
            )

            boxes = det_result.get(topic, [])
            print(f"RemoteSAM detection '{topic}': {len(boxes)} instances")

            # semantic_seg() returns {topic: (H, W) 0/1 mask}.
            seg_result = _REMOTESAM_MODEL.semantic_seg(
                image=image_rgb,
                classnames=[topic],
            )

            mask = seg_result.get(topic, None)
            if mask is not None:
                mask = np.array(mask, dtype=np.uint8)
                LAST_MASKS[topic] = mask.tolist()
            else:
                LAST_MASKS[topic] = [[0] * width for _ in range(height)]

            # Shapes come from the semantic mask (pixel-count areas, as in the
            # SQuID GT). Boxes are only a fallback: box w*h overestimates area.
            if mask is not None and mask.any():
                topic_shapes = _mask_to_shapes(
                    mask, topic, min_area_pixels, gsd, shape_id
                )
            else:
                topic_shapes = _boxes_to_shapes(
                    boxes, topic, min_area_pixels, gsd, shape_id, width, height
                )

            all_shapes.extend(topic_shapes)
            shape_id += len(topic_shapes)

        except Exception as e:
            logger.error(f"RemoteSAM failed for topic '{topic}': {e}")
            print(f"RemoteSAM error for '{topic}': {e}")
            import traceback
            traceback.print_exc()
            LAST_MASKS[topic] = [[0] * width for _ in range(height)]
            continue

    result = {
        "shapes": all_shapes,
        "image_width": width,
        "image_height": height,
        "total_pixels": total_pixels,
    }

    if VIZ_DIR is not None and all_shapes:
        VIZ_COUNTER += 1
        viz_path = VIZ_DIR / f"seg_{VIZ_COUNTER:04d}.png"
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        create_visualization(image_bgr, result, str(viz_path))

    return result


def _boxes_to_shapes(boxes, class_type, min_area_pixels, gsd, start_id, img_w, img_h):
    """Convert RemoteSAM detection boxes to rectangular shape dicts.

    Boxes are clamped to the image and their area is the box area ``w * h``.
    The confidence score is not copied into the shape dict.

    Args:
        boxes (list[list[float]]): Detections ``[x1, y1, x2, y2, confidence]``.
        class_type (str): Topic/class name for all shapes.
        min_area_pixels (float): Boxes with a smaller area are dropped.
        gsd (float): Ground sample distance in meters per pixel.
        start_id (int): First shape id.
        img_w (int): Image width used for clamping.
        img_h (int): Image height used for clamping.

    Returns:
        list[dict]: Shape dicts in the standard format, with ``holes=None``.
    """
    shapes = []
    shape_id = start_id

    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        x1 = max(0, min(float(x1), img_w))
        y1 = max(0, min(float(y1), img_h))
        x2 = max(0, min(float(x2), img_w))
        y2 = max(0, min(float(y2), img_h))

        box_w = x2 - x1
        box_h = y2 - y1
        area_pixels = box_w * box_h

        if area_pixels < min_area_pixels:
            continue

        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        shapes.append({
            "id": shape_id,
            "class_type": class_type,
            "area_pixels": float(area_pixels),
            "area_m2": float(area_pixels) * (gsd ** 2),
            "area_hectares": float(area_pixels) * (gsd ** 2) / 10000,
            "center": [float(cx), float(cy)],
            "polygon": [float(v) for v in polygon],
            "holes": None,
        })
        shape_id += 1

    return shapes


def create_visualization(image_bgr, segmentation_result, output_path):
    """Save the image with its segmented shapes overlaid and a topic legend.

    Unlike ``eval_visualization.create_visualization`` (used by the other
    backends), this variant does not paint holes. The output therefore differs
    for shapes with holes, so the two are kept separate.

    Args:
        image_bgr (np.ndarray): Input image (H, W, 3) in BGR order.
        segmentation_result (dict): Output of ``segment_image_from_path``.
            Only its ``shapes`` entry is used.
        output_path (str | Path): Destination image file.
    """
    import colorsys

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    shapes = segmentation_result.get("shapes", [])

    classes = list(set(s["class_type"] for s in shapes))
    colors = {}
    for i, cls in enumerate(classes):
        hue = i / max(len(classes), 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[cls] = tuple(int(c * 255) for c in rgb)

    for shape in shapes:
        poly = np.array(shape["polygon"]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [poly], colors[shape["class_type"]])

    result = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    if classes:
        legend_height = 30 * len(classes) + 20
        legend_width = 200
        h, w = result.shape[:2]
        final = np.ones(
            (max(h, legend_height), w + legend_width, 3), dtype=np.uint8
        ) * 255
        final[:h, :w] = result

        for i, cls in enumerate(sorted(classes)):
            y = 30 + i * 30
            cv2.rectangle(final, (w + 10, y - 10), (w + 30, y + 10), colors[cls], -1)
            cv2.rectangle(final, (w + 10, y - 10), (w + 30, y + 10), (0, 0, 0), 1)
            cv2.putText(
                final, cls, (w + 40, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )
        cv2.putText(
            final, "Topics", (w + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )
        Image.fromarray(final).save(output_path)
    else:
        Image.fromarray(result).save(output_path)
