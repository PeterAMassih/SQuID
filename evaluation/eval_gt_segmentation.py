"""Ground-truth-mask segmentation backend (oracle mode).

With ``USE_GT_MASKS=1``, ``eval_satellite_processing.segment_image_from_path``
routes here, so the LLM-generated code and the geometry API run unchanged on
the SQuID ground-truth masks instead of predicted ones. This isolates the
reasoning and API layer from segmentation quality.

The mask file is derived from the image path with the naming conventions of
the SQuID generators, which are also those of the benchmark JSON's ``mask``
field:

    DeepGlobe  <scene>_sat.jpg  -> <scene>_label.png   (RGB colors, +-10)
    EarthVQA   <scene>.png      -> <scene>_label2.png  (exact BGR colors)
    Solar/PV   <scene>.bmp      -> <scene>_label.bmp   (grayscale > 0)

NAIP images have no masks (their ground truth is a human consensus) and raise
a ValueError.

Relative image paths that do not exist from the working directory are
resolved against ``GT_DATA_ROOT``. It defaults to
``<repo>/data/benchmark/images_and_masks``, the layout produced by the dataset
download script (``data/benchmark/qvlm_benchmark.json`` stores image and mask
paths relative to it, e.g. ``earthvqa_0.3m/3444.png``). Set the ``GT_DATA_ROOT``
environment variable to override it, or to an empty string to disable it.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from .eval_shapes import mask_to_shapes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_DATA_ROOT = REPO_ROOT / "data" / "benchmark" / "images_and_masks"
GT_DATA_ROOT = os.environ.get("GT_DATA_ROOT", str(DEFAULT_GT_DATA_ROOT))

# Class colors exactly as in the SQuID generators.
# DeepGlobe: RGB with +-10 tolerance (mask converted to RGB on load).
DEEPGLOBE_CLASSES = {
    'urban': [0, 255, 255],
    'agriculture': [255, 255, 0],
    'rangeland': [255, 0, 255],
    'forest': [0, 255, 0],
    'water': [0, 0, 255],
    'barren': [255, 255, 255],
}
# EarthVQA: exact BGR equality (mask read as BGR, no conversion).
EARTHVQA_CLASSES = {
    'building': (0, 0, 255),
    'water': (255, 0, 0),
    'barren': (255, 0, 255),
    'forest': (0, 255, 0),
    'agriculture': (255, 255, 0),
}

# QASR topic -> GT class(es) per source. Topics with no GT class on a source
# yield an empty mask (zero shapes), which is the correct oracle answer for
# feature-absence questions.
DEEPGLOBE_TOPICS = {
    'urban': ['urban'], 'agric': ['agriculture'], 'grass': ['rangeland'],
    'forest': ['forest'], 'water': ['water'], 'barren': ['barren'],
    'solar': [], 'roof': [],
}
EARTHVQA_TOPICS = {
    'roof': ['building'], 'urban': ['building'], 'water': ['water'],
    'barren': ['barren'], 'forest': ['forest'], 'agric': ['agriculture'],
    'grass': [], 'solar': [],
}
SOLAR_TOPICS = {'solar': ['solar']}

# topic -> boolean mask of the last call, copied into
# eval_satellite_processing.LAST_MASKS.
LAST_GT_MASKS = {}

_mask_cache = {}


def _resolve(path):
    """Return ``path`` if it exists, else ``GT_DATA_ROOT / path``.

    Raises:
        FileNotFoundError: If neither location exists.
    """
    p = Path(path).expanduser()
    if p.exists():
        return p
    if GT_DATA_ROOT:
        q = Path(GT_DATA_ROOT).expanduser() / path
        if q.exists():
            return q
    raise FileNotFoundError(f"GT mode: image not found: {path}")


def gt_mask_path(image_path):
    """Return the ground-truth mask path and source dataset for an image.

    The source is inferred from the file name and from substrings of the full
    path (``naip``, ``deepglobe``, ``solar``). Everything else is EarthVQA.

    Args:
        image_path (str | Path): Image file path.

    Returns:
        tuple[Path, str]: Mask path and source, one of ``'deepglobe'``,
        ``'solar'`` or ``'earthvqa'``.

    Raises:
        ValueError: If the image is a NAIP image, which has no mask.
    """
    p = Path(image_path)
    s = str(p).lower()
    if 'naip' in s:
        raise ValueError(f"GT mode: NAIP images have no ground-truth masks: {p.name}")
    if p.stem.endswith('_sat') or 'deepglobe' in s:
        return p.with_name(p.stem.replace('_sat', '') + '_label.png'), 'deepglobe'
    if p.suffix.lower() == '.bmp' or 'solar' in s:
        return p.with_name(p.stem + '_label.bmp'), 'solar'
    return p.with_name(p.stem + '_label2.png'), 'earthvqa'


def _load_label(mask_path, source):
    """Read a label image (RGB for DeepGlobe, gray for Solar, BGR otherwise).

    Only the most recently loaded label is cached.
    """
    key = str(mask_path)
    if key in _mask_cache:
        return _mask_cache[key]
    img = cv2.imread(str(mask_path))
    if img is None:
        raise FileNotFoundError(f"GT mode: mask not found: {mask_path}")
    if source == 'deepglobe':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif source == 'solar':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _mask_cache.clear()          # keep only the current image's label in memory
    _mask_cache[key] = img
    return img


def _class_mask(label, source, class_name):
    """Return the boolean mask of one ground-truth class in a label image."""
    if source == 'deepglobe':
        color = DEEPGLOBE_CLASSES[class_name]
        lower = np.array([c - 10 for c in color])
        upper = np.array([c + 10 for c in color])
        return cv2.inRange(label, lower, upper) > 0
    if source == 'earthvqa':
        b, g, r = EARTHVQA_CLASSES[class_name]
        return (label[:, :, 0] == b) & (label[:, :, 1] == g) & (label[:, :, 2] == r)
    return label > 0             # solar: any foreground pixel


def segment_image_with_gt(image_input, topics, min_area_pixels=0, gsd=1.0):
    """Segment an image by reading its ground-truth mask.

    Same contract as ``segment_image_from_path``. Each QASR topic is mapped to
    zero or more ground-truth classes of the image's source dataset. Topics
    without a class yield an empty mask and no shapes, which is the correct
    oracle answer for feature-absence questions. Also fills ``LAST_GT_MASKS``.

    Args:
        image_input (str | Path): Image path. PIL images are not supported.
        topics (str | list[str]): Topic or list of topics.
        min_area_pixels (int): Minimum component size in pixels.
        gsd (float): Ground sample distance in meters per pixel.

    Returns:
        dict: ``shapes``, ``image_width``, ``image_height`` and
        ``total_pixels``, as in ``segment_image_from_path``.

    Raises:
        ValueError: If ``image_input`` is not a path, or is a NAIP image.
        FileNotFoundError: If the image or its mask cannot be found.
    """
    if not isinstance(image_input, (str, Path)):
        raise ValueError("GT mode requires a file path, got " + type(image_input).__name__)
    if isinstance(topics, str):
        topics = [topics]

    image_path = _resolve(image_input)
    mask_path, source = gt_mask_path(image_path)
    label = _load_label(mask_path, source)

    topic_map = {'deepglobe': DEEPGLOBE_TOPICS, 'earthvqa': EARTHVQA_TOPICS,
                 'solar': SOLAR_TOPICS}[source]

    h, w = label.shape[:2]
    LAST_GT_MASKS.clear()
    all_shapes = []
    shape_id = 0
    for topic in topics:
        topic_mask = np.zeros((h, w), dtype=bool)
        for cls in topic_map.get(topic, []):
            topic_mask |= _class_mask(label, source, cls)
        LAST_GT_MASKS[topic] = topic_mask
        shapes = mask_to_shapes(topic_mask, topic, min_area_pixels, gsd, shape_id)
        all_shapes.extend(shapes)
        shape_id += len(shapes)

    return {
        'shapes': all_shapes,
        'image_width': w,
        'image_height': h,
        'total_pixels': w * h,
    }
