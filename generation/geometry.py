"""Mask geometry shared by all question generators.

Areas are exact pixel counts of 8-connected components (holes are simply not
class pixels, so they are excluded), converted to hectares as
``pixels * gsd**2 / 1e4``. Distances use a Euclidean distance transform. This
is the same geometry as ``evaluation/eval_shapes.py``, so ground-truth answers
and QASR measurements are computed identically.
"""

import cv2
import numpy as np
from scipy import ndimage


def read_rgb(path):
    """Reads a colour mask as an RGB array.

    Args:
        path: Path to the mask image.

    Returns:
        A ``(H, W, 3)`` uint8 array in RGB order.
    """
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def color_mask(rgb, color, tol=10):
    """Selects the pixels whose colour is within ``tol`` of ``color``.

    Args:
        rgb: ``(H, W, 3)`` RGB array.
        color: Target RGB triplet.
        tol: Per-channel tolerance, absorbing JPEG-style colour noise.

    Returns:
        A boolean ``(H, W)`` mask.
    """
    lower = np.array([c - tol for c in color])
    upper = np.array([c + tol for c in color])
    return cv2.inRange(rgb, lower, upper) > 0


def component_areas(mask, gsd=None):
    """Returns the area of every 8-connected component of ``mask``.

    Args:
        mask: Boolean class mask.
        gsd: Ground sampling distance in metres per pixel. If given, areas are
            in hectares, otherwise in pixels.

    Returns:
        A list of component areas, in label order.
    """
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel(), minlength=n)[1:].astype(float)
    return (sizes * (gsd ** 2) / 10000).tolist() if gsd is not None else sizes.tolist()


def coverage_percentage(mask):
    """Returns the percentage of the image covered by ``mask``."""
    total_pixels = mask.shape[0] * mask.shape[1]
    return (np.sum(mask) / total_pixels) * 100


def distance_buffer(mask, distance_px):
    """Returns the pixels within ``distance_px`` of ``mask`` (mask included)."""
    return ndimage.distance_transform_edt(~mask) <= distance_px


def clipped_area(target, buffer, gsd):
    """Returns the area of ``target`` inside ``buffer``, in hectares."""
    return int(np.sum(target & buffer)) * (gsd ** 2) / 10000


def _components(mask, gsd):
    """Labels the components of ``mask`` and measures them in hectares.

    Returns:
        ``(labels, n_labels, area_per_label, hectares_per_pixel)``. Label 0 is
        the background.
    """
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    counts = np.bincount(labels.ravel(), minlength=n).astype(float)
    to_area = (gsd ** 2) / 10000
    return labels, n, counts * to_area, to_area


def filtered_clipped_area(mask, near, distance_m, min_area, gsd):
    """Area of the large patches of ``mask`` that lies within reach of ``near``.

    Only components of ``mask`` larger than ``min_area`` count, and of each one
    only the pixels within ``distance_m`` of ``near``.

    Args:
        mask: Boolean mask of the measured class.
        near: Boolean mask of the reference class.
        distance_m: Buffer distance in metres.
        min_area: Minimum component area in hectares (strict).
        gsd: Ground sampling distance in metres per pixel.

    Returns:
        The clipped area in hectares.
    """
    labels, n, area, to_area = _components(mask, gsd)
    inside = np.bincount(labels[distance_buffer(near, distance_m / gsd)], minlength=n).astype(float)
    return float(sum(inside[k] * to_area for k in range(1, n) if area[k] > min_area))


def filtered_clipped_count(mask, near, distance_m, min_area, gsd):
    """Counts the patches of ``mask`` larger than ``min_area`` touching the buffer.

    A component counts when any of its pixels is within ``distance_m`` of
    ``near``. Arguments are as in :func:`filtered_clipped_area`.

    Returns:
        The number of such components.
    """
    labels, n, area, _ = _components(mask, gsd)
    inside = np.bincount(labels[distance_buffer(near, distance_m / gsd)], minlength=n)
    return int(sum(1 for k in range(1, n) if area[k] > min_area and inside[k] > 0))
