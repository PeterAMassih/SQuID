"""Shape extraction and the geometry API exposed to the code-generating LLM.

``mask_to_shapes`` turns a binary mask into the shape dicts every segmentation
backend returns. ``find_shapes_within_distance`` and
``calculate_shape_distances`` are the two spatial primitives the LLM may call.
Their docstrings are inserted verbatim into the LLM prompt by
``build_api_docs`` and must not be edited without re-running the evaluation.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _trace_component(labels, stats, lbl):
    """Trace the outer contour and holes of one connected component.

    Contours are traced on the component's bounding-box crop and offset back
    to full-image coordinates.

    Args:
        labels (np.ndarray): Label image from ``cv2.connectedComponentsWithStats``.
        stats (np.ndarray): Per-label statistics from the same call.
        lbl (int): Label of the component to trace.

    Returns:
        tuple[list[float], list[list[float]]] | None: Flattened outer polygon
        ``[x1, y1, x2, y2, ...]`` and a (possibly empty) list of flattened hole
        polygons, or None if OpenCV finds no contour.
    """
    x, y, w, h = (int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP]),
                  int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT]))
    comp = (labels[y:y + h, x:x + w] == lbl).astype(np.uint8)
    contours, hierarchy = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    outer_idx = next(i for i in range(len(contours)) if hierarchy[i][3] == -1)

    polygon = []
    for pt in contours[outer_idx]:
        polygon.extend([float(pt[0][0] + x), float(pt[0][1] + y)])

    holes = []
    child_idx = hierarchy[outer_idx][2]
    while child_idx != -1:
        hole = []
        for pt in contours[child_idx]:
            hole.extend([float(pt[0][0] + x), float(pt[0][1] + y)])
        holes.append(hole)
        child_idx = hierarchy[child_idx][0]

    return polygon, holes


def mask_to_shapes(mask, class_type, min_area_pixels, gsd, start_id):
    """Convert a binary class mask to shape dicts.

    This is the single geometry function shared by all segmentation backends.
    The area of a shape is the exact pixel count of its 8-connected component.
    Holes are background pixels and are therefore excluded. This matches the
    area definition used by the SQuID ground-truth labeler. Contour tracing
    only fills the ``polygon`` and ``holes`` fields.

    Args:
        mask (np.ndarray): Binary mask (H, W). Any non-zero pixel is foreground.
        class_type (str): Topic/class name for all shapes from this mask.
        min_area_pixels (int): Minimum component size in pixels.
        gsd (float): Ground sample distance in meters per pixel.
        start_id (int): First shape id.

    Returns:
        list[dict]: One dict per component with keys ``id``, ``class_type``,
        ``area_pixels``, ``area_m2``, ``area_hectares``, ``center``,
        ``polygon`` and ``holes``.
    """
    shapes = []
    shape_id = start_id

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)

    for lbl in range(1, n_labels):
        pixel_area = int(stats[lbl, cv2.CC_STAT_AREA])
        if pixel_area < min_area_pixels:
            continue

        traced = _trace_component(labels, stats, lbl)
        if traced is None:
            continue
        polygon, holes = traced

        cx, cy = centroids[lbl]           # pixel-mass centroid (consistent with area)

        shapes.append({
            'id': shape_id,
            'class_type': class_type,
            'area_pixels': float(pixel_area),
            'area_m2': float(pixel_area) * (gsd ** 2),
            'area_hectares': float(pixel_area) * (gsd ** 2) / 10000,
            'center': [float(cx), float(cy)],
            'polygon': polygon,
            'holes': holes if holes else None
        })
        shape_id += 1

    return shapes


def get_shape_geometries(shapes, buffer_pixels=0):
    """Convert shape dicts to Shapely geometries.

    Not exposed to the LLM. Holes are subtracted, invalid polygons are repaired
    with ``buffer(0)``, and shapes without a polygon or whose geometry cannot be
    built are skipped silently.

    Args:
        shapes (list[dict]): Shape dicts with ``polygon`` and optional ``holes``.
        buffer_pixels (float): Outward buffer applied to each geometry, in
            pixels. No buffer is applied when it is not positive.

    Returns:
        list[dict]: One dict per successfully converted shape with keys
        ``geometry`` (a Shapely geometry) and ``original_shape``.
    """
    geometries = []
    for shape in shapes:
        if not shape.get("polygon"):
            continue

        try:
            coords = shape["polygon"]
            poly_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            poly = Polygon(poly_points)

            if shape.get("holes"):
                for hole in shape["holes"]:
                    hole_points = [(hole[i], hole[i+1]) for i in range(0, len(hole), 2)]
                    poly = poly.difference(Polygon(hole_points))

            # Repair self-intersections and other invalid geometries.
            if not poly.is_valid:
                poly = poly.buffer(0)

            if buffer_pixels > 0:
                poly = poly.buffer(buffer_pixels)

            geometries.append({
                "geometry": poly,
                "original_shape": shape
            })

        except Exception:
            continue

    return geometries


def find_shapes_within_distance(target_shapes, reference_shapes, distance_meters, resolution):
    """
    Find portions of target_shapes that fall within distance_meters of any reference_shape.

    This function CUTS/CLIPS target shapes to only return the portions that are within
    the specified distance of reference shapes. Shapes completely outside the distance
    are not included in the results.

    Distances and areas are computed on the pixel grid (rasterized shapes, exact
    euclidean distance transform, pixel-count areas) -- the same definitions as the
    SQuID GT labeler. Holes are background pixels, so they are excluded automatically.

    Args:
        target_shapes (list[dict]): List of shape dicts from segment_image_from_path.
            Each dict must have: polygon, class_type, id, area_pixels fields.
        reference_shapes (list[dict]): List of shape dicts to measure distance from.
            Each dict must have: polygon field.
        distance_meters (float): Buffer distance in meters.
        resolution (float): Ground sample distance in meters per pixel (gsd).

    Returns:
        list[dict]: List of NEW shape dicts representing ONLY the portions of target_shapes
        that fall within distance_meters of any reference_shape. Empty list if no intersections.
        Each returned dict contains (matching segment_image_from_path structure):
            - id (int): New unique shape identifier
            - class_type (str): Original class name (same as input target shape)
            - area_pixels (float): Area of the CLIPPED portion in pixels
            - area_m2 (float): Area of the CLIPPED portion in square meters
            - area_hectares (float): Area of the CLIPPED portion in hectares
            - center (list[float, float]): [x, y] centroid of the CLIPPED portion
            - polygon (list[float]): Flattened list of coordinates [x1, y1, x2, y2, ...]
            - holes (list[list[float]] | None): List of hole polygons (usually None for clipped shapes)

    Example:
        If a forest shape is 1000 pixels but only 300 pixels fall within 100m of water,
        the returned shape will have area_pixels=300 (not 1000).
    """
    if not target_shapes or not reference_shapes:
        return []

    buffer_pixels = distance_meters / resolution

    # Canvas covering every polygon plus the buffer radius.
    max_x = max_y = 0.0
    for s in list(target_shapes) + list(reference_shapes):
        poly = s.get("polygon") or []
        if poly:
            max_x = max(max_x, max(poly[0::2]))
            max_y = max(max_y, max(poly[1::2]))
    W = int(np.ceil(max_x + buffer_pixels)) + 2
    H = int(np.ceil(max_y + buffer_pixels)) + 2

    def _rasterize_one(shape, canvas):
        # Fill the shape (minus its holes) on an (H, W) canvas. Optionally OR it
        # into ``canvas``. Returns None for shapes without a polygon.
        poly = shape.get("polygon")
        if not poly:
            return None
        m = np.zeros((H, W), dtype=np.uint8)
        pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        for hole in (shape.get("holes") or []):
            hp = np.round(np.asarray(hole, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(m, [hp], 0)
        if canvas is not None:
            np.logical_or(canvas, m, out=canvas)
        return m

    # Buffer zone: pixels within buffer_pixels of ANY reference pixel (exact EDT).
    ref_mask = np.zeros((H, W), dtype=bool)
    for s in reference_shapes:
        _rasterize_one(s, ref_mask)
    if not ref_mask.any():
        return []
    dist = cv2.distanceTransform((~ref_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    buffer_mask = dist <= buffer_pixels

    clipped_shapes = []
    shape_id = 0

    for s in target_shapes:
        tgt = _rasterize_one(s, None)
        if tgt is None:
            continue
        inter = (tgt.astype(bool) & buffer_mask).astype(np.uint8)
        if not inter.any():
            continue

        # Clipping can split one shape into multiple pieces: one output shape per
        # 8-connected piece, with pixel-count area and traced polygon geometry.
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inter, connectivity=8)
        for lbl in range(1, n_labels):
            pixel_area = int(stats[lbl, cv2.CC_STAT_AREA])
            traced = _trace_component(labels, stats, lbl)
            if traced is None:
                continue
            polygon, holes = traced

            cx, cy = centroids[lbl]

            clipped_shapes.append({
                'id': shape_id,
                'class_type': s['class_type'],
                'area_pixels': float(pixel_area),
                'area_m2': float(pixel_area) * resolution**2,
                'area_hectares': float(pixel_area) * resolution**2 / 10000,
                'center': [float(cx), float(cy)],
                'polygon': polygon,
                'holes': holes if holes else None
            })
            shape_id += 1

    return clipped_shapes


def calculate_shape_distances(target_shapes, reference_shapes, resolution):
    """
    Calculate minimum distance from each target shape to nearest reference shape.
    
    This function MODIFIES target_shapes in place by adding a distance_meters field.
    All target shapes are returned, even those far from reference shapes.
    
    Args:
        target_shapes (list[dict]): List of shape dicts from segment_image_from_path.
            Each dict must have: polygon field.
        reference_shapes (list[dict]): List of shape dicts to measure distance to.
            Each dict must have: polygon field.
        resolution (float): Ground sample distance in meters per pixel (gsd).
    
    Returns:
        list[dict]: The SAME target_shapes list with added field:
            - distance_meters (float): Minimum distance in meters to nearest reference shape.
            All original fields are preserved unchanged.
    """
    if not target_shapes or not reference_shapes:
        return target_shapes

    ref_geoms = get_shape_geometries(reference_shapes)
    if not ref_geoms:
        return target_shapes

    ref_union = unary_union([g["geometry"] for g in ref_geoms])

    target_geoms = get_shape_geometries(target_shapes)

    for i, target_geom in enumerate(target_geoms):
        distance_pixels = target_geom["geometry"].distance(ref_union)
        target_shapes[i]["distance_meters"] = float(distance_pixels * resolution)

    return target_shapes
