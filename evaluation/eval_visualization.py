"""Segmentation overlay rendering shared by the evaluation backends."""

import colorsys

import cv2
import numpy as np
from PIL import Image


def create_visualization(image_bgr, segmentation_result, output_path):
    """Save the image with its segmented shapes overlaid and a topic legend.

    Each topic gets a distinct hue. Holes are painted white before blending.
    A legend panel listing the topics is appended to the right of the image
    when at least one shape is present.

    Args:
        image_bgr (np.ndarray): Input image (H, W, 3) in BGR order.
        segmentation_result (dict): Output of a ``segment_image_from_path``
            backend. Only its ``shapes`` entry is used.
        output_path (str | Path): Destination image file.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    shapes = segmentation_result.get("shapes", [])

    classes = list(set(s['class_type'] for s in shapes))
    colors = {}
    for i, cls in enumerate(classes):
        hue = i / max(len(classes), 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[cls] = tuple(int(c * 255) for c in rgb)

    for shape in shapes:
        poly = np.array(shape['polygon']).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [poly], colors[shape['class_type']])

        if shape.get('holes'):
            for hole in shape['holes']:
                hole_pts = np.array(hole).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [hole_pts], (255, 255, 255))

    result = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    if classes:
        legend_height = 30 * len(classes) + 20
        legend_width = 200

        h, w = result.shape[:2]
        final_width = w + legend_width
        final_height = max(h, legend_height)
        final = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255

        final[:h, :w] = result

        for i, cls in enumerate(sorted(classes)):
            y = 30 + i * 30
            cv2.rectangle(final, (w + 10, y - 10), (w + 30, y + 10), colors[cls], -1)
            cv2.rectangle(final, (w + 10, y - 10), (w + 30, y + 10), (0, 0, 0), 1)
            cv2.putText(final, cls, (w + 40, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.putText(final, "Topics", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        Image.fromarray(final).save(output_path)
    else:
        Image.fromarray(result).save(output_path)
