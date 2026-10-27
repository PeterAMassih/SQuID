"""Question generator for the solar-panel source (PV segmentation, 0.3 m).

Solar masks hold a single class, so the questions are about installations:
coverage, counts and sizes (Tier 1), power output and size bins (Tier 2), and
utility-scale filtering and averages (Tier 3). Coverage and power use every
panel pixel. Counts and sizes ignore installations below ``MIN_AREA``.
"""

import random

import cv2
import numpy as np

import geometry

MIN_AREA = 0.01
WATTS_PER_M2 = 200
UTILITY_SCALE_HA = 5


class SolarQA:
    """Generates the solar questions, one mask at a time.

    Attributes:
        questions: All questions generated so far.
    """

    MASK_SUFFIX = '_label.bmp'
    GSD = 0.3

    def __init__(self):
        self.questions = []

    def process(self, mask_path):
        """Generates all questions for one mask.

        Args:
            mask_path: Path to a ``<scene>_label.bmp`` (or ``.png``) mask.

        Returns:
            True. Images without panels still get zero-answer questions.

        Raises:
            IOError: If the mask cannot be read.
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"cannot read mask {mask_path}")
        panels = mask > 0
        image = f"{mask_path.stem.replace('_label', '')}.bmp"

        def add(question, answer, q_type, tier):
            self.questions.append({'image': image, 'question': f"{question} (GSD: {self.GSD}m)",
                                   'answer': answer, 'type': q_type, 'tier': tier})

        count_q = ("How many solar panel installations are there? "
                   f"When counting, ignore installations smaller than {MIN_AREA} hectares.")
        presence_q = f"Are there any solar panels larger than {MIN_AREA} hectares in this image?"
        percentage = geometry.coverage_percentage(panels)

        if percentage < 0.01:
            add("What percentage of the image is solar panels?", 0.0, 'percentage', 1)
            add(count_q, 0, 'count', 1)
            add(presence_q, 0, 'binary_presence', 1)
            return True

        sizes = [s for s in geometry.component_areas(panels, self.GSD) if s >= MIN_AREA]
        add("What percentage of the image is solar panels?", round(percentage, 2), 'percentage', 1)
        add(count_q, len(sizes), 'count', 1)

        total_area = 0
        if sizes:
            add("What is the area of the largest solar installation in hectares?",
                round(max(sizes), 2), 'size', 1)
            if len(sizes) > 1:
                add("What is the area of the smallest solar installation in hectares "
                    f"(excluding installations smaller than {MIN_AREA} hectares)?",
                    round(min(sizes), 2), 'size', 1)
            total_area = sum(sizes)
            add(f"What is the total solar panel area in hectares (excluding installations smaller than {MIN_AREA} hectares)?",
                round(total_area, 2), 'total_area', 1)

        area_m2 = np.sum(panels) * (self.GSD ** 2)
        add(f"Calculate the solar potential MW output assuming {WATTS_PER_M2}W/m² efficiency.",
            round(area_m2 * WATTS_PER_M2 / 1_000_000, 2), 'power_calculation', 2)
        add(presence_q, 1 if sizes else 0, 'binary_presence', 1)
        add(f"Is there more than 1 hectare of solar panels (excluding installations smaller than {MIN_AREA} hectares)?",
            1 if total_area > 1 else 0, 'binary_threshold', 1)
        add(f"Are there multiple separate solar installations larger than {MIN_AREA} hectares?",
            1 if len(sizes) > 1 else 0, 'binary_multiple', 1)

        # Drawn for every image with panels, so the RNG stream does not depend on sizes.
        upper = random.choice([0.5, 1, 2])
        if len(sizes) > 1:
            add(f"How many solar installations between {MIN_AREA} and {upper} hectares are there?",
                sum(1 for s in sizes if s < upper), 'connectivity', 2)
        # Asked only when a utility-scale panel is possible, else the answer is a forced 0.
        if total_area > UTILITY_SCALE_HA:
            add(f"What is the total area (in hectares) of solar installations larger than "
                f"{UTILITY_SCALE_HA} hectares (utility-scale)?",
                round(sum(s for s in sizes if s > UTILITY_SCALE_HA), 2), 'complex_size_filter', 3)
        if len(sizes) > 1:
            add("What is the average size of solar installations in hectares "
                f"(excluding installations smaller than {MIN_AREA} hectares)?",
                round(total_area / len(sizes), 2), 'complex_average', 3)
        return True
