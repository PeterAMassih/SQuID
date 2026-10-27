"""Question generators for the land-cover sources, DeepGlobe and EarthVQA.

Both sources share the same three tiers of questions:

* Tier 1: coverage percentages, patch counts, largest/smallest patch, class
  comparisons, plus zero-answer questions about two absent classes.
* Tier 2: proximity (percentage, area and presence of class A within d metres
  of class B), patch-size connectivity and fragmentation.
* Tier 3: area of the patches of A larger than s hectares that lies within d
  metres of B, plus source-specific risk scenarios.

EarthVQA additionally has a ``building`` object class, which is asked about with
counting questions instead of coverage questions.

Draws from :mod:`random` happen in a fixed order, so a seeded run is
reproducible question for question.
"""

import random

import cv2
import numpy as np

import geometry

DISTANCES_M = [50, 100, 200, 500]


class LandCoverQA:
    """Generates the questions of one land-cover source, one mask at a time.

    Subclasses set the class vocabulary and how masks are decoded, and may add
    source-specific questions through the ``_object_questions`` and
    ``_scenario_questions`` hooks.

    Attributes:
        TERMS: Class name -> phrase used in the question text. Its order is the
            order in which absent classes are considered.
        MIN_AREA: Class name -> minimum patch area in hectares. Smaller patches
            are ignored, matching the human annotation instructions.
        MASK_SUFFIX: File-name suffix of the masks to process.
        GSD: Ground sampling distance in metres per pixel.
        questions: All questions generated so far.
    """

    TERMS = {}
    MIN_AREA = {}
    MASK_SUFFIX = ''
    GSD = None

    def __init__(self):
        self.questions = []
        self._image = None
        self._masks = None

    def image_file(self, mask_path):
        """Returns the image file name that goes with ``mask_path``."""
        raise NotImplementedError

    def read_class_masks(self, mask_path):
        """Decodes a mask into one boolean layer per present class.

        Returns:
            Ordered dict, class name -> boolean mask. The order fixes the
            question order.
        """
        raise NotImplementedError

    def landcover_classes(self):
        """Returns the present classes that are areal (not countable objects)."""
        return list(self._masks)

    def process(self, mask_path):
        """Generates all questions for one mask.

        Args:
            mask_path: Path to the mask.

        Returns:
            False if no class is present (the mask is skipped), else True.
        """
        self._masks = self.read_class_masks(mask_path)
        if not self._masks:
            return False
        self._image = self.image_file(mask_path)
        self._basic_questions()
        self._spatial_questions()
        self._complex_questions()
        return True

    def _add(self, question, answer, q_type, tier):
        self.questions.append({
            'image': self._image,
            'question': f"{question} (GSD: {self.GSD}m)",
            'answer': answer,
            'type': q_type,
            'tier': tier,
        })

    def _filtered_sizes(self, name):
        sizes = geometry.component_areas(self._masks[name], self.GSD)
        return [s for s in sizes if s >= self.MIN_AREA[name]]

    def _region_percentage(self, mask, area_ha):
        return round((area_ha * 10000) / (mask.shape[0] * mask.shape[1] * self.GSD ** 2) * 100, 2)

    # Tier 1.

    def _basic_questions(self):
        for name in self._masks:
            self._class_questions(name)

        absent = [c for c in self.TERMS if c not in self._masks]
        for name in random.sample(absent, min(2, len(absent))):
            self._absent_class_questions(name)

        classes = self.landcover_classes()
        for i, a in enumerate(classes):
            for b in classes[i + 1:]:
                pixels_a, pixels_b = np.sum(self._masks[a]), np.sum(self._masks[b])
                if pixels_a > 0 and pixels_b > 0:
                    self._add(f"Is there more {self.TERMS[a]} than {self.TERMS[b]} in this image?",
                              1 if pixels_a > pixels_b else 0, 'binary_comparison', 1)

    def _class_questions(self, name):
        mask, term, min_area = self._masks[name], self.TERMS[name], self.MIN_AREA[name]
        sizes = self._filtered_sizes(name)

        percentage = geometry.coverage_percentage(mask)
        if percentage > 0.01:
            self._add(f"What percentage of the image is covered by {term}?",
                      round(percentage, 2), 'percentage', 1)
        self._add(f"How many separate {term} regions are there? "
                  f"When counting, ignore patches smaller than {min_area} hectares.",
                  len(sizes), 'count', 1)
        if sizes:
            self._add(f"What percentage of the image is covered by the largest {term} region "
                      f"(among regions larger than {min_area} hectares)?",
                      self._region_percentage(mask, max(sizes)), 'region_percentage', 1)
            if len(sizes) > 1:
                self._add(f"What percentage of the image is covered by the smallest {term} region "
                          f"(excluding patches smaller than {min_area} hectares)?",
                          self._region_percentage(mask, min(sizes)), 'region_percentage', 1)

    def _absent_class_questions(self, name):
        term, min_area = self.TERMS[name], self.MIN_AREA[name]
        self._add(f"What percentage of the image is covered by {term}?", 0.0, 'percentage', 1)
        self._add(f"How many separate {term} regions are there? "
                  f"When counting, ignore patches smaller than {min_area} hectares.",
                  0, 'count', 1)

    # Tier 2.

    def _spatial_questions(self):
        self._object_questions()
        classes = self.landcover_classes()

        for a in classes:
            for b in classes:
                if a == b:
                    continue
                mask_a = self._masks[a]
                for distance in random.sample(DISTANCES_M, 2):
                    buffer = geometry.distance_buffer(self._masks[b], distance / self.GSD)
                    overlap = mask_a & buffer
                    percentage = (np.sum(overlap) / (mask_a.shape[0] * mask_a.shape[1])) * 100
                    if 0 < percentage < 100:
                        self._add(f"What percentage of the image is {self.TERMS[a]} "
                                  f"within {distance}m of {self.TERMS[b]}?",
                                  round(percentage, 2), 'proximity_percentage', 2)
                    area = geometry.clipped_area(mask_a, buffer, self.GSD)
                    if area > 0.01:
                        self._add(f"What is the total {self.TERMS[a]} area (in hectares) "
                                  f"within {distance}m of {self.TERMS[b]}?",
                                  round(area, 2), 'proximity_area', 2)
                    self._add(f"Is there any {self.TERMS[a]} within {distance}m of {self.TERMS[b]}?",
                              1 if np.sum(overlap) > 0 else 0, 'binary_proximity', 2)

        for name in classes:
            term, min_area = self.TERMS[name], self.MIN_AREA[name]
            sizes = geometry.component_areas(self._masks[name], self.GSD)
            if len(sizes) > 1:
                upper = random.choice([3, 5, 10])
                self._add(f"How many separate {term} patches between {min_area} and {upper} hectares are there?",
                          sum(1 for s in sizes if min_area <= s < upper), 'connectivity', 2)
                n_large = sum(1 for s in sizes if s >= min_area)
                self._add(f"Is the {term} connected or fragmented "
                          f"(more than 5 separate patches larger than {min_area} hectares)?",
                          "fragmented" if n_large > 5 else "connected", 'fragmentation', 2)

    def _object_questions(self):
        """Hook for object-class questions, asked before the proximity ones."""

    # Tier 3.

    def _complex_questions(self):
        classes = self.landcover_classes()
        for a in classes:
            for b in classes:
                if a == b:
                    continue
                mask_a, mask_b = self._masks[a], self._masks[b]
                if np.sum(mask_a) < 1000 or np.sum(mask_b) < 1000:
                    continue
                size_threshold = random.choice([1, 5])
                distance = random.choice([200, 500])
                # No patch can exceed the class total: the answer would be a forced 0.
                if np.sum(mask_a) * self.GSD ** 2 / 10000 <= size_threshold:
                    continue
                area = geometry.filtered_clipped_area(mask_a, mask_b, distance, size_threshold, self.GSD)
                self._add(f"Find {self.TERMS[a]} patches larger than {size_threshold} hectares, then calculate "
                          f"how much of their area (in hectares) falls within {distance}m of {self.TERMS[b]}",
                          round(area, 2), 'complex_multi_condition', 3)
        self._scenario_questions()

    def _scenario_questions(self):
        """Hook for source-specific Tier-3 scenarios (flood risk, irrigation...)."""

    def _scenario(self, a, b, distance, min_area, question, q_type):
        """Adds one Tier-3 filter-and-clip question when both classes are present."""
        if a in self._masks and b in self._masks and np.sum(self._masks[a]) > 0 and np.sum(self._masks[b]) > 0:
            area = geometry.filtered_clipped_area(self._masks[a], self._masks[b], distance, min_area, self.GSD)
            self._add(question, round(area, 2), q_type, 3)


class DeepGlobeQA(LandCoverQA):
    """DeepGlobe Land Cover (0.5 m). Agriculture, rangeland and forest are
    merged into one ``vegetation`` class, whose members are spelled out in every
    question so that models are not scored on guessing the vocabulary."""

    PALETTE = {
        'urban': [0, 255, 255],
        'agriculture': [255, 255, 0],
        'rangeland': [255, 0, 255],
        'forest': [0, 255, 0],
        'water': [0, 0, 255],
        'barren': [255, 255, 255],
    }
    VEGETATION = ['agriculture', 'rangeland', 'forest']
    TERMS = {
        'urban': 'urban area',
        'vegetation': 'vegetation (forest, agricultural, or rangeland)',
        'water': 'water bodies',
        'barren': 'barren land',
    }
    MIN_AREA = {'urban': 0.1, 'vegetation': 0.125, 'water': 0.1, 'barren': 0.125}
    MASK_SUFFIX = '_label.png'
    GSD = 0.5

    def image_file(self, mask_path):
        return f"{mask_path.stem.replace('_mask', '').replace('_label', '')}_sat.jpg"

    def read_class_masks(self, mask_path):
        rgb = geometry.read_rgb(mask_path)
        layers = {name: geometry.color_mask(rgb, color) for name, color in self.PALETTE.items()}
        masks = {}
        if any(np.sum(layers[c]) > 100 for c in self.VEGETATION):
            masks['vegetation'] = layers['agriculture'] | layers['rangeland'] | layers['forest']
        for name in ['urban', 'water', 'barren']:
            if np.sum(layers[name]) > 100:
                masks[name] = layers[name]
        return masks

    def _scenario_questions(self):
        vegetation, water = self.TERMS['vegetation'], self.TERMS['water']
        self._scenario('urban', 'water', 100, 1,
                       "Find urban patches larger than 1 hectare, then calculate how much of their area "
                       "(in hectares) falls within 100m of water bodies (flood risk assessment)",
                       'complex_urban_flood_risk')
        self._scenario('urban', 'vegetation', 50, 1,
                       "Find urban patches larger than 1 hectare, then calculate how much of their area "
                       f"(in hectares) falls within 50m of {vegetation} (fire risk assessment)",
                       'complex_urban_fire_risk')
        self._scenario('vegetation', 'water', 200, 2,
                       f"Find {vegetation} patches larger than 2 hectares, then calculate how much of their area "
                       f"(in hectares) falls within 200m of {water}",
                       'complex_vegetation_water_access')


class EarthVQAQA(LandCoverQA):
    """EarthVQA (0.3 m), whose ``_label2`` masks include a ``building`` class.

    Buildings are counted (total, near water, near forest, near each land-cover
    class) and are left out of the areal questions.
    """

    PALETTE_BGR = {
        'building': (0, 0, 255),
        'water': (255, 0, 0),
        'barren': (255, 0, 255),
        'forest': (0, 255, 0),
        'agriculture': (255, 255, 0),
    }
    TERMS = {
        'building': 'buildings',
        'water': 'water bodies',
        'barren': 'barren land',
        'forest': 'forest area',
        'agriculture': 'agricultural land',
    }
    MIN_AREA = {'building': 0.01, 'water': 0.1, 'barren': 0.125, 'forest': 0.125, 'agriculture': 0.125}
    MASK_SUFFIX = '_label2.png'
    GSD = 0.3

    def image_file(self, mask_path):
        return f"{mask_path.stem.replace('_label2', '')}.png"

    def read_class_masks(self, mask_path):
        bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        masks = {}
        for name, color in self.PALETTE_BGR.items():
            mask = np.all(bgr == np.array(color, dtype=bgr.dtype), axis=-1)
            if np.sum(mask) > 100:
                masks[name] = mask
        return masks

    def landcover_classes(self):
        return [c for c in self._masks if c != 'building']

    def _building_count_question(self, count):
        min_area = self.MIN_AREA['building']
        self._add(f"How many buildings are there in the image? "
                  f"When counting, ignore buildings smaller than {min_area} hectares.",
                  count, 'count', 1)

    def _class_questions(self, name):
        if name == 'building':
            self._building_count_question(len(self._filtered_sizes(name)))
        else:
            super()._class_questions(name)

    def _absent_class_questions(self, name):
        if name == 'building':
            self._building_count_question(0)
        else:
            super()._absent_class_questions(name)

    def _object_questions(self):
        if 'building' not in self._masks:
            return
        buildings, min_area = self._masks['building'], self.MIN_AREA['building']

        def count_near(name, distance):
            return geometry.filtered_clipped_count(buildings, self._masks[name], distance, min_area, self.GSD)

        for name, distance, q_type, risk in [('water', 100, 'building_flood_risk', 'flood'),
                                             ('forest', 50, 'building_fire_risk', 'fire')]:
            if name in self._masks and np.sum(self._masks[name]) > 0:
                self._add(f"How many buildings (larger than {min_area} hectares) are located within "
                          f"{distance}m of {self.TERMS[name]} ({risk} risk assessment)?",
                          count_near(name, distance), q_type, 2)

        for name in self.landcover_classes():
            if np.sum(self._masks[name]) == 0:
                continue
            for distance in random.sample(DISTANCES_M, 2):
                self._add(f"How many buildings (larger than {min_area} hectares) are within "
                          f"{distance}m of {self.TERMS[name]}?",
                          count_near(name, distance), 'building_proximity', 2)

    def _scenario_questions(self):
        self._scenario('agriculture', 'water', 200, 2,
                       f"Find {self.TERMS['agriculture']} patches larger than 2 hectares, then calculate how much "
                       f"of their area (in hectares) falls within 200m of {self.TERMS['water']}",
                       'complex_agriculture_water_access')
