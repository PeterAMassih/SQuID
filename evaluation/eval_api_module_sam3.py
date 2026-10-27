"""API exposed to the code-generating LLM for the SAM3 agent.

Same as ``eval_api_module`` except that ``segment_image_from_path`` comes from
``eval_sam3_segmentation``. Since SAM3 is open-vocabulary, ``EVAL_DOMAIN_INFO``
lists example prompts instead of a closed topic set.
"""

import inspect
import json

from .eval_sam3_segmentation import segment_image_from_path
from .eval_shapes import find_shapes_within_distance, calculate_shape_distances


api_functions = [
    segment_image_from_path,
    find_shapes_within_distance,
    calculate_shape_distances,
]


def build_api_docs():
    """Render ``api_functions`` as text for the LLM prompt.

    Identical to ``eval_api_module.build_api_docs`` but bound to this module's
    ``api_functions``.

    Returns:
        str: The concatenated documentation of all API functions.
    """
    docs = []
    for func in api_functions:
        sig = str(inspect.signature(func))
        doc = inspect.getdoc(func) or "No description provided."
        docs.append(f"{func.__name__}{sig}:\n{doc}\n")
    return "\n".join(docs)


# Only AvailableTopics["examples"] is inserted into the SAM3 developer prompt.
EVAL_DOMAIN_INFO = {
    "description": "SAM3 open-vocabulary segmentation",
    "note": "Unlike closed-vocabulary models, SAM3 accepts ANY text prompt as a topic",
    "AvailableTopics": {
        "info": "SAM3 is OPEN VOCABULARY - use any descriptive noun phrase",
        "examples": {
            "urban": ["building", "house", "parking lot", "road", "highway", "sidewalk"],
            "vegetation": ["forest", "tree", "agricultural field", "grass", "vegetation"],
            "water": ["water", "river", "lake", "pond", "swimming pool", "ocean"],
            "infrastructure": ["solar panel", "tennis court", "stadium", "runway", "bridge"],
            "vehicles": ["car", "truck", "boat", "airplane", "train"],
        },
        "tips": [
            "Use descriptive terms: 'parking lot' works better than 'pavement'",
            "Be specific when possible: 'residential building' vs just 'building'",
            "Common nouns work best - SAM3 is trained on natural images"
        ]
    }
}


if __name__ == "__main__":
    print(build_api_docs())
    print(json.dumps(EVAL_DOMAIN_INFO, indent=2))
