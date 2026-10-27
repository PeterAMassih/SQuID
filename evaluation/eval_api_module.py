"""API exposed to the code-generating LLM for the default QASR agent.

``build_api_docs`` renders the signatures and docstrings of ``api_functions``
for the developer prompt, and ``EVAL_DOMAIN_INFO`` carries the segmentation
topics listed in it. The topics are queried from the segmentation server at
import time, except in oracle mode (``USE_GT_MASKS=1``), which uses the frozen
``STANDARD_TOPICS``.
"""

import inspect
import os

import requests

from .eval_satellite_processing import segment_image_from_path
from .eval_shapes import find_shapes_within_distance, calculate_shape_distances


# The topic dict the segmentation server serves, frozen so runs that never touch
# the server (USE_GT_MASKS=1) build the identical developer prompt.
STANDARD_TOPICS = {
    "urban": "urban areas (paved, built environment)",
    "forest": "areas with trees, forests",
    "agric": "agricultural areas, cultivated fields",
    "grass": "uncultivated rangeland, unforested grassy areas and lawns",
    "barren": "barren earth areas (no vegetation, undevelopped)",
    "water": "areas covered by water (lake, river, pond, flooded area)",
    "solar": "areas covered with solar panels",
    "roof": "roofs of individual buildings",
}


def get_available_topics():
    """Return the segmentation topics offered to the LLM.

    Returns:
        dict: Topic name -> description. ``STANDARD_TOPICS`` with reference
        masks. Otherwise the union of the ``topics`` of all models listed by
        the server's ``get-models`` endpoint, or ``{}`` with a warning when
        the server cannot be reached.
    """
    if os.environ.get("USE_GT_MASKS", "0") == "1":
        return STANDARD_TOPICS
    try:
        response = requests.get("http://localhost:8000/get-models")
        response.raise_for_status()
    except requests.RequestException:
        print("WARNING: no segmentation server on http://localhost:8000. "
              "Start it, or set USE_GT_MASKS=1 to use the reference masks.")
        return {}
    all_topics = {}
    for model in response.json():
        if 'topics' in model:
            all_topics.update(model['topics'])
    return all_topics


api_functions = [
    segment_image_from_path,
    find_shapes_within_distance,
    calculate_shape_distances,
]


def build_api_docs():
    """Render ``api_functions`` as text for the LLM prompt.

    Each entry is ``name(signature):`` followed by the function's cleaned
    docstring, so editing those docstrings changes the prompt.

    Returns:
        str: The concatenated documentation of all API functions.
    """
    docs = []
    for func in api_functions:
        sig = str(inspect.signature(func))
        doc = inspect.getdoc(func) or "No description provided."
        docs.append(f"{func.__name__}{sig}:\n{doc}\n")
    return "\n".join(docs)


EVAL_DOMAIN_INFO = {
    "AvailableTopics": get_available_topics()
}

if __name__ == "__main__":
    print(EVAL_DOMAIN_INFO)
