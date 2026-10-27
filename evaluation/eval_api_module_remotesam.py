"""API exposed to the code-generating LLM for the RemoteSAM agent.

Same as ``eval_api_module`` except that ``segment_image_from_path`` comes from
``eval_remotesam_segmentation``. There is no topic list: RemoteSAM takes
free-form text prompts.
"""

import inspect

from .eval_remotesam_segmentation import segment_image_from_path
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


if __name__ == "__main__":
    print(build_api_docs())
