"""QASR agent: GPT code generation over the default segmentation backends.

The LLM writes Python that calls ``segment_image_from_path`` and the geometry
API (see ``eval_api_module``). ``execute_code`` runs it. The segmentation
backend is selected by environment variables at import time:

* ``DINOV3_CHECKPOINT=PeterAM4/SQuID-DINOv3-Mask2Former`` (or a local file):
  DINOv3+Mask2Former for the topics it supports, the HTTP server for the rest
  (QASR-C in Table 3).
* unset: the ConvNeXt-UNet HTTP server at http://localhost:8000
  (QASR-A/B/D).
* ``USE_GT_MASKS=1``: the SQuID reference masks replace every
  model backend, so neither a segmenter nor the HTTP server is needed (see
  ``eval_gt_segmentation``).

The ConvNeXt-UNet checkpoint used in the paper is not redistributed. Any
segmenter that implements the HTTP API in ``eval_satellite_processing`` can
take its place.
"""

import io
import json
import os
import sys

from openai import OpenAI

from .eval_api_module import build_api_docs, EVAL_DOMAIN_INFO
from .eval_codegen import request_code
from .eval_satellite_processing import init_dinov3_model

DINOV3_CHECKPOINT = os.environ.get("DINOV3_CHECKPOINT")

if DINOV3_CHECKPOINT:
    init_dinov3_model(DINOV3_CHECKPOINT, device='cuda')
    print(f"DINOv3 hybrid mode enabled (checkpoint: {DINOV3_CHECKPOINT})")
elif os.environ.get("USE_GT_MASKS", "0") == "1":
    print("USE_GT_MASKS=1: reference masks, no segmenter needed")
else:
    print("DINOV3_CHECKPOINT not set, using the ConvNeXt HTTP server (http://localhost:8000)")


class EvalLLMCodeGenerator:
    """Generates and runs analysis code with the default segmentation backend."""

    def __init__(self, model="gpt-5-mini", api_key=None):
        """Create the OpenAI client.

        Args:
            model (str): OpenAI model used for code generation.
            api_key (str | None): OpenAI API key. Defaults to the
                ``OPENAI_API_KEY`` environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_code(self, question, answer_format="Store your final answer in a variable called 'answer'."):
        """Generate Python code that answers a question about an image.

        Args:
            question (str): The benchmark question.
            answer_format (str): Answer-format instructions appended to the
                developer prompt.

        Returns:
            tuple[str, dict, str]: The generated code, the token usage
            (``input_tokens``, ``output_tokens``, ``total_tokens``) and the
            developer prompt that was sent.
        """
        api_docs = build_api_docs()

        developer_prompt = f"""You are a code generator for geospatial image analysis.

        CRITICAL RULES:
        - Output ONLY executable Python code
        - NO explanatory text before or after code
        - NO imports (everything is already imported)
        - NO markdown code blocks
        - Store final result in variable 'answer'

        Available APIs (already imported, do not import anything):
        {api_docs}

        Available segmentation topics:
        {json.dumps(EVAL_DOMAIN_INFO.get('AvailableTopics', {}), indent=2)}

        IMAGE_PATH contains the image path (string or PIL Image object).
        Do not try to open IMAGE_PATH with PIL - it's already handled by the APIs.

        {answer_format}"""

        code, token_usage = request_code(self.client, self.model, developer_prompt, question)
        return code, token_usage, developer_prompt

    @staticmethod
    def execute_code(code, image_path, gsd):
        """Run generated code with the segmentation and geometry API in scope.

        ``IMAGE_PATH`` and ``gsd`` are bound and stdout is captured.

        Args:
            code (str): Python code from ``generate_code``.
            image_path (str | PIL.Image.Image): Image path or PIL image.
            gsd (float): Ground sampling distance of the image, in metres.

        Returns:
            dict: ``success`` (bool), ``answer`` (value of the ``answer``
            variable, or None), ``prints`` (captured stdout, empty on error)
            and ``error`` (exception message, or None).
        """
        namespace = {
            'IMAGE_PATH': image_path,
            'gsd': round(gsd, 6),  # the Hub parquet stores 0.3 as float32
        }

        exec("from evaluation.eval_satellite_processing import segment_image_from_path", namespace)
        exec("from evaluation.eval_shapes import find_shapes_within_distance, calculate_shape_distances", namespace)

        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            exec(code, namespace)
        except Exception as e:
            return {
                'success': False,
                'answer': None,
                'prints': '',
                'error': str(e)
            }
        finally:
            sys.stdout = old_stdout

        return {
            'success': True,
            'answer': namespace.get('answer'),
            'prints': buffer.getvalue(),
            'error': None
        }
