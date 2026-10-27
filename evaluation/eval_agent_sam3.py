"""QASR agent with SAM3 open-vocabulary segmentation (QASR-F).

Mirrors ``eval_agent`` with two differences: the developer prompt describes
SAM3's free-form text prompts instead of a closed topic list, and the generated
code calls ``eval_sam3_segmentation.segment_image_from_path``. SAM3 must be
loaded with ``eval_sam3_segmentation.init_sam3_model`` before running code.
"""

import io
import json
import os
import sys

from openai import OpenAI

from .eval_api_module_sam3 import build_api_docs, EVAL_DOMAIN_INFO
from .eval_codegen import request_code


class EvalLLMCodeGeneratorSAM3:
    """Generates and runs analysis code with the SAM3 backend."""

    def __init__(self, model="gpt-5", api_key=None):
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

IMPORTANT: This system uses SAM3 OPEN-VOCABULARY segmentation.
You can use ANY descriptive noun phrase as a topic, not just predefined classes.
Examples of valid topics:
{json.dumps(EVAL_DOMAIN_INFO.get('AvailableTopics', {}).get('examples', {}), indent=2)}

Tips for best results:
- Use common, descriptive terms: "building", "parking lot", "forest"
- Be specific when helpful: "agricultural field" instead of just "field"
- Multiple words often work: "swimming pool", "solar panel"

IMAGE_PATH contains the image path (string or PIL Image object).
Do not try to open IMAGE_PATH with PIL - it's already handled by the APIs.

{answer_format}"""

        code, token_usage = request_code(self.client, self.model, developer_prompt, question)
        return code, token_usage, developer_prompt

    def execute_code(self, code, image_path):
        """Run generated code with the SAM3 segmentation and geometry API in scope.

        ``IMAGE_PATH`` is bound to ``image_path`` and stdout is captured. Unlike
        the other agents, the stdout captured before an exception is kept in
        ``prints``.

        Args:
            code (str): Python code from ``generate_code``.
            image_path (str | PIL.Image.Image): Image path or PIL image.

        Returns:
            dict: ``success`` (bool), ``answer`` (value of the ``answer``
            variable, or None), ``prints`` (captured stdout) and ``error``
            (exception message, or None).
        """
        namespace = {
            'IMAGE_PATH': image_path,
        }

        exec("from evaluation.eval_sam3_segmentation import segment_image_from_path", namespace)
        exec("from evaluation.eval_shapes import find_shapes_within_distance, calculate_shape_distances", namespace)

        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            exec(code, namespace)
        except Exception as e:
            sys.stdout = old_stdout
            return {
                'success': False,
                'answer': None,
                'prints': buffer.getvalue(),
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
