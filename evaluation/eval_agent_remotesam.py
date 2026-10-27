"""QASR agent with RemoteSAM segmentation (QASR-E).

Mirrors ``eval_agent`` with two differences: the developer prompt describes
RemoteSAM's free-form text prompts, and the generated code calls
``eval_remotesam_segmentation.segment_image_from_path``.

Usage:
    from .eval_agent_remotesam import EvalLLMCodeGeneratorRemoteSAM
    from .eval_remotesam_segmentation import init_remotesam_model

    init_remotesam_model(remotesam_repo_path="/path/to/RemoteSAM", device="cuda")
    agent = EvalLLMCodeGeneratorRemoteSAM(model="gpt-5", api_key="...")
    code, usage, prompt = agent.generate_code("How many buildings?", "Store answer in 'answer'.")
    result = agent.execute_code(code, "/path/to/image.png")
"""

import io
import os
import sys

from openai import OpenAI

from .eval_api_module_remotesam import build_api_docs
from .eval_codegen import request_code


class EvalLLMCodeGeneratorRemoteSAM:
    """Generates and runs analysis code with the RemoteSAM backend."""

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

        IMPORTANT NOTES ON segment_image_from_path:
        - topics is a list of text prompts for what to detect, e.g. ["building"], ["vehicle", "car"]
        - Use simple, common nouns as topics (e.g. "building" not "residential structure")
        - RemoteSAM supports 1000+ remote sensing object categories natively
        - Each detected instance becomes a separate shape in the result
        - For counting: len(result["shapes"]) gives the count
        - For area: sum shape areas from result["shapes"]

        IMAGE_PATH contains the image path (string or PIL Image object).
        Do not try to open IMAGE_PATH with PIL - it's already handled by the APIs.

        {answer_format}"""

        code, token_usage = request_code(self.client, self.model, developer_prompt, question)
        return code, token_usage, developer_prompt

    def execute_code(self, code, image_path):
        """Run generated code with the RemoteSAM segmentation and geometry API in scope.

        ``IMAGE_PATH`` is bound to ``image_path`` and stdout is captured.

        Args:
            code (str): Python code from ``generate_code``.
            image_path (str | PIL.Image.Image): Image path or PIL image.

        Returns:
            dict: ``success`` (bool), ``answer`` (value of the ``answer``
            variable, or None), ``prints`` (captured stdout, empty on error)
            and ``error`` (exception message, or None).
        """
        namespace = {
            "IMAGE_PATH": image_path,
        }

        exec("from evaluation.eval_remotesam_segmentation import segment_image_from_path", namespace)
        exec("from evaluation.eval_shapes import find_shapes_within_distance, calculate_shape_distances", namespace)

        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            exec(code, namespace)
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "prints": "",
                "error": str(e),
            }
        finally:
            sys.stdout = old_stdout

        return {
            "success": True,
            "answer": namespace.get("answer"),
            "prints": buffer.getvalue(),
            "error": None,
        }
