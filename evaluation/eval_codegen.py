"""OpenAI Responses API call shared by the QASR code-generating agents."""

import re


def request_code(client, model, developer_prompt, question):
    """Ask the LLM for code answering ``question`` and strip markdown fences.

    The request uses the Responses API with minimal reasoning effort.

    Args:
        client (openai.OpenAI): Client used for the request.
        model (str): Model name, e.g. ``"gpt-5"``.
        developer_prompt (str): System/developer instructions.
        question (str): User input, the benchmark question.

    Returns:
        tuple[str, dict]: The generated code with any ```` ```python ```` fences
        removed, and the token usage with keys ``input_tokens``,
        ``output_tokens`` and ``total_tokens`` (zeros if the response has no
        usage information).
    """
    response = client.responses.create(
        model=model,
        instructions=developer_prompt,
        input=question,
        reasoning={"effort": "minimal"}
    )

    code = response.output_text if hasattr(response, 'output_text') else ""

    code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code)

    token_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }

    if hasattr(response, 'usage'):
        token_usage['input_tokens'] = response.usage.input_tokens
        token_usage['output_tokens'] = response.usage.output_tokens
        token_usage['total_tokens'] = response.usage.total_tokens

    return code.strip(), token_usage
