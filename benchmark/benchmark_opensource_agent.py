"""SQuID benchmark for QASR with an open-weight LLM as the code generator.

The agent loop, geometry API and output format follow benchmark_agent.py, but
the program is generated locally with Hugging Face transformers instead of
the OpenAI API. Three prompt formats are supported: gpt-oss (harmony format,
reasoning effort in the system message), Qwen thinking models (code is taken
after the last ``</think>`` token) and plain chat models. The full decoded
generation, special tokens included, is stored in ``full_model_response``.

Usage:
    python benchmark/benchmark_opensource_agent.py --model-name openai/gpt-oss-120b
    python benchmark/benchmark_opensource_agent.py --model-name openai/gpt-oss-120b --reasoning-effort medium
    python benchmark/benchmark_opensource_agent.py --model-name openai/gpt-oss-120b --resume
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import (accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    is_correct, load_benchmark, load_checkpoint, parse_answer,
                    pending_indices, print_accuracy, question_with_gsd,
                    save_last_masks, save_progress, write_results, write_summary)

sys.path.append(str(Path(__file__).parent.parent))
from evaluation.eval_satellite_processing import set_visualization_dir
import evaluation.eval_satellite_processing as seg_module
from evaluation.eval_api_module import build_api_docs, EVAL_DOMAIN_INFO


def format_answer_instruction(q_type):
    """Tell the code-generating LLM how to store its answer.

    Identical to ``common.format_answer_instruction`` except that the last
    sentence reads "each topic" instead of "each topics". Kept separate so the
    prompt stays byte-identical to the one used for the reported results.

    Args:
        q_type: SQuID question type.

    Returns:
        str: Instruction appended to the developer prompt.
    """
    if q_type in ['percentage', 'proximity_percentage']:
        ret = "Store your final answer as a number between 0 and 100 in 'answer'."
    elif 'binary' in q_type:
        ret = "Store your final answer as 1 for yes or 0 for no in 'answer'."
    elif q_type == 'fragmentation':
        ret = "Store your final answer as 'connected' or 'fragmented' in 'answer'."
    else:
        ret = "Store your final answer in 'answer'. answer must be simple numeric type (float or int)."

    return ret + " Print the number of initial shapes found for each topic, as well as other relevant intermediate findings."


class OpenSourceCodeGenerator:
    """Generates and executes geometry-API programs with a local causal LM.

    Args:
        model_name: Hugging Face model id. Names containing "gpt-oss" select
            the harmony format. Names containing both "qwen" and "thinking"
            select thinking-token handling. Anything else is a standard chat
            model.
        reasoning_effort: gpt-oss reasoning effort ("low", "medium", "high").
    """

    def __init__(self, model_name, reasoning_effort="medium"):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

        model_lower = model_name.lower()
        if 'gpt-oss' in model_lower:
            self.model_type = 'gpt_oss'
            print("Detected: GPT-OSS model (uses harmony format)")
        elif 'qwen' in model_lower and 'thinking' in model_lower:
            self.model_type = 'qwen_thinking'
            print("Detected: Qwen Thinking model (has thinking tokens)")
        else:
            self.model_type = 'standard'
            print("Detected: Standard model")

        print(f"Loading {model_name}...")
        print(f"  Model type: {self.model_type}")
        print(f"  Reasoning effort: {reasoning_effort}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_gpus = torch.cuda.device_count()
        print(f"  GPUs available: {num_gpus}")

        print("  Loading with BF16 + Tensor Parallelism...")

        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        try:
            from flash_attn import flash_attn_func
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Flash Attention: enabled")
        except:
            print("  Flash Attention: not available")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        self.model.eval()
        print("Model loaded successfully\n")

    def generate_code(self, question, answer_format):
        """Generate a program for one question with greedy decoding.

        The developer prompt is derived from the one in
        ``evaluation.eval_agent.EvalLLMCodeGenerator`` but also describes the
        ``IMAGE_PATH``/``gsd`` variables and embeds the question itself.

        Args:
            question: Question text (with GSD).
            answer_format: Output of :func:`format_answer_instruction`.

        Returns:
            tuple: ``(code, full_response, token_usage, developer_prompt)``
            where ``full_response`` keeps special tokens and ``token_usage``
            has ``input_tokens`` and ``output_tokens``.
        """
        api_docs = build_api_docs()

        developer_prompt = f"""You are a code generator for geospatial image analysis.

        CRITICAL RULES:
        - Output ONLY executable Python code
        - NO explanatory text before or after code
        - NO imports (everything is already imported)
        - NO markdown code blocks
        - Store final result in variable 'answer'
        - Use IMAGE_PATH variable (already defined, contains the image path)
        - Use gsd variable (already defined, contains the ground sampling distance)

        Available APIs (already imported, do not import anything):
        {api_docs}

        Available segmentation topics:
        {json.dumps(EVAL_DOMAIN_INFO.get('AvailableTopics', {}), indent=2)}

        Context variables available:
        - IMAGE_PATH: path to the image file
        - gsd: ground sampling distance in meters

        {answer_format}

        Question: {question}

        Generate ONLY the Python code to answer this question:"""

        if self.model_type == 'gpt_oss':
            messages = [
                {"role": "system", "content": f"You are an expert Python programmer.\nReasoning: {self.reasoning_effort}"},
                {"role": "user", "content": developer_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. Think step-by-step."},
                {"role": "user", "content": developer_prompt}
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]

        max_tokens = 4096 if self.model_type in ('gpt_oss', 'qwen_thinking') else 2048

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                num_beams=1
            )

        generated_ids = outputs[0, input_length:]
        output_tokens = len(generated_ids)

        # The full trace keeps special tokens. Code is extracted from the clean text.
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        response_clean = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.model_type == 'qwen_thinking':
            code = self._extract_code_qwen(generated_ids)
        else:
            code = self._extract_code(response_clean)

        token_usage = {
            'input_tokens': input_length,
            'output_tokens': output_tokens
        }

        return code, full_response, token_usage, developer_prompt

    def _extract_code_qwen(self, generated_ids):
        """Decode the tokens after the last ``</think>`` (id 151668) and strip fences."""
        output_ids = generated_ids.tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        code_ids = output_ids[index:]
        code_text = self.tokenizer.decode(code_ids, skip_special_tokens=True).strip()

        code_text = self._extract_code(code_text)

        return code_text

    def _extract_code(self, text):
        """Strip Markdown code fences from generated text."""
        if not text or not text.strip():
            return ""

        text = re.sub(r'^```python\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\s*$', '', text)
        text = text.strip()

        code_indicators = [
            'segment_image_from_path',
            'find_shapes_within_distance',
            'result = ',
            'shapes = ',
            'answer = '
        ]
        if any(indicator in text for indicator in code_indicators):
            return text

        return text

    def execute_code(self, code, image_path, gsd):
        """Run a generated program and collect its ``answer`` and stdout.

        Args:
            code: Python source.
            image_path: Value bound to ``IMAGE_PATH`` in the program's namespace.
            gsd: Ground sampling distance in metres, bound to ``gsd``.

        Returns:
            dict: ``success``, ``answer``, ``prints`` (captured stdout) and
            ``error`` (message or None).
        """
        import io

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
            success = True
            answer = namespace.get('answer')
            error = None
        except Exception as e:
            success = False
            answer = None
            error = str(e)
        finally:
            sys.stdout = old_stdout

        return {
            'success': success,
            'answer': answer,
            'prints': buffer.getvalue(),
            'error': error
        }


def main():
    parser = argparse.ArgumentParser(description="Open-Source Agent Benchmark")
    parser.add_argument("--model-name", default="openai/gpt-oss-120b",
                       help="HuggingFace model name")
    parser.add_argument("--reasoning-effort", default="medium",
                       choices=["low", "medium", "high"],
                       help="Reasoning effort")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples (default: all)")
    parser.add_argument("--output-dir", default="./opensource_agent_results",
                       help="Output directory")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--save-viz", action="store_true",
                       help="Save visualizations")

    args = parser.parse_args()

    agent = OpenSourceCodeGenerator(args.model_name, args.reasoning_effort)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.save_viz:
        viz_dir = output_dir / "visualizations"
        set_visualization_dir(viz_dir)
        print(f"Visualization enabled: {viz_dir}")

    processed_ids, results, totals = load_checkpoint(output_dir, args.resume)
    total_generation_time = totals['total_generation_time']
    total_execution_time = totals['total_execution_time']
    total_inference_time = totals['total_inference_time']
    total_input_tokens = totals['total_input_tokens']
    total_output_tokens = totals['total_output_tokens']

    print("Loading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    if not samples_to_process:
        print("All samples already processed!")
        return

    print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process):
        sample = dataset[idx]

        question = question_with_gsd(sample)
        answer_format = format_answer_instruction(sample['type'])

        try:
            start_time = time.time()
            code, full_response, token_usage, developer_prompt = agent.generate_code(question, answer_format)
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            input_tokens = token_usage['input_tokens']
            output_tokens = token_usage['output_tokens']
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            viz_counter_before = seg_module.VIZ_COUNTER if args.save_viz else 0

            start_time = time.time()
            result = agent.execute_code(code, sample['image'], sample['gsd'])
            execution_time = time.time() - start_time
            total_execution_time += execution_time

            mask_path = save_last_masks(seg_module, output_dir, sample['id'])

            viz_path = None
            if args.save_viz:
                viz_counter_after = seg_module.VIZ_COUNTER
                if viz_counter_after > viz_counter_before:
                    viz_path = f"visualizations/seg_{viz_counter_after-1:04d}.png"

            inference_time = generation_time + execution_time
            total_inference_time += inference_time

            prediction = parse_answer(result, sample['type'])
            raw_response = code if code else ""
            code_prints = result['prints']

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            code_prints = ''
            prediction = None
            raw_response = str(e)
            full_response = ''
            developer_prompt = ''
            generation_time = execution_time = inference_time = 0
            input_tokens = output_tokens = 0
            mask_path = None
            viz_path = None
            seg_module.LAST_MASKS = {}

        results.append({
            'id': sample['id'],
            'question': sample['question'],
            'type': sample['type'],
            'tier': sample['tier'],
            'ground_truth': sample['answer'],
            'acceptable_range_lower': sample.get('acceptable_range_lower'),
            'acceptable_range_upper': sample.get('acceptable_range_upper'),
            'prediction': prediction,
            'developer_prompt': developer_prompt,
            'raw_response': raw_response,
            'full_model_response': full_response,
            'intermediate_prints': code_prints,
            'is_correct': is_correct(
                prediction,
                sample['answer'],
                sample['type'],
                sample.get('acceptable_range_lower'),
                sample.get('acceptable_range_upper')
            ),
            'generation_time': generation_time,
            'execution_time': execution_time,
            'inference_time': inference_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'error_percentage': calculate_error_percentage(
                prediction, sample['answer'], sample['type']
            ),
            'visualization_path': viz_path,
            'segmentation_masks': mask_path,
        })

        save_progress(results, output_dir, results_every=10, checkpoint_every=100)
        if len(results) % 10 == 0:
            print(f"  Checkpoint: {len(results)} samples")

        if len(results) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    write_results(results, output_dir)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    correct = sum(1 for r in results if r['is_correct'])
    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc)

    print("\nToken Usage:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")

    print("\nTiming:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average per sample: {total_inference_time/len(results):.3f}s")
    print(f"  Code generation: {total_generation_time:.2f}s total, {total_generation_time/len(results):.3f}s avg")
    print(f"  Code execution: {total_execution_time:.2f}s total, {total_execution_time/len(results):.3f}s avg")

    summary = {
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'model': args.model_name,
        'reasoning_effort': args.reasoning_effort,
        'total_generation_time': total_generation_time,
        'total_execution_time': total_execution_time,
        'total_inference_time': total_inference_time,
        'avg_generation_time': total_generation_time / len(results),
        'avg_execution_time': total_execution_time / len(results),
        'avg_inference_time': total_inference_time / len(results),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc
    }
    write_summary(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
