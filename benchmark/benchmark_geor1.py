"""SQuID benchmark for Geo-R1 and other Qwen2.5-VL based models.

Geo-R1 (miniHui/Geo-R1) is Qwen2.5-VL-7B fine-tuned with GRPO for geospatial
reasoning. It thinks in ``<think>...</think>`` and answers in
``<answer>...</answer>``. For models whose name marks them as reasoning
models the trace is split off into a ``thinking_trace`` column, and the
*last* number of the remaining answer is taken as the prediction.

Supported models:
- miniHui/Geo-R1
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-3B-Instruct

Usage:
    python benchmark/benchmark_geor1.py --model-name miniHui/Geo-R1
    python benchmark/benchmark_geor1.py --model-name miniHui/Geo-R1 --num-samples 10
    python benchmark/benchmark_geor1.py --model-name miniHui/Geo-R1 --resume

Requirements:
    pip install transformers>=4.49.0 qwen-vl-utils torch accelerate
"""

import argparse
import gc
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from common import (accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    is_correct, load_benchmark, load_checkpoint, parse_response,
                    pending_indices, print_accuracy, question_with_gsd, save_progress,
                    write_results, write_summary)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geor1_benchmark")

# Substrings of a model name that mark it as producing <think> traces.
REASONING_MODELS = [
    "geo-r1", "geo_r1", "geor1",
    "r1", "thinking", "reasoning",
]


def is_reasoning_model(model_name: str) -> bool:
    """Whether ``model_name`` contains one of :data:`REASONING_MODELS`."""
    name_lower = model_name.lower()
    for pattern in REASONING_MODELS:
        if pattern in name_lower:
            return True
    return False


def parse_reasoning_response(response: str) -> tuple:
    """Split a Geo-R1 style response into reasoning trace and answer.

    The answer is the content of ``<answer>...</answer>``. Without those tags
    it is the text after the last ``</think>``, and otherwise the whole
    response.

    Args:
        response: Decoded model output.

    Returns:
        tuple[str, str]: ``(thinking_trace, final_answer)``. The trace is
        empty when there is no ``<think>...</think>`` block.
    """
    thinking = ""
    answer = response

    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()

    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        if '</think>' in response.lower():
            parts = re.split(r'</think>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[-1].strip()

    return thinking, answer


def create_prompt(question, q_type):
    """Build the direct-answer prompt.

    Same wording as ``common.create_prompt``, but binary types are matched
    by substring (``'binary' in q_type``) rather than by an explicit list.

    Args:
        question: Question text.
        q_type: SQuID question type.

    Returns:
        str: The question followed by an answer-format instruction.
    """
    if q_type in ['percentage', 'proximity_percentage']:
        return f"{question}\n\nAnswer with ONLY a number between 0 and 100:"
    elif 'binary' in q_type:
        return f"{question}\n\nAnswer with ONLY 'yes' or 'no':"
    elif q_type == 'fragmentation':
        return f"{question}\n\nAnswer with ONLY 'connected' or 'fragmented':"
    else:
        return f"{question}\n\nAnswer with ONLY a number:"


class Qwen25VLInference:
    """Greedy-decoding wrapper around a Qwen2.5-VL checkpoint.

    Args:
        model_name: Hugging Face model id.
        device: Torch device for single-GPU runs. With several GPUs the model
            is sharded with ``device_map="auto"``.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.is_reasoning = is_reasoning_model(model_name)

        print(f"Loading model: {model_name}")
        print(f"  -> Reasoning model: {self.is_reasoning}")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        device_map = "auto" if use_multi_gpu else (device if "cuda" in str(device) else None)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            print("  -> Flash Attention 2 enabled")
        except Exception as e:
            logger.info(f"Flash Attention not available: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )

        self.model.eval()

        if use_multi_gpu:
            print(f"  -> Multi-GPU enabled across {torch.cuda.device_count()} GPUs")

        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            self.use_qwen_vl_utils = True
            print("  -> Using qwen_vl_utils for image processing")
        except ImportError:
            self.use_qwen_vl_utils = False
            print("  -> Using direct PIL image processing (qwen_vl_utils not found)")

    def generate(self, image, prompt: str, max_new_tokens: int = 2048) -> tuple:
        """Answer one question.

        Args:
            image: PIL image or path to one.
            prompt: Text prompt.
            max_new_tokens: Generation budget.

        Returns:
            tuple: ``(final_response, input_tokens, output_tokens,
            thinking_trace, raw_response)``. For reasoning models
            ``final_response`` is the answer part of ``raw_response``. On
            error the tuple is ``(None, 0, 0, "", "")``.
        """
        try:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            if self.use_qwen_vl_utils:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                )

            inputs = inputs.to(self.model.device)
            input_tokens = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_tokens:]
            output_tokens = len(generated_ids)

            response = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            thinking_trace = ""
            final_response = response

            if self.is_reasoning:
                thinking_trace, final_response = parse_reasoning_response(response)
                if thinking_trace:
                    logger.debug(f"  -> Thinking trace: {len(thinking_trace)} chars")

            return final_response, input_tokens, output_tokens, thinking_trace, response

        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, 0, "", ""


def main():
    parser = argparse.ArgumentParser(description="Benchmark Geo-R1 and Qwen2.5-VL models on SQuID")
    parser.add_argument("--model-name", default="miniHui/Geo-R1",
                       help="HuggingFace model name (default: miniHui/Geo-R1)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./geor1_results",
                       help="Directory to save results")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                       help="Max new tokens (default: 2048)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")

    args = parser.parse_args()

    print("=" * 60)
    print("GEO-R1 / QWEN2.5-VL BENCHMARK ON SQuID")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    inference = Qwen25VLInference(args.model_name, args.device)

    processed_ids, results, totals = load_checkpoint(
        output_dir, args.resume,
        time_columns=(),
        optional_time_columns=('generation_time', 'inference_time'))
    total_input_tokens = totals['total_input_tokens']
    total_output_tokens = totals['total_output_tokens']
    total_generation_time = totals['total_generation_time']
    total_inference_time = totals['total_inference_time']

    print("\nLoading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process, desc=f"Evaluating {args.model_name.split('/')[-1]}"):
        sample = dataset[idx]

        image = sample['image']
        prompt = create_prompt(question_with_gsd(sample), sample['type'])

        try:
            start_time = time.time()
            response, input_tokens, output_tokens, thinking_trace, raw_response = inference.generate(
                image, prompt,
                max_new_tokens=args.max_new_tokens
            )
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            execution_time = 0
            inference_time = generation_time
            total_inference_time += inference_time

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            prediction = parse_response(response, sample['type'], number_index=-1)

        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            prediction = None
            raw_response = str(e)
            response = str(e)
            thinking_trace = ""
            input_tokens = output_tokens = 0
            generation_time = execution_time = inference_time = 0

        results.append({
            'id': sample['id'],
            'question': sample['question'],
            'type': sample['type'],
            'tier': sample['tier'],
            'ground_truth': sample['answer'],
            'acceptable_range_lower': sample.get('acceptable_range_lower'),
            'acceptable_range_upper': sample.get('acceptable_range_upper'),
            'prediction': prediction,
            'raw_response': raw_response,
            'thinking_trace': thinking_trace,
            'is_correct': is_correct(
                prediction,
                sample['answer'],
                sample['type'],
                sample.get('acceptable_range_lower'),
                sample.get('acceptable_range_upper')
            ),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'generation_time': generation_time,
            'execution_time': execution_time,
            'inference_time': inference_time,
            'error_percentage': calculate_error_percentage(prediction, sample['answer'], sample['type']),
            'visualization_path': None,
            'segmentation_masks': None,
            'model': args.model_name,
        })

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        save_progress(results, output_dir, results_every=10, checkpoint_every=100)

    write_results(results, output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    correct = sum(1 for r in results if r['is_correct'])
    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc)

    print("\nToken Usage:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")

    print("\nTiming:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    if results:
        print(f"  Average per sample: {total_inference_time/len(results):.3f}s")

    if inference.is_reasoning:
        thinking_samples = [r for r in results if r.get('thinking_trace')]
        print("\nReasoning Stats:")
        print(f"  Samples with reasoning: {len(thinking_samples)}/{len(results)}")
        if thinking_samples:
            avg_len = sum(len(r['thinking_trace']) for r in thinking_samples) / len(thinking_samples)
            print(f"  Average reasoning length: {avg_len:.0f} chars")

    summary = {
        'model': args.model_name,
        'is_reasoning_model': inference.is_reasoning,
        'base_architecture': 'Qwen2.5-VL-7B',
        'overall_accuracy': 100 * correct / len(results) if results else 0,
        'total_samples': len(results),
        'correct_samples': correct,
        'total_generation_time': total_generation_time,
        'total_inference_time': total_inference_time,
        'avg_generation_time': total_generation_time / len(results) if results else 0,
        'avg_inference_time': total_inference_time / len(results) if results else 0,
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
