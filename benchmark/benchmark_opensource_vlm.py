"""SQuID benchmark for open-source vision-language models answering directly.

Models whose name contains "thinking" or "reasoning" are treated as thinking
models: their reasoning trace is split off at the ``</think>`` token and
stored in a separate ``thinking_trace`` column, and only the final answer is
parsed. A thinking model that exhausts ``--max-new-tokens`` without closing
its trace gets ``</think>`` appended and 10 more tokens to answer.

Tested with:
- Qwen/Qwen2-VL-7B-Instruct
- Qwen/Qwen2-VL-2B-Instruct  
- Qwen/Qwen2-VL-72B-Instruct
- Qwen/Qwen3-VL-30B-A3B-Thinking (MoE thinking model)
- Qwen/Qwen3-VL-235B-A22B-Thinking (MoE thinking model)
- llava-hf/llava-1.5-7b-hf
- llava-hf/llava-1.5-13b-hf

Usage:
    python benchmark/benchmark_opensource_vlm.py --model-name Qwen/Qwen3-VL-30B-A3B-Thinking
    python benchmark/benchmark_opensource_vlm.py --model-name Qwen/Qwen2-VL-7B-Instruct --num-samples 10
    python benchmark/benchmark_opensource_vlm.py --model-name llava-hf/llava-1.5-7b-hf --resume
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Qwen3VLMoeForConditionalGeneration
)

from common import (accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    create_prompt, is_correct, load_benchmark, load_checkpoint,
                    parse_response, pending_indices, print_accuracy, question_with_gsd,
                    save_progress, write_results, write_summary)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm_benchmark")


def is_thinking_model(model_name):
    """Whether ``model_name`` names a thinking/reasoning model."""
    name_lower = model_name.lower()
    return "thinking" in name_lower or "reasoning" in name_lower


def parse_thinking_trace_qwen3(output_ids, tokenizer):
    """Split Qwen3-Thinking output into reasoning trace and final answer.

    The split is at the last ``</think>`` token (id 151668). If there is none,
    generation was truncated mid-thought. The last 100 characters are then
    searched for a trailing number or yes/no, which is taken as the answer.

    Args:
        output_ids: Generated token ids (list or tensor), prompt excluded.
        tokenizer: The model's tokenizer.

    Returns:
        tuple[str, str]: ``(thinking_content, final_content)``, with any
        literal ``<think>``/``</think>`` tags removed. ``final_content`` is
        empty when a truncated trace contains no recognisable answer.
    """
    try:
        think_end_token = 151668

        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.tolist()

        reversed_ids = output_ids[::-1]
        try:
            index = len(output_ids) - reversed_ids.index(think_end_token)
            thinking_was_closed = True
        except ValueError:
            logger.warning("No </think> token found - thinking trace was truncated by max_tokens")
            index = 0
            thinking_was_closed = False

        if index > 0:
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            final_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        else:
            full_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            if not thinking_was_closed:
                import re
                tail = full_text[-100:] if len(full_text) > 100 else full_text
                answer_patterns = [
                    r'(?:answer is|result is|therefore|thus|so)?\s*[:\s]*(\d+\.?\d*)\s*$',
                    r'(\d+\.?\d*)\s*(?:buildings?|structures?|objects?)?\s*$',
                    r'(?:yes|no)\s*$'
                ]

                final_content = ""
                for pattern in answer_patterns:
                    match = re.search(pattern, tail, re.IGNORECASE)
                    if match:
                        final_content = match.group(0).strip()
                        thinking_content = full_text[:full_text.rfind(final_content)].strip()
                        logger.info(f"  -> Extracted potential answer from truncated thinking: '{final_content}'")
                        break

                if not final_content:
                    thinking_content = full_text
                    final_content = ""
                    logger.warning("  -> Model was truncated mid-thinking, no clear answer found")
            else:
                thinking_content = ""
                final_content = full_text

        thinking_content = thinking_content.replace("<think>", "").replace("</think>", "").strip()
        final_content = final_content.replace("<think>", "").replace("</think>", "").strip()

        return thinking_content, final_content

    except Exception as e:
        logger.error(f"Error parsing thinking trace: {e}")
        final_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return "", final_content


def load_model_and_processor(model_name, device):
    """Load a VLM and its processor, picking the model class from the name.

    Qwen3-VL MoE checkpoints (``qwen3`` with ``-a``/``moe`` in the name) use
    ``Qwen3VLMoeForConditionalGeneration``. Other Qwen, LLaVA and generic
    models use ``AutoModelForImageTextToText``. Flash Attention 2 is tried
    first for Qwen models. With several GPUs the model is sharded with
    ``device_map="auto"``.

    Args:
        model_name: Hugging Face model id.
        device: Torch device for single-device runs.

    Returns:
        tuple: ``(model, processor, model_type, is_thinking)`` where
        ``model_type`` is "qwen", "llava" or "generic".

    Raises:
        ValueError: If a generic model cannot be loaded.
    """
    print(f"Loading model: {model_name}")

    is_thinking = is_thinking_model(model_name)
    if is_thinking:
        print("  -> Detected THINKING model - will parse thinking traces")

    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    device_map = "auto" if use_multi_gpu else (device if "cuda" in str(device) else None)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if "qwen" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        is_moe = "qwen3" in model_name.lower() and ("-a" in model_name.lower() or "moe" in model_name.lower())

        if is_moe:
            print("  -> Detected Qwen3-VL MoE model - using Qwen3VLMoeForConditionalGeneration")
            try:
                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                print("  -> Flash Attention 2 enabled for faster inference")
            except Exception as e:
                print(f"  -> Flash Attention 2 not available, using default attention: {e}")
                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
        else:
            print("  -> Using AutoModelForImageTextToText for Qwen2-VL")
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                print("  -> Flash Attention 2 enabled for faster inference")
            except Exception as e:
                print(f"  -> Flash Attention 2 not available, using default attention: {e}")
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
        model_type = "qwen"

    elif "llava" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map
        )
        model_type = "llava"

    else:
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            model_type = "generic"
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {e}")

    if not use_multi_gpu and device_map is None:
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully (type: {model_type}, thinking: {is_thinking})")
    if use_multi_gpu:
        print(f"  -> Using multi-GPU with device_map='auto' across {torch.cuda.device_count()} GPUs")
    return model, processor, model_type, is_thinking


def generate_response(model, processor, model_type, is_thinking, image, prompt, device, max_new_tokens=2048):
    """Answer one question with greedy decoding.

    For Qwen thinking models that use up the token budget without emitting
    ``</think>``, the token is appended and generation continues for up to 10
    tokens. The trace is then split off with
    :func:`parse_thinking_trace_qwen3`.

    Args:
        model: Loaded VLM.
        processor: Matching processor.
        model_type: "qwen", "llava" or "generic".
        is_thinking: Whether to split off a thinking trace.
        image: PIL image.
        prompt: Text prompt.
        device: Device the inputs are moved to.
        max_new_tokens: Generation budget.

    Returns:
        tuple: ``(final_response, input_tokens, output_tokens, thinking_trace)``.
        ``(None, 0, 0, "")`` if generation raised.
    """
    try:
        if model_type == "qwen":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

        elif model_type == "llava":
            prompt_with_image = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = processor(text=prompt_with_image, images=image, return_tensors="pt")

        else:
            inputs = processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None
            )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:].tolist()

        input_tokens = input_length
        output_tokens = len(generated_ids)

        think_end_token = 151668
        if is_thinking and model_type == "qwen" and len(generated_ids) >= max_new_tokens - 1:
            if think_end_token not in generated_ids:
                logger.warning("Thinking model hit token limit without </think> - forcing injection and continuation")

                full_sequence = torch.cat([
                    outputs[0],
                    torch.tensor([think_end_token], device=outputs.device)
                ], dim=0).unsqueeze(0)

                with torch.no_grad():
                    continued = model.generate(
                        input_ids=full_sequence,
                        max_new_tokens=10,
                        min_new_tokens=1,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None
                    )

                generated_ids = continued[0, input_length:].tolist()
                output_tokens = len(generated_ids)

        thinking_trace = ""
        if is_thinking and model_type == "qwen":
            thinking_trace, final_response = parse_thinking_trace_qwen3(
                generated_ids,
                processor.tokenizer
            )
            logger.info(f"  Thinking trace: {len(thinking_trace)} chars")
            logger.info(f"  Final response: {final_response[:100]}...")
        else:
            if hasattr(processor, 'batch_decode'):
                final_response = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
            else:
                final_response = processor.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]

        return final_response, input_tokens, output_tokens, thinking_trace

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None, 0, 0, ""


def main():
    parser = argparse.ArgumentParser(description="Benchmark open-source VLMs end to end on SQuID")
    parser.add_argument("--model-name", required=True,
                       help="HuggingFace model name (e.g., Qwen/Qwen3-VL-30B-A3B-Thinking)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./opensource_vlm_results",
                       help="Directory to save results")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                       help="Max new tokens (default: 2048 for thinking models - handles truncation gracefully)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if exists")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run model on (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inference (default: 1)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model, processor, model_type, is_thinking = load_model_and_processor(args.model_name, args.device)

    processed_ids, results, totals = load_checkpoint(
        output_dir, args.resume,
        time_columns=('inference_time',),
        optional_time_columns=('generation_time',))
    total_input_tokens = totals['total_input_tokens']
    total_output_tokens = totals['total_output_tokens']
    total_generation_time = totals['total_generation_time']
    total_inference_time = totals['total_inference_time']

    print("Loading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process, desc="Evaluating"):
        sample = dataset[idx]

        image = sample['image']
        prompt = create_prompt(question_with_gsd(sample), sample['type'])

        try:
            start_time = time.time()
            response, input_tokens, output_tokens, thinking_trace = generate_response(
                model, processor, model_type, is_thinking, image, prompt, args.device, args.max_new_tokens
            )
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            execution_time = 0
            inference_time = generation_time
            total_inference_time += inference_time

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            prediction = parse_response(response, sample['type'])
            raw_response = response if response else ""

        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            prediction = None
            raw_response = str(e)
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
        })

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        save_progress(results, output_dir, results_every=10, checkpoint_every=100)

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
    print(f"  Generation time: {total_generation_time:.2f}s total, {total_generation_time/len(results):.3f}s avg")

    if is_thinking:
        thinking_samples = [r for r in results if r.get('thinking_trace')]
        print("\nThinking Model Stats:")
        print(f"  Samples with thinking: {len(thinking_samples)}/{len(results)}")
        if thinking_samples:
            avg_thinking_len = sum(len(r['thinking_trace']) for r in thinking_samples) / len(thinking_samples)
            print(f"  Average thinking length: {avg_thinking_len:.0f} chars")

    summary = {
        'model': args.model_name,
        'is_thinking_model': is_thinking,
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'total_generation_time': total_generation_time,
        'total_inference_time': total_inference_time,
        'avg_generation_time': total_generation_time / len(results),
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
