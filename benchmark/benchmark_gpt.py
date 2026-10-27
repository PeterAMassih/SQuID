"""SQuID benchmark for OpenAI vision models answering directly (no tools).

Each question is sent with the image to the Responses API with minimal
reasoning effort and a one-line answer-format instruction.

Usage:
    python benchmark/benchmark_gpt.py --api-key YOUR_KEY
    python benchmark/benchmark_gpt.py --api-key YOUR_KEY --num-samples 10
    python benchmark/benchmark_gpt.py --api-key YOUR_KEY --resume
"""

import argparse
import base64
import io
import math
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from common import (accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    create_prompt, is_correct, load_benchmark, load_checkpoint,
                    parse_response, pending_indices, print_accuracy, question_with_gsd,
                    report_cost, save_progress, write_results, write_summary)

# USD per 1M tokens (uncached input).
MODEL_PRICING = {
    "gpt-5":       {"input": 1.25, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "gpt-5-nano":  {"input": 0.05, "output": 0.40},
    "gpt-4.1":       {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4o":        {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "o4-mini":       {"input": 1.10, "output": 4.40},
}


def calculate_image_tokens(image: Image.Image, model: str = "gpt-5-mini", detail: str = "high") -> int:
    """Estimate the input tokens OpenAI bills for one image.

    Smaller models count 32x32 patches (capped at 1536) times a model-specific
    multiplier. The others count 512px tiles after OpenAI's resizing rules.

    Args:
        image: Input image.
        model: OpenAI model id. Unknown models fall back to gpt-4o.
        detail: Image detail level ("high" or "low").

    Returns:
        int: Estimated image tokens.
    """
    PATCH_MODELS = {
        "gpt-5-mini": 1.62,
        "gpt-5-nano": 2.46,
        "gpt-4.1-mini": 1.62,
        "gpt-4.1-nano": 2.46,
        "o4-mini": 1.72,
    }

    TILE_MODELS = {
        "gpt-5": {"base": 70, "tile": 140},
        "gpt-5-chat-latest": {"base": 70, "tile": 140},
        "gpt-4o": {"base": 85, "tile": 170},
        "gpt-4.1": {"base": 85, "tile": 170},
        "gpt-4.5": {"base": 85, "tile": 170},
        "gpt-4o-mini": {"base": 2833, "tile": 5667},
        "o1": {"base": 75, "tile": 150},
        "o1-pro": {"base": 75, "tile": 150},
        "o3": {"base": 75, "tile": 150},
        "computer-use-preview": {"base": 65, "tile": 129},
    }

    w, h = image.size

    if model in PATCH_MODELS:
        # Low detail is not documented for patch models. The same count is assumed.
        patches = math.ceil(w / 32) * math.ceil(h / 32)

        if patches > 1536:
            # Shrink so the image fits 1536 patches, then snap to whole patches.
            r = math.sqrt((32 * 32 * 1536) / (w * h))
            r = r * min(
                math.floor(w * r / 32) / (w * r / 32) if w * r / 32 > 0 else 1,
                math.floor(h * r / 32) / (h * r / 32) if h * r / 32 > 0 else 1
            )
            resized_w = w * r
            resized_h = h * r
            patches = math.ceil(resized_w / 32) * math.ceil(resized_h / 32)
            patches = min(patches, 1536)

        multiplier = PATCH_MODELS[model]
        return int(round(patches * multiplier))

    elif model in TILE_MODELS:
        base_tokens = TILE_MODELS[model]["base"]
        tile_tokens = TILE_MODELS[model]["tile"]

        if detail == "low":
            return base_tokens

        # Fit in 2048x2048, then scale the shortest side down to 768px.
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            w = int(w * scale)
            h = int(h * scale)

        if min(w, h) > 768:
            scale = 768 / min(w, h)
            w = int(w * scale)
            h = int(h * scale)

        tiles_w = math.ceil(w / 512)
        tiles_h = math.ceil(h / 512)
        total_tiles = tiles_w * tiles_h

        return base_tokens + (tile_tokens * total_tiles)

    else:
        print(f"Warning: Unknown model '{model}', using gpt-4o calculation")
        return calculate_image_tokens(image, "gpt-4o", detail)


def encode_image(image):
    """Encode an image as base64 PNG (converted to RGB first)."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI models end to end on SQuID")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./results",
                       help="Directory to save results (default: ./results)")
    parser.add_argument("--max-output-tokens", type=int, default=0,
                       help="Max output tokens, 0 for unlimited (default: 0)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if exists")
    parser.add_argument("--model", default="gpt-5-mini",
                       help="Model to use (default: gpt-5-mini)")
    parser.add_argument("--detail", default="high",
                       help="Detail level for image input (default: high)")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_ids, results, totals = load_checkpoint(
        output_dir, args.resume,
        token_columns=('image_tokens', 'input_tokens', 'output_tokens'),
        time_columns=('inference_time',),
        optional_time_columns=('generation_time', 'execution_time'))
    total_image_tokens = totals['total_image_tokens']
    total_input_tokens = totals['total_input_tokens']
    total_output_tokens = totals['total_output_tokens']
    total_generation_time = totals['total_generation_time']
    total_execution_time = totals['total_execution_time']
    total_inference_time = totals['total_inference_time']

    print("Loading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process):
        sample = dataset[idx]

        image_tokens = calculate_image_tokens(sample['image'], args.model, args.detail)
        total_image_tokens += image_tokens
        base64_image = encode_image(sample['image'])
        prompt = create_prompt(question_with_gsd(sample), sample['type'])

        try:
            api_params = {
                "model": args.model,
                "reasoning": {"effort": "minimal"},
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                            "detail": args.detail
                        }
                    ]
                }]
            }

            if args.max_output_tokens > 0:
                api_params["max_output_tokens"] = args.max_output_tokens

            # A single API call: generation is the whole inference, nothing is executed.
            start_time = time.time()
            response = client.responses.create(**api_params)
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            execution_time = 0
            total_execution_time += execution_time

            inference_time = generation_time
            total_inference_time += inference_time

            raw_response = response.output_text
            prediction = parse_response(raw_response, sample['type'])

            if hasattr(response, 'usage'):
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            else:
                input_tokens = output_tokens = 0

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction = None
            raw_response = str(e)
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
            'is_correct': is_correct(
                prediction,
                sample['answer'],
                sample['type'],
                sample.get('acceptable_range_lower'),
                sample.get('acceptable_range_upper')
            ),
            'image_tokens': image_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'generation_time': generation_time,
            'execution_time': execution_time,
            'inference_time': inference_time,
            'error_percentage': calculate_error_percentage(prediction, sample['answer'], sample['type']),
            'visualization_path': None,
            'segmentation_masks': None,
        })

        time.sleep(0.05)
        save_progress(results, output_dir)

    write_results(results, output_dir)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    correct = sum(1 for r in results if r['is_correct'])
    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc)

    print("\nToken Usage:")
    print(f"  Estimated image tokens: {total_image_tokens:,}")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")

    cost = report_cost(args.model, total_input_tokens, total_output_tokens, pricing=MODEL_PRICING)

    print("\nTiming:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average per sample: {total_inference_time/len(results):.3f}s")
    print(f"  Generation time: {total_generation_time:.2f}s total, {total_generation_time/len(results):.3f}s avg")
    print(f"  Execution time: {total_execution_time:.2f}s total, {total_execution_time/len(results):.3f}s avg")

    summary = {
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'model': args.model,
        'image_tokens_estimated': total_image_tokens,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'total_generation_time': total_generation_time,
        'total_execution_time': total_execution_time,
        'total_inference_time': total_inference_time,
        'avg_generation_time': total_generation_time / len(results),
        'avg_execution_time': total_execution_time / len(results),
        'avg_inference_time': total_inference_time / len(results),
        'cost': cost,
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc
    }
    write_summary(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
