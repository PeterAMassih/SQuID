"""SQuID benchmark for QASR: an OpenAI model writes Python against the geometry API.

For each question the LLM generates a program (``EvalLLMCodeGenerator``),
which is executed on the image with the default segmentation backend
(``evaluation.eval_satellite_processing``). The ``answer`` variable it leaves
behind is scored. Segmentation masks of every run are saved under
``<output-dir>/masks/``.

Usage:
    python benchmark/benchmark_agent.py --api-key YOUR_KEY
    python benchmark/benchmark_agent.py --api-key YOUR_KEY --num-samples 10
    python benchmark/benchmark_agent.py --api-key YOUR_KEY --resume
"""

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from common import (accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    format_answer_instruction, is_correct, load_benchmark, load_checkpoint,
                    parse_answer, pending_indices, print_accuracy, question_with_gsd,
                    report_cost, save_last_masks, save_progress, write_results, write_summary)

sys.path.append(str(Path(__file__).parent.parent))
from evaluation.eval_satellite_processing import set_visualization_dir
import evaluation.eval_satellite_processing as seg_module
from evaluation.eval_agent import EvalLLMCodeGenerator


def main():
    parser = argparse.ArgumentParser(description="QASR Agent Benchmark")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./agent_results",
                       help="Directory to save results (default: ./agent_results)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if exists")
    parser.add_argument("--model", default="gpt-5-mini",
                       help="Model to use (default: gpt-5-mini)")
    parser.add_argument("--save-viz", action="store_true",
                       help="Save segmentation visualizations")

    args = parser.parse_args()

    agent = EvalLLMCodeGenerator(model=args.model, api_key=args.api_key)

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
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process):
        sample = dataset[idx]

        question = question_with_gsd(sample)
        answer_format = format_answer_instruction(sample['type'])

        try:
            start_time = time.time()
            code, token_usage, developer_prompt = agent.generate_code(question, answer_format)
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
            'error_percentage': calculate_error_percentage(prediction, sample['answer'], sample['type']),
            'visualization_path': viz_path,
            'segmentation_masks': mask_path,
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
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")

    cost = report_cost(args.model, total_input_tokens, total_output_tokens)

    print("\nTiming:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average per sample: {total_inference_time/len(results):.3f}s")
    print(f"  Code generation: {total_generation_time:.2f}s total, {total_generation_time/len(results):.3f}s avg")
    print(f"  Code execution: {total_execution_time:.2f}s total, {total_execution_time/len(results):.3f}s avg")

    summary = {
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'model': args.model,
        'total_generation_time': total_generation_time,
        'total_execution_time': total_execution_time,
        'total_inference_time': total_inference_time,
        'avg_generation_time': total_generation_time / len(results),
        'avg_execution_time': total_execution_time / len(results),
        'avg_inference_time': total_inference_time / len(results),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'cost': cost,
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc
    }
    write_summary(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
