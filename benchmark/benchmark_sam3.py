"""SQuID benchmark for QASR with SAM 3 as the segmentation backend.

Same LLM, prompts, scoring and output format as benchmark_agent.py, but the
geometry API segments with the open-vocabulary SAM 3 model
(``evaluation.eval_sam3_segmentation``). Rows carry an extra
``segmentation_backend`` column.

Usage:
    python benchmark/benchmark_sam3.py --api-key YOUR_KEY
    python benchmark/benchmark_sam3.py --api-key YOUR_KEY --num-samples 100
    python benchmark/benchmark_sam3.py --api-key YOUR_KEY --resume
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
from evaluation.eval_sam3_segmentation import set_visualization_dir, init_sam3_model
import evaluation.eval_sam3_segmentation as seg_module
from evaluation.eval_agent_sam3 import EvalLLMCodeGeneratorSAM3


def main():
    parser = argparse.ArgumentParser(description="QASR + SAM3 Open-Vocabulary Benchmark")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./sam3_results",
                       help="Directory to save results (default: ./sam3_results)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if exists")
    parser.add_argument("--model", default="gpt-5",
                       help="Model to use (default: gpt-5)")
    parser.add_argument("--save-viz", action="store_true",
                       help="Save segmentation visualizations")
    parser.add_argument("--sam3-device", default="cuda",
                       help="Device for SAM3 model (default: cuda)")

    args = parser.parse_args()

    print("=" * 60)
    print("QASR + SAM3 OPEN-VOCABULARY BENCHMARK")
    print("=" * 60)

    print("\nInitializing SAM3...")
    if not init_sam3_model(device=args.sam3_device):
        print("\nERROR: SAM3 initialization failed!")
        print("Make sure you have:")
        print("  1. pip install transformers>=4.45.0")
        print("  2. Accepted license: https://huggingface.co/facebook/sam3")
        print("  3. Run: huggingface-cli login")
        sys.exit(1)

    agent = EvalLLMCodeGeneratorSAM3(model=args.model, api_key=args.api_key)

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

    print("\nLoading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")
        print(f"LLM: {args.model}")
        print("Segmentation: SAM3 (open-vocabulary)")

    for idx in tqdm(samples_to_process):
        sample = dataset[idx]

        question = question_with_gsd(sample)
        answer_format = format_answer_instruction(sample['type'])

        try:
            viz_counter_before = seg_module.VIZ_COUNTER if args.save_viz else 0

            start_time = time.time()
            code, token_usage, developer_prompt = agent.generate_code(question, answer_format)
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            input_tokens = token_usage['input_tokens']
            output_tokens = token_usage['output_tokens']
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            start_time = time.time()
            result = agent.execute_code(code, sample['image'])
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
            print(f"\nError on sample {idx}: {e}")
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
            'segmentation_backend': 'SAM3',
        })

        time.sleep(0.05)
        save_progress(results, output_dir)

    write_results(results, output_dir)

    print("\n" + "=" * 60)
    print("QASR + SAM3 EVALUATION RESULTS")
    print("=" * 60)

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
        'segmentation_backend': 'SAM3 (open-vocabulary)',
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
    print("  - results.csv (detailed results)")
    print("  - summary.json (aggregated metrics)")
    if args.save_viz:
        print("  - visualizations/ (segmentation overlays)")


if __name__ == "__main__":
    main()
