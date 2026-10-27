"""SQuID benchmark for QASR on the reference masks.

Same agent, API and scoring as benchmark_agent.py, but
``segment_image_from_path`` serves the SQuID reference masks
(``USE_GT_MASKS=1``), so the run measures the LLM + geometry-API ceiling with
perfect perception. Questions come from the local benchmark JSON, which
carries image and mask paths, rather than from the Hub. The 50 NAIP
questions have no masks and are skipped.

With ``--replay``, the programs logged in the ``results.csv`` of a
predicted-mask run (benchmark_agent.py or benchmark_opensource_agent.py) run
again on the reference masks, with no new LLM call. This is the paper's
reference-mask mode. Without it, the LLM writes new programs.

By default the JSON is read from ``data/benchmark/qvlm_benchmark.json`` under
the repository root, with images and masks in
``data/benchmark/images_and_masks/``.

Usage:
    python benchmark/benchmark_agent_gt.py --replay agent_results/results.csv
    python benchmark/benchmark_agent_gt.py --api-key YOUR_KEY
    python benchmark/benchmark_agent_gt.py --api-key YOUR_KEY --num-samples 10
    python benchmark/benchmark_agent_gt.py --api-key YOUR_KEY --resume \
        --bench-json /path/to/qvlm_benchmark.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Must be set before the evaluation modules are imported.
os.environ["USE_GT_MASKS"] = "1"
os.environ.setdefault("USE_DINOV3", "0")

from tqdm import tqdm

from common import (REPO_ROOT, accuracy_by_tier, accuracy_by_type, calculate_error_percentage,
                    format_answer_instruction, is_correct, load_checkpoint, parse_answer,
                    print_accuracy, question_with_gsd, save_last_masks, save_progress,
                    write_results, write_summary)

sys.path.append(str(Path(__file__).parent.parent))
import evaluation.eval_satellite_processing as seg_module
from evaluation.eval_agent import EvalLLMCodeGenerator

# USD per 1M tokens. Unlisted models are billed at gpt-5 rates.
MODEL_PRICING = {
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00},
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
}

DEFAULT_BENCH_JSON = str(REPO_ROOT / "data" / "benchmark" / "qvlm_benchmark.json")


def load_bench(bench_json, data_root):
    """Read the local benchmark JSON into rows shaped like the Hub dataset.

    Args:
        bench_json: Path to ``qvlm_benchmark.json``.
        data_root: Directory that the JSON's relative image paths resolve against.

    Returns:
        tuple: ``(samples, skipped)``: the questions that have a ground-truth
        mask, with ``image`` as an absolute path string, and the number of
        questions skipped for lacking one (NAIP).
    """
    with open(Path(bench_json).expanduser()) as f:
        records = json.load(f)
    samples, skipped = [], 0
    for r in records:
        if not r.get('mask'):
            skipped += 1
            continue
        lo, hi = (r['acceptable_range'] if r.get('acceptable_range') else (None, None))
        samples.append({
            'id': r['id'],
            'question': r['question'],
            'type': r['type'],
            'tier': r['tier'],
            'answer': r['answer'],
            'gsd': r.get('gsd', 1.0),
            'acceptable_range_lower': lo,
            'acceptable_range_upper': hi,
            'image': str(Path(data_root).expanduser() / r['image']),
        })
    return samples, skipped


def main():
    parser = argparse.ArgumentParser(description="QASR on the SQuID reference masks")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--bench-json", default=DEFAULT_BENCH_JSON,
                        help="Benchmark JSON with image/mask paths")
    parser.add_argument("--data-root", default=None,
                        help="Root of images_and_masks (default: sibling of --bench-json)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all masked)")
    parser.add_argument("--replay", metavar="RESULTS_CSV",
                        help="Re-run the programs of a predicted-mask run instead of generating new ones")
    parser.add_argument("--output-dir", default="./agent_gt_results",
                        help="Directory to save results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if exists")
    parser.add_argument("--model", default="gpt-5",
                        help="Model to use (default: gpt-5)")
    args = parser.parse_args()

    data_root = args.data_root or str(Path(args.bench_json).expanduser().parent / "images_and_masks")
    os.environ["GT_DATA_ROOT"] = data_root

    if args.replay:
        programs = pd.read_csv(args.replay).set_index('id')['raw_response'].fillna('').to_dict()
    else:
        agent = EvalLLMCodeGenerator(model=args.model, api_key=args.api_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_ids, results, totals = load_checkpoint(output_dir, args.resume)
    total_generation_time = totals['total_generation_time']
    total_execution_time = totals['total_execution_time']
    total_inference_time = totals['total_inference_time']
    total_input_tokens = totals['total_input_tokens']
    total_output_tokens = totals['total_output_tokens']

    print(f"Loading SQuID benchmark from {args.bench_json} ...")
    samples, n_naip = load_bench(args.bench_json, data_root)
    print(f"{len(samples)} mask-backed questions ({n_naip} NAIP questions skipped: no reference mask)")
    if args.replay:
        samples = [s for s in samples if s['id'] in programs]
        if not samples:
            sys.exit(f"No question of {args.bench_json} is in {args.replay}. Replay a run on the same benchmark.")
        print(f"Replaying the {len(samples)} programs logged in {args.replay}")
    num_samples = args.num_samples or len(samples)

    samples_to_process = [s for s in samples[:num_samples] if s['id'] not in processed_ids]
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for sample in tqdm(samples_to_process):
        question = question_with_gsd(sample)
        answer_format = format_answer_instruction(sample['type'])

        try:
            start_time = time.time()
            if args.replay:
                code, token_usage, developer_prompt = programs[sample['id']], {'input_tokens': 0, 'output_tokens': 0}, ''
            else:
                code, token_usage, developer_prompt = agent.generate_code(question, answer_format)
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            input_tokens = token_usage['input_tokens']
            output_tokens = token_usage['output_tokens']
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            start_time = time.time()
            result = EvalLLMCodeGenerator.execute_code(code, sample['image'], sample['gsd'])
            execution_time = time.time() - start_time
            total_execution_time += execution_time

            mask_path = save_last_masks(seg_module, output_dir, sample['id'])

            inference_time = generation_time + execution_time
            total_inference_time += inference_time

            prediction = parse_answer(result, sample['type'])
            raw_response = code if code else ""
            code_prints = result['prints']
            exec_error = result['error']

        except Exception as e:
            print(f"Error on sample {sample['id']}: {e}")
            code_prints = ''
            prediction = None
            raw_response = str(e)
            developer_prompt = ''
            exec_error = str(e)
            generation_time = execution_time = inference_time = 0
            input_tokens = output_tokens = 0
            mask_path = None
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
            'execution_error': exec_error,
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
            'visualization_path': None,
            'segmentation_masks': mask_path,
        })

        time.sleep(0.05)
        save_progress(results, output_dir)

    write_results(results, output_dir)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (REFERENCE MASKS)")
    print("=" * 50)

    correct = sum(1 for r in results if r['is_correct'])
    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc, n_best=0, n_worst=8)

    pricing = MODEL_PRICING.get(args.model, MODEL_PRICING['gpt-5'])
    input_cost = (total_input_tokens / 1_000_000) * pricing['input']
    output_cost = (total_output_tokens / 1_000_000) * pricing['output']
    if not args.replay:
        print(f"\nTokens: {total_input_tokens:,} in / {total_output_tokens:,} out"
              f"  |  Cost ({args.model}): ${input_cost + output_cost:.2f}")

    summary = {
        'mode': 'replay' if args.replay else 'reference_masks',
        'replay': args.replay,
        'bench_json': str(Path(args.bench_json).expanduser()),
        'naip_skipped': n_naip,
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'model': None if args.replay else args.model,
        'total_generation_time': total_generation_time,
        'total_execution_time': total_execution_time,
        'total_inference_time': total_inference_time,
        'avg_inference_time': total_inference_time / len(results),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'cost': {'total': input_cost + output_cost, 'input': input_cost, 'output': output_cost},
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc,
    }
    write_summary(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
