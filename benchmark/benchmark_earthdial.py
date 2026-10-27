"""SQuID benchmark for EarthDial, a remote-sensing vision-language model.

EarthDial (MBZUAI + IBM) is an RS-specialized VLM: InternVL2-4B
lineage (InternViT-300M @ 448px dynamic tiling + Phi-3-mini-128k LLM),
instruction-tuned on 11.11M RS pairs. Free-form answers, greedy decoding
(matches the paper's own evaluation protocol).

Requires the EarthDial repository, which vendors the model code (the HF
checkpoint's auto_map files are not on the Hub):
    git clone https://github.com/hiyamdebary/EarthDial.git ~/EarthDial
    cd ~/EarthDial && pip install -e .        # pins transformers==4.37.2
    export PYTHONPATH=~/EarthDial/src:$PYTHONPATH   # src-layout quirk
Checkpoint (8.3 GB, not gated, MIT):
    huggingface-cli download akshaydudhane/EarthDial_4B_RGB \
        --local-dir ~/checkpoints/EarthDial_4B_RGB

The question is suffixed with the image GSD before the answer-format
instruction is added.

Usage:
    python benchmark/benchmark_earthdial.py --model-path ~/checkpoints/EarthDial_4B_RGB --num-samples 10
    python benchmark/benchmark_earthdial.py --model-path ~/checkpoints/EarthDial_4B_RGB --resume
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from common import (accuracy_by_tier, accuracy_by_type, create_prompt, is_correct,
                    load_benchmark, parse_response, pending_indices, print_accuracy,
                    question_with_gsd, save_progress, write_results, write_summary)

try:
    from transformers import AutoTokenizer
    from earthdial.model.internvl_chat import InternVLChatModel
    from earthdial.train.dataset import build_transform, dynamic_preprocess
except ImportError as e:
    print("=" * 70)
    print(f"ERROR: EarthDial package not importable ({e}).")
    print("  git clone https://github.com/hiyamdebary/EarthDial.git ~/EarthDial")
    print("  cd ~/EarthDial && pip install -e .")
    print("  export PYTHONPATH=~/EarthDial/src:$PYTHONPATH")
    print("=" * 70)
    sys.exit(1)


def load_checkpoint(output_dir):
    """Restore a previous run from ``<output_dir>/results.csv``.

    Args:
        output_dir: Run directory (a ``Path``).

    Returns:
        tuple: ``(processed_ids, results, total_inference_time)``. Empty and
        0.0 when there is no results file.
    """
    results_file = output_dir / "results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        processed_ids = set(df['id'].tolist())
        print(f"Resuming from checkpoint: {len(processed_ids)} samples already processed")
        return processed_ids, df.to_dict('records'), df['inference_time'].sum()
    return set(), [], 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark EarthDial on SQuID")
    parser.add_argument("--model-path", default="~/checkpoints/EarthDial_4B_RGB",
                        help="Local snapshot of akshaydudhane/EarthDial_4B_RGB")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./earthdial_results",
                        help="Directory to save results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=6,
                        help="Dynamic tiling max_num (training used 6)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_path = str(Path(args.model_path).expanduser())
    print(f"Loading EarthDial from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    transform = build_transform(is_train=False, input_size=448, normalize_type='imagenet')

    generation_config = dict(num_beams=1, do_sample=False,
                             max_new_tokens=args.max_new_tokens, min_new_tokens=1)

    def ask(pil_image, question):
        tiles = dynamic_preprocess(pil_image.convert("RGB"), min_num=1,
                                   max_num=args.max_tiles, image_size=448, use_thumbnail=True)
        pixel_values = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).cuda()
        with torch.inference_mode():
            return model.chat(tokenizer=tokenizer, pixel_values=pixel_values,
                              question=question, generation_config=generation_config)

    if args.resume:
        processed_ids, results, total_inference_time = load_checkpoint(output_dir)
    else:
        processed_ids, results, total_inference_time = set(), [], 0.0

    print("Loading SQuID benchmark...")
    dataset = load_benchmark()
    num_samples = args.num_samples or len(dataset)

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)
    print(f"Evaluating {len(samples_to_process)} remaining samples (out of {num_samples} total)...")

    for idx in tqdm(samples_to_process):
        sample = dataset[idx]

        prompt = create_prompt(question_with_gsd(sample), sample['type'])

        try:
            start_time = time.time()
            response = ask(sample['image'], prompt)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            prediction = parse_response(response, sample['type'])
            raw_response = response
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction = None
            raw_response = str(e)
            inference_time = 0

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
            'inference_time': inference_time,
        })

        save_progress(results, output_dir)

    write_results(results, output_dir)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS - EarthDial")
    print("=" * 50)

    correct = sum(1 for r in results if r['is_correct'])
    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc, n_best=0, n_worst=0)

    summary = {
        'model': 'EarthDial_4B_RGB',
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'total_inference_time': total_inference_time,
        'avg_inference_time': total_inference_time / len(results),
        'max_tiles': args.max_tiles,
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc,
    }
    write_summary(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
