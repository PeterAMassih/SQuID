"""Evaluates Qwen2.5-VL-7B-Instruct with a LoRA adapter from training/train_lora.py.

Same prompts, parsing, scoring and output files as benchmark_opensource_vlm.py.
The only difference is the PEFT adapter loaded on top of the base model.

Usage:
    python benchmark/benchmark_lora.py --lora_path runs/squid_lora --output_dir results/squid_lora
"""

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import (DATASET_ID, accuracy_by_tier, accuracy_by_type, calculate_error_percentage, create_prompt,
                    is_correct, load_benchmark, load_checkpoint, parse_response, pending_indices, print_accuracy,
                    question_with_gsd, save_progress, write_results, write_summary)

# cuDNN 9 fails on the bf16 Conv3d patch embedding on Ampere consumer GPUs (RTX A6000).
torch.backends.cudnn.enabled = False


def load_model(base_model, lora_path, device):
    """Loads the base model with the adapter and the adapter's processor.

    Args:
        base_model: Hugging Face id of the base model.
        lora_path: Directory with ``adapter_config.json``.
        device: Torch device.

    Returns:
        ``(model, processor)``.
    """
    try:
        import flash_attn  # noqa: F401
        attn = 'flash_attention_2'
    except ImportError:
        attn = 'sdpa'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=attn)
    model = PeftModel.from_pretrained(model, lora_path).eval()
    processor_path = lora_path if (Path(lora_path) / 'preprocessor_config.json').exists() else base_model
    return model, AutoProcessor.from_pretrained(processor_path)


@torch.no_grad()
def generate(model, processor, image, prompt, device, max_new_tokens):
    """Greedy-decodes one answer.

    Returns:
        ``(response, input_tokens, output_tokens)``.
    """
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id)
    n_in = inputs['input_ids'].shape[1]
    generated = output[0, n_in:].tolist()
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip(), n_in, len(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base_model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--lora_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.base_model, args.lora_path, args.device)
    processed_ids, results, totals = load_checkpoint(
        output_dir, args.resume, time_columns=('inference_time',))

    dataset = load_benchmark()
    for idx in tqdm(pending_indices(dataset, args.num_samples or len(dataset), processed_ids)):
        sample = dataset[idx]
        start = time.time()
        response, n_in, n_out = generate(model, processor, sample['image'],
                                         create_prompt(question_with_gsd(sample), sample['type']),
                                         args.device, args.max_new_tokens)
        elapsed = time.time() - start
        totals['total_inference_time'] += elapsed
        totals['total_input_tokens'] += n_in
        totals['total_output_tokens'] += n_out
        prediction = parse_response(response, sample['type'])
        lower, upper = sample['acceptable_range_lower'], sample['acceptable_range_upper']
        results.append({
            'id': sample['id'], 'question': sample['question'], 'type': sample['type'], 'tier': sample['tier'],
            'ground_truth': sample['answer'], 'acceptable_range_lower': lower, 'acceptable_range_upper': upper,
            'prediction': prediction, 'raw_response': response,
            'is_correct': is_correct(prediction, sample['answer'], sample['type'], lower, upper),
            'input_tokens': n_in, 'output_tokens': n_out, 'inference_time': elapsed,
            'error_percentage': calculate_error_percentage(prediction, sample['answer'], sample['type']),
        })
        save_progress(results, output_dir, results_every=10, checkpoint_every=100)
    write_results(results, output_dir)

    type_acc = accuracy_by_type(results)
    print_accuracy(results, type_acc)
    correct = sum(bool(r['is_correct']) for r in results)
    write_summary({
        'model': f"{args.base_model} + LoRA[{args.lora_path}]",
        'dataset': DATASET_ID,
        'overall_accuracy': 100 * correct / len(results),
        'total_samples': len(results),
        'correct_samples': correct,
        'accuracy_by_tier': accuracy_by_tier(results),
        'accuracy_by_type': type_acc,
        'total_inference_time': float(totals['total_inference_time']),
        'total_input_tokens': int(totals['total_input_tokens']),
        'total_output_tokens': int(totals['total_output_tokens']),
    }, output_dir)


if __name__ == '__main__':
    main()
