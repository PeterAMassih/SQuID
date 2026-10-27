"""LoRA fine-tuning of Qwen2.5-VL-7B-Instruct on SQuID-Train.

Plain supervised fine-tuning: one image and question in, the exact answer out,
cross-entropy on the answer tokens only. The recipe follows the LoRA SFT stage
of CoVT (Qwen-VL-Series-Finetune), without its auxiliary tokens and losses:

* LoRA r=16, alpha=32, dropout=0.01 on every linear layer of the language
  model. The vision tower, embeddings and LM head stay frozen.
* AdamW, lr 5e-5, weight decay 0.1, cosine schedule, 5% warmup, 2 epochs.
* bf16, TF32, FlashAttention-2 when available, gradient checkpointing.
* Images keep their native resolution within [256, 1280] x 28 x 28 pixels
  instead of being resized to 448 x 448, which would erase small objects.

Usage (4 GPUs, effective batch 4 x 2 x 4 = 32):
    torchrun --nproc_per_node=4 training/train_lora.py --output_dir runs/squid_lora
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

# cuDNN 9 fails on the bf16 Conv3d patch embedding on Ampere consumer GPUs
# (RTX A6000). That layer is frozen, so disabling cuDNN costs < 5% speed.
torch.backends.cudnn.enabled = False

REPO = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
SYSTEM_MESSAGE = "You are a helpful assistant."
SOURCE_DIRS = {'deepglobe': 'deepglobe_0.5m', 'earthvqa': 'earthvqa_0.3m', 'solar': 'solar_0.3m'}
LORA_EXCLUDE = ['embed_tokens', 'lm_head', 'visual']


def image_path(record):
    """Returns a training record's image path relative to ``images_and_masks/``."""
    return f"{SOURCE_DIRS[record['dataset']]}/{record['image']}"


class SquidDataset(Dataset):
    """SQuID-Train as tokenized (prompt + answer) sequences.

    Args:
        data_path: Training JSON.
        image_root: Directory holding the per-source image folders.
        processor: Qwen2.5-VL processor.
        min_pixels: Lower bound on the image resolution.
        max_pixels: Upper bound on the image resolution.
        cutoff_len: Maximum sequence length. Longer samples raise.
        seed: Seed of the one-off shuffle.
        max_samples: Optional cap on the number of records.
    """

    def __init__(self, data_path, image_root, processor, min_pixels, max_pixels,
                 cutoff_len, seed=42, max_samples=None):
        with open(data_path) as f:
            self.records = json.load(f)[:max_samples]
        np.random.default_rng(seed).shuffle(self.records)
        self.image_root = Path(image_root)
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.cutoff_len = cutoff_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        path = self.image_root / image_path(record)
        if not path.exists():
            raise FileNotFoundError(path)
        images, _ = process_vision_info([{'role': 'user', 'content': [
            {'type': 'image', 'image': str(path), 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}]}])

        prompt = (f"<|im_start|>system\n{SYSTEM_MESSAGE}\n<|im_end|>\n"
                  f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{record['question']}\n<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        response = f"{record['answer']}\n<|im_end|>\n"

        inputs = self.processor(text=[prompt], images=images, videos=None, padding=False, return_tensors='pt')
        prompt_ids = inputs['input_ids'][0]
        response_ids = self.processor.tokenizer(response, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        input_ids = torch.cat([prompt_ids, response_ids])
        if input_ids.size(0) > self.cutoff_len:
            raise RuntimeError(f"{record['id']}: {input_ids.size(0)} tokens > cutoff_len={self.cutoff_len}")
        return {
            'input_ids': input_ids,
            'labels': torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), response_ids]),
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
        }


@dataclass
class Collator:
    """Right-pads a batch and concatenates the image patches."""

    pad_token_id: int

    def __call__(self, examples):
        length = max(ex['input_ids'].size(0) for ex in examples)
        input_ids = torch.full((len(examples), length), self.pad_token_id, dtype=examples[0]['input_ids'].dtype)
        labels = torch.full((len(examples), length), IGNORE_INDEX, dtype=examples[0]['labels'].dtype)
        attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
        for i, ex in enumerate(examples):
            n = ex['input_ids'].size(0)
            input_ids[i, :n] = ex['input_ids']
            labels[i, :n] = ex['labels']
            attention_mask[i, :n] = 1
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': torch.cat([ex['pixel_values'] for ex in examples]),
            'image_grid_thw': torch.cat([ex['image_grid_thw'] for ex in examples]),
        }


def lora_targets(model):
    """Returns the linear and embedding layers that receive LoRA adapters."""
    return [name for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Embedding)) and not any(k in name for k in LORA_EXCLUDE)]


def check_disjoint(train_path, benchmark_path):
    """Raises if any benchmark image also appears in the training set."""
    with open(train_path) as f:
        train = {image_path(r) for r in json.load(f)}
    with open(benchmark_path) as f:
        overlap = train & {r['image'] for r in json.load(f)}
    if overlap:
        raise SystemExit(f"{len(overlap)} benchmark images are in the training set, e.g. {sorted(overlap)[:3]}")


def parse_args():
    """Parses the command line. Defaults point at the downloaded data/ folder."""
    data = REPO / 'data'
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_path', default=data / 'train' / 'qvlm_benchmark_training.json')
    p.add_argument('--image_root', default=data / 'train' / 'images_and_masks')
    p.add_argument('--benchmark_path', default=data / 'benchmark' / 'qvlm_benchmark.json',
                   help='benchmark JSON checked for image overlap (empty string to skip)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.01)
    p.add_argument('--min_pixels', type=int, default=256 * 28 * 28)
    p.add_argument('--max_pixels', type=int, default=1280 * 28 * 28)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--num_train_epochs', type=float, default=2.0)
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--cutoff_len', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--save_total_limit', type=int, default=6)
    p.add_argument('--dataloader_num_workers', type=int, default=4)
    p.add_argument('--report_to', default='none')
    p.add_argument('--ddp_backend', default=None, choices=['nccl', 'gloo', 'mpi'],
                   help="'gloo' where NCCL does not match the CUDA driver")
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry_run', action='store_true', help='100 steps on 200 samples')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.benchmark_path:
        check_disjoint(args.data_path, args.benchmark_path)

    try:
        import flash_attn  # noqa: F401
        attn = 'flash_attention_2'
    except ImportError:
        attn = 'sdpa'
    processor = AutoProcessor.from_pretrained(args.model_id, padding_side='right')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none',
        target_modules=lora_targets(model), task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    dataset = SquidDataset(args.data_path, args.image_root, processor, args.min_pixels, args.max_pixels,
                           args.cutoff_len, args.seed, max_samples=200 if args.dry_run else None)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=100 if args.dry_run else -1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type='cosine',
            optim='adamw_torch',
            logging_steps=10,
            save_strategy='steps',
            save_steps=50 if args.dry_run else args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': True},
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=args.report_to,
            seed=args.seed,
            ddp_find_unused_parameters=False,
            ddp_backend=args.ddp_backend,
        ),
        train_dataset=dataset,
        data_collator=Collator(processor.tokenizer.pad_token_id),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
