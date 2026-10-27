"""SQuID benchmark for GeoChat, a remote-sensing vision-language model.

GeoChat (arXiv 2311.15826) is LLaVA-1.5 with a Vicuna-1.5-7B
language model and a CLIP ViT-L/14 encoder at 336x336 (interpolated to
504x504), LoRA-tuned on GeoChat_Instruct (318k samples). The model classes
come from the GeoChat repository:

    git clone https://github.com/mbzuai-oryx/GeoChat.git ~/GeoChat
    cd ~/GeoChat && pip install -e .

Every run writes a timestamped ``benchmark_*.log`` next to ``results.csv``
and ``summary.json``. Scoring is more permissive than in the other runners
(see :func:`is_correct`), and the parser drops "approximately", "about" and
"around" before reading a number (see :func:`parse_response`).

Usage:
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B --num-samples 10
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B --resume

Requirements:
    pip install transformers torch accelerate pillow datasets pandas tqdm sentencepiece
"""

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import (DATASET_ID, calculate_error_percentage, create_prompt,
                    load_benchmark, pending_indices, question_with_gsd)


def check_geochat_installed():
    """Whether the GeoChat package and the modules this script needs are importable."""
    try:
        from geochat.model.builder import load_pretrained_model
        from geochat.mm_utils import get_model_name_from_path
        from geochat.conversation import conv_templates, SeparatorStyle
        from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        return True
    except ImportError:
        return False


def install_geochat():
    """Clone GeoChat into ``~/GeoChat`` (if absent) and ``pip install -e`` it.

    Returns:
        bool: True on success.
    """
    print("=" * 70)
    print("GeoChat not found. Attempting automatic installation...")
    print("=" * 70)

    geochat_path = os.path.expanduser("~/GeoChat")

    if not os.path.exists(geochat_path):
        print(f"Cloning GeoChat repository to {geochat_path}...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/mbzuai-oryx/GeoChat.git", geochat_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr}")
            return False

    print(f"Installing GeoChat from {geochat_path}...")
    result = subprocess.run(
        ["pip", "install", "-e", geochat_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Pip install failed: {result.stderr}")
        return False

    print("GeoChat installed successfully!")
    return True


if not check_geochat_installed():
    print("\n" + "=" * 70)
    print("ERROR: GeoChat repository not installed!")
    print("=" * 70)
    print("\nGeoChat requires custom model classes from their repository.")
    print("\nTo install manually:")
    print("  1. git clone https://github.com/mbzuai-oryx/GeoChat.git ~/GeoChat")
    print("  2. cd ~/GeoChat && pip install -e .")
    print("\nOr run this script with --auto-install flag to attempt automatic installation.")
    print("=" * 70 + "\n")

    if "--auto-install" in sys.argv and install_geochat():
        print("\nPlease restart the script now that GeoChat is installed.")
    sys.exit(1)

from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from geochat.conversation import conv_templates
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def setup_logging(output_dir, model_name):
    """Log to stdout (INFO) and to a timestamped file in ``output_dir`` (DEBUG).

    The header records the model and the Python/PyTorch/CUDA/GPU environment.

    Args:
        output_dir: Run directory (a ``Path``).
        model_name: Model path, for the header.

    Returns:
        logging.Logger: The ``geochat_benchmark`` logger.
    """

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    log_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger("geochat_benchmark")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("GEOCHAT ON THE SQUID BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Log file: {log_file}")

    logger.info("-" * 50)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("-" * 50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            logger.info(f"    Compute capability: {props.major}.{props.minor}")

    return logger


# Recorded verbatim in summary.json under 'model_info'.
GEOCHAT_INFO = {
    "model_name": "MBZUAI/geochat-7B",
    "architecture": "LLaVA-1.5",
    "visual_encoder": "CLIP ViT-L/14",
    "language_model": "Vicuna-1.5-7B",
    "input_resolution": "336x336 (interpolated to 504x504)",
    "training_data": "GeoChat_Instruct (318k samples)",
    "training_method": "LoRA fine-tuning",
    "paper": "GeoChat: Grounded Large Vision-Language Model for Remote Sensing",
    "arxiv": "2311.15826",
    "authors": "Kuckreja et al. (MBZUAI)",
    "key_capabilities": [
        "Image-level VQA",
        "Region-level reasoning",
        "Scene classification",
        "Object grounding",
        "Referring detection"
    ],
    "limitations_for_quantitative_tasks": [
        "CLIP ViT-L/14 uses 14x14 patches -> loses fine-grained spatial info",
        "336x336 base resolution -> downsampling destroys small objects",
        "Patch embedding compresses 14x14=196 pixels into single token",
        "No pixel-level precision for counting/measurement tasks"
    ]
}


def parse_response(response, q_type):
    """Extract a prediction from GeoChat's free-text answer.

    Like ``common.parse_response``, but "approximately", "about" and "around"
    are removed before the first number is taken.

    Args:
        response: Raw model answer (may be None or empty).
        q_type: SQuID question type.

    Returns:
        int, str, float or None: 0/1 for binary questions, 'connected' or
        'fragmented' for fragmentation, a float otherwise. None if nothing
        could be parsed.
    """
    if not response:
        return None

    response_clean = response.strip().lower()

    if 'binary' in q_type:
        if 'yes' in response_clean:
            return 1
        elif 'no' in response_clean:
            return 0
        else:
            return None

    elif q_type == 'fragmentation':
        if 'connected' in response_clean:
            return 'connected'
        elif 'fragmented' in response_clean:
            return 'fragmented'
        else:
            return None

    else:
        cleaned = response_clean.replace(',', '').replace('%', '').replace('approximately', '')
        cleaned = cleaned.replace('about', '').replace('around', '')
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)

        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None


def is_correct(prediction, ground_truth, q_type, lower_bound, upper_bound):
    """Score one prediction, casting both sides before comparing.

    Unlike ``common.is_correct``, fragmentation answers are compared as
    lower-cased strings, binary answers via ``int()``, numeric answers via
    ``float()``, and a missing upper bound counts as no range. Conversion
    errors score as incorrect.

    Args:
        prediction: Parsed prediction, or None.
        ground_truth: Reference answer.
        q_type: SQuID question type.
        lower_bound: Lower end of the acceptable range.
        upper_bound: Upper end of the acceptable range.

    Returns:
        bool: Whether the prediction counts as correct.
    """
    if prediction is None:
        return False

    if q_type == 'fragmentation':
        return str(prediction).lower() == str(ground_truth).lower()
    elif 'binary' in q_type:
        try:
            return int(prediction) == int(ground_truth)
        except (ValueError, TypeError):
            return False
    else:
        if lower_bound is None or upper_bound is None:
            return False
        try:
            return float(lower_bound) <= float(prediction) <= float(upper_bound)
        except (ValueError, TypeError):
            return False


def load_checkpoint(output_dir, logger):
    """Restore a previous run from ``<output_dir>/results.csv``.

    Args:
        output_dir: Run directory (a ``Path``).
        logger: Logger for the resume message and the previous accuracy.

    Returns:
        tuple: ``(processed_ids, results, stats)`` with ``stats`` holding
        ``total_input_tokens``, ``total_output_tokens``,
        ``total_generation_time`` and ``total_inference_time`` (0 when the
        column or the file is absent).
    """
    results_file = output_dir / "results.csv"

    if results_file.exists():
        df = pd.read_csv(results_file)
        existing_results = df.to_dict('records')
        processed_ids = set(df['id'].tolist())

        stats = {
            'total_input_tokens': int(df['input_tokens'].sum()) if 'input_tokens' in df.columns else 0,
            'total_output_tokens': int(df['output_tokens'].sum()) if 'output_tokens' in df.columns else 0,
            'total_generation_time': df['generation_time'].sum() if 'generation_time' in df.columns else 0,
            'total_inference_time': df['inference_time'].sum() if 'inference_time' in df.columns else 0
        }

        logger.info(f"Resuming from checkpoint: {len(processed_ids)} samples already processed")
        logger.info(f"  Checkpoint file: {results_file}")
        logger.info(f"  Previous accuracy: {100*df['is_correct'].mean():.1f}%")

        return processed_ids, existing_results, stats

    return set(), [], {
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_generation_time': 0,
        'total_inference_time': 0
    }


def save_checkpoint(results, output_dir, logger, milestone=False):
    """Write ``results.csv``, and also ``checkpoint_<n>.csv`` when ``milestone``."""
    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)

    if milestone:
        checkpoint_file = output_dir / f"checkpoint_{len(results)}.csv"
        df.to_csv(checkpoint_file, index=False)
        logger.info(f"Saved milestone checkpoint: {checkpoint_file}")


class GeoChatInference:
    """Greedy-decoding wrapper around a GeoChat checkpoint.

    The model is loaded with GeoChat's own ``load_pretrained_model``. The
    conversation template is chosen from the model name.

    Args:
        model_path: Hugging Face id or local path of the checkpoint.
        model_base: Base model for LoRA-only checkpoints, else None.
        load_8bit: Load the model in 8-bit.
        load_4bit: Load the model in 4-bit.
        device: Torch device.
        logger: Logger. Defaults to the ``geochat_benchmark`` logger.
    """

    def __init__(self, model_path: str, model_base: str = None,
                 load_8bit: bool = False, load_4bit: bool = False,
                 device: str = "cuda", logger=None):
        self.model_path = model_path
        self.device = device
        self.logger = logger or logging.getLogger("geochat_benchmark")

        self.logger.info("-" * 50)
        self.logger.info("LOADING GEOCHAT MODEL")
        self.logger.info("-" * 50)
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Model base: {model_base}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Load 8-bit: {load_8bit}")
        self.logger.info(f"Load 4-bit: {load_4bit}")

        model_name = get_model_name_from_path(model_path)
        self.logger.info(f"Model name: {model_name}")

        load_start = time.time()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name,
            load_8bit, load_4bit,
            device=device
        )
        load_time = time.time() - load_start

        self.logger.info(f"  -> Model loaded in {load_time:.1f}s")
        self.logger.info(f"  -> Context length: {self.context_len}")

        if hasattr(self.model, 'parameters'):
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"  -> Total parameters: {total_params / 1e9:.2f}B")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.info(f"  -> GPU memory allocated: {memory_allocated:.2f} GB")
            self.logger.info(f"  -> GPU memory reserved: {memory_reserved:.2f} GB")

        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.logger.info(f"  -> Conversation mode: {self.conv_mode}")

        if hasattr(self.image_processor, 'crop_size'):
            self.logger.info(f"  -> Image size: {self.image_processor.crop_size}")
        elif hasattr(self.image_processor, 'size'):
            self.logger.info(f"  -> Image size: {self.image_processor.size}")

    def generate(self, image, prompt: str, max_new_tokens: int = 256) -> tuple:
        """Answer one question.

        Args:
            image: PIL image or path to one.
            prompt: Text prompt.
            max_new_tokens: Generation budget.

        Returns:
            tuple: ``(response, input_tokens, output_tokens, generation_time)``.
            ``(None, 0, 0, 0.0)`` if generation raised.
        """
        try:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            image_tensor = process_images([image], self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            conv = conv_templates[self.conv_mode].copy()

            if self.model.config.mm_use_im_start_end:
                from geochat.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.model.device)

            input_tokens = input_ids.shape[1]

            gen_start = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            generation_time = time.time() - gen_start

            generated_ids = output_ids[0][input_tokens:]
            output_tokens = len(generated_ids)

            response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            return response, input_tokens, output_tokens, generation_time

        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            self.logger.debug(traceback.format_exc())
            return None, 0, 0, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GeoChat on SQuID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # First install GeoChat (one-time):
    git clone https://github.com/mbzuai-oryx/GeoChat.git ~/GeoChat
    cd ~/GeoChat && pip install -e .
    
    # Quick test with 10 samples
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B --num-samples 10
    
    # Full benchmark (2000 samples)
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B
    
    # Resume interrupted run
    python benchmark/benchmark_geochat.py --model-path MBZUAI/geochat-7B --resume
    
    # Auto-install GeoChat if not present
    python benchmark/benchmark_geochat.py --auto-install --model-path MBZUAI/geochat-7B
        """
    )

    parser.add_argument("--model-path", default="MBZUAI/geochat-7B",
                       help="HuggingFace model path or local path (default: MBZUAI/geochat-7B)")
    parser.add_argument("--model-base", default=None,
                       help="Base model path (for LoRA weights)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all 2000)")
    parser.add_argument("--output-dir", default="./geochat_results",
                       help="Directory to save results")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                       help="Max new tokens (default: 256)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run model on")
    parser.add_argument("--load-8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--load-4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                       help="Save checkpoint every N samples (default: 10)")
    parser.add_argument("--milestone-interval", type=int, default=100,
                       help="Save milestone checkpoint every N samples (default: 100)")
    parser.add_argument("--auto-install", action="store_true",
                       help="Automatically install GeoChat if not found")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logging(output_dir, args.model_path)

    logger.info("-" * 50)
    logger.info("BENCHMARK CONFIGURATION")
    logger.info("-" * 50)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    logger.info("-" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("-" * 50)
    for key, value in GEOCHAT_INFO.items():
        if isinstance(value, list):
            logger.info(f"  {key}:")
            for item in value:
                logger.info(f"    - {item}")
        else:
            logger.info(f"  {key}: {value}")

    inference = GeoChatInference(
        args.model_path,
        args.model_base,
        args.load_8bit,
        args.load_4bit,
        args.device,
        logger
    )

    if args.resume:
        processed_ids, results, stats = load_checkpoint(output_dir, logger)
        total_input_tokens = stats['total_input_tokens']
        total_output_tokens = stats['total_output_tokens']
        total_generation_time = stats['total_generation_time']
        total_inference_time = stats['total_inference_time']
    else:
        processed_ids = set()
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_generation_time = 0
        total_inference_time = 0

    logger.info("-" * 50)
    logger.info("LOADING SQUID BENCHMARK DATASET")
    logger.info("-" * 50)

    dataset = load_benchmark()
    total_samples = len(dataset)
    num_samples = args.num_samples or total_samples

    logger.info(f"Dataset size: {total_samples} samples")
    logger.info(f"Evaluating: {num_samples} samples")

    type_counts = {}
    tier_counts = {1: 0, 2: 0, 3: 0}
    for i in range(min(num_samples, total_samples)):
        q_type = dataset[i]['type']
        tier = dataset[i]['tier']
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    logger.info("Dataset composition:")
    logger.info(f"  Tier 1 (Basic): {tier_counts[1]} samples")
    logger.info(f"  Tier 2 (Spatial): {tier_counts[2]} samples")
    logger.info(f"  Tier 3 (Complex): {tier_counts[3]} samples")
    logger.info(f"  Question types: {len(type_counts)}")

    samples_to_process = pending_indices(dataset, num_samples, processed_ids)

    if not samples_to_process:
        logger.info("All samples already processed!")
    else:
        logger.info(f"Processing {len(samples_to_process)} remaining samples...")

    logger.info("-" * 50)
    logger.info("STARTING EVALUATION")
    logger.info("-" * 50)

    eval_start_time = time.time()

    for idx in tqdm(samples_to_process, desc="Evaluating GeoChat"):
        sample = dataset[idx]
        sample_start = time.time()

        image = sample['image']
        prompt = create_prompt(question_with_gsd(sample), sample['type'])

        try:
            response, input_tokens, output_tokens, generation_time = inference.generate(
                image, prompt,
                max_new_tokens=args.max_new_tokens
            )

            total_generation_time += generation_time
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            prediction = parse_response(response, sample['type'])
            raw_response = response if response else ""

        except Exception as e:
            logger.error(f"Error on sample {sample['id']}: {e}")
            prediction = None
            raw_response = ""
            response = ""
            input_tokens = output_tokens = 0
            generation_time = 0.0

        sample_time = time.time() - sample_start
        total_inference_time += sample_time

        correct = is_correct(
            prediction,
            sample['answer'],
            sample['type'],
            sample.get('acceptable_range_lower'),
            sample.get('acceptable_range_upper')
        )

        error_pct = calculate_error_percentage(
            prediction,
            sample['answer'],
            sample['type']
        )

        result = {
            'id': sample['id'],
            'question': sample['question'],
            'type': sample['type'],
            'tier': sample['tier'],
            'ground_truth': sample['answer'],
            'acceptable_range_lower': sample.get('acceptable_range_lower'),
            'acceptable_range_upper': sample.get('acceptable_range_upper'),
            'prediction': prediction,
            'raw_response': raw_response,
            'thinking_trace': "",
            'is_correct': correct,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'generation_time': generation_time,
            'execution_time': 0,
            'inference_time': sample_time,
            'error_percentage': error_pct,
            'visualization_path': None,
            'segmentation_masks': None,
            'model': args.model_path,
        }

        results.append(result)

        if len(results) <= 5:
            logger.debug(f"Sample {sample['id']}:")
            logger.debug(f"  Type: {sample['type']}, Tier: {sample['tier']}")
            logger.debug(f"  GT: {sample['answer']}, Pred: {prediction}, Correct: {correct}")
            logger.debug(f"  Response: {raw_response[:100]}...")

        if len(results) % args.checkpoint_interval == 0:
            save_checkpoint(results, output_dir, logger, milestone=False)

            current_acc = 100 * sum(1 for r in results if r['is_correct']) / len(results)
            logger.info(f"Progress: {len(results)}/{len(samples_to_process)} samples, "
                       f"Accuracy: {current_acc:.1f}%, "
                       f"Avg time: {total_inference_time/len(results):.2f}s/sample")

        if len(results) % args.milestone_interval == 0:
            save_checkpoint(results, output_dir, logger, milestone=True)

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    total_eval_time = time.time() - eval_start_time
    save_checkpoint(results, output_dir, logger, milestone=True)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    if not results:
        logger.warning("No results to analyze!")
        return

    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = 100 * correct_count / len(results)

    logger.info(f"\nOVERALL ACCURACY: {correct_count}/{len(results)} = {accuracy:.2f}%")

    logger.info("\nACCURACY BY TIER:")
    tier_results = {}
    for tier in [1, 2, 3]:
        tier_data = [r for r in results if r['tier'] == tier]
        if tier_data:
            tier_correct = sum(1 for r in tier_data if r['is_correct'])
            tier_acc = 100 * tier_correct / len(tier_data)
            tier_name = ['Basic', 'Spatial', 'Complex'][tier - 1]
            tier_results[f'tier_{tier}'] = {
                'name': tier_name,
                'correct': tier_correct,
                'total': len(tier_data),
                'accuracy': tier_acc
            }
            logger.info(f"  Tier {tier} ({tier_name}): {tier_correct}/{len(tier_data)} = {tier_acc:.1f}%")

    logger.info("\nACCURACY BY QUESTION TYPE:")
    type_results = {}
    for q_type in sorted(set(r['type'] for r in results)):
        type_data = [r for r in results if r['type'] == q_type]
        type_correct = sum(1 for r in type_data if r['is_correct'])
        type_acc = 100 * type_correct / len(type_data)
        type_results[q_type] = {
            'correct': type_correct,
            'total': len(type_data),
            'accuracy': type_acc
        }
        logger.info(f"  {q_type}: {type_correct}/{len(type_data)} = {type_acc:.1f}%")

    sorted_types = sorted(type_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    logger.info("\nBEST PERFORMING TYPES:")
    for q_type, data in sorted_types[:5]:
        logger.info(f"  {q_type}: {data['accuracy']:.1f}%")

    logger.info("\nWORST PERFORMING TYPES:")
    for q_type, data in sorted_types[-5:]:
        logger.info(f"  {q_type}: {data['accuracy']:.1f}%")

    numeric_results = [r for r in results if r['error_percentage'] is not None]
    errors = []
    if numeric_results:
        errors = [r['error_percentage'] for r in numeric_results]
        logger.info("\nERROR ANALYSIS (numeric questions):")
        logger.info(f"  Mean error: {sum(errors)/len(errors):.2f}%")
        logger.info(f"  Median error: {sorted(errors)[len(errors)//2]:.2f}%")
        logger.info(f"  Max error: {max(errors):.2f}%")

    logger.info("\nTOKEN USAGE:")
    logger.info(f"  Total input tokens: {total_input_tokens:,}")
    logger.info(f"  Total output tokens: {total_output_tokens:,}")
    logger.info(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")
    logger.info(f"  Avg input tokens/sample: {total_input_tokens/len(results):.0f}")
    logger.info(f"  Avg output tokens/sample: {total_output_tokens/len(results):.0f}")

    logger.info("\nTIMING:")
    logger.info(f"  Total evaluation time: {total_eval_time:.1f}s ({total_eval_time/60:.1f} min)")
    logger.info(f"  Total generation time: {total_generation_time:.1f}s")
    logger.info(f"  Avg time per sample: {total_inference_time/len(results):.3f}s")
    logger.info(f"  Throughput: {len(results)/total_eval_time:.2f} samples/sec")

    null_responses = sum(1 for r in results if r['prediction'] is None)
    logger.info("\nRESPONSE ANALYSIS:")
    logger.info(f"  Valid predictions: {len(results) - null_responses}/{len(results)}")
    logger.info(f"  Failed to parse: {null_responses}/{len(results)}")

    summary = {
        'benchmark_info': {
            'script': 'benchmark_geochat.py',
            'timestamp': datetime.now().isoformat(),
            'squid_version': DATASET_ID,
        },
        'model_info': GEOCHAT_INFO,
        'configuration': {
            'model_path': args.model_path,
            'model_base': args.model_base,
            'max_new_tokens': args.max_new_tokens,
            'device': args.device,
            'load_8bit': args.load_8bit,
            'load_4bit': args.load_4bit,
        },
        'results': {
            'overall_accuracy': accuracy,
            'total_samples': len(results),
            'correct_samples': correct_count,
            'null_predictions': null_responses,
        },
        'accuracy_by_tier': tier_results,
        'accuracy_by_type': {k: v['accuracy'] for k, v in type_results.items()},
        'timing': {
            'total_evaluation_time_sec': total_eval_time,
            'total_generation_time_sec': total_generation_time,
            'avg_inference_time_sec': total_inference_time / len(results),
            'throughput_samples_per_sec': len(results) / total_eval_time,
        },
        'token_usage': {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'avg_input_tokens': total_input_tokens / len(results),
            'avg_output_tokens': total_output_tokens / len(results),
        },
        'error_analysis': {
            'mean_error_pct': sum(errors)/len(errors) if errors else None,
            'median_error_pct': sorted(errors)[len(errors)//2] if errors else None,
        } if errors else None,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("  - results.csv: Full results")
    logger.info("  - summary.json: Summary statistics")
    logger.info("  - benchmark_*.log: Detailed logs")


if __name__ == "__main__":
    main()
