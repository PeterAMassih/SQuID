"""Shared helpers for the SQuID benchmark runners.

Each ``benchmark_*.py`` script evaluates one system of Table 3 on the SQuID
benchmark split. This module holds the parts they share: loading the
benchmark, the answer-format prompts and parsers, scoring against the
acceptable ranges, resumable ``results.csv`` files and the accuracy report.

Runners are launched from the repository root and import this module by its
bare name, which works because Python puts the script's directory
(``benchmark/``) on ``sys.path``::

    python benchmark/benchmark_gpt.py --api-key $OPENAI_API_KEY --num-samples 10

Some helpers have runner-specific variants. These are selected through
explicit arguments whose defaults reproduce the most common caller, and each
docstring names the runners that use a non-default value.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_REPO = "PeterAM4/SQuID"
DATASET_CONFIG = "benchmark"
DATASET_SPLIT = "test"
DATASET_REVISION = "v2.0"
DATASET_ID = f"{DATASET_REPO}:{DATASET_CONFIG}/{DATASET_SPLIT}"

TIER_NAMES = {1: 'Basic', 2: 'Spatial', 3: 'Complex'}

# Prices in USD per 1M tokens, used by the agent runners (benchmark_agent.py,
# benchmark_remotesam.py, benchmark_sam3.py). Unlisted models are billed at
# gpt-5-mini rates.
MODEL_PRICING = {
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00},
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
    'o4-mini': {'input': 1.10, 'output': 4.40},
}


def load_benchmark():
    """Load the SQuID benchmark split from the Hugging Face Hub.

    Returns:
        datasets.Dataset: One row per question with columns ``id``, ``image``
        (PIL), ``mask`` (PIL or None), ``question``, ``answer`` (string),
        ``type``, ``tier``, ``gsd``, ``acceptable_range_lower``,
        ``acceptable_range_upper`` and ``source``.
    """
    return load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)


def question_with_gsd(sample):
    """Append the ground sampling distance to a question unless it already states it.

    Used by every runner, so that each system receives the GSD. Only some
    NAIP questions lack it.

    Args:
        sample: Benchmark row with ``question`` and optionally ``gsd``.

    Returns:
        str: The question, suffixed with ``" (GSD: <gsd>m)"`` when it contains
        neither ``"GSD:"`` nor ``"gsd:"``.
    """
    question = sample['question']
    gsd = sample.get('gsd', 1.0)
    if "GSD:" not in question and "gsd:" not in question:
        question = f"{question} (GSD: {gsd}m)"
    return question


def create_prompt(question, q_type):
    """Build the direct-answer prompt for a VLM.

    Used by benchmark_gpt.py, benchmark_opensource_vlm.py,
    benchmark_geochat.py and benchmark_earthdial.py. benchmark_geor1.py keeps
    its own version, which matches binary types by substring.

    Args:
        question: Question text.
        q_type: SQuID question type.

    Returns:
        str: The question followed by an answer-format instruction.
    """
    if q_type in ['percentage', 'proximity_percentage']:
        return f"{question}\n\nAnswer with ONLY a number between 0 and 100:"
    elif q_type in ['binary_comparison', 'binary_presence', 'binary_threshold',
                    'binary_multiple', 'binary_proximity']:
        return f"{question}\n\nAnswer with ONLY 'yes' or 'no':"
    elif q_type == 'fragmentation':
        return f"{question}\n\nAnswer with ONLY 'connected' or 'fragmented':"
    else:
        return f"{question}\n\nAnswer with ONLY a number:"


def parse_response(response, q_type, number_index=0):
    """Extract a prediction from a free-text VLM answer.

    Binary questions map to 1 if "yes" occurs anywhere in the answer, else 0
    if "no" occurs. Fragmentation questions look for "connected" before
    "fragmented". Everything else takes a number from the answer after
    removing commas and percent signs.

    Args:
        response: Raw model answer (may be None or empty).
        q_type: SQuID question type.
        number_index: Which number to keep when several appear. 0 (first) for
            benchmark_gpt.py, benchmark_opensource_vlm.py and
            benchmark_earthdial.py. -1 (last, i.e. after any reasoning) for
            benchmark_geor1.py.

    Returns:
        int, str, float or None: 0/1 for binary questions, 'connected' or
        'fragmented' for fragmentation, a float otherwise. None if nothing
        could be parsed.
    """
    if not response:
        return None

    response = response.strip().lower()

    if 'binary' in q_type:
        return 1 if 'yes' in response else (0 if 'no' in response else None)
    elif q_type == 'fragmentation':
        return 'connected' if 'connected' in response else ('fragmented' if 'fragmented' in response else None)
    else:
        numbers = re.findall(r'-?\d+\.?\d*', response.replace(',', '').replace('%', ''))
        return float(numbers[number_index]) if numbers else None


def format_answer_instruction(q_type):
    """Tell the code-generating LLM how to store its answer.

    Used by benchmark_agent.py, benchmark_agent_gt.py, benchmark_remotesam.py
    and benchmark_sam3.py. benchmark_opensource_agent.py keeps its own copy,
    whose wording differs by one word.

    Args:
        q_type: SQuID question type.

    Returns:
        str: Instruction appended to the agent's developer prompt.
    """
    if q_type in ['percentage', 'proximity_percentage']:
        ret = "Store your final answer as a number between 0 and 100 in 'answer'."
    elif 'binary' in q_type:
        ret = "Store your final answer as 1 for yes or 0 for no in 'answer'."
    elif q_type == 'fragmentation':
        ret = "Store your final answer as 'connected' or 'fragmented' in 'answer'."
    else:
        ret = "Store your final answer in 'answer'. answer must be simple numeric type (float or int)."

    return ret + " Print the number of initial shapes found for each topics, as well as other relevant intermediate findings."


def parse_answer(result, q_type):
    """Convert the ``answer`` variable left by executed agent code into a prediction.

    Args:
        result: Dict returned by the agent's ``execute_code`` with keys
            ``success`` and ``answer``.
        q_type: SQuID question type.

    Returns:
        int, str, float or None: 0/1 for binary questions (bools, or the
        numbers 0 and 1), a lower-cased 'connected'/'fragmented' for
        fragmentation, a float otherwise. None if execution failed or the
        answer has the wrong form.
    """
    if not result['success'] or result['answer'] is None:
        return None

    answer = result['answer']

    if q_type == 'fragmentation':
        return answer.lower() if isinstance(answer, str) and answer.lower() in ['connected', 'fragmented'] else None
    elif 'binary' in q_type:
        if isinstance(answer, bool):
            return 1 if answer else 0
        elif isinstance(answer, (int, float)):
            return int(answer) if answer in [0, 1] else None
    else:
        try:
            return float(answer)
        except:
            return None


def is_correct(prediction, ground_truth, q_type, lower_bound, upper_bound):
    """Score one prediction.

    Used by every runner except benchmark_geochat.py, which keeps a more
    permissive version that casts both sides before comparing.

    Args:
        prediction: Parsed prediction, or None.
        ground_truth: Reference answer as stored in the dataset (string).
        q_type: SQuID question type.
        lower_bound: Lower end of the acceptable range (numeric questions).
        upper_bound: Upper end of the acceptable range (numeric questions).

    Returns:
        bool: Exact match for fragmentation and binary questions. For numeric
        questions, whether the prediction lies in the closed acceptable range
        (False when the question has no range).
    """
    if prediction is None:
        return False

    if q_type == 'fragmentation':
        return prediction == ground_truth
    elif 'binary' in q_type:
        return prediction == int(ground_truth)
    else:
        return lower_bound <= prediction <= upper_bound if lower_bound is not None else False


def calculate_error_percentage(prediction, ground_truth, q_type):
    """Error of a numeric prediction.

    Args:
        prediction: Parsed prediction, or None.
        ground_truth: Reference answer (string or number).
        q_type: SQuID question type.

    Returns:
        float or None: Absolute difference in percentage points for
        percentage types, relative deviation in percent for counts and other
        numeric types. None for binary and fragmentation questions, missing
        predictions, non-numeric references, and counts whose reference is 0.
    """
    if prediction is None or 'binary' in q_type or q_type == 'fragmentation':
        return None

    try:
        ground_truth = float(ground_truth)
    except (ValueError, TypeError):
        return None

    if 'percentage' in q_type:
        return abs(prediction - ground_truth)

    if ground_truth == 0:
        return None

    return abs(prediction - ground_truth) / abs(ground_truth) * 100


def load_checkpoint(output_dir, resume,
                    token_columns=('input_tokens', 'output_tokens'),
                    time_columns=('generation_time', 'execution_time', 'inference_time'),
                    optional_time_columns=()):
    """Restore a previous run from ``<output_dir>/results.csv``.

    The defaults match the agent runners. Other callers:
    benchmark_gpt.py uses ``token_columns=('image_tokens', 'input_tokens',
    'output_tokens')``, ``time_columns=('inference_time',)``,
    ``optional_time_columns=('generation_time', 'execution_time')``.
    benchmark_opensource_vlm.py uses ``time_columns=('inference_time',)``,
    ``optional_time_columns=('generation_time',)``. benchmark_geor1.py uses
    ``time_columns=()``, ``optional_time_columns=('generation_time',
    'inference_time')``.

    Args:
        output_dir: Run directory.
        resume: If False, start fresh without reading anything.
        token_columns: Columns summed as ints, 0 when absent from the file.
        time_columns: Columns summed as floats. Must be present in the file.
        optional_time_columns: Columns summed as floats, 0 when absent.

    Returns:
        tuple: ``(processed_ids, results, totals)``: the set of finished
        question ids, the finished rows as dicts, and ``totals`` mapping
        ``'total_<column>'`` to the column sum for every requested column
        (all 0 when starting fresh).

    Raises:
        KeyError: If a column listed in ``time_columns`` is missing.
    """
    totals = {f'total_{c}': 0 for c in (*token_columns, *time_columns, *optional_time_columns)}
    results_file = Path(output_dir) / "results.csv"
    if not resume or not results_file.exists():
        return set(), [], totals

    df = pd.read_csv(results_file)
    existing_results = df.to_dict('records')
    processed_ids = set(df['id'].tolist())

    for column in token_columns:
        totals[f'total_{column}'] = int(df[column].sum()) if column in df.columns else 0
    for column in time_columns:
        totals[f'total_{column}'] = df[column].sum()
    for column in optional_time_columns:
        totals[f'total_{column}'] = df[column].sum() if column in df.columns else 0

    print(f"Resuming from checkpoint: {len(processed_ids)} samples already processed")
    return processed_ids, existing_results, totals


def pending_indices(dataset, num_samples, processed_ids):
    """Indices among the first ``num_samples`` rows that still need a prediction.

    Args:
        dataset: Benchmark dataset (indexable, rows with an ``id``).
        num_samples: Number of leading rows to consider.
        processed_ids: Ids already present in ``results.csv``.

    Returns:
        list[int]: Row indices in dataset order.
    """
    return [idx for idx in range(num_samples) if dataset[idx]['id'] not in processed_ids]


def save_progress(results, output_dir, results_every=200, checkpoint_every=200):
    """Periodically persist the rows collected so far.

    Call once after appending each row. API runners use the default
    (every 200 rows). The local open-source runners write ``results.csv``
    every 10 rows and a numbered checkpoint every 100.

    Args:
        results: Rows collected so far.
        output_dir: Run directory (a ``Path``).
        results_every: Rewrite ``results.csv`` when ``len(results)`` is a
            multiple of this.
        checkpoint_every: Write ``checkpoint_<n>.csv`` when ``len(results)``
            is a multiple of this.
    """
    if len(results) % checkpoint_every == 0:
        pd.DataFrame(results).to_csv(output_dir / f"checkpoint_{len(results)}.csv", index=False)
    if len(results) % results_every == 0:
        pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)


def write_results(results, output_dir):
    """Write all rows to ``<output_dir>/results.csv`` (no-op when empty)."""
    if results:
        pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)


def save_last_masks(seg_module, output_dir, sample_id):
    """Store the masks produced while executing one agent program, then clear them.

    Args:
        seg_module: Segmentation module whose ``LAST_MASKS`` dict maps topic to
            mask for the most recent program.
        output_dir: Run directory (a ``Path``).
        sample_id: Question id, used in the file name.

    Returns:
        str or None: Path of the ``.npz`` file relative to ``output_dir``, or
        None if no masks were produced.
    """
    mask_path = None
    if seg_module.LAST_MASKS:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        mask_path = f"masks/{sample_id}_masks.npz"
        bool_masks = {topic: np.array(mask, dtype=bool) for topic, mask in seg_module.LAST_MASKS.items()}
        np.savez_compressed(output_dir / mask_path, **bool_masks)
    seg_module.LAST_MASKS = {}
    return mask_path


def accuracy_by_type(results):
    """Accuracy in percent for each question type present in ``results``."""
    type_acc = {}
    for q_type in set(r['type'] for r in results):
        type_results = [r for r in results if r['type'] == q_type]
        type_acc[q_type] = 100 * sum(1 for r in type_results if r['is_correct']) / len(type_results)
    return type_acc


def accuracy_by_tier(results):
    """Accuracy in percent keyed ``'tier_1'`` .. ``'tier_3'``, skipping empty tiers."""
    return {
        f'tier_{tier}': 100 * sum(1 for r in results if r['tier'] == tier and r['is_correct']) /
                        len([r for r in results if r['tier'] == tier])
        for tier in [1, 2, 3] if [r for r in results if r['tier'] == tier]
    }


def print_accuracy(results, type_acc, n_best=5, n_worst=5):
    """Print overall, per-tier and best/worst per-type accuracy.

    Args:
        results: All result rows.
        type_acc: Output of :func:`accuracy_by_type`.
        n_best: Number of best types to list (0 to omit the section).
        n_worst: Number of worst types to list (0 to omit the section).

    Raises:
        ZeroDivisionError: If ``results`` is empty.
    """
    correct = sum(1 for r in results if r['is_correct'])
    print(f"\nOverall Accuracy: {correct}/{len(results)} = {100*correct/len(results):.1f}%")

    print("\nAccuracy by Tier:")
    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r['tier'] == tier]
        if tier_results:
            tier_correct = sum(1 for r in tier_results if r['is_correct'])
            print(f"  Tier {tier} ({TIER_NAMES[tier]}): {tier_correct}/{len(tier_results)} = {100*tier_correct/len(tier_results):.1f}%")

    if n_best:
        print("\nBest performing types:")
        for q_type, acc in sorted(type_acc.items(), key=lambda x: x[1], reverse=True)[:n_best]:
            print(f"  {q_type}: {acc:.1f}%")

    if n_worst:
        print("\nWorst performing types:")
        for q_type, acc in sorted(type_acc.items(), key=lambda x: x[1])[:n_worst]:
            print(f"  {q_type}: {acc:.1f}%")


def report_cost(model, input_tokens, output_tokens, pricing=MODEL_PRICING, default_model='gpt-5-mini'):
    """Print and return the API cost of a run.

    Args:
        model: Model id used for the run.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
        pricing: Price table in USD per 1M tokens. benchmark_gpt.py passes its
            own, larger table.
        default_model: Entry of ``pricing`` used for unlisted models.

    Returns:
        dict: ``{'total', 'input', 'output'}`` cost in USD.
    """
    rates = pricing.get(model, pricing[default_model])
    input_cost = (input_tokens / 1_000_000) * rates['input']
    output_cost = (output_tokens / 1_000_000) * rates['output']

    print(f"\nCost ({model}):")
    print(f"  Total: ${input_cost + output_cost:.4f}")
    print(f"  Input: ${input_cost:.4f} (${rates['input']:.2f}/1M tokens)")
    print(f"  Output: ${output_cost:.4f} (${rates['output']:.2f}/1M tokens)")

    return {'total': input_cost + output_cost, 'input': input_cost, 'output': output_cost}


def write_summary(summary, output_dir):
    """Write ``summary`` as indented JSON to ``<output_dir>/summary.json``."""
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
