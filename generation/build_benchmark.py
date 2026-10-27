"""Builds the final 2,000-question benchmark.

Adds the 50 human-annotated NAIP questions to the 1,950 selected questions,
attaches image and mask paths, and gives every numeric question an acceptable
range from the human-agreement tolerances in ``agreement_metrics.json``:

* percentages: +/- MAD percentage points (proximity MAD for "within ... m of").
* counts: [C - max(1, C * MADc), C + max(1, C * MADc)], floored / ceiled.
* areas and power: +/- MAD percent of the value, at least 0.01.
* binary and categorical answers: exact match, no range.

Ids are content hashes of image and question, so they are stable across
regenerations.

Usage:
    python generation/build_benchmark.py     # -> build/benchmark/qvlm_benchmark.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GSD = {'deepglobe': 0.5, 'earthvqa': 0.3, 'solar': 0.3}
MASK_NAME = {
    'deepglobe': lambda stem: f"{stem.removesuffix('_sat')}_label.png",
    'earthvqa': lambda stem: f"{stem}_label2.png",
    'solar': lambda stem: f"{stem}_label.bmp",
}
# Minimum patch area (ha) human annotators were told to ignore when counting.
MIN_COUNT_AREA = {'urban': 0.1, 'forest': 0.125, 'agriculture': 0.125, 'water': 0.1,
                  'barren': 0.125, 'rangeland': 0.125, 'building': 0.01, 'solar': 0.01}
NO_RANGE_TYPES = {'binary', 'binary_comparison', 'binary_presence', 'binary_threshold',
                  'binary_multiple', 'binary_proximity', 'fragmentation'}
COUNT_TYPES = {'count', 'connectivity', 'building_proximity', 'building_flood_risk', 'building_fire_risk'}


def counted_class(question):
    """Returns the land-cover class a question is about, or None."""
    q = question.lower()
    for key, name in [('urban', 'urban'), ('forest', 'forest'), ('agricultur', 'agriculture'),
                      ('water', 'water'), ('barren', 'barren'), ('desert', 'barren'),
                      ('grass', 'rangeland'), ('rangeland', 'rangeland'),
                      ('building', 'building'), ('solar', 'solar')]:
        if key in q:
            return name
    return None


def add_count_threshold(question):
    """Appends the "ignore patches smaller than ..." clause to a count question.

    The human interface showed this instruction separately. Spelling it out in
    the text makes the NAIP questions consistent with the generated ones.
    """
    q = question.lower()
    name = counted_class(question)
    if 'how many' not in q or 'when counting' in q or 'ignore' in q or name is None:
        return question
    clause = f"When counting, ignore patches smaller than {MIN_COUNT_AREA[name]} hectares."
    if '(GSD:' in question:
        head, tail = question.split('(GSD:')
        return f"{head.rstrip()} {clause} (GSD:{tail}"
    if '?' in question:
        return question.replace('?', f'? {clause}')
    return f"{question} {clause}"


def human_questions(csv_path):
    """Aggregates the human annotations into one question per item.

    Numeric answers are the median over annotators. Categorical answers are the
    majority vote (yes/no mapped to 1/0). Items keep the annotation order.

    Args:
        csv_path: Path to ``human_annotations_anonymized.csv``.

    Returns:
        A list of benchmark entries without id and range.
    """
    df = pd.read_csv(csv_path)
    df = df[df['Answer.answer_type'].notna()]
    questions = []
    for (image, question), group in df.groupby(['Input.image', 'Input.question'], sort=False):
        if group.iloc[0]['Answer.answer_type'] == 'numeric':
            answers = pd.to_numeric(group['Answer.answer'], errors='coerce').dropna()
            if answers.empty:
                continue
            answer = float(np.median(answers))
        else:
            answer = group['Answer.answer'].mode().iloc[0]
            answer = {'yes': 1, 'true': 1, 'no': 0, 'false': 0}.get(answer.lower(), answer)

        question = add_count_threshold(question)
        q = question.lower()
        if 'percentage' in q:
            q_type, tier = ('proximity_percentage', 2) if 'within' in q else ('percentage', 1)
        elif 'how many' in q:
            q_type, tier = ('building_proximity', 2) if 'within' in q else ('count', 1)
        elif 'connected' in q or 'fragmented' in q:
            q_type, tier = 'fragmentation', 2
        else:
            q_type, tier = 'binary', 1
        questions.append({'image': f"naip_1.0m/{Path(image).name}", 'mask': None, 'question': question,
                          'answer': answer, 'type': q_type, 'tier': tier, 'gsd': 1.0})
    return questions


def count_range(value, madc):
    """Returns the integer tolerance range of a count answer."""
    dr = max(1, value * madc)
    return [max(0, math.floor(value - dr)), math.ceil(value + dr)]


def relative_range(value, fraction):
    """Returns a +/- ``fraction`` relative range, at least +/- 0.01, rounded to 3 dp."""
    dr = max(0.01, value * fraction)
    return [round(max(0.0, value - dr), 3), round(value + dr, 3)]


def acceptable_range(value, q_type, question, mad):
    """Returns the acceptable answer range of a question.

    Args:
        value: Ground-truth answer.
        q_type: Question type.
        question: Question text.
        mad: The ``tolerances`` entry of ``agreement_metrics.json``.

    Returns:
        ``[lower, upper]``, or None when the answer must match exactly.
    """
    if not isinstance(value, (int, float)) or q_type in NO_RANGE_TYPES:
        return None
    q = question.lower()
    spatial = 'within' in q and 'm of' in q
    if q_type in ('percentage', 'region_percentage') or 'percentage' in q:
        m = mad['proximity'] if spatial else mad['percentage']
        return [round(max(0.0, value - m), 3), round(min(100.0, value + m), 3)]
    if q_type in COUNT_TYPES or ('how many' in q and ('within' not in q or 'm of' in q)):
        return count_range(value, mad['count_madc'])
    if 'hectare' in q or 'area' in q:
        return relative_range(value, (mad['proximity'] if spatial else mad['percentage']) / 100.0)
    if 'mw' in q or q_type == 'power_calculation':
        return relative_range(value, mad['percentage'] / 100.0)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--build-dir', type=Path, default=REPO / 'build')
    args = parser.parse_args()

    with open(HERE / 'agreement_metrics.json') as f:
        mad = json.load(f)['tolerances']

    with open(args.build_dir / 'qvlm_benchmark_auto.json') as f:
        generated = json.load(f)
    entries = []
    for q in generated:
        source_dir = f"{q['dataset']}_{GSD[q['dataset']]}m"
        name = Path(q['image']).name
        entries.append({'image': f"{source_dir}/{name}",
                        'mask': f"{source_dir}/{MASK_NAME[q['dataset']](Path(name).stem)}",
                        'question': q['question'], 'answer': q['answer'], 'type': q['type'],
                        'tier': q['tier'], 'gsd': GSD[q['dataset']]})
    entries += human_questions(HERE / 'human_annotations_anonymized.csv')

    benchmark = []
    for q in entries:
        entry = {'id': "SQuID_" + hashlib.sha1(f"{q['image']}|{q['question']}".encode()).hexdigest()[:12], **q}
        rng = acceptable_range(q['answer'], q['type'], q['question'], mad)
        if rng:
            entry['acceptable_range'] = rng
        benchmark.append(entry)

    out = args.build_dir / 'benchmark' / 'qvlm_benchmark.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(benchmark, f, indent=2)
    print(f"{len(benchmark)} questions ({len(entries) - len(generated)} human), "
          f"{sum('acceptable_range' in q for q in benchmark)} with ranges -> {out}")


if __name__ == '__main__':
    main()
