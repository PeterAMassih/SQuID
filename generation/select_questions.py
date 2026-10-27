"""Merges the per-source questions and selects a split.

Benchmark: a balanced 1,950-question sample under per-category quotas (35%
complex, 20% proximity, 15% percentage, 10% each count / connectivity /
binary), at least two questions of every type, a capped share of near-zero
answers, balanced tiers and at least 5% solar questions. The 50 human NAIP
questions are added afterwards by ``build_benchmark.py``.

Train: every valid question, no cap. The result is the released training file.

Usage:
    python generation/select_questions.py            # -> build/qvlm_benchmark_auto.json
    python generation/select_questions.py --train    # -> build/train/qvlm_benchmark_training.json
"""

import argparse
import collections
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = ['deepglobe', 'earthvqa', 'solar']
QUOTAS = {'complex': 0.35, 'proximity': 0.20, 'percentage': 0.15,
          'count': 0.10, 'connectivity': 0.10, 'binary': 0.10, 'other': 0.00}
CATEGORIES = {
    'proximity': {'proximity_percentage', 'binary_proximity', 'building_proximity',
                  'building_flood_risk', 'building_fire_risk', 'proximity_area'},
    # Solar 'size' / 'total_area' are areas, grouped with percentages for quotas only.
    'percentage': {'percentage', 'region_percentage', 'size', 'total_area'},
    'count': {'count'},
    'binary': {'binary_presence', 'binary_comparison', 'binary_threshold', 'binary_multiple'},
    'connectivity': {'connectivity', 'fragmentation'},
}


def load_questions(build_dir, train):
    """Loads the generator outputs and tags each question with its source."""
    suffix = '_train' if train else ''
    questions = []
    for source in SOURCES:
        with open(build_dir / f"{source}_qa{suffix}.json") as f:
            for q in json.load(f):
                q['dataset'] = source
                questions.append(q)
    return questions


def category(q):
    """Maps a question type to its quota category."""
    for name, types in CATEGORIES.items():
        if q['type'] in types:
            return name
    if 'complex' in q['type'] or q['type'] == 'power_calculation':
        return 'complex'
    return 'other'


def is_valid(q):
    """Keeps numeric answers with a tier and the two categorical answers."""
    if q.get('answer') is None:
        return False
    if isinstance(q['answer'], str):
        return q['answer'] in ['fragmented', 'connected']
    return q.get('tier') in [1, 2, 3]


def is_near_zero(q):
    """True for near-empty answers: coverage <= 2 points, counts <= 1, else exactly 0."""
    answer = q['answer']
    if not isinstance(answer, (int, float)):
        return False
    if q['type'] in ('percentage', 'region_percentage', 'proximity_percentage'):
        return answer <= 2.0
    if category(q) == 'count':
        return answer <= 1
    return answer == 0


def select_balanced(questions, target):
    """Draws the balanced benchmark sample.

    Args:
        questions: All generated questions (consumes the global RNG).
        target: Number of questions to select.

    Returns:
        The selected questions, at most ``target``.
    """
    valid = [q for q in questions if is_valid(q)]
    random.shuffle(valid)
    quotas = {name: int(target * share) for name, share in QUOTAS.items()}
    zero_budget = int(target * 0.05)

    # Two questions of every exact type first, so no type is missing.
    by_type = collections.defaultdict(list)
    for q in valid:
        by_type[q['type']].append(q)
    selected = [q for qs in by_type.values() for q in qs[:2]]

    by_category = collections.defaultdict(list)
    for q in valid:
        if q not in selected:
            by_category[category(q)].append(q)
    counts = collections.Counter(category(q) for q in selected)
    zeros = sum(is_near_zero(q) for q in selected)

    # Fill each quota, with near-zero answers capped at 10% of it and 5% overall.
    for name, quota in quotas.items():
        remaining = quota - counts[name]
        if remaining <= 0:
            continue
        pool = by_category[name]
        random.shuffle(pool)
        pool_zeros = [q for q in pool if is_near_zero(q)]
        pool_others = [q for q in pool if not is_near_zero(q)]
        n_zeros = min(len(pool_zeros), int(remaining * 0.10), zero_budget - zeros)
        selected.extend(pool_zeros[:n_zeros])
        zeros += n_zeros
        counts[name] += n_zeros
        selected.extend(pool_others[:remaining - n_zeros])
        counts[name] += min(remaining - n_zeros, len(pool_others))

    # Rebalance tiers by swapping questions of the same category.
    tiers = collections.Counter(q['tier'] for q in selected)
    per_tier = len(selected) // 3
    for tier in [1, 2, 3]:
        if tiers[tier] >= per_tier:
            continue
        need = per_tier - tiers[tier]
        for other in [1, 2, 3]:
            if other == tier or tiers[other] <= per_tier:
                continue
            for name in quotas:
                ours = [q for q in selected if q['tier'] == tier and category(q) == name]
                theirs = [q for q in valid if q['tier'] == other and category(q) == name and q not in selected]
                swaps = min(need, len(theirs))
                if swaps > 0 and len(ours) >= swaps:
                    for i in range(swaps):
                        selected.remove(ours[i])
                        selected.append(theirs[i])
                    need -= swaps
                    tiers[tier] += swaps
                    tiers[other] -= swaps
                if need <= 0:
                    break
            if need <= 0:
                break

    # At least 5% solar questions, swapped in for abundant categories.
    min_solar = int(target * 0.05)
    n_solar = sum(q['dataset'] == 'solar' for q in selected)
    if n_solar < min_solar:
        solar = [q for q in valid if q['dataset'] == 'solar' and q not in selected]
        random.shuffle(solar)
        need = min_solar - n_solar
        for name in ['count', 'proximity', 'percentage']:
            candidates = [q for q in selected if category(q) == name and q['dataset'] != 'solar']
            if len(candidates) >= need:
                for q in candidates[:need]:
                    selected.remove(q)
                selected.extend(solar[:need])
                break

    # Top up to the target, within quotas first.
    if len(selected) < target:
        rest = [q for q in valid if q not in selected]
        random.shuffle(rest)
        for q in rest:
            if counts[category(q)] < quotas[category(q)]:
                selected.append(q)
                counts[category(q)] += 1
                if len(selected) >= target:
                    break
        if len(selected) < target:
            leftover = [q for q in rest if q not in selected]
            selected.extend(leftover[:target - len(selected)])

    missing = set(by_type) - {q['type'] for q in selected}
    if missing:
        print(f"warning: types missing from the sample: {sorted(missing)}")
    return selected[:target]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--train', action='store_true', help='keep every valid training question')
    parser.add_argument('--target', type=int, default=1950)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--build-dir', type=Path, default=REPO / 'build')
    args = parser.parse_args()

    random.seed(args.seed)
    questions = load_questions(args.build_dir, args.train)
    if args.train:
        selected = [q for q in questions if is_valid(q)]
        random.shuffle(selected)
        out = args.build_dir / 'train' / 'qvlm_benchmark_training.json'
    else:
        selected = select_balanced(questions, args.target)
        out = args.build_dir / 'qvlm_benchmark_auto.json'
    random.shuffle(selected)
    for i, q in enumerate(selected):
        q['id'] = f"qvlm_{i + 1:06d}"

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(selected, f, indent=2)
    tiers = collections.Counter(q['tier'] for q in selected)
    print(f"{len(selected)} questions, tiers {dict(sorted(tiers.items()))} -> {out}")


if __name__ == '__main__':
    main()
