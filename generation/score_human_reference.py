"""Scores each of the 500 human annotations like a model prediction.

A response is correct when a numeric answer falls in the question's acceptable
range (exact match when it has none) or a categorical answer matches. This
gives the human reference accuracy reported in the paper (62.6%).

The benchmark text extends the annotation-interface wording (count threshold,
GSD tag), so annotations are matched to questions by prefix on the same image.
Every annotation must match exactly one question.

Usage:
    python generation/score_human_reference.py [data/benchmark/qvlm_benchmark.json]
"""

import argparse
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = HERE.parent / 'data' / 'benchmark' / 'qvlm_benchmark.json'


def match(row, naip):
    """Returns the unique NAIP benchmark question an annotation answers."""
    q = row['Input.question'].strip()
    found = [r for r in naip if r['image'] == row['Input.image']
             and (r['question'].strip().startswith(q) or q.startswith(r['question'].strip()))]
    assert len(found) == 1, f"no unique match for {row['Input.image']} | {q!r}"
    return found[0]


def is_correct(row, question):
    """Applies the model scoring rule to one human response."""
    if question['type'] == 'fragmentation':
        return row['Answer.categorical-answer'].strip().lower() == str(question['answer']).strip().lower()
    value = float(row['Answer.answer'])
    rng = question.get('acceptable_range')
    return rng[0] <= value <= rng[1] if rng else value == float(question['answer'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('benchmark', nargs='?', type=Path, default=DEFAULT_BENCHMARK)
    with open(parser.parse_args().benchmark) as f:
        naip = [r for r in json.load(f) if r['image'].startswith('naip')]
    with open(HERE / 'human_annotations_anonymized.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 500, f"expected 500 annotations, got {len(rows)}"

    per_annotator = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        stats = per_annotator[row['Turkle.Username']]
        stats[0] += is_correct(row, match(row, naip))
        stats[1] += 1

    correct = sum(c for c, _ in per_annotator.values())
    accuracies = {name: 100 * c / n for name, (c, n) in sorted(per_annotator.items())}
    print(f"pooled accuracy: {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%")
    for name, acc in accuracies.items():
        print(f"  {name}: {acc:.1f}%")
    print(f"per-annotator range: {min(accuracies.values()):.0f}% to {max(accuracies.values()):.0f}%")


if __name__ == '__main__':
    main()
