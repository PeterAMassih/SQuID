"""Generates the raw questions of one source and split from its masks.

Reads ``<data-root>/<split>/images_and_masks/<source dir>/`` and writes
``<out-dir>/<source>_qa.json`` (benchmark) or ``<source>_qa_train.json``
(train). For the benchmark split, scenes that also appear in the training
split are dropped, which keeps the two splits image-disjoint.

Usage:
    python generation/generate.py --source earthvqa --split benchmark
"""

import argparse
import collections
import json
import random
from pathlib import Path

from landcover_qa import DeepGlobeQA, EarthVQAQA
from solar_qa import SolarQA

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    'deepglobe': (DeepGlobeQA, 'deepglobe_0.5m'),
    'earthvqa': (EarthVQAQA, 'earthvqa_0.3m'),
    'solar': (SolarQA, 'solar_0.3m'),
}


def scene_id(mask_path):
    """Returns the scene id shared by an image and its masks."""
    return mask_path.stem.replace('_label2', '').replace('_label', '').replace('_mask', '')


def list_masks(directory, source):
    """Lists the masks of one source directory, sorted, one per scene.

    Solar scenes may ship a ``.bmp`` and a ``.png`` mask. The ``.bmp`` is kept so
    a scene is never processed twice.
    """
    if source != 'solar':
        return sorted(directory.glob(f"*{SOURCES[source][0].MASK_SUFFIX}"))
    by_stem = {}
    for path in sorted(directory.glob('*_label.bmp')) + sorted(directory.glob('*_label.png')):
        by_stem.setdefault(path.stem, path)
    return [by_stem[stem] for stem in sorted(by_stem)]


def generate(source, split, data_root):
    """Runs one generator over every mask of a split.

    Args:
        source: One of ``SOURCES``.
        split: ``'benchmark'`` or ``'train'``.
        data_root: Directory holding ``benchmark/`` and ``train/``.

    Returns:
        The list of generated questions.
    """
    generator_cls, subdir = SOURCES[source]
    masks = list_masks(data_root / split / 'images_and_masks' / subdir, source)
    if not masks:
        raise FileNotFoundError(f"no {source} masks under {data_root / split}")
    if split == 'benchmark':
        train_scenes = {scene_id(p) for p in list_masks(data_root / 'train' / 'images_and_masks' / subdir, source)}
        masks = [p for p in masks if scene_id(p) not in train_scenes]

    random.seed(42)
    generator = generator_cls()
    used = sum(generator.process(path) for path in masks)
    tiers = collections.Counter(q['tier'] for q in generator.questions)
    print(f"{source}/{split}: {len(generator.questions)} questions from {used}/{len(masks)} masks, "
          f"tiers {dict(sorted(tiers.items()))}")
    return generator.questions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--split', choices=['benchmark', 'train'], required=True)
    parser.add_argument('--data-root', type=Path, default=REPO / 'data')
    parser.add_argument('--out-dir', type=Path, default=REPO / 'build')
    args = parser.parse_args()

    questions = generate(args.source, args.split, args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_train' if args.split == 'train' else ''
    with open(args.out_dir / f"{args.source}_qa{suffix}.json", 'w') as f:
        json.dump(questions, f, indent=2)


if __name__ == '__main__':
    main()
