"""Downloads SQuID from Hugging Face into this folder.

Fills ``data/benchmark/`` and ``data/train/`` with the question files and the
images and ground-truth masks they reference:

    data/benchmark/qvlm_benchmark.json             2,000 questions
    data/benchmark/images_and_masks/<source>/      images + masks
    data/train/qvlm_benchmark_training.json        75,687 questions
    data/train/images_and_masks/<source>/          images + masks

Usage:
    python data/download.py                  # both splits (~5 GB)
    python data/download.py --split benchmark
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = 'PeterAM4/SQuID'
REVISION = 'v2.0'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--split', choices=['benchmark', 'train'], nargs='*', default=['benchmark', 'train'])
    args = parser.parse_args()
    snapshot_download(REPO_ID, repo_type='dataset', revision=REVISION, local_dir=Path(__file__).parent,
                      allow_patterns=[f'{split}/*' for split in args.split])


if __name__ == '__main__':
    main()
