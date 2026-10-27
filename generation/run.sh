#!/bin/bash
# Regenerates both SQuID splits from the masks in data/ and checks that the
# result is byte-identical to the released files.
#
#   python data/download.py && bash generation/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for source in deepglobe earthvqa solar; do
  for split in benchmark train; do
    python generation/generate.py --source "$source" --split "$split" &
  done
done
wait

python generation/select_questions.py
python generation/select_questions.py --train
python generation/build_benchmark.py

cmp build/benchmark/qvlm_benchmark.json data/benchmark/qvlm_benchmark.json
cmp build/train/qvlm_benchmark_training.json data/train/qvlm_benchmark_training.json
echo "Both splits match the release."
