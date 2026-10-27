#!/bin/bash
# QASR with the SQuID reference masks in place of the segmentation model.
# GPT-5 writes new programs. Needs only the downloaded benchmark
# (python data/download.py --split benchmark) and an OpenAI API key:
#
#   OPENAI_API_KEY=... ./run_oracle_gt.sh
#
# To re-run the programs of a predicted-mask run instead, with no LLM call:
#
#   python benchmark/benchmark_agent_gt.py --replay agent_results/results.csv
set -euo pipefail
cd "$(dirname "$0")"

export USE_GT_MASKS=1
export USE_DINOV3=0

python benchmark/benchmark_agent_gt.py \
  --bench-json "${BENCH_JSON:-data/benchmark/qvlm_benchmark.json}" \
  --output-dir "${OUT_DIR:-results/oracle_gt}" \
  --model "${MODEL:-gpt-5}" \
  --resume
