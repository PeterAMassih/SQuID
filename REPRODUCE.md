# Running the paper's systems

Every runner evaluates on the SQuID benchmark (`PeterAM4/SQuID`, config `benchmark`) and writes `results.csv` and `summary.json`. A numeric answer is correct within its tolerance range. Categorical answers must match exactly. Add `--num-samples 10` for a quick check and `--resume` to continue a run.

One command per row of Table 3:

```bash
export OPENAI_API_KEY=...

# QASR, predicted masks
python benchmark/benchmark_agent.py --model gpt-5 --output-dir results/qasr_a
python benchmark/benchmark_opensource_agent.py --model-name openai/gpt-oss-120b --output-dir results/qasr_b
DINOV3_CHECKPOINT=PeterAM4/SQuID-DINOv3-Mask2Former \
  python benchmark/benchmark_agent.py --model gpt-5 --output-dir results/qasr_c
python benchmark/benchmark_opensource_agent.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-dir results/qasr_d
python benchmark/benchmark_remotesam.py --model gpt-5 --remotesam-repo <repo> --output-dir results/qasr_e
python benchmark/benchmark_sam3.py --output-dir results/qasr_f

# End to end, zero-shot
python benchmark/benchmark_gpt.py --api-key $OPENAI_API_KEY --model gpt-5 --output-dir results/gpt5
python benchmark/benchmark_opensource_vlm.py --model-name Qwen/Qwen2.5-VL-7B-Instruct --output-dir results/qwen2.5_vl_7b
python benchmark/benchmark_opensource_vlm.py --model-name Qwen/Qwen3-VL-8B-Instruct --output-dir results/qwen3_vl_8b
python benchmark/benchmark_opensource_vlm.py --model-name Qwen/Qwen3-VL-30B-A3B-Thinking --max-new-tokens 4096 \
  --output-dir results/qwen3_vl_30b
python benchmark/benchmark_geor1.py --output-dir results/geo_r1
python benchmark/benchmark_geochat.py --output-dir results/geochat
python benchmark/benchmark_earthdial.py --model-path <EarthDial_4B_RGB> --output-dir results/earthdial
# CoVT has no runner here.

# End to end, trained on SQuID-Train
torchrun --nproc_per_node=4 training/train_lora.py --output_dir runs/squid_lora
python benchmark/benchmark_lora.py --lora_path runs/squid_lora/checkpoint-2500 --output_dir results/lora

# QASR, reference masks: the programs of QASR-A and QASR-D run again, with no new LLM call
python benchmark/benchmark_agent_gt.py --replay results/qasr_a/results.csv --output-dir results/qasr_a_ref
python benchmark/benchmark_agent_gt.py --replay results/qasr_d/results.csv --output-dir results/qasr_d_ref
```

`./run_oracle_gt.sh` instead has GPT-5 write new programs on the reference masks.

## Segmentation backends

- **ConvNeXt-UNet** (QASR-A, B and D, and QASR-C for urban, grass and solar) is served over HTTP on `localhost:8000` (API in `evaluation/eval_satellite_processing.py`). Its checkpoint is not released. Any segmenter with the same API can replace it.
- **DINOv3 + Mask2Former** (QASR-C) serves forest, agriculture, barren land, water and roofs. Its weights load from [PeterAM4/SQuID-DINOv3-Mask2Former](https://huggingface.co/PeterAM4/SQuID-DINOv3-Mask2Former) and its backbone from the gated [facebook/dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m), whose license must be accepted first. It needs `pip install git+https://github.com/facebookresearch/dinov3@76c051f5732cb4500bff5c23ac3c54a3695f15ff`.
- **Reference masks** come from `python data/download.py --split benchmark`. They cover the 1,950 mask-derived questions. The 50 NAIP questions have human-consensus answers.
- **RemoteSAM, SAM-3, GeoChat, Geo-R1 and EarthDial** need their upstream code or weights. Each runner's header explains the setup.
