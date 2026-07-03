#!/usr/bin/env bash
# Single-concurrency (c1) long-context scaling sweep for MiniMax-M3 (native MXFP4
# W4A4). Port of bench/deepseek-v4-flash_W300_TP4_sglang/s.sh — same shared engine
# (src/vllm_bench/bench_sweep.py), same telemetry, so the M3 vs DSV4 long-context
# columns in docs/DEPLOY-MXFP4-W4A4-MINIMAX-M3-SM120.md are apples-to-apples.
#
# Doubles input length 2K -> 1M-1024, output 1024, ONE prompt per point (one
# warmup, since num-warmups == concurrency == 1). Reports prefill (TTFT) and
# decode (TPOT) per prompt length. Run AFTER launch-bwrap-highconc.sh is serving
# (/v1/models up on :8000). Results land in a SEPARATE dir so they don't clobber
# the throughput matrix sweep's c1 cells.
#
# M3 specifics vs the DSV4 port:
#   --model-id      minimax-m3   (== server --served-model-name; becomes the bench
#                                 request's `--model`, so it MUST match, NOT the path)
#   --tokenizer     .../minimax-m3-mxfp4  (passed explicitly: model-id != dir name)
#   input ceiling   1,047,552 = 1M-1024, so input+output == 1,048,576 = M3 max ctx
set -euo pipefail

# Client venv (HTTP only) — the repo's .venv with vllm bench serve.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate

# 2K,4K,...,512K then 1,047,552 (= 1M - 1024, so input+output == 1,048,576 ctx).
INPUT_LENS="${INPUT_LENS:-2048,4096,8192,16384,32768,65536,131072,262144,524288,1047552}"

nohup python src/vllm_bench/bench_sweep.py --matrix --telemetry \
  --model-id minimax-m3 --watt 300 \
  --tokenizer /mnt/hot/ambientlight/models/minimax-m3-mxfp4 \
  --input-lens "$INPUT_LENS" --output-len 1024 \
  --min-concurrency 1 --max-concurrency 1 \
  --num-prompts 1 --max-error-rate 0.0 \
  --result-dir "$HERE/single" \
  > "$HERE/sweep-single.log" 2>&1 &
echo "single-concurrency sweep PID $! -> $HERE/sweep-single.log"
echo "results -> $HERE/single/"
