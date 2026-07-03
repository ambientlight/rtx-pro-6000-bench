#!/usr/bin/env bash
# Throughput MATRIX sweep for MiniMax-M3 (native MXFP4 W4A4). Port of the DSV4
# bench command recorded in bench/deepseek-v4-flash_W300_TP4_sglang/b.log — same
# shared engine (src/vllm_bench/bench_sweep.py), same concurrency ladder + output
# -len 1024 as the minimax/qwen/devstral/DSV4 benches, for cross-model comparison.
#
# Sweeps fixed input lengths x a concurrency ladder (geometric 1,2,4,8 then +8 up
# to --max-concurrency), output 1024. Larger-input rows auto-stop early when
# saturation is detected. Run AFTER launch-bwrap-highconc.sh is serving (:8000).
#
# M3 specifics:
#   --model-id   minimax-m3   (== server --served-model-name -> bench `--model`)
#   --tokenizer  .../minimax-m3-mxfp4  (model-id != dir name, so pass explicitly)
#   The highconc launcher serves max-running-requests=16 + cuda-graph-max-bs=16 by
#   default, so the USEFUL concurrency ceiling is ~16; set MAX_CONC=16 by default
#   (raise the server's MAX_RUNNING/CG_MAX_BS to match if you bump it here).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate

INPUT_LENS="${INPUT_LENS:-2048,4096,8192,16384,32768,65536}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
MAX_CONC="${MAX_CONC:-128}"          # match server max-running-requests / cuda-graph-max-bs
STEP_SIZE="${STEP_SIZE:-8}"         # geometric 1,2,4,8 then +8 to MAX_CONC
NUM_PROMPTS="${NUM_PROMPTS:-128}"

nohup python src/vllm_bench/bench_sweep.py --matrix --telemetry \
  --model-id minimax-m3 --watt 250 \
  --tokenizer /mnt/hot/ambientlight/models/minimax-m3-mxfp4 \
  --input-lens "$INPUT_LENS" --output-len "$OUTPUT_LEN" \
  --max-concurrency "$MAX_CONC" --step-size "$STEP_SIZE" \
  --num-prompts "$NUM_PROMPTS" --max-error-rate 0.1 \
  --result-dir "$HERE" \
  > "$HERE/bench_sweep.log" 2>&1 &
echo "throughput matrix sweep PID $! -> $HERE/bench_sweep.log"
echo "results -> $HERE/minimax-m3_random_*in_${OUTPUT_LEN}out_c*_W300/"
