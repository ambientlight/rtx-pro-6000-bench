#!/usr/bin/env bash
# MiniMax-M3 entrypoint (unified image). NCCL is already set by the dispatcher
# (entrypoint.sh) — this only sets M3's MoE-routing env (dynamic-M JIT +
# force-static) and the long-context server args. Mirrors
# bench/minimax-m3-mxfp4_TP4_sglang/launch-bwrap-highconc.sh. Model MOUNTED at
# $MODEL_DIR. Env-overridable.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/model}
SERVED_NAME=${SERVED_NAME:-minimax-m3}
PORT=${PORT:-8000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}     # full 1M default
MAX_RUNNING=${MAX_RUNNING:-16}
CG_MAX_BS=${CG_MAX_BS:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.95}   # fits one full 1M sequence
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: no model at $MODEL_DIR (mount it: -v /host/minimax-m3-mxfp4:/model:ro)" >&2; exit 1; }

# --- MoE kernel routing (native MXFP4 static path + dynamic-M JIT) ---
export SGLANG_M3_FORCE_STATIC=${SGLANG_M3_FORCE_STATIC:-1}
export FLASHINFER_B12X_STATIC_DYNAMIC_M=${FLASHINFER_B12X_STATIC_DYNAMIC_M:-1}  # =0 -> exact-M A/B

echo "MiniMax-M3: model=$MODEL_DIR ctx=$CONTEXT_LENGTH mrr=$MAX_RUNNING cg-bs=$CG_MAX_BS kv=$KV_CACHE_DTYPE port=$PORT"

exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --tokenizer-path "$MODEL_DIR" \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --tp-size 4 \
  --context-length "$CONTEXT_LENGTH" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --quantization compressed-tensors \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-shared-experts-fusion \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode "$CG_MAX_BS" \
  --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens 16384 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --watchdog-timeout 1200 \
  --skip-server-warmup \
  --enable-metrics \
  ${EXTRA_SERVER_ARGS:-}
