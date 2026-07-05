#!/usr/bin/env bash
# DeepSeek-V4-Flash entrypoint (unified image). NCCL is already set by the
# dispatcher (entrypoint.sh) — this only sets the DSV4-specific SM120 sparse-attn
# + reasoning env and the long-context server args. Mirrors
# bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh. Model MOUNTED at
# $MODEL_DIR. Env-overridable.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/model}
SERVED_NAME=${SERVED_NAME:-deepseek-v4-flash}
PORT=${PORT:-8000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}     # full 1M
MAX_RUNNING=${MAX_RUNNING:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: no model at $MODEL_DIR (mount it: -v /host/DeepSeek-V4-Flash:/model:ro)" >&2; exit 1; }

# --- SM120 sparse-attention + indexer selection (HMMA decode/prefill) ---
export SGLANG_SM120_SPARSE_DECODE=${SGLANG_SM120_SPARSE_DECODE:-hmma}
export SGLANG_SM120_SPARSE_PREFILL=${SGLANG_SM120_SPARSE_PREFILL:-hmma}
export SGLANG_OPT_USE_TILELANG_INDEXER=${SGLANG_OPT_USE_TILELANG_INDEXER:-1}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-1}
export SGLANG_SM120_INDEXER_SPLIT=${SGLANG_SM120_INDEXER_SPLIT:-1}            # split-KV: the long-ctx win
export SGLANG_SM120_INDEXER_SPLIT_COUNT=${SGLANG_SM120_INDEXER_SPLIT_COUNT:-256}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP:-0}
export SGLANG_OPT_USE_FUSED_HASH_TOPK=${SGLANG_OPT_USE_FUSED_HASH_TOPK:-0}

# --- reasoning (thinking + MAX effort; pairs with the parsers below) ---
export SGLANG_DEFAULT_THINKING=${SGLANG_DEFAULT_THINKING:-1}
export SGLANG_DSV4_REASONING_EFFORT=${SGLANG_DSV4_REASONING_EFFORT:-max}

echo "DSV4-Flash: model=$MODEL_DIR ctx=$CONTEXT_LENGTH mrr=$MAX_RUNNING kv=$KV_CACHE_DTYPE port=$PORT"

exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --tp 4 --trust-remote-code --host 0.0.0.0 --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --moe-runner-backend triton \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --page-size 256 \
  --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --dsa-topk-backend torch \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --enable-metrics \
  --watchdog-timeout 3600 --log-level info \
  ${EXTRA_SERVER_ARGS:-}
