#!/usr/bin/env bash
# Native latest-main DSV4/SM120 control.  It intentionally omits the legacy
# HMMA/indexer selectors, the nominal Triton MoE override, and the torch top-k
# override so the comparison exercises main's intended SM120 paths.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/model}
SERVED_NAME=${SERVED_NAME:-deepseek-v4-flash}
PORT=${PORT:-8000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}
MAX_RUNNING=${MAX_RUNNING:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.65}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}

test -f "${MODEL_DIR}/config.json" || {
  echo "ERROR: no model config at ${MODEL_DIR}/config.json" >&2
  exit 1
}

# Match the deployed PCIe TP4 collective setup.
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_PROTO=${NCCL_PROTO:-LL}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_NTHREADS=${NCCL_NTHREADS:-512}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Keep the production chat semantics while taking parser behavior from main.
export SGLANG_DEFAULT_THINKING=${SGLANG_DEFAULT_THINKING:-1}
export SGLANG_DSV4_REASONING_EFFORT=${SGLANG_DSV4_REASONING_EFFORT:-max}

echo "DSV4 main control: commit=${SGLANG_BUILD_COMMIT:-unknown} model=${MODEL_DIR} ctx=${CONTEXT_LENGTH} mrr=${MAX_RUNNING} kv=${KV_CACHE_DTYPE} port=${PORT}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --tp 4 --trust-remote-code --host 0.0.0.0 --port "${PORT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" --page-size 256 \
  --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --enable-metrics \
  --watchdog-timeout 3600 --log-level info \
  ${EXTRA_SERVER_ARGS:-}
