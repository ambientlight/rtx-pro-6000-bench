#!/usr/bin/env bash
# Quest-5 PERF-THESIS launcher: native MXFP4 W4A4 M3 with cuda-graph decode
# captured up to HIGH concurrency, so we can finally measure the regime the
# native path was built for: conc>=11 (routed_rows = conc*top_k >= 44), where
# the prefill-validated STATIC MoE kernel runs and Marlin's no-batch
# serialization should lose badly.
#
# Differences vs launch-bwrap.sh (production):
#   --context-length          250000 -> ${CONTEXT_LENGTH:-1048576}    (full 1M default)
#   --max-running-requests         4 -> ${MAX_RUNNING:-16}            (matches the DSV4 config)
#   --chunked-prefill-size      4096 -> ${CHUNKED_PREFILL_SIZE:-8192}
#   --cuda-graph-max-bs-decode  (new) -> ${CG_MAX_BS:-16}             (capture decode graphs to bs=16)
#   --mem-fraction-static       0.90 -> ${MEM_FRACTION_STATIC:-0.95}  (fits one full 1M-token sequence)
#   --kv-cache-dtype                  -> ${KV_CACHE_DTYPE:-fp8_e4m3}
#
# NOTE: takes port 8000. Stop any existing server first (it will refuse to bind
# otherwise). The first real request at each NEW batch shape pays a one-time
# CuteDSL JIT (~tens of s); the bench warmup pass absorbs it.
#
# DEBUG (opt-in, off by default — the CuteDSL JIT is otherwise SILENT, which once
# masked a per-`m` MoE recompile storm as mysterious GPU-idle stalls under agentic
# load). Set these in the env to surface compiles:
#   CUTE_DSL_JIT_TIME_PROFILING=1   -> logs IR-gen/compile/exec TIMES per kernel
#                                      (auto-enables console output; THE one to use)
#   CUTE_DSL_LOG_TO_CONSOLE=1 CUTE_DSL_LOG_LEVEL=10  -> full DEBUG (10=DEBUG,20=INFO;
#                                      LOG_LEVEL needs LOG_TO_CONSOLE or it no-ops)
set -euo pipefail

ROOTFS=${ROOTFS:-/mnt/hot/minimax-m3-rootfs}
NVHOST=${NVHOST:-/mnt/hot/minimax-m3-nvhost}
REPO_FI=${REPO_FI:-/mnt/hot/ambientlight/repos/flashinfer}
MODEL_DIR=${MODEL_DIR:-$HOME/models/minimax-m3-mxfp4}
FICACHE=${FICACHE:-/mnt/hot/minimax-m3-ficache}
PORT=${PORT:-8000}
MAX_RUNNING=${MAX_RUNNING:-16}
CG_MAX_BS=${CG_MAX_BS:-16}

[ -d "$ROOTFS/sgl-workspace" ] || { echo "rootfs missing at $ROOTFS" >&2; exit 1; }
[ -e "$NVHOST/lib/libcuda.so.1" ] || { echo "host driver libs missing at $NVHOST" >&2; exit 1; }
[ -f "$REPO_FI/flashinfer/__init__.py" ] || { echo "repo flashinfer missing at $REPO_FI" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "MXFP4 checkpoint missing at $MODEL_DIR" >&2; exit 1; }
mkdir -p "$FICACHE"

DEV_BINDS=()
for d in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [ -e "$d" ] && DEV_BINDS+=( --dev-bind "$d" "$d" )
done

echo "HIGH-CONC launch: max-running=$MAX_RUNNING cuda-graph-max-bs-decode=$CG_MAX_BS port=$PORT"

exec bwrap \
  --ro-bind "$ROOTFS"/usr /usr \
  --ro-bind "$ROOTFS"/lib /lib \
  --ro-bind "$ROOTFS"/lib64 /lib64 \
  --ro-bind "$ROOTFS"/bin /bin \
  --ro-bind "$ROOTFS"/sbin /sbin \
  --ro-bind "$ROOTFS"/etc /etc \
  --bind "$ROOTFS"/sgl-workspace /sgl-workspace \
  --bind "$ROOTFS"/opt /opt \
  --ro-bind "$NVHOST" /nvhost \
  --ro-bind "$REPO_FI" "$REPO_FI" \
  --bind "$FICACHE" "$FICACHE" \
  --ro-bind /sys /sys \
  --proc /proc \
  --tmpfs /tmp \
  --tmpfs /run \
  --dev /dev \
  --bind /dev/shm /dev/shm \
  "${DEV_BINDS[@]}" \
  --ro-bind "$MODEL_DIR" "$MODEL_DIR" \
  --setenv LD_LIBRARY_PATH "/nvhost/lib:/usr/local/cuda-13.0/lib64" \
  --setenv PATH "/nvhost/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --setenv CUDA_HOME /usr/local/cuda \
  --setenv PYTORCH_CUDA_ALLOC_CONF expandable_segments:True \
  --setenv PYTHONPATH "$REPO_FI:/sgl-workspace/sglang/python" \
  --setenv FLASHINFER_WORKSPACE_BASE "$FICACHE" \
  --setenv CUDA_VISIBLE_DEVICES 0,1,2,3 \
  --setenv HF_HUB_OFFLINE 1 \
  --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv FLASHINFER_DISABLE_VERSION_CHECK 1 \
  --setenv NCCL_SHM_DISABLE "${NCCL_SHM_DISABLE:-0}" \
  --setenv NCCL_CUMEM_ENABLE "${NCCL_CUMEM_ENABLE:-0}" \
  --setenv NCCL_P2P_DISABLE 0 \
  --setenv NCCL_IB_DISABLE 1 \
  --setenv NCCL_SOCKET_IFNAME "${NCCL_SOCKET_IFNAME:-lo}" \
  ${NCCL_PROTO:+--setenv NCCL_PROTO "$NCCL_PROTO"} \
  ${NCCL_ALGO:+--setenv NCCL_ALGO "$NCCL_ALGO"} \
  ${NCCL_MIN_NCHANNELS:+--setenv NCCL_MIN_NCHANNELS "$NCCL_MIN_NCHANNELS"} \
  ${NCCL_NTHREADS:+--setenv NCCL_NTHREADS "$NCCL_NTHREADS"} \
  --setenv SGLANG_M3_FORCE_STATIC "${SGLANG_M3_FORCE_STATIC:-1}" \
  ${SGLANG_M3_FORCE_MARLIN:+--setenv SGLANG_M3_FORCE_MARLIN "$SGLANG_M3_FORCE_MARLIN"} \
  ${FLASHINFER_B12X_STATIC_DYNAMIC_M:+--setenv FLASHINFER_B12X_STATIC_DYNAMIC_M "$FLASHINFER_B12X_STATIC_DYNAMIC_M"} \
  ${CUTE_DSL_JIT_TIME_PROFILING:+--setenv CUTE_DSL_JIT_TIME_PROFILING "$CUTE_DSL_JIT_TIME_PROFILING"} \
  ${CUTE_DSL_LOG_TO_CONSOLE:+--setenv CUTE_DSL_LOG_TO_CONSOLE "$CUTE_DSL_LOG_TO_CONSOLE"} \
  ${CUDA_LAUNCH_BLOCKING:+--setenv CUDA_LAUNCH_BLOCKING "$CUDA_LAUNCH_BLOCKING"} \
  ${TORCH_USE_CUDA_DSA:+--setenv TORCH_USE_CUDA_DSA "$TORCH_USE_CUDA_DSA"} \
  ${SGLANG_M3_DEBUG_PREFILL:+--setenv SGLANG_M3_DEBUG_PREFILL "$SGLANG_M3_DEBUG_PREFILL"} \
  ${CUTE_DSL_LOG_LEVEL:+--setenv CUTE_DSL_LOG_LEVEL "$CUTE_DSL_LOG_LEVEL"} \
  --chdir /sgl-workspace \
  python -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --tokenizer-path "$MODEL_DIR" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name minimax-m3 \
    --tp-size 4 \
    --context-length "${CONTEXT_LENGTH:-1048576}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
    --quantization compressed-tensors \
    --reasoning-parser minimax-m3 \
    --tool-call-parser minimax-m3 \
    --disable-shared-experts-fusion \
    --cuda-graph-backend-decode full \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs-decode "$CG_MAX_BS" \
    --max-running-requests "$MAX_RUNNING" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
    --max-prefill-tokens 16384 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.95}" \
    --watchdog-timeout 1200 \
    --skip-server-warmup \
    ${ENABLE_METRICS:---enable-metrics} \
    ${EXTRA_SERVER_ARGS:-}
