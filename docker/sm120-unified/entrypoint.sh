#!/usr/bin/env bash
# Unified dispatcher: pick the model with MODEL={dsv4|m3} (default m3), set the
# ONE optimized NCCL config both models share, then exec the model entrypoint.
#
# The unified NCCL block is the whole point of this image: in a real container
# /sys is native (no bwrap workaround), so both models get direct-P2P transport
# (SHM+legacy-IPC) PLUS the DSV4 latency tuning (LL/Ring). Same collective config
# for both -> apples-to-apples benchmarks. Everything env-overridable.
set -euo pipefail

# ---- shared optimized NCCL (PCIe, no NVLink; TP=4) ----
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}          # SHM ON -> legacy-IPC direct P2P (45us vs 277us socket)
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}        # legacy IPC path
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}     # single-host TP=4 loopback
export NCCL_PROTO=${NCCL_PROTO:-LL}                     # low-latency proto (small 12KB all-reduces)
export NCCL_ALGO=${NCCL_ALGO:-Ring}                     # Ring for 4-way PCIe
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_NTHREADS=${NCCL_NTHREADS:-512}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-m3}
case "$MODEL" in
  dsv4|deepseek|deepseek-v4|deepseek-v4-flash)
    echo "[unified] dispatching -> DeepSeek-V4-Flash"
    exec /usr/local/bin/entrypoint-dsv4.sh "$@"
    ;;
  m3|minimax|minimax-m3)
    echo "[unified] dispatching -> MiniMax-M3"
    exec /usr/local/bin/entrypoint-m3.sh "$@"
    ;;
  *)
    echo "ERROR: unknown MODEL='$MODEL' (use MODEL=dsv4 or MODEL=m3)" >&2
    exit 1
    ;;
esac
