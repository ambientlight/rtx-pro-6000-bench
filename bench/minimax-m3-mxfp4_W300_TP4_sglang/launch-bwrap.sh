#!/usr/bin/env bash
# Launch MiniMax-M3 (MXFP4, MSA) on 4x RTX PRO 6000 (SM120, TP=4) via sglang +
# the NATIVE MXFP4 W4A4 MoE path (clamped SwiGLU-OAI), under bubblewrap.
#
# This is the Quest-5 native path: it serves the olka-fi MXFP4 checkpoint
# (compressed-tensors, routed experts mxfp4-pack-quantized) through our
# Mxfp4W4A4MoEScheme -> launch_sm120_moe(quant_mode="mxfp4", activation="swigluoai")
# instead of the slow Marlin W4A16 dequant. Requires the REPO flashinfer (0.6.13,
# native mxfp4 + the clamp epilogue edits) on PYTHONPATH, since the image's
# baked 0.6.12 lacks native mxfp4.
#
# A/B: SGLANG_M3_FORCE_MARLIN=1 reverts to the Marlin path for comparison.
#
# CONCURRENCY: configured to MATCH the DeepSeek-V4-Flash throughput test
# (max-running 128, cuda-graph capture to bs=128, ctx 131072, chunked-prefill
# 16384, mem-fraction 0.80, fp8 KV) so the M3 vs DSV4 throughput matrix (b.sh) is
# apples-to-apples. The shorter 131072 ctx (vs M3's native 1M) is deliberate — it
# frees KV-cache pool for high concurrency at the 2K-64K bench inputs, exactly as
# the DSV4 config notes. All knobs are env-overridable (MAX_RUNNING, CG_MAX_BS,
# CONTEXT_LENGTH, CHUNKED_PREFILL_SIZE, MEM_FRACTION_STATIC, KV_CACHE_DTYPE).
# `--enable-metrics` is on by default (ENABLE_METRICS to override) for bench_sweep
# telemetry (kv_cache plots). For the 1M-context single-stream sweep use
# launch-bwrap-highconc.sh instead (max-running 16, 1M ctx).
set -euo pipefail

ROOTFS=${ROOTFS:-/mnt/hot/minimax-m3-rootfs}                 # sglang image rootfs (has M3 model + MSA)
NVHOST=${NVHOST:-/mnt/hot/minimax-m3-nvhost}                 # host driver libs @ 595.80
REPO_FI=${REPO_FI:-/mnt/hot/ambientlight/repos/flashinfer}   # native-mxfp4 + clamp-patched flashinfer
MODEL_DIR=${MODEL_DIR:-$HOME/models/minimax-m3-mxfp4}        # MXFP4 (compressed-tensors) checkpoint
FICACHE=${FICACHE:-/mnt/hot/minimax-m3-ficache}              # persistent flashinfer JIT cache (writable)
PORT=${PORT:-8000}

# Throughput-bench concurrency config — MATCHES the DSV4 throughput test
# (bench/deepseek-v4-flash_W300_TP4_sglang/sglang.yaml): max-running 128 +
# cuda-graph capture to bs=128, context 131072 (NOT M3's native 1M — a shorter
# ctx frees KV-cache pool for high concurrency at the 2K-64K bench inputs),
# chunked-prefill 16384, mem-fraction 0.80, fp8 KV. Overridable via env.
MAX_RUNNING=${MAX_RUNNING:-128}
CG_MAX_BS=${CG_MAX_BS:-128}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}

[ -d "$ROOTFS/sgl-workspace" ] || { echo "rootfs missing at $ROOTFS" >&2; exit 1; }
[ -e "$NVHOST/lib/libcuda.so.1" ] || { echo "host driver libs missing at $NVHOST" >&2; exit 1; }
[ -f "$REPO_FI/flashinfer/__init__.py" ] || { echo "repo flashinfer missing at $REPO_FI" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "MXFP4 checkpoint missing at $MODEL_DIR" >&2; exit 1; }
mkdir -p "$FICACHE"

DEV_BINDS=()
for d in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [ -e "$d" ] && DEV_BINDS+=( --dev-bind "$d" "$d" )
done

echo "binding repo flashinfer (native mxfp4 + clamp) + $NVHOST + $(( ${#DEV_BINDS[@]} / 2 )) GPU nodes"

# NCCL: PCIe-only rig under bwrap. SHM transport uses CUDA-IPC which fails in the
# user namespace (cuda error 801); route over P2P/NET. (Same fix as the vLLM path.)
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
  --setenv SGLANG_M3_FORCE_STATIC "${SGLANG_M3_FORCE_STATIC:-1}" \
  ${SGLANG_M3_FORCE_MARLIN:+--setenv SGLANG_M3_FORCE_MARLIN "$SGLANG_M3_FORCE_MARLIN"} \
  ${FLASHINFER_B12X_STATIC_DYNAMIC_M:+--setenv FLASHINFER_B12X_STATIC_DYNAMIC_M "$FLASHINFER_B12X_STATIC_DYNAMIC_M"} \
  --chdir /sgl-workspace \
  python -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --tokenizer-path "$MODEL_DIR" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name minimax-m3 \
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
    ${ENABLE_METRICS:---enable-metrics}
