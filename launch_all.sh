#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — Bring up all inference backends on 4x RTX PRO 6000
# (Blackwell / sm120), sharing GPUs, in the correct order.
#
#   :8000  DeepSeek-V4-Flash    (dsv4,   TP=4, all GPUs, mem-fraction-static 0.65)
#   :8010  DSV4 reasoning bridge (dsv4-reasoning-bridge, CPU-only relay -> :8000)
#   :8001  Qwen3-Embedding-0.6B (qwen3-embed, GPU3 only, mem-fraction-static 0.10)
#   :8002  Gemma-4-31B-IT-NVFP4 (gemma4, TP=4, all GPUs, mem-fraction-static 0.70)
#
# ORDER MATTERS. In this SGLang build, --mem-fraction-static is a fraction of
# the memory that is FREE WHEN THE PROCESS STARTS, not of total VRAM. Each
# server must launch after the previous one has claimed its share, so launch
# strictly: DeepSeek -> Qwen -> Gemma. Re-running out of order will mis-size
# the KV pools and can OOM the triple-tenant card (GPU3).
#
# REBOOT SAFETY. All three containers use --restart unless-stopped, so Docker
# brings them back after a reboot -- but Docker restarts them CONCURRENTLY in
# arbitrary order, which would violate the ordering above. To keep reboots
# safe, the later two containers SELF-SEQUENCE: qwen waits for :8000/health and
# gemma waits for :8001/health (via the host gateway) before launching sglang.
# So regardless of the order Docker starts them, each blocks until its
# predecessor has claimed its VRAM. The wait loops are baked into the container
# command below, so manual `./launch_all.sh` and reboot auto-start behave the
# same way.
#
# Usage:
#   ./launch_all.sh                  # launch all (skips any already healthy)
#   ./launch_all.sh --restart        # force-remove existing containers first
#   ./launch_all.sh dsv4             # (dsv4|bridge|qwen|gemma) launch just one
#   DSV4_MODEL_DIR=/path ./launch_all.sh dsv4   # override the DSV4 checkpoint
# =============================================================================
set -euo pipefail

# ---- shared paths ----------------------------------------------------------
MODELS_DIR=/mnt/hot/ambientlight/models
HF_CACHE=/mnt/hot/ambientlight/.cache/huggingface

# DSV4 checkpoint (env-overridable): default to the 0731 weights.
DSV4_MODEL_DIR=${DSV4_MODEL_DIR:-${MODELS_DIR}/DeepSeek-V4-Flash-0731}

# ---- images ----------------------------------------------------------------
# Benchmark-qualified SGLang main; IMG_DSV4 remains env-overridable.
IMG_DSV4=${IMG_DSV4:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a}
IMG_BRIDGE=ambientlight/dsv4-reasoning-bridge:latest   # local reasoning-bridge relay
IMG_EMBED=sglang-embed:v0.5.15-distro                  # stock lmsys + `pip install distro`
IMG_GEMMA=sglang-gemma:v0.5.15-distro                  # stock lmsys + `pip install distro`

# ---- helper: wait until a server's /health returns 200 ---------------------
wait_healthy() {  # $1=name  $2=port  $3=timeout_s
  local name=$1 port=$2 timeout=${3:-900} i=0
  echo "  waiting for $name on :$port (timeout ${timeout}s)..."
  while [ "$i" -lt "$timeout" ]; do
    if curl -s -m 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "  ✅ $name healthy after ~${i}s"; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${name}\$"; then
      echo "  ❌ $name container exited. Last logs:"; docker logs --tail 25 "$name" 2>&1 | grep -viE 'torchcodec|libav' | tail -20
      return 1
    fi
    sleep 10; i=$((i+10))
  done
  echo "  ❌ $name did not become healthy in ${timeout}s"; return 1
}

# ---------------------------------------------------------------------------
# 1) DeepSeek-V4-Flash  ::8000  (TP=4)
#    Model-specific NCCL tuning + SM120 sparse-attn env live inside the
#    image's entrypoint (MODEL=dsv4). We only override the mem fraction.
# ---------------------------------------------------------------------------
launch_dsv4() {
  echo "[1/4] DeepSeek-V4-Flash -> :8000"
  docker rm -f dsv4 >/dev/null 2>&1 || true
  docker run -d --name dsv4 \
    --gpus all --ipc=host --shm-size 32g --restart unless-stopped \
    -e MODEL=dsv4 \
    -e MEM_FRACTION_STATIC=0.65 \
    -e NCCL_DEBUG=WARN \
    -p 8000:8000 \
    -v "${DSV4_MODEL_DIR}:/model:ro" \
    "$IMG_DSV4"
  wait_healthy dsv4 8000 900
}

# ---------------------------------------------------------------------------
# 1b) DSV4 reasoning bridge  ::8010  (plain-HTTP FastAPI relay, no GPU)
#     Forces thinking, separates reasoning, re-emits DSML tool-calls as
#     OpenAI tool_calls, repairs streaming tool-call artifacts. LiteLLM's
#     deepseek-v4-flash route points here (:8010), which forwards to :8000.
#     Reaches the dsv4 sglang server via the host gateway. TLS is terminated
#     upstream at LiteLLM; this stays HTTP. Waits for dsv4 :8000 first.
# ---------------------------------------------------------------------------
launch_bridge() {
  echo "[1b/4] DSV4 reasoning bridge -> :8010"
  docker rm -f dsv4-reasoning-bridge >/dev/null 2>&1 || true
  docker run -d --name dsv4-reasoning-bridge \
    --add-host=host:host-gateway --restart unless-stopped \
    -e SGLANG_BASE_URL=http://host:8000/v1 \
    -e BRIDGE_DEBUG=1 \
    -p 8010:8010 \
    --entrypoint bash \
    "$IMG_BRIDGE" \
    -c 'echo "  [bridge] waiting for DeepSeek :8000/health before launch..."; \
        until curl -sf -m3 http://host:8000/health >/dev/null 2>&1; do sleep 5; done; \
        echo "  [bridge] DeepSeek healthy — launching."; \
        exec python deepseek_reasoning_bridge.py'
  wait_healthy dsv4-reasoning-bridge 8010 120
}

# ---------------------------------------------------------------------------
# 2) Qwen3-Embedding-0.6B  ::8001  (TP=1, pinned to GPU3)
# ---------------------------------------------------------------------------
launch_qwen() {
  echo "[2/4] Qwen3-Embedding-0.6B -> :8001 (GPU3)"
  docker rm -f qwen3-embed >/dev/null 2>&1 || true
  docker run -d --name qwen3-embed \
    --gpus all --shm-size 4g --restart unless-stopped \
    --add-host=host:host-gateway \
    -e CUDA_VISIBLE_DEVICES=3 \
    -e HF_HOME=/hf \
    -p 8001:8001 \
    -v "${HF_CACHE}:/hf" \
    --entrypoint bash \
    "$IMG_EMBED" \
    -c 'echo "  [qwen] waiting for DeepSeek :8000/health before launch..."; \
        until curl -sf -m3 http://host:8000/health >/dev/null 2>&1; do sleep 5; done; \
        echo "  [qwen] DeepSeek healthy — launching."; \
        exec python3 -m sglang.launch_server \
          --model-path Qwen/Qwen3-Embedding-0.6B \
          --is-embedding --host 0.0.0.0 --port 8001 --tp 1 \
          --mem-fraction-static 0.10 --attention-backend triton'
  wait_healthy qwen3-embed 8001 300
}

# ---------------------------------------------------------------------------
# 3) Gemma-4-31B-IT-NVFP4  ::8002  (TP=4, full multimodal)  == Config B ==
#    - SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 : GPU3 is co-tenant; the >10%
#      free-mem imbalance across ranks is expected, so downgrade abort->warn.
#    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : reclaims fragmented
#      reserve; without it warmup OOMs by ~300MB on GPU3.
#    - mem-fraction-static 0.70 : fraction of free-at-startup (see header).
#    - cuda-graph-max-bs 16 : bound graph-capture memory on the tight GPU3.
#    - tool-call-parser / reasoning-parser gemma4 : parse the model's native
#      tool-call + thinking syntax into structured OpenAI tool_calls. Without
#      these, SGLang leaks raw "<|tool_call>..." as plain content and clients
#      (Claude Code via LiteLLM) can't execute tools.
#    NOTE: no NCCL tuning and no --disable-custom-all-reduce — benchmarking
#    showed both hurt or were neutral for this dense model (Config B was best).
# ---------------------------------------------------------------------------
launch_gemma() {
  echo "[3/4] Gemma-4-31B-IT-NVFP4 -> :8002 (TP=4, multimodal)"
  docker rm -f gemma4 >/dev/null 2>&1 || true
  docker run -d --name gemma4 \
    --gpus all --ipc=host --shm-size 32g --restart unless-stopped \
    --add-host=host:host-gateway \
    -e HF_HOME=/hf \
    -e NCCL_DEBUG=WARN \
    -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -p 8002:8002 \
    -v "${MODELS_DIR}/gemma-4-31b-it-nvfp4:/model:ro" \
    -v "${HF_CACHE}:/hf" \
    --entrypoint bash \
    "$IMG_GEMMA" \
    -c 'echo "  [gemma] waiting for Qwen :8001/health before launch..."; \
        until curl -sf -m3 http://host:8001/health >/dev/null 2>&1; do sleep 5; done; \
        echo "  [gemma] Qwen healthy — launching."; \
        exec python3 -m sglang.launch_server \
          --model-path /model --served-model-name gemma-4-31b-it \
          --tp 4 --trust-remote-code --host 0.0.0.0 --port 8002 \
          --context-length 262144 --mem-fraction-static 0.70 \
          --kv-cache-dtype fp8_e4m3 --attention-backend triton \
          --cuda-graph-max-bs 16 \
          --tool-call-parser gemma4 --reasoning-parser gemma4'
  wait_healthy gemma4 8002 900
}

# ---- dispatch --------------------------------------------------------------
FORCE=0; TARGET=all
for arg in "$@"; do
  case "$arg" in
    --restart) FORCE=1 ;;
    dsv4|bridge|qwen|gemma|all) TARGET=$arg ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

# When launching a single target, --restart is implied (we always recreate it).
case "$TARGET" in
  dsv4)   launch_dsv4 ;;
  bridge) launch_bridge ;;
  qwen)   launch_qwen ;;
  gemma)  launch_gemma ;;
  all)
    # If not forcing, skip any backend already healthy so we don't disturb it.
    if [ "$FORCE" -eq 0 ] && curl -s -m3 http://localhost:8000/health >/dev/null 2>&1; then
      echo "[1/4] DeepSeek already healthy — skipping (use --restart to force)"
    else launch_dsv4; fi
    if [ "$FORCE" -eq 0 ] && curl -s -m3 http://localhost:8010/health >/dev/null 2>&1; then
      echo "[1b/4] DSV4 bridge already healthy — skipping (use --restart to force)"
    else launch_bridge; fi
    if [ "$FORCE" -eq 0 ] && curl -s -m3 http://localhost:8001/health >/dev/null 2>&1; then
      echo "[2/4] Qwen already healthy — skipping (use --restart to force)"
    else launch_qwen; fi
    if [ "$FORCE" -eq 0 ] && curl -s -m3 http://localhost:8002/health >/dev/null 2>&1; then
      echo "[3/4] Gemma already healthy — skipping (use --restart to force)"
    else launch_gemma; fi
    ;;
esac

echo
echo "=== all requested backends up ==="
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'NAMES|dsv4|bridge|qwen|gemma'
echo
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
