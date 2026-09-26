#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — Bring up LiteLLM and the enabled inference backends on
# 4x RTX PRO 6000 (Blackwell / sm120), in the correct order.
#
#   :4000  LiteLLM local proxy      (litellm-local, HTTPS, CPU-only)
#   :8000  DeepSeek-V4.1-Flash (dsv41, TP4/EP4; public name deepseek-v4-flash)
#   :8001  Qwen3-Embedding-0.6B (DISABLED; stopped container retained for rollback)
#   :8002  Gemma-4-31B-IT-NVFP4 (DISABLED FOR NOW; launcher retained below)
#
# Only dsv41 is an enabled GPU inference backend. LiteLLM is CPU-only.
# Both enabled services use unless-stopped. Do not automatically revive the
# preserved V12, canary, or embedding containers: they compete for the same GPUs.
#
# Usage:
#   ./launch_all.sh                  # launch all (skips any already healthy)
#   ./launch_all.sh --restart        # force-recreate existing containers
#   ./launch_all.sh litellm          # (litellm|dsv41) launch one
#   ./launch_all.sh dsv4             # compatibility alias for dsv41, not V12
# =============================================================================
set -euo pipefail

# ---- shared paths ----------------------------------------------------------
MODELS_DIR=/mnt/hot/ambientlight/models
HF_CACHE=/mnt/hot/ambientlight/.cache/huggingface
LITELLM_REPO=${LITELLM_REPO:-/mnt/hot/ambientlight/repos/litellm}
LITELLM_COMPOSE_FILE=${LITELLM_COMPOSE_FILE:-${LITELLM_REPO}/docker-compose.local.yml}
LAUNCH_REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DSV41_COMPOSE_FILE=${DSV41_COMPOSE_FILE:-${LAUNCH_REPO_DIR}/docker/deepseek-v41/compose.api.yaml}

# DSV4 checkpoint (env-overridable): default to the 0731 weights.
DSV4_MODEL_DIR=${DSV4_MODEL_DIR:-${MODELS_DIR}/DeepSeek-V4-Flash-0731}

# ---- images ----------------------------------------------------------------
# Benchmark-qualified SGLang main; IMG_DSV4 remains env-overridable.
IMG_DSV4=${IMG_DSV4:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480}
IMG_EMBED=sglang-embed:v0.5.15-distro                  # stock lmsys + `pip install distro`
# IMG_GEMMA=sglang-gemma:v0.5.15-distro                # disabled for now

# ---- helper: wait until a server's /health returns 200 ---------------------
wait_healthy() {  # $1=name  $2=port  $3=timeout_s
  local name=$1 port=$2 timeout=${3:-900} i=0
  echo "  waiting for $name on :$port (timeout ${timeout}s)..."
  while [ "$i" -lt "$timeout" ]; do
    if curl -sf -m 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
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
# 1) LiteLLM local proxy  ::4000  (HTTPS, host network, CPU-only)
#    Compose owns its config, TLS, auth-cache mounts, and runtime environment.
#    Start it before the inference backends; routes become usable as each
#    backend comes online.
# ---------------------------------------------------------------------------
launch_litellm() {
  echo "[1/2] LiteLLM local proxy -> https://:4000"
  if [ ! -f "$LITELLM_COMPOSE_FILE" ]; then
    echo "  ❌ LiteLLM Compose file not found: $LITELLM_COMPOSE_FILE"
    return 1
  fi
  docker compose -f "$LITELLM_COMPOSE_FILE" config --quiet
  docker compose -f "$LITELLM_COMPOSE_FILE" up -d \
    --no-build --force-recreate litellm

  local timeout=180 i=0
  echo "  waiting for litellm-local on https://:4000 (timeout ${timeout}s)..."
  while [ "$i" -lt "$timeout" ]; do
    if curl -ksf -m 3 https://localhost:4000/health/liveliness >/dev/null 2>&1; then
      echo "  ✅ litellm-local healthy after ~${i}s"; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q '^litellm-local$'; then
      echo "  ❌ litellm-local container exited. Last logs:"
      docker logs --tail 25 litellm-local 2>&1 | tail -20
      return 1
    fi
    sleep 5; i=$((i+5))
  done
  echo "  ❌ litellm-local did not become healthy in ${timeout}s"; return 1
}

# ---------------------------------------------------------------------------
# 2) Promoted DeepSeek-V4.1-Flash ::8000 (legacy public model name)
# ---------------------------------------------------------------------------
launch_dsv41() {
  echo "[2/2] DeepSeek-V4.1-Flash -> :8000 (container dsv41)"
  local other
  for other in dsv4 dsv41-api dsv41-baseline qwen3-embed; do
    if [ "$(docker inspect --format '{{.State.Running}}' "$other" 2>/dev/null || true)" = true ]; then
      echo "  ❌ $other is still running; stop it explicitly before launching dsv41."
      return 1
    fi
  done
  docker compose -f "$DSV41_COMPOSE_FILE" config --quiet
  docker compose -f "$DSV41_COMPOSE_FILE" up -d --no-build --force-recreate deepseek
  wait_healthy dsv41 8000 1800
}

# Historical launchers are retained as inert rollback documentation. In
# particular, normal startup must neither delete V12 nor enable embeddings.
: <<'LEGACY_GPU_BACKENDS_DISABLED'
# 2) DeepSeek-V4-Flash  ::8000  (TP=4)
#    Model-specific NCCL tuning + SM120 sparse-attn env live inside the
#    image's entrypoint (MODEL=dsv4). We only override the mem fraction.
# ---------------------------------------------------------------------------
launch_dsv4() {
  echo "[2/3] DeepSeek-V4-Flash -> :8000"
  docker rm -f dsv4 >/dev/null 2>&1 || true
  docker run -d --name dsv4 \
    --gpus all --ipc=host --shm-size 32g --restart unless-stopped \
    -e MODEL=dsv4 \
    -e MEM_FRACTION_STATIC=0.85 \
    -e CHUNKED_PREFILL_SIZE=4096 \
    -e SGLANG_DEFAULT_THINKING=0 \
    -e NCCL_DEBUG=WARN \
    -p 8000:8000 \
    -v "${DSV4_MODEL_DIR}:/model:ro" \
    "$IMG_DSV4"
  wait_healthy dsv4 8000 900
}

# ---------------------------------------------------------------------------
# 3) Qwen3-Embedding-0.6B  ::8001  (TP=1, pinned to GPU3)
# ---------------------------------------------------------------------------
launch_qwen() {
  echo "[3/3] Qwen3-Embedding-0.6B -> :8001 (GPU3)"
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
LEGACY_GPU_BACKENDS_DISABLED

# GEMMA4 DISABLED FOR NOW. This entire here-document is a block comment.
# Remove the two GEMMA4_DISABLED lines and restore its dispatch entries to
# re-enable it.
: <<'GEMMA4_DISABLED'
# ---------------------------------------------------------------------------
# Gemma-4-31B-IT-NVFP4  ::8002  (TP=4, full multimodal)  == Config B ==
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
  echo "Gemma-4-31B-IT-NVFP4 -> :8002 (TP=4, multimodal)"
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
GEMMA4_DISABLED

# ---- dispatch --------------------------------------------------------------
FORCE=0; TARGET=all
for arg in "$@"; do
  case "$arg" in
    --restart) FORCE=1 ;;
    litellm|dsv41|dsv4|qwen|all) TARGET=$arg ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

# When launching a single target, --restart is implied (we always recreate it).
case "$TARGET" in
  litellm) launch_litellm ;;
  dsv41|dsv4) launch_dsv41 ;;
  qwen) echo "Qwen embeddings are disabled; only dsv41 is enabled for GPU inference."; exit 1 ;;
  all)
    # If not forcing, skip any backend already healthy so we don't disturb it.
    if [ "$FORCE" -eq 0 ] && curl -ksf -m3 https://localhost:4000/health/liveliness >/dev/null 2>&1; then
      echo "[1/2] LiteLLM already healthy — skipping (use --restart to force)"
    else launch_litellm; fi
    if [ "$FORCE" -eq 0 ] && [ "$(docker inspect --format '{{.State.Running}}' dsv41 2>/dev/null || true)" = true ] && curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1; then
      echo "[2/2] dsv41 already healthy — skipping (use --restart to force)"
    else launch_dsv41; fi
    # Gemma4 disabled for now. Restore its health check/launch branch here.
    ;;
esac

echo
echo "=== all requested services up ==="
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'NAMES|litellm-local|dsv41'
echo
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
