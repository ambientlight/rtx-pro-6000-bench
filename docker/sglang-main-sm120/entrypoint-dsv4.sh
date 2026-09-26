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
# Keep long Claude Code replays below the DSV4 indexer's peak scratch-memory
# cliff. At TP=4, 8192-token chunks can require ~2.6 GiB per rank once a
# prompt reaches ~350k tokens; 4096 keeps the same allocation near ~1.3 GiB.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}
# A very long chunked prefill otherwise wins every scheduler round and can
# leave an already-streaming decode silent for minutes. Four decode rounds
# after each prefill chunk is the best-balanced setting in SGLang's published
# DSV4 long-context sweep; callers can still override it per deployment.
PREFILL_DECODE_INTERVAL=${PREFILL_DECODE_INTERVAL:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
# Missing request fields inherit these operator defaults. Explicit sampling
# values sent by a caller remain authoritative inside SGLang. These match the
# checkpoint's documented recommendation for agentic workloads; low-temperature
# experiments remain opt-in at the LiteLLM route.
DSV4_PREFERRED_SAMPLING_PARAMS=${DSV4_PREFERRED_SAMPLING_PARAMS:-'{"temperature":1.0,"top_p":0.95}'}

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

# Anthropic requests that explicitly enable thinking are emitted as structured
# thinking blocks. When the field is omitted, SGLang's Anthropic adapter does
# not activate the response parser; defaulting thinking on in that case leaks
# the model's closing </think> marker into normal text. Keep omitted thinking
# off, while preserving explicit Claude Code extended-thinking requests.
export SGLANG_DEFAULT_THINKING=${SGLANG_DEFAULT_THINKING:-0}
export SGLANG_DSV4_REASONING_EFFORT=${SGLANG_DSV4_REASONING_EFFORT:-high}
# Malformed/incomplete tool protocol gets one isolated, grammar-constrained
# retry. Generic SGLang images remain opt-in; this qualified DSV4 image enables
# it with bounded decode, concurrency, timeout, and SSE heartbeat defaults for
# both Anthropic Messages and OpenAI Responses traffic.
export SGLANG_ANTHROPIC_TOOL_RECOVERY=${SGLANG_ANTHROPIC_TOOL_RECOVERY:-1}
export SGLANG_ANTHROPIC_TOOL_RECOVERY_TIMEOUT_S=${SGLANG_ANTHROPIC_TOOL_RECOVERY_TIMEOUT_S:-180}
export SGLANG_ANTHROPIC_TOOL_RECOVERY_MAX_TOKENS=${SGLANG_ANTHROPIC_TOOL_RECOVERY_MAX_TOKENS:-8192}
export SGLANG_ANTHROPIC_TOOL_RECOVERY_CONCURRENCY=${SGLANG_ANTHROPIC_TOOL_RECOVERY_CONCURRENCY:-2}
export SGLANG_ANTHROPIC_TOOL_RECOVERY_PING_INTERVAL_S=${SGLANG_ANTHROPIC_TOOL_RECOVERY_PING_INTERVAL_S:-10}
export SGLANG_RESPONSES_TOOL_RECOVERY=${SGLANG_RESPONSES_TOOL_RECOVERY:-1}
export SGLANG_RESPONSES_TOOL_RECOVERY_TIMEOUT_S=${SGLANG_RESPONSES_TOOL_RECOVERY_TIMEOUT_S:-180}
export SGLANG_RESPONSES_TOOL_RECOVERY_MAX_TOKENS=${SGLANG_RESPONSES_TOOL_RECOVERY_MAX_TOKENS:-8192}
export SGLANG_RESPONSES_TOOL_RECOVERY_CONCURRENCY=${SGLANG_RESPONSES_TOOL_RECOVERY_CONCURRENCY:-2}
export SGLANG_RESPONSES_TOOL_RECOVERY_PING_INTERVAL_S=${SGLANG_RESPONSES_TOOL_RECOVERY_PING_INTERVAL_S:-10}
# These dumps are deliberately bounded and contain only the raw model suffix
# needed to diagnose a parser failure. The preserved V12 container mounts the
# protected host dump root here so artifacts survive container replacement.
export SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR=${SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR:-/mnt/hot/dsv4_dumps/parser-failures}
export SGLANG_RESPONSES_TOOL_FAILURE_TAIL_CHARS=${SGLANG_RESPONSES_TOOL_FAILURE_TAIL_CHARS:-8192}

echo "DSV4 main control: commit=${SGLANG_BUILD_COMMIT:-unknown} model=${MODEL_DIR} ctx=${CONTEXT_LENGTH} mrr=${MAX_RUNNING} kv=${KV_CACHE_DTYPE} port=${PORT} anthropic_tool_recovery=${SGLANG_ANTHROPIC_TOOL_RECOVERY} responses_tool_recovery=${SGLANG_RESPONSES_TOOL_RECOVERY} responses_failure_dump=${SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR}"

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
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --prefill-decode-interval "${PREFILL_DECODE_INTERVAL}" \
  --page-size 256 \
  --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --preferred-sampling-params "${DSV4_PREFERRED_SAMPLING_PARAMS}" \
  --enable-metrics \
  --watchdog-timeout 3600 --log-level info \
  ${EXTRA_SERVER_ARGS:-}
