#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash for the SWE-bench Verified eval (thinking + MAX reasoning
# effort) on the served :8000 endpoint. Config: sglang-single.yaml (1M context,
# max-running-requests 16). Wait for /v1/models before sending requests.
set -euo pipefail

source ~/.venvs/dsv4-test/bin/activate

# Selection
export SGLANG_SM120_SPARSE_DECODE=hmma           # HMMA tensor-core sparse decode
export SGLANG_SM120_SPARSE_PREFILL=hmma           # HMMA tensor-core sparse prefill (> 11673-token batches)

# SM120 path (SM120+DeepseekV4 auto-sets FP8_WO_A_GEMM, USE_TOPK_V2, TILELANG_MHC_PRE,
# DEEPGEMM_HC_PRENORM, FP8_PAGED_MQA_LOGITS_TORCH at startup).
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_SM120_INDEXER_SPLIT=1               # split-KV indexer logits scan (single-stream long-ctx)
export SGLANG_SM120_INDEXER_SPLIT_COUNT=128
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0

# --- Reasoning (thinking + MAX effort) ---
# dsv4 chat encoder injects the MAX-effort prefix only when BOTH are set; the
# --reasoning-parser below splits <think>...</think> into reasoning_content so a
# client only sees clean `content`.
export SGLANG_DEFAULT_THINKING=1                 # thinking_mode -> "thinking"
export SGLANG_DSV4_REASONING_EFFORT=max          # inject "Reasoning Effort: Absolute maximum ..."

# Version guards
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL (PCIe, no NVLink)
export NCCL_PROTO=LL NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=8 NCCL_NTHREADS=512
export CUDA_VISIBLE_DEVICES=0,1,2,3

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU coredump on device exception (backtrace-only, ~few MB). Read with:
#   cuda-gdb python <CUDA_COREDUMP_FILE>
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS="skip_global_memory,skip_shared_memory,skip_local_memory"
export CUDA_COREDUMP_FILE="$HERE/cudacore.%h.%p"

# --reasoning-parser deepseek-v4 splits <think> out of responses.
# --tool-call-parser deepseekv4 parses the model's native <｜DSML｜...> tool-call
# markup into structured tool_calls (required for tool-calling clients, e.g.
# mini-swe-agent v2 — without it the calls come back as raw content and fail to parse).
exec python -m sglang.launch_server --config "$HERE/sglang-single.yaml" \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4
