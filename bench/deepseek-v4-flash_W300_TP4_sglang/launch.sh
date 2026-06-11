#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash (MXFP4 W4A4 + HMMA sparse decode) for bench_sweep.
# Canonical deployment: docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md
# Wait for /v1/models before running bench-sweep. Startup ~2 min (weight load + graph capture).
set -euo pipefail

source ~/.venvs/dsv4-test/bin/activate

# Selection
export SGLANG_SM120_SPARSE_DECODE=hmma           # HMMA tensor-core sparse decode (else triton)
export SGLANG_SM120_SPARSE_PREFILL=hmma           # HMMA tensor-core sparse prefill (> 11673-token batches)

# SM120 path (SM120+DeepseekV4 auto-sets FP8_WO_A_GEMM, USE_TOPK_V2, TILELANG_MHC_PRE,
# DEEPGEMM_HC_PRENORM, FP8_PAGED_MQA_LOGITS_TORCH at startup).
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0

# Version guards
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL (PCIe, no NVLink)
export NCCL_PROTO=LL NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=8 NCCL_NTHREADS=512
export CUDA_VISIBLE_DEVICES=0,1,2,3

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU coredump on device exception. Captures the faulting kernel of any
# single-rank device-side fault (e.g. the transient that killed a long sweep at
# c104/8192in) that otherwise dies without a traceback. Dump is backtrace/PC/
# registers only (skip_*_memory), so it stays a few MB, not 96 GB. Read with:
#   cuda-gdb python <CUDA_COREDUMP_FILE>
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS="skip_global_memory,skip_shared_memory,skip_local_memory"
export CUDA_COREDUMP_FILE="$HERE/cudacore.%h.%p"

exec python -m sglang.launch_server --config "$HERE/sglang.yaml"
