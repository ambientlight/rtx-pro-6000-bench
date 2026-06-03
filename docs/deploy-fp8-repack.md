# DeepSeek-V4-Flash FP8 Deployment Guide

**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, FP8 checkpoint)
**Stack:** SGLang (local, branch `sm120-nvfp4-rebase`) + Triton MoE + HMMA sparse decode kernel
**Repo:** `/mnt/hot/ambientlight/repos/sglang`

---

## Prerequisites

- Model checkpoint: `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8`
- DeepSeek-V4 custom HMMA kernels: `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker`
- SGLang repo: `/mnt/hot/ambientlight/repos/sglang` (branch: `sm120-nvfp4-rebase`)
- FlashInfer repo: `/mnt/hot/ambientlight/repos/flashinfer` (symlinked into venv, AOT precompiled)
- Python venv: `/mnt/hot/ambientlight/.venvs/sglang-cu130`

## Server Launch

Launch script at `repos/sglang/debug/launch-fp8.sh`:

```bash
#!/bin/bash
source /mnt/hot/ambientlight/.venvs/sglang-cu130/bin/activate
export PYTHONPATH="/mnt/hot/ambientlight/repos/sglang/python:/mnt/hot/ambientlight/repos/deepseek-v4-flash-sm120/build-docker/deepseek_v4_kernel:/mnt/hot/ambientlight/repos/deepseek-v4-flash-sm120/build-docker"
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0
export SGLANG_OPT_USE_TOPK_V2=0
export NCCL_PROTO=LL
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=512
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_DISABLE_VERSION_CHECK=1

LOGFILE="/mnt/hot/ambientlight/repos/sglang/debug/fp8.log"

exec python -m sglang.launch_server \
  --model-path /mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --host 0.0.0.0 --port 8000 \
  --context-length 131072 --mem-fraction-static 0.85 \
  --max-running-requests 4 \
  --kv-cache-dtype fp8_e4m3 \
  --fp8-gemm-backend triton --moe-runner-backend triton \
  --chunked-prefill-size 32768 --page-size 256 \
  --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --dsa-topk-backend torch \
  --watchdog-timeout 3600 --log-level info \
  > "$LOGFILE" 2>&1
```

Run with:
```bash
nohup bash /mnt/hot/ambientlight/repos/sglang/debug/launch-fp8.sh &
```

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SGLANG_DSV4_FP4_EXPERTS=0` | Disabled | Use FP8 weights for MoE experts, not FP4 |
| `SGLANG_OPT_USE_TOPK_V2=0` | Disabled | SM120: `topk_v2.cuh` JIT kernel crashes on SM120 |
| `SGLANG_OPT_USE_FUSED_HASH_TOPK=0` | Disabled | SM120: fused hash topk not compatible |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` | Enabled | SM120: use torch fallback for paged MQA logits |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | Enabled | FlashInfer repo (0.6.12) vs cubin wheel (0.6.11.post3) mismatch |
| `--fp8-gemm-backend triton` | Triton | FP8 GEMM via Triton kernels |
| `--moe-runner-backend triton` | Triton | MoE dispatch via Triton (not Marlin — crashes on SM120) |
| `--dsa-topk-backend torch` | Torch | SM120: `sgl-kernel` topk crashes during CUDA graph capture |
| `--disable-shared-experts-fusion` | Disabled | Required for SM120 compatibility |
| `--chunked-prefill-size 32768` | 32K | Larger chunks for better prefill throughput |
| `--mem-fraction-static 0.85` | 85% | Higher KV cache allocation |
| `--max-running-requests 4` | 4 | FP8 memory constraints |
| `SGLANG_OPT_USE_TILELANG_INDEXER=1` | Enabled | **Critical**: tilelang compiled indexer. Without this, falls back to torch Python loop (15-20× slower decode) |
| `SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1` | Enabled | Tilelang SWA prepare (requires tilelang==0.1.8) |
| `SGLANG_OPT_USE_TILELANG_MHC_PRE=1` | Enabled | Tilelang MHC pre (requires tilelang==0.1.8) |
| `SGLANG_OPT_USE_TILELANG_MHC_POST=1` | Enabled | Tilelang MHC post (requires tilelang==0.1.8) |

## SM120-Specific Patches (in `repos/sglang` branch `sm120-nvfp4-rebase`)

| File | Change |
|------|--------|
| `layers/attention/deepseek_v4_backend.py` | Route SM120 sparse decode to HMMA kernel instead of `flash_mla.cuda` |
| `layers/attention/dsv4/indexer.py` | CUDA-graph-compatible SM120 indexer (no `.item()` calls during capture) |
| `layers/quantization/mxfp4_marlin_moe.py` | NVFP4 W4A4 + Triton MoE dispatch for FP4 path |
| `entrypoints/warmup.py` | `moe_w4a4` warmup for NVFP4 kernel compilation |

## How SM120 FP8 Decode Works

1. **Sparse decode attention** → Custom HMMA tensor-core kernel in `deepseek-v4-flash-sm120/deepseek_v4_kernel/` (compiled `.so`). Monkey-patched via `sitecustomize.py` → `_patch.py` which intercepts `flash_mla.flash_mla_with_kvcache` on SM120.
2. **FP8 MoE** → SGLang Triton fused MoE runner
3. **FP8 dense GEMMs** → SGLang Triton FP8 backend
4. **Indexer/TopK** → Torch fallbacks (SM120 JIT kernels not compatible)

## FlashInfer AOT Kernels

FlashInfer ops are AOT-precompiled at `repos/flashinfer/flashinfer/data/aot/` (653 ops, 412MB). Built with:

```bash
cd /mnt/hot/ambientlight/repos/flashinfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
export TMPDIR=/mnt/hot/ambientlight/repos/flashinfer/build/tmp
mkdir -p $TMPDIR
FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_CUDA_ARCH_LIST="12.0f" \
python -m flashinfer.aot --out-dir flashinfer/data/aot --build-dir /tmp/flashinfer-aot-build
```

Without AOT, the first request triggers a ~3 min JIT compilation hang for `sampling.so`.

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Startup time | ~75s (AOT kernels) / ~4 min (first run, tilelang JIT) |
| Single decode | ~67 tok/s |
| 4-concurrent decode | ~170 tok/s |
| Prefill throughput (1K) | ~500+ tok/s |
| Prefill throughput (4K+) | ~800-1000+ tok/s |
| Prefix cache hit rate | >96% for multi-turn |
| GPU utilization | 100% sustained |
| 3-min throughput sample | ~10,800 tok/s aggregate |

## Critical: tilelang Version Pinning

The tilelang indexer kernel (`SGLANG_OPT_USE_TILELANG_INDEXER=1`) is **the single most important setting** for FP8 decode performance. Without it, decode falls back to a pure Python torch loop at 7-12 tok/s instead of 133-173 tok/s.

**Required versions** (must match Docker image `lmsysorg/sglang:deepseek-v4-blackwell`):
```bash
pip install tilelang==0.1.8 apache-tvm-ffi==0.1.9
```

tilelang 0.1.10 + tvm-ffi 0.1.11 have a TVM buffer shape regression that crashes SM120 CUDA graph capture. Symptoms: `Check failed: (buffer->shape.size() >= 2) is false: The dimension of Buffer "k_smem_u8" with shape (8192,) should be at least 2`

## Monitoring

```bash
# Server logs
tail -f /mnt/hot/ambientlight/repos/sglang/debug/fp8.log

# GPU utilization
watch -n1 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# Health check
curl http://localhost:8000/health
```

## Head-to-Head: FP8 vs NVFP4 (Same Hardware, Same Repo)

| Metric | FP8 (tilelang indexer) | NVFP4 (CuTe-DSL + _StaticMoELaunch) | Notes |
|--------|----------------------|--------------------------------------|-------|
| 1w decode | ~67 tok/s | **81 tok/s** | NVFP4 +21% (W4A4 tensor cores) |
| 4w decode | ~170 tok/s | **259 tok/s** | NVFP4 +52% |
| 8w decode | ~260 tok/s (est) | **442 tok/s** | NVFP4 +70% |
| Warm prefill (256-tok chunks) | ~1,550 tok/s | **~1,680 tok/s** | Comparable |
| JIT stalls after warmup | None | **None** | `_StaticMoELaunch` eliminates per-M JIT |
| GPU utilization | 100% sustained | 100% sustained | Same |
| Prefix cache hit rate | 96.2% | 96.2% | Same (shared attention) |
| Checkpoint size | 274 GB (shared) | 274 GB (shared) | Same FP8 checkpoint |
| VRAM per GPU | ~82 GB | ~86 GB | NVFP4 +4 GB (re-quantized MoE) |
| Startup time | ~75s | ~210s | NVFP4 slower (re-quant + CuTe-DSL JIT) |
| MoE kernel | Triton FP8 (pre-compiled) | FlashInfer CuTe-DSL W4A4 (JIT once) | NVFP4 uses SM120 W4A4 tensor cores |

**NVFP4 is the recommended path for all workloads** — it delivers 50-100% faster decode, comparable warm prefill throughput, and 3.6× more KV cache capacity (from FP4 weight compression). FP8 remains useful only when minimal startup time matters (75s vs 210s) or as a simpler fallback without the FlashInfer `_StaticMoELaunch` patch.

## Appendix: E2E Kernel Pipeline (FP8 Decode, 1 Token)

```
TOKEN IN
    │
    ▼
┌─ EMBEDDING ─────────────────────────────────────────── [Torch] ─┐
│  VocabParallelEmbedding → embed + repeat for HC                  │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ PER LAYER ×61 ─────────────────────────────────────────────────┐
│                                                                   │
│  ┌─ HC_PRE (attention) ────────────────────────────────────────┐ │
│  │  deep_gemm tf32 prenorm GEMM              [DeepGEMM]        │ │
│  │  mhc_pre fused Sinkhorn+RMSNorm           [TileLang 0.1.8] │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      │                                                            │
│      ▼                                                            │
│  ┌─ MQA ATTENTION ─────────────────────────────────────────────┐ │
│  │  Q proj: wq_a FP8 GEMM                    [Triton FP8]      │ │
│  │  Q norm + RoPE fused                       [Triton]          │ │
│  │  Q proj: wq_b FP8 GEMM                    [Triton FP8]      │ │
│  │  KV proj + cache write                     [Triton]          │ │
│  │                                                               │ │
│  │  C4 Indexer:                                                  │ │
│  │    compressor gate GEMM                    [Triton]          │ │
│  │    fused_q_indexer_rope_hadamard_quant     [Triton]          │ │
│  │    fp8_paged_mqa_logits             ★      [TileLang 0.1.8] │ │
│  │    topk_transform_512                      [Triton]          │ │
│  │                                                               │ │
│  │  Sparse Decode:                                               │ │
│  │    sparse_decode_fwd                ★★     [HMMA custom .so] │ │
│  │                                                               │ │
│  │  Output:                                                      │ │
│  │    wo_a FP8 einsum                         [DeepGEMM]        │ │
│  │    wo_b FP8 GEMM + AllReduce               [Triton + NCCL]  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      │                                                            │
│      ▼                                                            │
│  ┌─ HC_POST (attention) ───────────────────────────────────────┐ │
│  │  mhc_post fused combine+residual          [TileLang 0.1.8] │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      │                                                            │
│      ▼                                                            │
│  ┌─ HC_PRE (FFN) ──────────────────────────────────────────────┐ │
│  │  (same as attention HC_PRE)          [DeepGEMM + TileLang]  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      │                                                            │
│      ▼                                                            │
│  ┌─ MoE FFN (43 MoE layers) / Dense FFN (3+15 layers) ────────┐ │
│  │  Router gate GEMM + topk                   [Triton]          │ │
│  │  Fused MoE experts:                                           │ │
│  │    gate_proj FP8 GEMM                      [Triton fused]   │ │
│  │    up_proj FP8 GEMM                        [Triton fused]   │ │
│  │    SiLU activation                         [Triton fused]   │ │
│  │    down_proj FP8 GEMM                      [Triton fused]   │ │
│  │  TP AllReduce                              [NCCL LL/Ring]   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      │                                                            │
│      ▼                                                            │
│  ┌─ HC_POST (FFN) ─────────────────────────────────────────────┐ │
│  │  mhc_post fused combine+residual          [TileLang 0.1.8] │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ LM HEAD ───────────────────────────────────────────────────────┐
│  fused_hc_head (weighted sum + RMSNorm)       [Triton]           │
│  lm_head FP8 GEMM → logits                   [Triton FP8]       │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ SAMPLING ──────────────────────────────────────────────────────┐
│  top_k_top_p_sampling_from_probs              [FlashInfer AOT]   │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
TOKEN OUT
```

★ = tilelang C4 indexer (15-20× bottleneck if using torch fallback)
★★ = custom HMMA sparse decode kernel (2× faster than stock flash_mla)

### Kernel Backend Summary

| Backend | Ops | Notes |
|---------|-----|-------|
| **Triton** | FP8 GEMMs, MoE, RoPE, norms, topk | Bulk of compute |
| **TileLang 0.1.8** | HC pre/post, C4 indexer logits | Version-pinned — 0.1.10 crashes on SM120 |
| **HMMA custom** | Sparse decode attention | `deepseek-v4-flash-sm120/deepseek_v4_kernel/` |
| **DeepGEMM** | HC prenorm, wo_a einsum | SM100 TF32 paths, works on SM120 |
| **NCCL** | TP all-reduce (LL/Ring over PCIe) | ~13% of decode time |
| **FlashInfer** | Sampling only | AOT precompiled for SM120 |
| **Torch** | Embedding, topk_v2 fallback | Minimal |
