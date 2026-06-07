# DeepSeek-V4-Flash Native MXFP4×MXFP4 (W4A4-mx CuTe-DSL) Deployment Guide

**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, 256 experts, top_k=6 — **native MXFP4 checkpoint, used as-is**)
**Stack:** SGLang (local, branch `sm120-nvfp4-rebase`) + FlashInfer CuTe-DSL SM120 fused MoE + HMMA sparse decode kernel
**Repos:**
- SGLang: `/mnt/hot/ambientlight/repos/sglang` (branch: `sm120-nvfp4-rebase`)
- FlashInfer: `/mnt/hot/ambientlight/repos/flashinfer` (branch: `sm120-nvfp4-rebase`, symlinked into venv)

---

## Prerequisites

- Model checkpoint: `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash` (the **native MXFP4** checkpoint — E2M1 int8 expert weights + E8M0 block scales; NOT the FP8 repack)
- DeepSeek-V4 custom HMMA kernels: `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker`
- SGLang repo: branch `sm120-nvfp4-rebase` (carries `Mxfp4W4A4MoEMethod` + the `fp8.py` routing)
- FlashInfer repo: branch `sm120-nvfp4-rebase` (carries the `MmaMXF4Op` MoE kernels + E8M0 quantizers; symlinked into the venv, no rebuild)
- Python venv: `/mnt/hot/ambientlight/.venvs/sglang-cu130`
- **Required versions:** `tilelang==0.1.8` `apache-tvm-ffi==0.1.9` (0.1.10+ crashes SM120 CUDA graphs)

## How Native MXFP4×MXFP4 (W4A4-mx) Works

DeepSeek-V4-Flash ships its routed experts as **native MXFP4**: E2M1 4-bit weights (packed 2-per-byte)
+ E8M0 32-element block scales. This path loads those tensors **as-is** and runs the experts with
FlashInfer's fused-SwiGLU CuTe-DSL MoE kernels in **MXFP4 weights × MXFP4 activations** — the activations
are quantized at runtime to MXFP4 with E8M0 self-scaling. Result:

1. FP8 dense layers (attention projections, embeddings, LM head) — same as the FP8 path
2. FP8 KV cache — same as the FP8 path
3. **Native MXFP4 MoE experts** — W4A4 tensor-core kernels via FlashInfer CuTe-DSL SM120, via the
   `mma.kind::mxf4 .scale_vec::2X .ue8m0` instruction (`MmaMXF4Op`)
4. Same HMMA sparse decode + tilelang indexer as the FP8 path

**No re-quantization. No calibration.** Unlike the NVFP4 path (`MXFP4 → dequant BF16 → re-quantize NVFP4
+ one-shot activation calibration`), W4A4-mx feeds the checkpoint's E2M1 weights + E8M0 scales straight
into the tensor cores. E8M0 block scales are computed per-block at runtime from the activation `max_abs`
with no learned/measured parameter — so there are **no `input_gs` / `down_input_scale` activation global
scales to calibrate**. Weight prep at load time is just a gate/up reorder (`[w1,w3]→[w3,w1]`) + an
E8M0→MMA-layout swizzle (no dequant, ~instant).

This is the *same fused kernel family* the retired NVFP4 path used to hit 759 tok/s @16w — but on the
native checkpoint format.

## Server Launch

Launch script at `repos/sglang/debug/launch-mxfp4-w4a4-prod.sh`:

```bash
#!/bin/bash
source /mnt/hot/ambientlight/.venvs/sglang-cu130/bin/activate
export PYTHONPATH="/mnt/hot/ambientlight/repos/sglang/python:/mnt/hot/ambientlight/repos/deepseek-v4-flash-sm120/build-docker/deepseek_v4_kernel:/mnt/hot/ambientlight/repos/deepseek-v4-flash-sm120/build-docker"

# --- Native MXFP4 W4A4-mx fused MoE (takes precedence over SGLANG_MXFP4_W4A8) ---
export SGLANG_MXFP4_W4A4=1
export SGLANG_OPT_FP8_WO_A_GEMM=0

# --- Tilelang indexer + attention (CRITICAL for decode perf) ---
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1

# --- SM120 fallbacks ---
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0
export SGLANG_OPT_USE_TOPK_V2=0

# --- NCCL PCIe tuning ---
export NCCL_PROTO=LL
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=512

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Let the allocator expand segments to avoid fragmentation during graph capture.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGFILE="/mnt/hot/ambientlight/repos/sglang/debug/mxfp4-w4a4-prod.log"

exec python -m sglang.launch_server \
  --model-path /mnt/hot/ambientlight/models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --host 0.0.0.0 --port 8000 \
  --context-length 1048576 --mem-fraction-static 0.80 \
  --max-running-requests 16 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend triton \
  --chunked-prefill-size 16384 --page-size 256 \
  --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --dsa-topk-backend torch \
  --watchdog-timeout 3600 --log-level info \
  > "$LOGFILE" 2>&1
```

Run with:
```bash
nohup bash /mnt/hot/ambientlight/repos/sglang/debug/launch-mxfp4-w4a4-prod.sh &
```

Auto-detection: the native checkpoint reports `is_fp4_experts=True`, and `SGLANG_MXFP4_W4A4=1` + SM120
routes the MoE to `Mxfp4W4A4MoEMethod`. On startup every layer logs:
`SM120 W4A4-mx: native MXFP4 experts ready (E=256, H=4096, I=512, fused MXFP4xMXFP4, no re-quant, no calibration; layer: ...)`.

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SGLANG_MXFP4_W4A4=1` | Enabled | Route native-MXFP4 experts to the fused W4A4-mx method. Takes precedence over `SGLANG_MXFP4_W4A8`. |
| `--model-path .../DeepSeek-V4-Flash` | **native MXFP4** | The E2M1+E8M0 checkpoint, NOT the FP8 repack. `is_fp4_experts` auto-detected. |
| `--mem-fraction-static 0.80` | 0.80 | The fused W4A4 graph capture needs headroom. After the memory fixes capture costs ~4 GB; 0.80 leaves room for KV + long-context attention scratch. (Capture survives at 0.88, but 0.80 gives soak margin for 16-wide + long context.) |
| `SGLANG_MXFP4_STATIC_WS_CAP` | 640 (default) | Pre-allocate the static MoE workspace for 640 routed rows (≥ the static/dynamic cutover). Read by `Mxfp4W4A4MoEMethod.apply()`; keeps the decode workspace fixed-capacity and outside graph capture. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Set | Avoids allocator fragmentation during the multi-batch-size graph capture. |
| `--cuda-graph-bs 1 2 4 8 16` | Include bs=8,16 | **Critical**: captured graphs are what deliver the 453/742 tok/s at 8/16-wide; without them decode falls to eager. |
| `--moe-runner-backend triton` | Triton | FP8-compatible weight loading; the W4A4 `apply()` calls `launch_sm120_moe` directly. |
| `SGLANG_OPT_USE_TILELANG_INDEXER=1` | Enabled | **Critical**: shared C4 indexer infrastructure (same as FP8/NVFP4). |
| All other SM120 flags | Same as FP8/NVFP4 | HMMA sparse decode, topk torch fallback, etc. |

## Startup Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Weight loading (native MXFP4, no requant) | ~64s | 46 shards; weights load as-is, gate/up swap + E8M0→MMA swizzle per layer (no dequant) |
| CUDA graph capture (bs=1,2,4,8,16) | ~185s | The fused MoE compiles once and is **shared across all 5 batch sizes** (RT wrapper). avail-mem stays flat after the first capture. |
| Warmup + ready | ~30s | dynamic-kernel JIT for the prefill path (one-time) |
| **Total to "fired up"** | **~280s (~4.7 min)** | |

For comparison FP8 startup is ~75s; NVFP4 is ~210s. The W4A4-mx overhead is CuTe-DSL kernel JIT during
graph capture (compiled once, then cached). The native-weight load is slightly slower than FP8 (no
quantization saved) but adds **no** re-quant step.

## Performance (live server, mem-fraction 0.88, CUDA graphs on)

### Single-stream decode (steady-state ITL, streaming)

| Context | Steady ITL | tok/s |
|--------:|-----------:|------:|
| 256  | 12.3 ms | **81.0** |
| 1024 | 13.9 ms | **71.9** |
| 4096 | 16.0 ms | **62.4** |
| 8192 | 16.3 ms | **61.2** |
| 16384 | 17.2 ms | **57.9** |

### Concurrent streaming throughput (aggregate)

| Concurrency | W4A4-mx | NVFP4 (re-quant) | W4A8-mx | FP8 |
|------------:|--------:|-----------------:|--------:|----:|
| 1  | **76.9** | 82  | 52  | 67 |
| 4  | **253.1**| 259 | 163 | 170 |
| 8  | **452.8**| 445 | 302 | 260 |
| 16 | **742.5**| **759** | **515** | — |

**@16-wide = 742 tok/s — matches the NVFP4 re-quant path within 2%, and beats the shipped W4A8-mx
(515) by +44%** — on the native checkpoint, with no re-quant and no calibration. Server decode logs show
`cuda graph: True, gen throughput (token/s): ...` matching these.

### Numerical accuracy

| | vs BF16 (dequantized native weights) |
|--|--|
| W4A4-mx (FP4 activations) | **cos 0.975** |
| W4A8-mx (FP8 activations) | cos 0.999 |

cos 0.975 is the genuine **FP4-activation floor** — the cost of quantizing activations to 4-bit E8M0
blocks vs FP8. It is uniform across M (1→256), all backends, and skewed routing (768 rows/expert), with
no NaN. The weight numerics are lossless (native E2M1, no double-quantization).

## The Two Engineering Problems (and their fixes)

W4A4-mx required solving one correctness bug and one throughput/memory bug. Both are documented in
Quest 2; summarized here because they explain the launch config and the committed code.

### 1. Correctness — `num_m_tiles` dropped the 2nd 64-row M-quadrant

The fused MoE kernels hardcoded `num_m_tiles = tile_shape[0] // (16 * 4)`, copied from the **dense**
kernel whose `atom_layout=(4,2,1)` has 4 M-warps. The MoE kernels use `atom_layout=(2,2,1)` (2 M-warps),
so the GEMM filled only M-tiles {0,1} while the accumulator + epilogue (`MmaMPerEpiM=4`) expected
{0,1,2,3} — **rows 64–127 stayed at `fill(0.0)`** (exact-zero output, a sharp cos cliff at >64
rows/expert). Fix: derive from `atom_shape` like dense — `num_m_tiles = tile_m // (mma_m * atom_shape[0])`.
This was also a **latent upstream NVFP4 bug** (sf_vec_size-independent; only bit at >64 routed rows/expert
with a 128-wide tile, i.e. large skewed prefill).

### 2. Throughput/memory — 22 GB capture + 29 tok/s (invocation, not kernels)

The first live bring-up was correct but slow (29 tok/s @1w) and OOM'd at mem-fraction 0.88 (CUDA-graph
capture of just bs {1,2,4} cost 22 GB). The kernels were fine — the NVFP4 path runs the *same* kernels at
0.88 — so it was the *invocation*. Three W4A4-specific host-side fixes:

1. **`del` + `torch.cuda.empty_cache()`** in `process_weights_after_loading` — per-layer f32 scale upcasts
   + ~2 GB `torch.cat` weight copies ×43 layers were pinning ~8–10 GB of allocator high-water reserve.
2. **Pre-materialize `_get_weight_views` + a capped static `_workspace` at load** (passed via
   `_weight_views=`/`_workspace=` to `launch_sm120_moe`), instead of building them lazily inside graph
   capture (where ~2 GB landed in the graph-private pool).
3. **Parameterize the runtime-m (RT) static wrapper `_get_static_kernel_rt` for `sf_vec_size=32`** and
   drop the mxfp4 exclusion — so mxfp4 compiles **one shared module across all M** instead of a fresh
   per-M module per batch size.

**Result:** CUDA-graph capture **22 GB → 4.17 GB (5×)** (the avail-mem curve flattens after the first
batch size — proof the per-M compiles collapsed to one), and decode **29 → 77–81 tok/s @1w**. The 29 was
host-side per-M JIT + eager dispatch outside the captured graph, NOT a slower MXF4 kernel.

### How the captured decode graph actually runs

`_use_rt = not _is_cuda_graph_capturing()`. So the RT wrapper runs on **eager/prefill** paths (one shared
module); **inside a captured graph** `_use_rt=False` by design (a graph must replay fixed-shape launches).
Captured decode therefore uses fixed-shape per-M kernels: **micro** for bs {1,2,4} (routed ≤ 40), **per-M
static** for bs {8,16}. The 742 @16w number is with those captured per-M kernels; the RT wrapper's win is
on eager/prefill and on reducing the process-wide compiled-module count (which fed the memory fix).

## Code Changes

### FlashInfer (`repos/flashinfer`, branch `sm120-nvfp4-rebase`) — 5 commits, +1147/−427

| File | Change |
|------|--------|
| `cute_dsl/fp4_common.py` | +127: 8 E8M0/32-block activation quantizers (`quantize_block_mxf4`, `quantize_and_pack_32`→2×uint64, `max_abs_32`, `silu_mul_32`/`relu2_32` + fused). Reuse `cvt_f32_to_ue8m0`/`ue8m0_to_output_scale`. |
| `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` | +53: `MmaMXF4Op` atom branch, `can_implement` gate lift, SFA/SFB fragment rank-collapse for mma_nsf=2. |
| `gemm/gemm_base.py` | +15: `B12xFp4GemmRunner` sf_vec_size/sf_dtype from `use_nvfp4`. |
| `fused_moe/.../moe_static_kernel.py` | +315: atom swap, `tile_k=128` for MXF4, 32-block E8M0 Phase-1 + Phase-2 quant, **`num_m_tiles` fix**. |
| `fused_moe/.../moe_micro_kernel.py` | +335: same MXF4 datapath (3 sites) + `num_m_tiles` fix. |
| `fused_moe/.../moe_dynamic_kernel.py` | +648: same across 3 Phase-1 routing sub-paths + `num_m_tiles` fix. |
| `fused_moe/.../moe_dispatch.py` | +81: `quant_mode="mxfp4"` threaded through `_normalize_quant_mode`/`_sf_params_for_quant_mode`, all kernel-getter cache keys (+sf_vec_size), workspaces, `_get_weight_views`; **RT wrapper parameterized for sf_vec_size=32**; backend gates allow mxfp4 static/dynamic/micro. |

### SGLang (`repos/sglang`, branch `sm120-nvfp4-rebase`) — 2 commits, +398

| File | Change |
|------|--------|
| `layers/quantization/mxfp4_w4a4_moe.py` (new, +366) | `Mxfp4W4A4MoEMethod`. `create_weights` = native MXFP4 buffers. `process_weights_after_loading` = gate/up swap + E8M0→128×4 swizzle→`convert_sf_to_mma_layout` + pre-materialize `_get_weight_views` + del/empty_cache. `apply` = `launch_sm120_moe(quant_mode="mxfp4")` with precomputed `_weight_views=`/capped `_workspace=`. |
| `layers/quantization/fp8.py` (+32) | Route `SGLANG_MXFP4_W4A4=1` + `is_fp4_experts` + SM120 → `Mxfp4W4A4MoEMethod` (before the W4A8 branch). |
| `layers/attention/deepseek_v4_backend.py` | SM120 HMMA sparse decode routing (shared with FP8/NVFP4 — unchanged). |
| `entrypoints/warmup.py` | MoE warmup (shared — unchanged). |

All FlashInfer/SGLang edits are gated on `sf_vec_size==32` / `quant_mode=="mxfp4"` — **NVFP4 and the
shipped W4A8-mx paths are untouched.**

## When to Use W4A4-mx vs W4A8-mx vs NVFP4 vs FP8

| Workload | Recommended | Why |
|----------|-------------|-----|
| **Decode-heavy + want native format, no calibration** | **W4A4-mx** | NVFP4-class decode (742 @16w) on the native checkpoint, zero re-quant/calibration |
| **Decode-heavy + max activation accuracy** | **W4A8-mx** | FP8-grade activations (cos 0.999), FP8-class decode (515 @16w), zero calibration |
| **Already on the NVFP4 re-quant path** | **W4A4-mx** | Same throughput, strictly cleaner (no double-quant, no calibration sidecar) |
| **Minimal startup time** | **FP8** | ~75s vs ~280s |

## Head-to-Head Summary

| Metric | FP8 | W4A8-mx | W4A4-mx (this) | NVFP4 (re-quant) |
|--------|-----|---------|----------------|------------------|
| 1w decode | ~67 | 52 | **77** | 82 |
| 8w decode | ~260 | 302 | **453** | 445 |
| 16w decode | — | 515 | **742** | 759 |
| Weight numerics | FP8 | native MXFP4 | **native MXFP4** | double-quantized |
| Activation precision | FP8 per-token | MXFP8 (E8M0/32) | MXFP4 (E8M0/32) | NVFP4 (E4M3/16) |
| Re-quant at load | no | no | **no** | yes (FP8→BF16→NVFP4) |
| Calibration | no | no | **no** | yes (1-shot per-layer amax) |
| Startup | ~75s | ~210s | ~280s | ~210s |

**Bottom line:** W4A4-mx delivers **NVFP4-class decode (742 vs 759 @16w)** on the native MXFP4 checkpoint
with **zero re-quant and zero calibration** — it supersedes the NVFP4 re-quant path on cleanliness at the
same speed, and beats the shipped W4A8-mx by +44% at 16-wide. The cost is the FP4-activation floor (cos
0.975 vs W4A8's 0.999) and a few minutes of one-time graph-capture JIT.

## Monitoring

```bash
# Server logs
tail -f /mnt/hot/ambientlight/repos/sglang/debug/mxfp4-w4a4-prod.log

# Confirm all 43 layers loaded native MXFP4
grep -c "W4A4-mx: native MXFP4 experts ready" /mnt/hot/ambientlight/repos/sglang/debug/mxfp4-w4a4-prod.log   # expect 43

# CUDA-graph capture memory (should be ~4 GB, not 22)
grep "Capture cuda graph end" /mnt/hot/ambientlight/repos/sglang/debug/mxfp4-w4a4-prod.log

# Decode throughput + confirm graphs are replaying
grep "Decode batch" /mnt/hot/ambientlight/repos/sglang/debug/mxfp4-w4a4-prod.log | tail   # cuda graph: True

# GPU utilization
watch -n1 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## Appendix: E2E Kernel Pipeline (W4A4-mx Decode, 1 Token)

Blocks marked ★ differ from the FP8 path. Only the **MoE FFN expert GEMM** is W4A4-mx; everything else
(embedding, MQA attention + C4 indexer + HMMA sparse decode, dense FFN, LM head, sampling) is identical
to the FP8/NVFP4 paths.

```
TOKEN IN
    |
    v
+-- EMBEDDING ------------------------------------------------ [Torch] --+
|  VocabParallelEmbedding -> embed + repeat for HC                        |
+-------------------------------------------------------------------------+
    |
    v
+-- PER LAYER x61 -------------------------------------------------------+
|                                                                          |
|  +-- HC_PRE (attention) ----------------------------------------------+ |
|  |  deep_gemm tf32 prenorm GEMM              [DeepGEMM]               | |
|  |  mhc_pre fused Sinkhorn+RMSNorm           [TileLang 0.1.8]        | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- MQA ATTENTION ---------------------------------------------------+ |
|  |  Q proj: wq_a FP8 GEMM                    [Triton FP8]             | |
|  |  Q norm + RoPE fused                       [Triton]                 | |
|  |  Q proj: wq_b FP8 GEMM                    [Triton FP8]             | |
|  |  KV proj + cache write                     [Triton]                 | |
|  |                                                                      | |
|  |  C4 Indexer:                                                         | |
|  |    compressor gate GEMM                    [Triton]                 | |
|  |    fused_q_indexer_rope_hadamard_quant     [Triton]                 | |
|  |    fp8_paged_mqa_logits                    [TileLang 0.1.8]        | |
|  |    topk_transform_512                      [Triton]                 | |
|  |                                                                      | |
|  |  Sparse Decode:                                                      | |
|  |    sparse_decode_fwd                       [HMMA custom .so]        | |
|  |                                                                      | |
|  |  Output:                                                             | |
|  |    wo_a FP8 einsum                         [DeepGEMM]               | |
|  |    wo_b FP8 GEMM + AllReduce               [Triton + NCCL]         | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- HC_POST (attention) ---------------------------------------------+ |
|  |  mhc_post fused combine+residual          [TileLang 0.1.8]        | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- HC_PRE (FFN) ----------------------------------------------------+ |
|  |  (same as attention HC_PRE)          [DeepGEMM + TileLang]         | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- MoE FFN (43 MoE layers) ----------------------------------- ★ ---+ |
|  |  Router gate GEMM + topk                   [Triton]                 | |
|  |                                                                      | |
|  |  ★ Native MXFP4xMXFP4 Fused MoE (FlashInfer CuTe-DSL SM120):       | |
|  |     weights: E2M1 int8 + E8M0/32 block scales (loaded as-is)        | |
|  |     MmaMXF4Op  -> mma.kind::mxf4 .scale_vec::2X .ue8m0              | |
|  |                                                                      | |
|  |  +-- DECODE (captured CUDA graph replay) -----------------------+   | |
|  |  |  bs 1/2/4 (routed<=40) -> MICRO kernel   (per-M, fixed shape) |   | |
|  |  |  bs 8/16  (routed<=640) -> STATIC per-M  (fixed shape)        |   | |
|  |  |    Phase-1: quantize x -> MXFP4 (E8M0/32, runtime self-scale) |   | |
|  |  |    FC1 (w3|w1 gate/up) GEMM                                   |   | |
|  |  |    SiLU(gate)*up  +  Phase-2 requant -> MXFP4                 |   | |
|  |  |    FC2 (w2 down) GEMM                                         |   | |
|  |  |    (all fused, MmaMXF4Op W4A4 TC)            [CuTe-DSL JIT]   |   | |
|  |  +---------------------------------------------------------------+   | |
|  |  +-- EAGER / PREFILL (non-graph) -------------------------------+   | |
|  |  |  routed<=640 -> STATIC via _StaticMoELaunch RT wrapper        |   | |
|  |  |                 (ONE module, M-independent, sf_vec_size=32)    |   | |
|  |  |  routed >640 -> DYNAMIC W4A4 fused MoE (M-independent)        |   | |
|  |  +---------------------------------------------------------------+   | |
|  |                                                                      | |
|  |  TP AllReduce                              [NCCL LL/Ring]          | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- Dense FFN (3+15 layers) -----------------------------------------+ |
|  |  (Same as FP8: Triton FP8 fused MoE runner)                        | |
|  +--------------------------------------------------------------------+ |
|      |                                                                   |
|      v                                                                   |
|  +-- HC_POST (FFN) ---------------------------------------------------+ |
|  |  mhc_post fused combine+residual          [TileLang 0.1.8]        | |
|  +--------------------------------------------------------------------+ |
+-------------------------------------------------------------------------+
    |
    v
+-- LM HEAD --------------------------------------------------------------+
|  fused_hc_head (weighted sum + RMSNorm)       [Triton]                   |
|  lm_head FP8 GEMM -> logits                   [Triton FP8]              |
+-------------------------------------------------------------------------+
    |
    v
+-- SAMPLING --------------------------------------------------------------+
|  top_k_top_p_sampling_from_probs              [FlashInfer AOT]           |
+-------------------------------------------------------------------------+
    |
    v
TOKEN OUT
```

### Kernel Backend Summary

| Backend | Ops | Notes |
|---------|-----|-------|
| ★ **CuTe-DSL (JIT)** | **MoE W4A4-mx fused experts** | `MmaMXF4Op` (E8M0/32). Micro/static per-M in captured graphs; `_StaticMoELaunch` RT wrapper + dynamic on eager/prefill. |
| **Triton** | FP8 GEMMs (dense), RoPE, norms, topk, router gate | Bulk of non-MoE compute |
| **TileLang 0.1.8** | HC pre/post, C4 indexer logits | Version-pinned (0.1.10 crashes SM120) |
| **HMMA custom** | Sparse decode attention | `deepseek-v4-flash-sm120/deepseek_v4_kernel/` |
| **DeepGEMM** | HC prenorm, wo_a einsum | SM100 TF32 paths, work on SM120 |
| **NCCL** | TP all-reduce (LL/Ring over PCIe) | ~13% of decode time |
| **FlashInfer** | Sampling only | AOT precompiled for SM120 |
| **Torch** | Embedding, topk_v2 fallback | Minimal |

### ★ FP8 vs NVFP4 vs W4A4-mx: What Changed in the MoE Block

| Component | FP8 Path | NVFP4 Path (re-quant) | W4A4-mx Path (this, native) |
|-----------|----------|------------------------|------------------------------|
| Weight format | FP8 (e4m3) + block scales | NVFP4 (MXFP4→BF16→NVFP4 re-quant) | **native MXFP4** (E2M1 int8 + E8M0, as-shipped) |
| Weight numerics | FP8 | double-quantized (lossy) | **lossless** (no conversion) |
| MMA instruction | FP8 W8A8 TC | `mma.kind::mxf4nvf4 .scale_vec::4X .ue4m3` (`MmaMXF4NVF4Op`) | `mma.kind::mxf4 .scale_vec::2X .ue8m0` (`MmaMXF4Op`) |
| MoE dispatch | SGLang Triton fused MoE runner | FlashInfer CuTe-DSL (`Mxfp4MarlinMoEMethod`) | FlashInfer CuTe-DSL (`Mxfp4W4A4MoEMethod`, clean) |
| Activation quant | dynamic FP8 (per-token) | NVFP4 + **one-shot calibration** (per-layer amax) | MXFP4 E8M0/32, **self-scaling, no calibration** |
| Load-time prep | none | FP8 dequant → BF16 → `nvfp4_quantize` (~8s) | gate/up swap + E8M0→MMA swizzle (~instant) |
| Decode kernel | Triton FP8 (W8A8) | CuTe-DSL static W4A4 (RT wrapper) | CuTe-DSL static/micro W4A4 (RT wrapper) |
| Prefill kernel | Triton FP8 (W8A8) | CuTe-DSL dynamic W4A4 | CuTe-DSL dynamic W4A4 |
| Tensor cores | FP8 W8A8 | FP4 W4A4 | **FP4 W4A4** |
| Decode @16w | — | 759 tok/s | **742 tok/s** |
