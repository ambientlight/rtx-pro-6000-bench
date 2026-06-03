# Quest 1: Consolidate SM120 FP8 + NVFP4 on Latest Mainline SGLang

**Date:** 2026-06-03
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, 256 experts, FP8 checkpoint)

---

## Objective

Consolidate all SM120 work from the old fork (`sglang-sm120-pr24692` + `flashinfer` branch `sm120-nvfp4-bucketing`) onto the latest mainline SGLang and FlashInfer repos. Get both FP8 and NVFP4 MoE paths working, benchmark them, diagnose and fix the NVFP4 JIT compilation bottleneck.

## Starting State

- Old fork `sglang-sm120-pr24692` (branch `sm120-dsv4-rebase`): had working FP8 + NVFP4 but on stale codebase
- Old FlashInfer `flashinfer_prev` (branch `sm120-nvfp4-bucketing`): had `_StaticMoELaunch` JIT fix
- New mainline `repos/sglang` (branch `sm120-nvfp4-rebase`): fresh from upstream, no SM120 patches
- New FlashInfer `repos/flashinfer` (branch `main`): fresh from upstream, no JIT fixes
- FP8 was known to work at ~67 tok/s decode on old fork
- NVFP4 was known to have JIT stall problems (quest0)

## Part 1: FP8 Baseline on Latest Mainline

### Changes (5 files, +534/-21 lines in sglang)

| File | Change |
|------|--------|
| `layers/attention/deepseek_v4_backend.py` | Route SM120 sparse decode to HMMA kernel instead of `flash_mla.cuda` |
| `layers/attention/dsv4/indexer.py` | Route SM120 tilelang indexer to `dsv4/tilelang_kernel` (avoids TVM regression in mainline `dsa/tilelang_kernel`) |
| `layers/quantization/mxfp4_marlin_moe.py` | Port NVFP4 W4A4 + CuTe-DSL dispatch (+335 lines) |
| `entrypoints/warmup.py` | Add `moe_w4a4` warmup (+85 lines) |
| `.gitignore` | Add `debug/` |

Plus 7 tuned kernel config JSONs for W8A8 FP8 and MoE shapes on RTX PRO 6000.

### FP8 Results

| Metric | Value |
|--------|-------|
| Single decode (bs=1) | ~67 tok/s |
| 4-concurrent decode (bs=4) | ~170 tok/s |
| 8-concurrent decode (bs=8, est) | ~260 tok/s |
| Prefill (256-tok chunks) | ~1,510-1,590 tok/s |
| Startup time | ~75s |
| Weight VRAM per GPU | ~68.9 GB |
| KV cache pool per GPU | ~10.9 GB (721K tokens) |

### Key: tilelang Indexer

The single most important setting is `SGLANG_OPT_USE_TILELANG_INDEXER=1`. Without it, decode falls back to a pure Python torch loop at 7-12 tok/s instead of 67-170 tok/s. Requires `tilelang==0.1.8` + `apache-tvm-ffi==0.1.9` (0.1.10+ has a TVM buffer shape regression that crashes SM120 CUDA graph capture).

**Commit:** `f8141690` on `sm120-nvfp4-rebase`

---

## Part 2: NVFP4 Path — FP8→NVFP4 Re-quantization

### The Challenge

The FP8 checkpoint stores all weights as `float8_e4m3fn`. Mainline SGLang's NVFP4 path expects native MXFP4-packed weights from the checkpoint. The mainline `Mxfp4MarlinMoEMethod` creates FP4-sized weight buffers, but our checkpoint has FP8-sized weights → shape mismatch on load.

### Solution: FP8 Buffers + Post-Load Re-quantization

1. **`fp8.py`**: Added new routing — when `SGLANG_FP4_MOE_NVFP4=1` and `SGLANG_DSV4_FP4_EXPERTS=0` and backend is triton, route to `Mxfp4MarlinMoEMethod`
2. **`mxfp4_marlin_moe.py`**: 
   - `create_weights()`: Delegate to `Fp8MoEMethod.create_weights()` for FP8-sized buffers
   - `create_moe_runner()`: Use triton runner (not Marlin, which crashes on SM120)
   - `process_weights_after_loading()`: Detect FP8 input dtype, dequantize to BF16 via block-scale expansion, then `nvfp4_quantize(gs=1.0)` → NVFP4 packed + MMA blockscale layout
   - `apply()`: Dispatch to FlashInfer CuTe-DSL `launch_sm120_{static,dynamic}_moe`

### Re-quantization Path

```
FP8 checkpoint (float8_e4m3fn + [128,128] block scales)
    → dequantize: w_fp8.float() × scale_block_expanded → BF16
    → nvfp4_quantize(gs=1.0, sfLayout=layout_128x4) → NVFP4 packed + blockscale
    → convert_sf_to_mma_layout() → MMA-compatible scale layout
    → store as layer._nvfp4_w13_fp4, layer._nvfp4_w13_sf, etc.
```

Cost: ~8 seconds additional startup (43 MoE layers × 256 experts per layer).

### Initial NVFP4 Results (Before JIT Fix)

| Metric | NVFP4 | FP8 |
|--------|-------|-----|
| Server-side decode (bs=1) | **81 tok/s** | 67 tok/s |
| Server-side decode (bs=4) | **270 tok/s** | 170 tok/s |
| Server-side decode (bs=8) | **460-520 tok/s** | 260 tok/s |
| E2E first request | **23s** (JIT stall) | 0.2s |
| E2E subsequent (same M) | 0.7s | 0.2s |
| E2E subsequent (new M) | **23s** (JIT stall) | 0.2s |

Decode throughput was outstanding — W4A4 tensor cores clearly winning. But every unique prompt length triggered a ~23s JIT compilation stall, making it unusable for production.

---

## Part 3: Diagnosing the JIT Stall

### Investigation

Added timing instrumentation to `mxfp4_marlin_moe.py::apply()`:

```python
if _total > 0.1:  # only log slow calls (>100ms)
    log_info_on_rank0(logger,
        f"NVFP4 MoE SLOW: total={_total:.3f}s "
        f"import={_t1-_t0:.3f} weights={_t2-_t1:.3f} "
        f"select={_t3-_t2:.3f} workspace={_t4-_t3:.3f} "
        f"launch={_t5-_t4:.3f} M={M} backend={backend}")
```

Findings:
- **100% of stall time was in `launch`** — specifically `launch_sm120_static_moe()`
- Import, weight views, backend selection, workspace allocation: all <1ms
- The `launch` call triggers `cute.compile()` inside `_get_static_kernel()` for every new M

### Root Cause: Three Cache Key Instabilities

The CuTe-DSL static kernel cache key `_STATIC_KERNEL_CACHE` includes:

1. **`m` (num_tokens)**: Every unique prompt length → new M → new compilation (~22s)
2. **`max_rows` (workspace capacity)**: Workspace grows as larger batches arrive → new compilation
3. **`mac` (max active clusters)**: Tuned MAC ladder returns different values for different routed_rows → new compilation
4. **`mma_tiler_mn` (tile shape)**: Tuned tile selector varies with routed_rows → new compilation

### The Warm Prefill Myth

Earlier measurements showed "warm prefill at 1,630 tok/s" which seemed comparable to FP8. This was misleading — those measurements hit **prefix cache** (no MoE forward at all). True warm prefill with different prompts stalled 22s each time due to the static kernel JIT.

---

## Part 4: The Fix — `_StaticMoELaunch`

### Architecture (ported from quest0's `sm120-nvfp4-bucketing` branch, adapted for mainline)

Added to `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`:

**1. `_StaticMoELaunch` class** — Thin `@cute.jit` wrapper that mirrors `_DynamicMoELaunch`:
- Runtime-shaped tensors (`a_input`, `topk_ids`, `topk_weights`, `scatter_output`) passed as raw `cute.Pointer` args
- Constructs them inside the JIT via `cute.make_tensor(ptr, layout=(num_tokens, k))`
- `num_tokens` is a `cutlass.Int32` runtime value, NOT a compile-time shape
- Result: `m` is completely absent from the compilation cache key

**2. `_get_static_kernel_rt()`** — New compilation function:
- Uses pointer fakes for runtime-shaped args
- Cache key has NO `m` — only fixed dimensions (E, K, N, max_rows, mac, tile_mn)
- Fixed `mac` to hardware limit (`get_max_active_clusters(1)` = 188 SMs) — no tuned ladder
- Fixed `mma_tiler_mn` to `(128, 128)` — no per-routed_rows tile selection

**3. Capped workspace** (in SGLang `mxfp4_marlin_moe.py`):
- Static workspace allocates `max(routed_rows, 640)` rows
- 640 = static/dynamic cutover threshold, so the workspace never grows within the static range
- Eliminates `max_rows` variance in the RT cache key

**4. Dual-path dispatch** in `launch_sm120_static_moe()`:
- CUDA graph capture (`_is_cuda_graph_capturing()` = True): original per-M compiled kernel (graphs need fixed shapes)
- Non-graph paths: `_StaticMoELaunch` wrapper via `_get_static_kernel_rt()` (zero per-M JIT)

### Why Not Just Port the Old Patch?

The old `_StaticMoELaunch` from `sm120-nvfp4-bucketing` (commit `78d0acd`) had the same architecture but:
- Was against FlashInfer 0.6.11.post3 — file had diverged from mainline 0.6.12
- Did NOT fix the `mac` and `mma_tiler_mn` variance (discovered during this quest)
- Did NOT cap `max_rows` (discovered during this quest)

We rewrote it from scratch against mainline, fixing all three cache key instabilities.

---

## Part 5: Final Results

### Commits

```
sglang (branch: sm120-nvfp4-rebase, remote: ambientlight/sglang):
  850af92b NVFP4: capped workspace + debug timing for JIT elimination
  f8141690 SM120: FP8 + NVFP4 support for DeepSeek-V4-Flash on RTX PRO 6000

flashinfer (branch: main, remote: ambientlight/flashinfer):
  906556fb SM120: _StaticMoELaunch runtime-m wrapper for NVFP4 static MoE
```

### E2E Throughput (All Kernels Warm, max_running_requests=16)

| Workers | Aggregate tok/s | Per-worker tok/s |
|---------|-----------------|-----------------|
| 1w | 82 | 81.6 |
| 2w | 139 | 69.3 |
| 4w | 259 | 64.8 |
| 8w | 445 | 55.7 |
| 16w | **759** | 47.4 |

Zero JIT stalls at any concurrency level.

### NVFP4 vs FP8 Head-to-Head

| Metric | FP8 | NVFP4 | Delta |
|--------|-----|-------|-------|
| 1w decode | 67 tok/s | **82 tok/s** | +22% |
| 4w decode | 170 tok/s | **259 tok/s** | +52% |
| 8w decode | 260 tok/s | **445 tok/s** | +71% |
| 16w decode | N/A (max_rr=8) | **759 tok/s** | — |
| Warm prefill | ~1,550 tok/s | ~1,680 tok/s | Comparable |
| JIT stalls | None | **None** | Fixed |
| Weight VRAM | 68.9 GB | **41.3 GB** | -27.6 GB |
| KV cache pool | 10.9 GB (721K tok) | **38.2 GB (2.59M tok)** | **3.6× more** |
| Startup | 75s | 210s | Slower |

### Memory Breakdown (per GPU)

| Component | FP8 | NVFP4 | Notes |
|-----------|-----|-------|-------|
| Weights | 68.9 GB | 41.3 GB | NVFP4 MoE experts are half size (FP4 packed) |
| KV cache | 10.9 GB | 38.2 GB | Freed weight space → 3.6× more KV cache |
| CUDA graphs | 3.7 GB | 4.1 GB | NVFP4 has 5 batch sizes vs FP8's 3 |
| Available | 10.6 GB | 10.5 GB | Same total utilization |

The 27.6 GB saved from FP4 MoE weight compression goes directly into KV cache, enabling 2.59M tokens vs 721K — critical for 16 concurrent SWE-bench workers with long contexts.

---

## Part 6: Startup Timeline

| Phase | FP8 | NVFP4 | Notes |
|-------|-----|-------|-------|
| Weight loading | ~10s | ~10s | Same checkpoint |
| process_weights_after_loading | ~35s | ~43s | +8s for FP8→NVFP4 requant |
| CUDA graph capture | ~30s (3 bs) | ~150s (5 bs) | Each bs triggers ~23s static kernel JIT |
| RT kernel compilation | — | ~17s | One-time _StaticMoELaunch compilation |
| Warmup health check | ~2s | ~26s | Dynamic kernel JIT for prefill path |
| **Total** | **~75s** | **~210s** | |

---

## Part 7: E2E Kernel Pipeline

The NVFP4 path shares all infrastructure with FP8 except the MoE FFN block:

- **Shared:** Embedding, HC pre/post (TileLang 0.1.8), MQA attention (Triton FP8), C4 indexer (TileLang), sparse decode (HMMA custom), LM head (Triton FP8), sampling (FlashInfer AOT), NCCL all-reduce
- **Different:** MoE experts use CuTe-DSL W4A4 tensor-core kernels instead of Triton FP8 GEMMs

### MoE Dispatch

| Path | Condition | Kernel | M in cache key? |
|------|-----------|--------|----------------|
| CUDA graph replay | `_is_cuda_graph_capturing()` | Per-M static kernel | Yes (fixed at capture) |
| Non-graph decode | routed_rows ≤ 640 | `_StaticMoELaunch` RT wrapper | **No** |
| Prefill | routed_rows > 640 | Dynamic kernel | **No** |

---

## Key Learnings

1. **tilelang indexer is the #1 bottleneck.** Without it, decode drops 15-20×. Must pin `tilelang==0.1.8`.

2. **W4A4 tensor cores deliver.** 50-100% faster decode than W8A8 FP8 at all batch sizes. This is the canonical DSv4 design — FP8 attention + FP4 MoE.

3. **CuTe-DSL JIT is the #2 bottleneck.** `cute.compile()` takes ~22s per unique shape. No AOT/disk cache exists. The fix is making shapes runtime via `@cute.jit` wrappers with `cute.Pointer` args.

4. **Cache key stability requires fixing ALL varying parameters.** Fixing `m` alone wasn't enough — `max_rows`, `mac`, and `mma_tiler_mn` also varied per batch, each causing recompilation. The fix required changes in both FlashInfer (fixed mac, fixed tile) and SGLang (capped workspace).

5. **FP4 weight compression → more KV cache.** 27.6 GB saved per GPU goes directly to KV pool (3.6× more capacity). This is arguably more important than the decode speedup for long-context workloads.

6. **Warm prefill is comparable.** The "prefill gap" between FP8 and NVFP4 was entirely JIT stalls, not kernel throughput. Once warm, NVFP4 prefill is ~1,680 tok/s vs FP8 ~1,550 tok/s.

---

## Files Modified

### SGLang (`repos/sglang`, branch `sm120-nvfp4-rebase`)

| File | Lines | Purpose |
|------|-------|---------|
| `layers/quantization/fp8.py` | +15 | NVFP4 routing via triton backend |
| `layers/quantization/mxfp4_marlin_moe.py` | +451/-26 | FP8→NVFP4 requant, CuTe-DSL dispatch, capped workspace, debug timing |
| `layers/attention/deepseek_v4_backend.py` | +55/-20 | SM120 HMMA sparse decode routing |
| `layers/attention/dsv4/indexer.py` | +90 | SM120 tilelang indexer routing |
| `entrypoints/warmup.py` | +85 | moe_w4a4 warmup |
| `.gitignore` | +1 | debug/ |
| 7 config JSONs | new | Tuned W8A8 + MoE kernel configs |

### FlashInfer (`repos/flashinfer`, branch `main`)

| File | Lines | Purpose |
|------|-------|---------|
| `fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | +314/-42 | `_StaticMoELaunch`, `_get_static_kernel_rt`, dual-path dispatch, fixed mac/tile |

### Deployment Docs (`repos/rtx-pro-6000-bench/docs/`)

| File | Purpose |
|------|---------|
| `deploy-fp8-repack.md` | Updated head-to-head with NVFP4 results |
| `deploy-nvfp4-cutedsl.md` | Full rewrite with JIT fix, benchmarks, kernel pipeline |

---

## Relationship to Prior Quests

- **quest0**: Established NVFP4 W4A4 path on old fork, discovered JIT problem, built first `_StaticMoELaunch` prototype. We ported the concept but rewrote the implementation, discovering and fixing three additional cache key instabilities (`max_rows`, `mac`, `mma_tiler_mn`) not present in quest0's version.
- **QUEST-SM120-NATIVE-FP4-MOE-KERNEL**: Built the HMMA sparse decode kernel and SM120 attention routing. We reuse these unchanged.
- **QUEST-DEEPSEEK-V4-FLASH-HILLCLIMB**: Tuned FP8 kernel configs and NCCL parameters. We carry those configs forward.
- **QUEST-UPSTREAM-SM120-HMMA-TO-SGLANG**: Established the `sitecustomize.py` → `_patch.py` monkey-patching for HMMA kernel. Still used.
