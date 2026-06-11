# Quest 0: FP4 Tensor-Core MoE on Blackwell SM120 — Full Journey

**Date:** 2026-05-31 → 2026-06-01  
**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)  
**Model:** DeepSeek-V4-Flash (291B MoE, 256 routed experts, 43 MoE layers, MXFP4 checkpoint)  
**Stack:** SGLang + FlashInfer CuTe-DSL SM120 MoE kernels  
**Target:** Maximize throughput on 8-concurrent agentic (SWE-bench) workloads using FP4 tensor cores

---

## Part 1: W4A16 — Closed-Form MXFP4→NVFP4 Conversion

### The Problem
DeepSeek-V4-Flash ships with MXFP4-quantized expert weights. On Blackwell SM120, Marlin kernel produces NaN. Triton GEMV fallback works but is slow. FlashInfer has a W4A16 CuTe-DSL kernel that dequants FP4→BF16 in SMEM and uses BF16 MMA tensor cores.

### Key Insight: Nibbles Are Identical
NVIDIA's `cast_mxfp4_to_nvfp4.py` (Model-Optimizer) revealed MXFP4 and NVFP4 use the **same E2M1 nibble encoding**. The conversion is pure scale remapping:
- MXFP4: `value = nibble * 2^k_j` (one E8M0 per 32 elements)
- NVFP4: `value = nibble * sf_e4m3 * ws2` (one E4M3 per 16 + global)

Choose `ws2 = 2^m` where `m = k_max - 8`, then `sf_e4m3 = 2^(k_j - m)` — bit-exact reconstruction.

### 8 Iterations to Correct Output

| Version | Approach | Result |
|---------|----------|--------|
| v1 | `nvfp4_quantize(gs=amax/448)` | 100% nibble saturation, null output |
| v2 | `nvfp4_quantize(gs=1.0)` | Unsaturated but null output — requant noise |
| v3 | Offline NVFP4 checkpoint conversion | Same requant quality issue |
| v4 | Closed-form scales, `global_sf=2^(m-119)` | Overflow to inf |
| v5 | Closed-form, `global_sf=2^m`, swizzled | `prepare_w4a16_packed_weights` succeeds, short answers OK, long outputs degenerate |
| v6 | Fallback to `nvfp4_quantize(gs=1.0)` | Same degradation pattern |
| v7 | Report-guided, batched conversion | Validation: 100% bit-exact MXFP4==NVFP4 dequant, but same long output degradation |
| **v8** | **Gate/up reorder fix** | **Perfect output quality** |

### The Gate/Up Ordering Bug
`prepare_w4a16_packed_weights` calls `reorder_w13_to_gate_up` which swaps halves. It expects input `[up, gate]` and reorders TO `[gate, up]` for the kernel. Our checkpoint had `[w1=gate, w3=up]` — feeding this directly gave `[up, gate]` after reorder, making the kernel compute `silu(up) * gate` instead of `silu(gate) * up`.

**Fix:** Swap W13 from `[w1, w3]` to `[w3, w1]` before passing to `prepare_w4a16_packed_weights`.

### W4A16 Performance

| Metric | GEMV Fallback | W4A16 Tensor-Core |
|--------|--------------|-------------------|
| Single decode | ~15 tok/s | **66.5 tok/s** |
| 8-concurrent aggregate | ~150 tok/s | **240 tok/s** |
| Startup time | ~60s | ~170s (CPU scale conversion) |
| Output quality | ✅ | ✅ (perfect) |

---

## Part 2: SWE-bench Reality — Prefill Is the Bottleneck

### The Discovery
Synthetic benchmarks (8 concurrent, 512-token completions) showed 240 tok/s. Real SWE-bench workloads showed catastrophic regression: ~80K tokens in 10 minutes vs FP8's 700K.

### Root Cause: Prefill Starvation
Server logs during SWE-bench:
- **72 prefill batches vs 26 decode batches** — 73% of time in prefill
- Decode median dropped to 11 tok/s
- W4A16 prefill throughput: **40-60 tok/s** for fresh chunks

The W4A16 path uses BF16 activations — prefill is **memory-bandwidth bound**, not compute bound. Each 1-3K token prefill takes 20-55 seconds.

---

## Part 3: W4A4 (NVFP4) — FP4×FP4 Tensor Cores

### Goal
Use FP4×FP4 native Blackwell tensor cores for both prefill AND decode, achieving higher prefill throughput.

### Attempt 1: CuteDSL v2 Wrapper
ModelOpt's NVFP4 path uses `CuteDslMoEWrapper` which pre-quantizes activations outside the kernel. Failed with `No supported CUDA architectures found for major versions [10]` — SM120 not supported by the v2 wrapper.

### Attempt 2: Direct Static/Dynamic Kernel
Bypassed the dispatcher, called `launch_sm120_static_moe` directly with separate `input_gs` (FC1 activation scale) and `weights.w1_alpha` (GEMM alpha). This avoids the `input_gs=w1_alpha` collision in the default dispatcher.

### Activation Calibration Challenge
`fc2_input_scale` (FC2 intermediate activation scale) cannot be computed from weight statistics alone. The FC2 input is `SiLU(gate) * up` which depends on runtime token distributions.

**Measured**: FC2 intermediate amax ≈ 3× FC1 input amax (for random input). Used `a2_raw = 5 * a1_raw` as heuristic.

### Gate/Up Ordering (Again!)
The W4A4 static kernel expects W13 as `[up, gate]` — opposite from what we initially fed. Research report confirmed: *"W13 is packed as [up, gate] across the concatenated N dimension."*

**Fix:** Swap W13 to `[w3=up, w1=gate]` before `nvfp4_quantize`.

### W4A4 Output Quality
After gate/up fix: correct output (Fibonacci, TCP/UDP, math). Slightly degraded vs W4A16 due to FP4 activation quantization noise — acceptable for 4-bit model.

---

## Part 4: The JIT Compilation Disaster

### Discovery
W4A4 requests took **22-53 seconds** for simple prompts. GPUs at **0% utilization** during processing. The CuTe-DSL NVFP4 static/micro kernels JIT-compile on first use, with `m` (num_tokens) in the cache key.

**Every unique `m` value triggers a ~25-second kernel compilation.**

### Why W4A16 Doesn't Have This Problem
W4A16 uses `_W4A16_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)` — only **5 bucketed sizes**. All compiled during CUDA graph capture. NVFP4 static/micro have NO bucketing.

### The Dynamic Kernel Solution
Research report revealed: **`_get_dynamic_kernel()` does NOT include `m` in its cache key** — already shape-agnostic.

```bash
export FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0
```

Forces all NVFP4 traffic to the dynamic kernel. Results:
- Single cold request: **0.5s (32 tok/s)** — vs 22.8s before
- 8 concurrent: **1.0s (123 tok/s)** — vs 53s before
- **Zero cold-start penalty**

### Dynamic Kernel Throughput Problem
The dynamic kernel is shape-agnostic but **significantly slower**:
- Decode: ~50 tok/s aggregate at 8 concurrent (vs 240 tok/s W4A16)
- GPU utilization: 16-50% (vs 99% W4A16 decode)

The shape-agnostic pointer-based dispatch has higher overhead than the specialized static kernel.

### Persistent Cache: Empty
`CUTE_DSL_CACHE_DIR` was set but remains empty. The research report confirmed: `cute.compile()` (used by FlashInfer) **bypasses CuTe DSL's implicit file cache**.

---

## Part 5: Current Performance Comparison

### Throughput on SWE-bench 8-worker Workload

| Path | Decode tok/s | Prefill tok/s | SWE-bench throughput | GPU utilization |
|------|-------------|---------------|---------------------|-----------------|
| **FP8 repack** (baseline) | ~67 | ~200+ | ~700K/10min | High, sustained |
| **FP4 GEMV v3** | ~64 | ~100 | ~500K/10min | Medium |
| **W4A16** | ~240 (8-conc) | **40-60** | Poor — prefill-bound | Spiky, lots of idle |
| **W4A4 dynamic** | ~50 (8-conc) | **120-190** | ~765K/13min | 16-50%, spiky |
| **W4A4 static** | ~271 (8-conc) | ~200+ (est.) | Not tested (workspace OOM) | Should be high |
| **W4A4 static+rt wrapper** | ~271 (8-conc, graph) | dynamic fallback | Decode: static, Prefill: dynamic | Stable, no JIT |

### Key Insight
- **W4A16**: Great decode, terrible prefill → poor SWE-bench throughput
- **W4A4 dynamic**: No JIT penalty, but slow per-token → moderate SWE-bench throughput
- **W4A4 static with bucketing**: Should give best of both (great decode + great prefill) → needs implementation

---

## Part 6: The Path Forward — Static Kernel Bucketing

### The Solution
W4A16 avoids JIT by bucketing into 5 fixed routed sizes. Apply the same approach to NVFP4 static kernel:

```python
_NVFP4_STATIC_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

def _bucket_m(m: int) -> int:
    for b in _NVFP4_STATIC_BUCKETS:
        if m <= b:
            return b
    return ((m + 1023) // 1024) * 1024
```

This requires:
1. Change `_get_static_kernel` to use `compile_m = _bucket_m(runtime_m)` for cache key and fake tensors
2. Pass `runtime_m` to the compiled kernel for bounds checking
3. Pad runtime tensors or add predicates in the kernel

### Alternative: AOT Compilation
CuTe DSL supports `export_to_c()` and `cute.runtime.load_module()`. After bucketing reduces variants to ~13, export those as AOT artifacts for instant loading.

### Estimated Impact
With bucketed static kernel:
- **13 compilations at startup** (~5 min one-time cost, can be parallelized)
- **Zero runtime JIT** for any request
- **220+ tok/s decode** (matching static kernel performance)
- **200+ tok/s prefill** (FP4×FP4 compute-bound)
- **Sustained high GPU utilization** (80%+ like FP8)

---

## Files Modified

| File | Changes |
|------|---------|
| `mxfp4_marlin_moe.py` | W4A16 closed-form conversion, W4A4 nvfp4_quantize path, gate/up reorder, calibration, direct kernel dispatch |
| `warmup.py` | `moe_w4a4` warmup function (phases 1-3) |

## Environment Variables Discovered

| Variable | Purpose |
|----------|---------|
| `SGLANG_FP4_MOE_W4A16=1` | Enable W4A16 tensor-core MoE |
| `SGLANG_FP4_MOE_NVFP4=1` | Enable W4A4 tensor-core MoE |
| `FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0` | Force NVFP4 dynamic kernel (shape-agnostic) |
| `CUTE_DSL_CACHE_DIR=<path>` | Persistent CuTe DSL cache (doesn't work with explicit `cute.compile`) |
| `FLASHINFER_B12X_FORCE_MOE_W4A16=1` | Force W4A16 even for NVFP4 quant mode |

## Deep Research Reports

1. `mxfp4_to_nvfp4_flashinfer_w4a16_report.md` — W4A16 scale chain, kernel dequant formula, correct conversion strategy
2. `flashinfer_sm120_nvfp4_fc2_input_scale_research.md` — FC2 activation scale calibration, alpha conventions, scale chain
3. `flashinfer_sm120_nvfp4_moe_research.md` — W4A4 gate/up ordering, block scale convention, standalone test
4. `drs/flashinfer_sm120_nvfp4_moe_jit_research.md` — JIT compilation root cause, dynamic kernel, bucketing, AOT
5. `drs/flashinfer_sm120_nvfp4_static_moe_shape_agnostic_research.md` — Static kernel architecture analysis, _StaticMoELaunch wrapper design, runtime-m implementation plan

---

## Part 7: _StaticMoELaunch — Runtime-Shaped Static Kernel

### The Problem
The NVFP4 static kernel includes `m` (num_tokens) in its CuTe DSL cache key. Every unique `m` triggers ~25s JIT compilation. The dynamic kernel avoids this but is 5× slower for decode.

### Key Insight: m Is Not a Compile-Time Requirement
Deep research revealed that `m` is only baked into the kernel through host-side fake tensor shapes in `_get_static_kernel()`. The kernel itself derives `num_tokens = a_input.shape[0]` at runtime. The dynamic kernel already proves the fix: pass runtime-shaped tensors as **raw pointers** and construct them with `cute.make_tensor()` inside a `@cute.jit` wrapper.

### Implementation: `_StaticMoELaunch` Wrapper
Added to `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`:

1. **`_StaticMoELaunch` class**: Thin `@cute.jit` wrapper that takes `a_ptr`, `topk_ids_ptr`, `topk_weights_ptr`, `scatter_ptr` as `cute.Pointer` + `num_tokens` as `cutlass.Int32`. Constructs runtime-shaped tensors via `cute.make_tensor(ptr, layout=(num_tokens, k))`, then calls `MoEStaticKernel`.

2. **`_get_static_kernel_rt()`**: New compilation function that uses pointer fakes instead of shaped fakes. Cache key has NO `m`. Compiles `_StaticMoELaunch(kernel)` instead of `kernel` directly.

3. **Dual-path dispatch in `launch_sm120_static_moe()`**:
   - CUDA graph capture: original per-`m` compiled kernel (`_get_static_kernel`)
   - Non-graph paths: `_StaticMoELaunch` wrapper (`_get_static_kernel_rt`)

### Results
- **Decode**: 271 tok/s at 8-concurrent (CUDA graph replay, unchanged)
- **Wrapper compiled successfully** for non-graph paths
- **Zero per-`m` JIT** for the wrapper path

### Remaining Limitation: Static Workspace Memory
The static workspace allocates `packed_input[E, max_rows, K//2]`. For prefill with `routed_rows=65536` (8K chunk × top-8), this requires ~32 GB — doesn't fit. Current cutover keeps prefill on dynamic kernel (120-190 tok/s) and decode on static (271 tok/s via CUDA graphs).

### Path Forward: Block-Packed Static Kernel (Track B)
To use static for prefill, need a more compact workspace layout (Track B from research). The `_StaticMoELaunch` wrapper is ready for this — once workspace is redesigned, prefill can use static at ~200+ tok/s with zero JIT.

## Commits

```
flashinfer (branch: sm120-nvfp4-bucketing):
  78d0acd feat: _StaticMoELaunch runtime-m wrapper for NVFP4 static MoE kernel

sglang-sm120-pr24692 (branch: sm120-dsv4-rebase):
  3f4f3f4 feat: NVFP4 W4A4 MoE with warmup for SM120 Blackwell
  e9cb8f1 feat: MXFP4→NVFP4 closed-form W4A16 tensor-core MoE for SM120

rtx-pro-6000-bench (branch: main):
  871a435 docs: MXFP4→NVFP4 W4A16 quest log, research report, and conversion script
  3e8190b docs: add SWE-bench reality check to W4A16 quest log
```
