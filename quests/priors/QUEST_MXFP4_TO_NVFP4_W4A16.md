# Quest: MXFP4 → NVFP4 W4A16 Tensor-Core MoE on Blackwell

**Date:** 2026-05-31 → 2026-06-01  
**Hardware:** 4× NVIDIA RTX PRO 6000 (Blackwell B200 / SM120)  
**Model:** DeepSeek-V4-Flash (MoE, 256 routed experts, MXFP4 quantized)  
**Stack:** SGLang + FlashInfer CuTe-DSL B12x W4A16 kernel  
**Outcome:** Correct, high-quality inference at 38 tok/s decode with tensor-core MoE — up from scalar GEMV fallback

---

## The Problem

DeepSeek-V4-Flash ships with MXFP4-quantized expert weights (packed E2M1 nibbles + E8M0 per-32-element block scales). On Blackwell SM120, the existing Marlin kernel produces NaN, so SGLang falls back to a scalar Triton GEMV dequant path — functional but slow and unable to use tensor cores.

FlashInfer ships a **W4A16 CuTe-DSL kernel** for SM120/SM121 that runs MoE through Blackwell's tensor cores using NVFP4 format. But NVFP4 ≠ MXFP4 — different scale encoding, different block granularity (16 vs 32), different layout expectations. Nobody had done the conversion for this checkpoint before.

**Goal:** Convert MXFP4 weights to NVFP4 at load time and route through FlashInfer's `prepare_w4a16_packed_weights` → `launch_sm120_moe` pipeline.

---

## The Journey

### Phase 1: First Contact — nvfp4_quantize Attempt

**Approach:** Dequant MXFP4 to float, then requantize via FlashInfer's `nvfp4_quantize(global_sf, sfLayout=layout_128x4)`.

**Problem 1 — What is `global_sf`?**  
We tried `global_sf = amax / 448.0` (the FP8 E4M3 max). This produced 100% saturated nibbles — every value clamped to `0x77` (max positive). The `global_sf` divides input values before block quantization, and `amax/448 ≈ 0.0004` turned tiny weight values (0.01–0.18 range) into huge scaled values that overflowed FP4.

We tried `global_sf = 1.0`. This produced unsaturated nibbles (16 unique values, 11% saturation) and `prepare_w4a16_packed_weights` accepted the tensors. But the nibbles were **different** from the original MXFP4 nibbles — `nvfp4_quantize` computes its own block-level quantization with 16-element blocks vs MXFP4's 32-element blocks.

**Result:** Server launched, generated tokens at 35-38 tok/s, but output was **garbage** — `content: null` on all responses. The requantization noise was too severe.

### Phase 2: The NVIDIA ModelOpt Discovery

We cloned NVIDIA's Model-Optimizer repo and found `examples/llm_ptq/cast_mxfp4_to_nvfp4.py` — a closed-form conversion script that revealed the critical insight:

> **MXFP4 and NVFP4 use the same E2M1 nibble encoding. The data bytes are identical. Only the scale representation changes.**

The conversion is pure scale remapping:
- MXFP4: `value = nibble * 2^k_j` (one E8M0 per 32 elements)
- NVFP4: `value = nibble * sf_e4m3 * weight_scale_2` (one E4M3 per 16 + global)

Choose `m = k_max - 8`, then:
- `weight_scale_2 = 2^m` (per-expert global scale)
- `sf_e4m3 = 2^(k_j - m)` (per-block, exactly representable in E4M3)
- Each MXFP4 block of 32 → two NVFP4 blocks of 16 with the same scale

This gives **bit-exact reconstruction** — zero numerical error.

### Phase 3: The Scale Format Maze

**Problem 2 — What format does `prepare_w4a16_packed_weights` expect for block scales?**

We tried passing flat `[N, K/16]` E4M3 bytes. Failed — `prepare_w4a16_packed_weights` calls `unswizzle_expert_scales` which expects a specific **swizzled** layout matching `nvfp4_quantize`'s `SfLayout.layout_128x4` output.

The swizzle is a 5D tile permutation:
```
[rows_padded//128, cols_padded//4, 32, 4, 4] → permute(0,3,2,1,4) → [rows_padded, cols_padded]
```

We implemented the inverse (swizzle) and verified round-trip identity with `unswizzle_block_scale`.

**Problem 3 — Float32 vs E8M0 scales in SGLang**

Our `mxfp4_to_nvfp4_weights` assumed E8M0 input but SGLang's weight loader promotes E8M0 scales to float32 (`2^k_j` values). Doing `.view(torch.uint8)` on float32 gave 4× the expected bytes, producing shape mismatches (`[1024, 1024]` instead of `[1024, 256]`).

Fixed by detecting `dtype == torch.float32` and recovering `k_j = round(log2(scale))`.

### Phase 4: The 4/3 Ratio Red Herring

After fixing the scale format, the server produced output — but **degraded on longer generations**. Short answers ("4", "Paris") worked; code/reasoning degenerated into repetition loops.

We compared our analytical block scales against `nvfp4_quantize`'s output and found a consistent **4/3 ratio**. Panicked, thinking the kernel used a different E2M1 interpretation than the standard LUT.

**This was a red herring.** The 4/3 ratio came from comparing two different things:
- Closed-form: `block_scale = 2^k_j` (full E2M1 range, preserving original nibbles)
- `nvfp4_quantize`: `block_scale = block_max / 6.0` (data-derived, different for each 16-element sub-block)

These are different by design. The closed-form is correct for preserving nibbles.

### Phase 5: The Deep Research Report

We commissioned a detailed research report analyzing the entire FlashInfer W4A16 kernel pipeline. Key findings:

1. **The kernel's BF16 subnormal trick cancels out.** The E2M1 helper produces `standard_E2M1 * 2^-126`, block scales get `* 2^7`, global gets `* 2^119`. Product: `E2M1 * block_scale * global_scale` — standard dequant. Our scales were correct.

2. **The `combined_scale_factor` is product-preserving.** `_nvfp4_compute_scale_factor` multiplies block scales and divides global by the same factor. Net effect: zero.

3. **No calibration needed.** The analytical values are the source of truth.

4. **The likely cause of degraded output: W13 gate/up ordering.**

### Phase 6: The Gate/Up Ordering Bug — The Final Fix

The research report warned: *"A wrong half order can still produce short factual answers while destroying longer generations."*

This was exactly our symptom. Investigation revealed:

- SGLang loads W13 as `[w1=gate, w3=up]` (w1 at idx 0, w3 at idx 1)
- `prepare_w4a16_packed_weights` calls `reorder_w13_to_gate_up` for gated activations
- `reorder_w13_to_gate_up` does `[second_half, first_half]` — it swaps halves
- The function **assumes** input is `[up, gate]` and reorders TO `[gate, up]`
- With our `[gate, up]` input, the output became `[up, gate]`
- The kernel does `silu(gate) * up` using first half as gate — so it computed `silu(up) * gate`
- `silu(up) * gate ≈ up * gate` for small values (silu ≈ identity near 0), explaining why short factual answers still worked but longer reasoning degraded

**Fix:** Swap W13 halves before passing to `prepare_w4a16_packed_weights`:
```python
w13_reordered = torch.cat([w13[:, I:], w13[:, :I]], dim=1)  # [w3=up, w1=gate]
```

After the reorder inside `prepare_w4a16_packed_weights`, this becomes `[w1=gate, w3=up]` — correct for the kernel.

---

## Final Architecture

```
MXFP4 Checkpoint
  ├── w13_weight: [E, 2I, H/2] int8 (packed E2M1 nibbles)
  ├── w13_scale:  [E, 2I, H/32] float32 (promoted from E8M0)
  ├── w2_weight:  [E, H, I/2] int8
  └── w2_scale:   [E, H, I/32] float32

        │ (1) Reorder W13: [w1,w3] → [w3,w1]
        │ (2) Nibbles: view as uint8 (unchanged!)
        │ (3) Scales: closed-form k_j → E4M3 block scale + float32 global
        │     - k_j = round(log2(float32_scale))
        │     - m = k_max - 8 per expert
        │     - weight_scale_2 = 2^m
        │     - sf_e4m3 = 2^(k_j - m), repeat_interleave(2)
        │ (4) Swizzle: 6D tile permutation for 128×4 layout
        ▼

NVFP4 Logical Tensors
  ├── w13_fp4:          [E, 2I, H/2] uint8
  ├── w13_blockscale:   [E, 2I_pad, H/16_pad] f8e4m3 (swizzled)
  ├── w13_global_scale: [E] float32
  ├── w2_fp4:           [E, H, I/2] uint8
  ├── w2_blockscale:    [E, H_pad, I/16_pad] f8e4m3 (swizzled)
  └── w2_global_scale:  [E] float32

        │ prepare_w4a16_packed_weights(source_format="modelopt")
        │   - unswizzle scales
        │   - reorder_w13_to_gate_up (swaps halves → [gate, up])
        │   - repack weights to int32 tile layout
        │   - process scales: permute, ×csf, ×2^7, bit-shift, E4M3 encode
        │   - process global: ×2^119, ÷csf
        ▼

W4A16PackedWeights → launch_sm120_moe() → Blackwell tensor-core MoE
```

---

## Changelog

### v1 — nvfp4_quantize with global_sf = amax/448
- **What:** Dequant MXFP4 → float → `nvfp4_quantize(gs=amax/448)`
- **Result:** 100% nibble saturation, all values clamped to max. Server produces `content: null`.

### v2 — nvfp4_quantize with global_sf = 1.0
- **What:** Same dequant path but `gs=1.0`
- **Result:** Unsaturated nibbles (11.4%). Server runs at 35 tok/s but output still `content: null` — requantization noise too high.

### v3 — Offline NVFP4 checkpoint conversion
- **What:** Pre-converted all 33,792 expert tensors to NVFP4 format offline (53 min GPU time)
- **Result:** Server loads checkpoint but output is `content: null`. Same requantization quality issue.

### v4 — Closed-form scale remapping (first attempt)
- **What:** Nibbles unchanged, E8M0 → E4M3 scale conversion with `global_sf = 2^(m-119)`
- **Result:** `global_sf` overflows to `inf` for small-valued experts (m=-13 → 2^132). Abandoned.

### v5 — Closed-form with correct global_sf convention
- **What:** `global_sf = 2^m` (weights_scaling_factor_2), E4M3 block scales, swizzled layout
- **Result:** `prepare_w4a16_packed_weights` succeeds! Server runs. Short answers work ("4", "Paris") but longer outputs degenerate into repetition loops.

### v6 — Fallback to nvfp4_quantize(gs=1.0) for comparison
- **What:** Reverted to requantization to compare quality
- **Result:** Same degradation pattern. Confirmed the issue isn't scale-specific — it's structural.

### v7 — Report-guided implementation with batched conversion
- **What:** Implemented report's `mxfp4_to_nvfp4_scales()` + `_swizzle_128x4_expert_scales()`. Validated 100% bit-exact MXFP4==NVFP4 dequant. Validated swizzle round-trip.
- **Result:** All validation passes. Server runs. Same degradation on longer outputs.

### v8 — Gate/up reorder fix ✅
- **What:** Swapped W13 from `[w1=gate, w3=up]` to `[w3=up, w1=gate]` before `prepare_w4a16_packed_weights`, so `reorder_w13_to_gate_up` produces correct `[gate, up]` for the kernel.
- **Result:** **Perfect output quality.** Fibonacci code, TCP/UDP explanation, math reasoning — all coherent and correct.

---

## Performance

| Metric | GEMV Fallback | W4A16 Tensor-Core |
|--------|--------------|-------------------|
| Startup time | ~60s | ~170s (scale conversion) |
| Decode throughput | ~15 tok/s | ~38 tok/s |
| GPU utilization | ~40% | ~99% |
| Kernel type | Scalar Triton dequant | SM120 B12x CuTe-DSL MMA |
| Nibble preservation | N/A (original format) | Bit-exact (no requant) |

---

## Key Lessons

1. **Read the vendor's reference implementation.** NVIDIA's `cast_mxfp4_to_nvfp4.py` contained the entire solution — same nibbles, analytical scales. We wasted hours on requantization before finding it.

2. **Hardware dequant tricks cancel out.** The BF16 subnormal reinterpretation in the W4A16 kernel looked terrifying but is just an exponent-bias trick that `_process_nvfp4_packed_global_scale` compensates for. The external interface is standard `E2M1 * block_scale * global_scale`.

3. **Gate/up ordering is a silent killer.** Swapping gate and up in a gated SiLU MoE produces output that is "close enough" for short factual answers but degrades catastrophically on longer generations. The symptom ("short answers work, reasoning loops") is a reliable diagnostic for this specific bug class.

4. **Don't compare analytical scales to data-derived scales.** The 4/3 ratio between closed-form `2^k_j` and `nvfp4_quantize`'s `block_max/6.0` is expected and correct — they're different scale conventions for the same numerical result. Chasing this red herring cost significant time.

5. **Validate before serving.** The report's checklist (MXFP4 dequant == NVFP4 dequant → swizzle round-trip → prepare succeeds → single-expert numerical test) would have caught the gate/up bug much earlier if we'd included a reference matmul comparison.

6. **Weight loaders promote dtypes silently.** SGLang's loader converts E8M0 scales to float32. Any conversion code must handle both formats, and `.view(torch.uint8)` on float32 gives 4× the bytes — a shape mismatch that fails at runtime, not at import.

---

## Files Modified

- `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`
  - Added `_raw_e8m0_to_k()`, `_is_nonzero_block()`, `mxfp4_to_nvfp4_scales()`, `_swizzle_128x4_expert_scales()`
  - Updated `process_weights_after_loading()` for W4A16 path with gate/up reorder
  - Updated `apply()` to dispatch through `launch_sm120_moe`

## Dependencies

- FlashInfer with SM120 W4A16 CuTe-DSL kernel (`flashinfer.fused_moe.cute_dsl.blackwell_sm12x`)
- `prepare_w4a16_packed_weights` and `launch_sm120_moe` from FlashInfer
- No ModelOpt dependency required — conversion is pure PyTorch

---

*Total debug time: ~8 hours across 8 iterations. The final fix was 4 lines of code.*

---

## Post-Victory: SWE-bench Reality Check

### The Problem

Synthetic benchmarks (8 concurrent, 512-token completions) showed 240 tok/s aggregate. But real SWE-bench workloads (8 concurrent agents, multi-KB tool outputs) showed **catastrophic regression**: ~80K tokens generated in 10 minutes vs 500K with FP4 GEMV v3 and 700K with FP8.

### Root Cause: Prefill Starvation

Server logs during SWE-bench revealed:
- **72 prefill batches vs 26 decode batches** — 73% of time spent prefilling
- Decode median dropped to **11 tok/s** (vs 233 tok/s in synthetic)
- Prefill throughput: median 46 tok/s for new tokens

The W4A16 tensor-core kernel excels at **decode** but doesn't help prefill. SWE-bench workloads are prefill-dominated (large code contexts every turn). The `--chunked-prefill-size 8192` blocks decode during each prefill chunk.

### Throughput During SWE-bench

| Metric | Synthetic (8 concurrent) | SWE-bench (8 agents) |
|---|---|---|
| Decode tok/s (median) | 233 | 11 |
| Prefill:Decode batch ratio | 1:3 | 3:1 |
| Total tokens / 10 min | ~240K (decode only) | ~80K |

### Key Insight

The W4A16 path optimizes the wrong bottleneck for agentic workloads. SWE-bench agents send large contexts (1-8K tokens) and receive moderate completions (200-500 tokens). The workload is **prefill-bound**, not decode-bound. Previous GEMV v3 handled prefill more efficiently because:
1. No weight repacking overhead at load time
2. GEMV works for both prefill and decode (same kernel)
3. Prefill is memory-bandwidth-bound anyway — tensor cores don't help

### Next Steps

To get beyond 500K tokens/10min, we need to optimize **prefill throughput** or **prefill-decode overlap**, not decode speed. Options:
- Increase `--chunked-prefill-size` to reduce prefill interruptions
- Use the W4A16 kernel for decode but GEMV for prefill (hybrid path)
- Optimize the FP4 prefill GEMM itself (currently ~46 tok/s at 8K chunks)
- Reduce prefill volume via better prompt caching / context reuse

### Commits

```
sglang-sm120-pr24692 (branch: sm120-dsv4-rebase):
  e9cb8f1 feat: MXFP4→NVFP4 closed-form W4A16 tensor-core MoE for SM120

rtx-pro-6000-bench (branch: main):
  871a435 docs: MXFP4→NVFP4 W4A16 quest log, research report, and conversion script
```
