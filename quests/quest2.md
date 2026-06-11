# Quest 2: Native MXFP4×MXFP4 MoE on SM120 — SHIPPED at NVFP4-class throughput

**Status (2026-06-07):** ✅ **DONE — W4A4-mx is the ship candidate.** Native MXFP4 weights × MXFP4
activations (E8M0 self-scaling, **zero re-quant, zero calibration**) running the fused-SwiGLU CuTe-DSL
MoE kernels on a live 4-GPU SGLang server, hitting **742 tok/s @16-wide** (NVFP4 was 759; shipped W4A8-mx
is 515). Logit cos 0.975 vs BF16 (the genuine FP4-activation floor). All correctness committed.

**Supersedes:** the original "offline NVFP4 activation calibration" framing (Appendix A) — that whole
problem disappears with native MXFP4: there are no activation global scales to calibrate.

---

## The Core Insight

The old path was `MXFP4 checkpoint → dequant to BF16 → re-quantize to NVFP4 → calibrate activation
scales`. Every step is avoidable. **SM120 tensor cores natively execute MXFP4×MXFP4**, the checkpoint is
already MXFP4, and the activation format (E8M0 block scales) is self-scaling — **nothing to calibrate**.

- **The hardware atom exists.** `nvidia_cutlass_dsl/.../cute/nvgpu/warp/mma.py` defines `MmaMXF4Op`
  (sf_vec_size=**32**, `Float8E8M0FNU`, `mma.kind::mxf4 .scale_vec::2X .ue8m0`) alongside `MmaMXF4NVF4Op`
  (16, E4M3). Both `admissible_archs = ["sm_120a","sm_121a","sm_120f"]`. MXFP4×MXFP4 is a first-class
  SM120 instruction.
- **The checkpoint is already native MXFP4.** DSV4-Flash expert tensors: `w*.weight I8` (packed E2M1) +
  `w*.scale F8_E8M0` with 32-element blocks — an exact layout match for `MmaMXF4Op`.
- **The only work was FlashInfer wiring** — every W4A4 MoE kernel hardcoded the NVF4 atom even though
  `sf_vec_size` is otherwise threaded through `tile_k`, the SFA/SFB SMEM layouts, etc.

### Why this beats the NVFP4-calibration path
| | NVFP4 re-quant + calibration | Native MXFP4×MXFP4 (this) |
|--|--|--|
| Weight numerics | Double-quantized (MXFP4→BF16→NVFP4), lossy | Native, zero conversion loss |
| Activation scale | E4M3 16-block + per-expert **global scale** (must calibrate) | E8M0 32-block, **self-scaling, no calibration** |
| Calibration corpus | 16M tokens, staged pipeline | **None** |
| `input_gs`/`down_input_scale` | Must be measured & shipped | Do not exist |
| Matches checkpoint intent | No | Yes |
| **Decode @16w** | 759 tok/s | **742 tok/s (within 2%)** |

W4A4-mx delivers the 759-class decode throughput **without** the double-quantization or the fragile
one-shot calibration — native checkpoint format AND NVFP4 throughput.

---

## What was built (final implementation)

### FlashInfer (4 commits on `sm120-nvfp4-rebase`)
`1c2cefc1` (WIP base) → `39e86f19` → `b8fdbd02` → `aac5a172`. 7 files, +1147/-427:

| File | Change |
|--|--|
| `cute_dsl/fp4_common.py` | +127: 8 E8M0/32-block activation quantizers (`quantize_block_mxf4`, `quantize_and_pack_32` → 2×uint64, `max_abs_32`, `silu_mul_32`/`relu2_32` + fused variants). Reuse existing `cvt_f32_to_ue8m0`/`ue8m0_to_output_scale`. |
| `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` | +53: atom-select branch (`MmaMXF4Op` if sf_vec_size==32), `can_implement` gate lift, the SFA/SFB fragment rank-collapse (`if rank>3: group_modes`) for mma_nsf=2 at tile_k=256. |
| `gemm/gemm_base.py` | +15: `B12xFp4GemmRunner` sf_vec_size/sf_dtype from `use_nvfp4`. |
| `fused_moe/.../moe_static_kernel.py` | +315: atom swap, **tile_k=128 for MXF4** (not sf_vec_size*8=256 — preserves the FC1-epi-N == FC2-K-tile coupling), 32-block E8M0 Phase-1 input quant + Phase-2 FC1 requant, **the `num_m_tiles` quadrant fix**. |
| `fused_moe/.../moe_micro_kernel.py` | +335: same MXF4 datapath (3 sites) + num_m_tiles fix. |
| `fused_moe/.../moe_dynamic_kernel.py` | +648: same across **3 Phase-1 routing sub-paths** + Phase-2 + num_m_tiles fix. |
| `fused_moe/.../moe_dispatch.py` | +81: `quant_mode="mxfp4"` threaded through `_normalize_quant_mode`, `_sf_params_for_quant_mode`→(32,E8M0,32), all kernel-getter cache keys (+sf_vec_size), workspaces, `_get_weight_views`; **RT static wrapper parameterized for sf_vec_size=32**; backend gates allow mxfp4 static/dynamic/micro. |

### SGLang (2 commits on `sm120-nvfp4-rebase`)
`1bec556c` → `7b20149a`. +398 lines:

| File | Change |
|--|--|
| `quantization/mxfp4_w4a4_moe.py` (new, +366) | `Mxfp4W4A4MoEMethod`. `create_weights` = native MXFP4 buffers (shared w/ W4A8). `process_weights_after_loading`: **gate/up swap [w1,w3]→[w3,w1]** + E8M0→128×4 swizzle→`convert_sf_to_mma_layout` + pre-materialize `_get_weight_views` + **del/empty_cache**. `apply`: `launch_sm120_moe(quant_mode="mxfp4")` with precomputed `_weight_views=`/capped `_workspace=`. |
| `quantization/fp8.py` (+32) | Routing: `SGLANG_MXFP4_W4A4=1` + `is_fp4_experts` + SM120 → `Mxfp4W4A4MoEMethod`, before the W4A8 branch. |
| `debug/launch-mxfp4-w4a4-prod.sh` (new, gitignored/local) | TP=4, `SGLANG_MXFP4_W4A4=1`, CUDA graphs on, mem-fraction 0.88. |

All edits gated on `sf_vec_size==32` / `quant_mode=="mxfp4"` — **NVFP4 and shipped W4A8-mx untouched.**

### Validation harnesses (bench repo `spikes/`)
`stageA_dense_mxf4.py`, `stageA2_dequant_ref.py` (dense atom, cos 0.999998), `stageB_quantizer.py`
(E8M0 quant bit-exact), `stageC_static_moe.py`/`stageC_relu2.py`/`stageC_phase1_probe.py` (static MoE +
the bug hunt), `stageC_sfa_tv_probe.py`/`stageC_sfa_frag_probe.py`/`_sfa_layout_extract.py` (SFA layout
probes), `stageD_micro_dynamic.py` (all 3 backends via public entry), `stageF_w4a4_method.py` (the SGLang
method E2E), `nvfp4_b12x_regression.py` (NVFP4 unbroken).

---

## The two bugs that mattered (both solved)

### 1. The numerics bug: `num_m_tiles` dropped the 2nd 64-row M-quadrant
**Symptom:** a sharp cliff at >64 rows/expert (relu2 single-expert: 64 rows cos 0.98, 128 rows **0.69**).
After exhaustively eliminating every host-inspectable layout (Phase-1 quant 0.993, weight scales, both
Phase-2 writes bijection-proven, SFA SMEM layout, fragment ranks) and **two deep-research detours that
wrongly fingered the PTX `scale_vec::2X` scale-operand semantics**, an epilogue-shape dump found the real
cause: the upper quadrant was `|out| = 0.000` **exactly** — a *dropped write*, not a scale error.

Root cause (`moe_static_kernel.py`): `num_m_tiles = tile_shape[0] // (16 * 4)` = 2, **copy-pasted from
the dense kernel** whose `atom_layout=(4,2,1)` has 4 M-warps. The MoE uses `atom_layout=(2,2,1)` → 2
M-warps → needs `(16 * 2)` → **num_m_tiles=4**. The GEMM filled only M-tiles {0,1}; the accumulator +
epilogue (`MmaMPerEpiM=4`) expected {0,1,2,3}, so rows 64–127 stayed at `fill(0.0)`.

**Fix:** derive from `atom_shape` like dense — `num_m_tiles = tile_m // (mma_m * atom_shape[0])`. Applied
to all 3 kernels (static/micro/dynamic). Cliff gone: M=128 0.69→0.98, M=256 0.70→0.98.

**Bonus — it was a latent NVFP4 bug too** (sf_vec_size-independent). NVFP4 decode never tripped it
(`_select_moe_mma_tiler_mn` returns tile_m=64 for ≤128 routed rows → one quadrant), but any static-MoE
call with >64 rows/expert AND a 128-wide tile (large skewed prefill) would have silently zeroed rows
64–127. Confirmed + fixed via `nvfp4_b12x_regression.py` (skewed routing 256–1024 rows/expert,
within=1.0000 after the fix; unregressed under random routing).

### 2. The throughput/memory bug: 22 GB capture + 29 tok/s (invocation, not kernels)
After correctness, the live server was correct but **slow (29 tok/s @1w) and OOM'd at mem-fraction 0.88**
(CUDA-graph capture of just bs {1,2,4} cost 22 GB). Decisive observation (the user's): NVFP4 ran the
**same** fused kernels at 0.88, captured bs {1..16}, hit 759 — so it was the *invocation*, not the
kernels. A code-verified rescan + DR#6 (`drs/sm120_w4a4_mxfp4_moe_perf_memory_research.md`) found three
W4A4-specific host-side issues, all fixed:

1. **No `del`/`empty_cache`** in `process_weights_after_loading` — per-layer f32 scale upcasts + ~2 GB
   `torch.cat` weight copies ×43 layers pinned ~8–10 GB of allocator high-water reserve.
2. **`apply()` built `_get_weight_views` + workspace lazily inside capture** → ~2 GB landed in the
   graph-private pool. Fix: pre-materialize weight-views + ONE capped (640) static workspace at load,
   pass via `_weight_views=`/`_workspace=`; hoist the `ones` tensor.
3. **RT static wrapper forced off for mxfp4** (`_use_rt = ... and not _is_mxfp4`) because
   `_get_static_kernel_rt` hardcoded sf_vec_size=16 → mxfp4 compiled a fresh per-M module per batch size.
   Fix: parameterize the RT wrapper by quant_mode, drop the exclusion → **one shared module across all
   M**.

**Result — both solved at once:**
- **Capture memory 22 GB → 4.17 GB (5×).** Relaunched at the original aggressive config (mem-fraction
  0.88, bs {1,2,4,8,16}); the avail-mem curve FLATTENED after bs=16 (proof the per-M compiles are gone).
- **Throughput 29 → NVFP4-class.** The 29 was host-side per-M JIT + eager dispatch outside the graph,
  NOT a slower MXF4 kernel.

---

## Final results (live 4-GPU server, mem-fraction 0.88, CUDA graphs on)

**Single-stream decode (steady ITL):** 256ctx 81.0 tok/s, 1024 71.9, 4096 62.4, 8192 61.2, 16384 57.9.

**Concurrent streaming throughput:**
| concurrency | **W4A4-mx** | NVFP4 (target) | W4A8 (shipped) | FP8 |
|--:|--:|--:|--:|--:|
| 1  | **76.9** | 82  | 52  | 67 |
| 4  | **253.1**| 259 | 163 | 170 |
| 8  | **452.8**| 445 | 302 | 260 |
| 16 | **742.5**| **759** | **515** | — |

**@16-wide = 742 — matches NVFP4 within 2%, beats shipped W4A8 by +44%.** Passes the Stage-G ship gate
(@16w > 515). All 43 W4A4 layers load clean; correct generation ("2+2"→"Four", "primary colors"→
"red, blue, yellow"). Numerics: cos 0.975 vs BF16 = the FP4-activation floor (W4A8 with FP8 acts is
0.999; the coarser FP4 activations are the expected, accepted cost).

### Remaining before flipping the default over W4A8
1. **Accuracy gate** — live SWE-bench / logit-cos vs W4A8 (offline already 0.975).
2. **Soak** — long-context stability at 0.88 (16384 single-stream now completes at 57.9 tok/s; OOM'd
   before the fixes).

---

## Shipped fallback: W4A8-mx (the un-fused grouped-GEMM path)

Before the fused W4A4 path was throughput-viable, **W4A8-mx** (MXFP4 weights × MXFP8 activations via the
`group_gemm_mxfp4_nt_groupwise` cubin) was built and SHIPPED as the safe answer. It remains a valid
fallback (native weights, FP8-grade activations cos 0.9986, zero calibration), at FP8-class decode:

| workers | W4A8-mx | vs FP8 | vs NVFP4 |
|--:|--:|--|--|
| 1w | 51.8 | −23% | −37% |
| 4w | 163.2 | −4% | −37% |
| 8w | 302.0 | +16% | −32% |
| 16w | 515.5 | — | −32% |

Prefill ~4,700–4,900 tok/s (well above FP8 ~1,550). Files: `mxfp4_w4a8_sm120.py` (forward),
`mxfp4_w4a8_moe.py` (`Mxfp4W4A8MoEMethod`), `fp8.py` routing (`SGLANG_MXFP4_W4A8=1`),
`debug/launch-mxfp4-w4a8-prod.sh`. Validated graph-safe (fixed-CAP forward, dynamic routing replays
bit-exactly). W4A4-mx now supersedes it on throughput while keeping the same native-weight,
zero-calibration properties.

---

## Appendix A — What this retires
The original Quest 2 (offline NVFP4 activation calibration, see
`drs/deepseek_v4_flash_nvfp4_offline_activation_calibration.md`) was sound *conditional on staying in
NVFP4*. Native MXFP4 makes it unnecessary: E8M0 block scales are computed per-block at runtime from the
activation `max_abs` with no learned/measured parameter. Kept as fallback record.

## Relationship to Other Quests
- **Quest 0:** NVFP4 W4A4 on the old fork, discovered the JIT problem.
- **Quest 1:** Mainline, fixed JIT (`_StaticMoELaunch`), 759 tok/s — but on re-quantized NVFP4 + fragile calibration.
- **Quest 2 (this):** Native MXFP4×MXFP4 — eliminate re-quant + calibration, match the checkpoint format, at NVFP4 throughput.
- **Quest 3:** Upstream the clean SM120 stack into SGLang (now cleaner — native MXFP4 loader, no calibration sidecar).
