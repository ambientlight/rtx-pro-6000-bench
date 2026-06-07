# Quest 3: Upstream the native MXFP4×MXFP4 (W4A4-mx) SM120 MoE into SGLang + FlashInfer

**Date:** 2026-06-03 (planning) → 2026-06-07 (re-scoped to the shipped W4A4-mx path)
**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, 256 experts, top_k=6, native MXFP4 checkpoint)
**Status:** Implementation DONE locally (Quest 2) — this quest upstreams it. Plan ready.

---

## Objective

Upstream the **native MXFP4×MXFP4 (W4A4-mx) fused MoE** path built and validated in Quest 2 —
**742 tok/s @16-wide** (matches NVFP4's 759), native checkpoint format, **zero re-quant, zero
calibration** — as clean, first-class code in FlashInfer and SGLang. Eliminate every env var,
monkey-patch, and Marlin-class hijack so `--moe-runner-backend auto` (or a single clean flag) just works
on SM120.

The local branches (`sm120-nvfp4-rebase` in both repos) already carry the full implementation:
- **FlashInfer** (5 commits, `1c2cefc1`→`91e527af`): the `MmaMXF4Op` atom + E8M0/32 activation
  quantizers + the 3 fused MoE kernels (static/micro/dynamic) wired for `sf_vec_size=32` /
  `quant_mode="mxfp4"`, the RT runtime-m wrapper parameterized for E8M0, and **a standalone NVFP4 bug
  fix** (`num_m_tiles` dropped the 2nd 64-row M-quadrant — latent in the shipped NVFP4 path too).
- **SGLang** (2 commits, `1bec556c`→`7b20149a`): a clean `Mxfp4W4A4MoEMethod` (NOT a Marlin hijack) +
  `fp8.py` routing, with the load-time/capture-memory fixes (del+empty_cache, pre-materialized
  `_weight_views` + capped `_workspace`).

Upstreaming means splitting these into reviewable PRs, generalizing the DSV4-specific assumptions, and
deleting the Quest-1 NVFP4 re-quant + calibration hacks that W4A4-mx makes obsolete.

### What changed vs the original (NVFP4-re-quant) upstreaming plan
The original Quest-3 plan upstreamed the **NVFP4 re-quant** path (`Fp8ToNvfp4MoEMethod`: FP8→BF16→NVFP4 +
one-shot activation calibration). That path is now **superseded** — W4A4-mx is strictly cleaner (native
weights, no double-quantization, no calibration sidecar, no `input_gs`/`down_input_scale`) at the same
throughput. So PR 4 below is rewritten from "clean `Fp8ToNvfp4MoEMethod`" to "clean
`Mxfp4W4A4MoEMethod`", and a new PR 0 lands the standalone `num_m_tiles` correctness fix.

## Current Hacks to Eliminate

| Hack | Where | Why it's bad |
|------|-------|--------------|
| `SGLANG_FP4_MOE_NVFP4=1` env var | `fp8.py` | Routes to Marlin class even though we don't use Marlin |
| Hijacked `Mxfp4MarlinMoEMethod` | `mxfp4_marlin_moe.py` | 451 lines in a class named "Marlin" that never touches Marlin |
| `SGLANG_DSV4_FP4_EXPERTS=0` env var | launch script | Overrides auto-detection to prevent FP4 buffer sizing |
| `SGLANG_NVFP4_STATIC_WS_CAP=640` env var | `mxfp4_marlin_moe.py` | Caps workspace to prevent JIT recompilation |
| `sitecustomize.py` monkey-patch | External `deepseek_v4_kernel/` | Swaps `flash_mla.flash_mla_with_kvcache` at interpreter startup |
| `sys.path.insert` for HMMA kernel | `deepseek_v4_backend.py` | Hardcoded local path to pre-compiled .so |
| SM120 tilelang routing hack | `dsv4/indexer.py` | Works around TVM 0.1.10 regression with conditional import |
| PYTHONPATH manipulation | launch script | Two extra paths for the kernel package and sitecustomize |

## What Upstream Already Has

### SGLang (mainline)
- `is_sm120_supported()` — SM120 detection cached helper (`utils/common.py`)
- `_is_sm120` flag in `deepseek_v4_backend.py` — already routes decode differently for SM120
- `flash_mla_sm120.py` — Built-in SM120 sparse decode fallback stack (Triton + PyTorch)
- `MoeRunnerBackend.FLASHINFER_CUTEDSL` — Enum value exists in `moe/utils.py`
- `moe_runner/flashinfer_cutedsl.py` — Dedicated CuTe-DSL runner with `CuteDslFp4MoeQuantInfo`
- `ModelOptNvFp4FusedMoEMethod` — Official NVFP4 path for native MXFP4 checkpoints (`modelopt_quant.py`)
- `FusedMoEMethodBase` — Clean interface: `create_weights`, `create_moe_runner`, `process_weights_after_loading`, `apply`

### SGLang Gaps (bugs / missing features)
- `auto` backend selection in `server_args.py` does NOT pick CuTe-DSL on SM120 (falls to `triton_kernel`)
- `flashinfer_cutedsl` validation incorrectly requires `is_sm100_supported()` only, blocking SM120
- No FP8→NVFP4 re-quantization path — only native MXFP4 checkpoints supported for FP4 MoE
- HMMA sparse decode requires external monkey-patched .so instead of sgl-kernel built-in

### FlashInfer (mainline)
- All CuTe-DSL MoE functions we use: `launch_sm120_static_moe`, `launch_sm120_dynamic_moe`, `select_sm120_moe_backend`, `allocate_sm120_moe_workspace`, `_get_weight_views`, `_get_cached_workspace`
- `_DynamicMoELaunch` — The pattern we copied for our `_StaticMoELaunch`

### FlashInfer Gaps
- The **native MXFP4×MXFP4 (W4A4-mx) MoE path is local-only** (`sm120-nvfp4-rebase`): the `MmaMXF4Op`
  atom branch, E8M0/32 activation quantizers (`fp4_common.py`), the `sf_vec_size==32` datapath in all 3
  fused kernels, and the `quant_mode="mxfp4"` dispatch threading. Upstream `MmaMXF4Op` exists in the
  cutlass DSL but is never called from the FP4 MoE kernels.
- The **`num_m_tiles` quadrant-drop bug is in upstream** — `num_m_tiles = tile_m // (16*4)` is copied
  from the dense kernel's 4-M-warp layout into the MoE kernels' 2-M-warp layout, silently zeroing rows
  64–127 for any static-MoE tile with >64 routed rows/expert (affects upstream NVFP4 at large skewed
  prefill, not just W4A4-mx). This is a standalone upstreamable fix.
- `_StaticMoELaunch` / `_get_static_kernel_rt` (the per-M JIT fix) — Quest-1 work; W4A4-mx extended it to
  be `sf_vec_size`-parameterized. Still not upstream.

---

## Plan: PRs Across the Repos

### PR 0: FlashInfer — `num_m_tiles` correctness fix (standalone, ships first)

**Independent, ~13 lines, pure bug fix — no W4A4 dependency.**
**File:** `fused_moe/cute_dsl/blackwell_sm12x/{moe_static,moe_micro,moe_dynamic}_kernel.py`

The MoE kernels hardcode `num_m_tiles = tile_shape_mnk[0] // (16 * 4)`, copied from the **dense** kernel
whose `atom_layout=(4,2,1)` has 4 M-warps. The MoE kernels use `atom_layout=(2,2,1)` (2 M-warps), so the
GEMM fills only M-tiles {0,1} while the accumulator + epilogue (`MmaMPerEpiM=4`) expect {0,1,2,3} —
**rows 64–127 stay at `fill(0.0)`**. Derive from `atom_shape` like dense:
`num_m_tiles = tile_m // (mma_m * atom_shape[0])`. This is a genuine upstream NVFP4 bug (latent because
decode uses tile_m=64 → one quadrant; bites at >64 routed rows/expert with a 128-wide tile). Local commit
`39e86f19`; validated by `nvfp4_b12x_regression.py` (skewed routing 256–1024 rows/expert, within=1.0000
after fix, unregressed under random). **Land this first — it's a clean fix reviewers can accept without
the whole MXFP4 feature.**

### PR 1: FlashInfer — native MXFP4×MXFP4 (W4A4-mx) MoE kernels + dispatch

**Depends on:** PR 0 (the kernels need the quadrant fix). **Blocks:** PR 4.
**Files:** `cute_dsl/fp4_common.py` (+127), `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` (+53),
`gemm/gemm_base.py` (+15), `fused_moe/.../moe_{static,micro,dynamic}_kernel.py`, `.../moe_dispatch.py`
(+81). Local commits `1c2cefc1`/`b8fdbd02`/`aac5a172`/`91e527af`.

- **`fp4_common.py`:** 8 E8M0/32-block activation quantizers (`quantize_block_mxf4`,
  `quantize_and_pack_32`→2×uint64, `max_abs_32`, `silu_mul_32`/`relu2_32` + fused), reusing the existing
  `cvt_f32_to_ue8m0`/`ue8m0_to_output_scale` intrinsics.
- **Atom selection** (`if sf_vec_size==32: MmaMXF4Op`) in the dense GEMM + all 3 MoE kernels; the
  `can_implement` gate lift; the SFA/SFB fragment rank-collapse for mma_nsf=2.
- **MXF4 datapath** in static/micro/dynamic: `tile_k=128` (not sf_vec_size*8, to preserve the
  FC1-epi-N == FC2-K coupling), 32-block E8M0 Phase-1 input quant + Phase-2 FC1 requant. The dynamic
  kernel has 3 Phase-1 routing sub-paths.
- **Dispatch:** `quant_mode="mxfp4"` through `_normalize_quant_mode`/`_sf_params_for_quant_mode`, all
  kernel-getter cache keys (+sf_vec_size), workspaces, `_get_weight_views`, and the RT wrapper
  (`_get_static_kernel_rt` parameterized for E8M0). All gated on `sf_vec_size==32` — NVFP4 untouched.
- **Generalize for upstream:** the path is already DSV4-agnostic (keys on shapes, not the model), but
  add a unit test mirroring `tests/moe/test_b12x_fused_moe.py` for the mxfp4 quant_mode, and confirm the
  `MmaMXF4Op` admissibility check covers sm_120/121.

This subsumes the original PR-1 (`_StaticMoELaunch` JIT fix is already upstream-shaped from Quest 1; here
it's just extended to sf_vec_size=32).

### PR 2: sgl-kernel — HMMA Sparse Decode Kernel

**Blocks:** PR 5
**New files:**
- `sgl-kernel/src/sgl-kernel/csrc/sm120_sparse_decode/` — Move .cu/.cuh from `deepseek-v4-flash-sm120/build-docker/`
- `sgl-kernel/python/sgl_kernel/sm120_sparse_decode.py` — Python wrapper
- CMakeLists.txt entry to compile for sm_120

Expose as `sgl_kernel.sm120_flash_mla_with_kvcache()`.
Eliminates the external .so + sitecustomize.py monkey-patch entirely.

### PR 3: SGLang — SM120 Auto-Detection Fix

**Independent, low-risk, ~10 lines total**

Files:
- `server_args.py` — Add SM120 to `auto` moe_runner_backend selection and `flashinfer_cutedsl` validation
- `pyproject.toml` — Pin `tilelang>=0.1.8,<0.1.10` (TVM regression on SM120)

Fixes the bug where SM120 falls back to `triton_kernel` instead of `flashinfer_cutedsl`.

### PR 4: SGLang — Clean `Mxfp4W4A4MoEMethod` (native MXFP4×MXFP4)

**Depends on:** PR 1 (FlashInfer mxfp4 kernels), PR 3 (SM120 validation).
**New file:** `layers/quantization/mxfp4_w4a4_moe.py` (~366 lines, local commits `1bec556c`/`7b20149a`).

Clean class implementing `FusedMoEMethodBase` — **NOT a Marlin hijack, no re-quant, no calibration**:
- `create_weights()` → native-MXFP4 buffers (E2M1 int8 weights + E8M0 block scales), sized to the
  checkpoint (no FP8 dequant, no buffer-size mismatch).
- `process_weights_after_loading()` → gate/up swap `[w1,w3]→[w3,w1]` (the fused kernel wants [up,gate])
  + E8M0 → 128×4 swizzle → `convert_sf_to_mma_layout` + pre-materialize `_get_weight_views` +
  `del`/`empty_cache`. **No nvfp4_quantize, no activation calibration.**
- `apply()` → `launch_sm120_moe(quant_mode="mxfp4")` with precomputed `_weight_views=` and a capped
  `_workspace=` (decode reuses a fixed-capacity workspace outside graph capture; prefill/dynamic pass
  `_workspace=None`). `ones` alpha (MXF4 self-scales). Applies `routed_scaling_factor` post-hoc.

**Modified:** `fp8.py` (+32 lines) — Route `is_fp4_experts + SM120 + (SGLANG_MXFP4_W4A4=1 or auto) →
Mxfp4W4A4MoEMethod`. For upstream, replace the env flag with `auto`-backend selection once PR 3 lands.

**Reverted:** the Quest-1 NVFP4 re-quant + calibration code in `mxfp4_marlin_moe.py` (the
`_calibrate_nvfp4` one-shot hack, the `SGLANG_FP4_MOE_NVFP4` routing) goes back to upstream — W4A4-mx
makes it obsolete. The `SGLANG_NVFP4_STATIC_WS_CAP` pattern is generalized to `SGLANG_MXFP4_STATIC_WS_CAP`
inside the new method (or `auto`-derived).

### PR 5: SGLang — Sparse Decode Cleanup

**Depends on:** PR 2 (sgl-kernel HMMA)

**Modified:** `deepseek_v4_backend.py` — Replace `sys.path.insert` + external import with:
```python
if _is_sm120:
    from sgl_kernel import sm120_flash_mla_with_kvcache
```

**Reverted:** `dsv4/indexer.py` changes (tilelang pin in pyproject.toml handles the TVM regression)

**Folded:** `warmup.py` moe_w4a4 warmup into `CuteDslMoEWrapper.warmup()`

---

## Dependency Graph

```
FlashInfer PR 0                    sgl-kernel PR 2
(num_m_tiles fix, ~13 lines)       (HMMA sparse decode)
       │                                │
       ▼                                │
FlashInfer PR 1                         │
(MXFP4×MXFP4 kernels + dispatch)        │
       │                                │
       ▼                                │
SGLang PR 3 ◄──── independent           │
(auto-detect fix, ~10 lines)            │
       │                                │
       ▼                                │
SGLang PR 4                             │
(Mxfp4W4A4MoEMethod, ~366 lines)        │
       │                                │
       ▼                                ▼
             SGLang PR 5
       (sparse decode + cleanup)
```

PR 0, PR 2, PR 3 can start in parallel. **PR 0 ships first** (standalone correctness fix). PR 1 (the
MXFP4 kernels) is the critical-path blocker for PR 4.

## Net Impact

| Metric | Quest 1/2 (local, hacky) | Quest 3 (upstream target) |
|--------|--------------------------|---------------------------|
| MoE path | NVFP4 re-quant + 1-shot calibration | **native MXFP4×MXFP4, no re-quant, no calibration** |
| Lines of hack code | ~700 | 0 |
| New clean code | 0 | ~366 (W4A4 method) + ~600 (FlashInfer kernels) |
| Env vars required | 3 (`FP4_MOE_NVFP4`, `DSV4_FP4_EXPERTS`, `NVFP4_STATIC_WS_CAP`) | 0 (`auto`) |
| External dependencies | monkey-patched .so + sitecustomize.py | 0 |
| sys.path manipulation | yes | no |
| Marlin class hijacking | yes | no |
| Calibration sidecar | yes (16M-token corpus) | **none** |
| Launch command | 10+ env vars + complex script | `--moe-runner-backend auto` |
| Decode @16w | 759 (NVFP4 re-quant) | **742 (native MXFP4, within 2%)** |

## Target Launch Command

After the PRs merge:
```bash
python -m sglang.launch_server \
  --model-path DeepSeek-V4-Flash \
  --tp 4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 16
```

No `--moe-runner-backend` needed (auto-detects SM120 native-MXFP4 experts → fused W4A4-mx). No env vars.
No monkey patches. No PYTHONPATH manipulation. No calibration. **742 tok/s @16-concurrent on the native
checkpoint.**

## Key Design Decisions

1. **Native MXFP4, not NVFP4 re-quant.** W4A4-mx runs the checkpoint's native E2M1 weights + E8M0 scales
   directly through `MmaMXF4Op` — no MXFP4→BF16→NVFP4 double-quantization, and E8M0 block scales are
   self-scaling so there is **nothing to calibrate**. This is strictly cleaner than the Quest-1
   `Fp8ToNvfp4MoEMethod` plan it replaces, at the same throughput.

2. **New class, not Marlin hijack.** `Mxfp4W4A4MoEMethod` implements `FusedMoEMethodBase` cleanly. The
   weight prep (gate/up reorder + E8M0→MMA-layout swizzle) is a load-time concern; runtime calls the
   public `launch_sm120_moe(quant_mode="mxfp4")`.

3. **`num_m_tiles` fix lands separately (PR 0).** It is a genuine upstream NVFP4 correctness bug
   (independent of MXFP4) and should be reviewable on its own merits before the larger feature.

4. **sgl-kernel for HMMA decode, not FlashInfer.** The sparse decode kernel is tightly coupled to
   SGLang's decode path. sgl-kernel is the right home for vendor-specific CUDA kernels.

5. **tilelang pin in pyproject.toml.** Replaces the conditional import hack in `dsv4/indexer.py`. The TVM
   buffer-shape regression in 0.1.10+ is a dependency-level issue.

## Relationship to Prior Quests

- **Quest 0**: NVFP4 W4A4 path on the old fork, discovered the JIT problem, first `_StaticMoELaunch` prototype.
- **Quest 1**: Mainline, fixed JIT cache-key instabilities, 759 tok/s — working but hacky (NVFP4 re-quant + calibration).
- **Quest 2**: Built + validated the **native MXFP4×MXFP4 (W4A4-mx)** fused path — 742 tok/s @16w, zero re-quant, zero calibration (DONE, local).
- **Quest 3** (this): Upstream Quest-2's W4A4-mx into FlashInfer + SGLang as first-class SM120 code, deleting the Quest-1 re-quant/calibration hacks.
