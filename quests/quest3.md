# Quest 3: Upstream the native MXFP4×MXFP4 (W4A4-mx) SM120 MoE into FlashInfer + SGLang

**Date:** 2026-06-07
**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, 256 experts, top_k=6 — **native MXFP4 checkpoint, used as-is**)
**Status:** ✅ **COMPLETE (2026-06-09)** — all upstream-bound branches prepared, polished, and pushed to the
fork; human-opened drafts live (FlashInfer #3533 + #3541; SGLang branch ready). Carryover → **Quest 4** (SM120
sparse-prefill kernel). See §10 (Completion) at the bottom.
**Scope decision:** Upstream **only** the native MXFP4×MXFP4 (W4A4-mx) E2E MoE path. The NVFP4 re-quant +
calibration path and every Quest-1 hack are **deliberately not upstreamed** — they are superseded and only
weaken the upstream argument.

> ## ⛔ EXECUTION POLICY (read first)
> - **NEVER open a PR automatically.** Every PR is reviewed and opened **manually by the human.** Tooling
>   may prepare branches, diffs, commit messages, and PR-body drafts — but `gh pr create` / pushing to a
>   PR is a human action only.
> - **The PRs are a STACK, not parallel.** They build on each other (A → B → B½ → C → D → E → F). Prepare
>   them as **stacked branches**, each branched off the previous PR's branch (not all off `upstream/main`),
>   so each diff shows only its own delta. Rebase the stack forward when an earlier PR changes in review.
> - Assistant deliverables per PR: the prepared stacked branch (local), the curated commit(s), a draft PR
>   title + body, and the test files. **Stop there.** The human opens it.

---


## 1. Objective

Land the native MXFP4×MXFP4 fused MoE — **588 tok/s @16-wide sustained** (`bench_serving` output throughput;
the earlier 742 was a prod *instantaneous-peak* gen-throughput figure — both real, see §10), native checkpoint
format, **zero re-quant, zero calibration**, and (now validated) **best SWE-bench Lite accuracy of any local
DeepSeek-V4-Flash deployment** — as clean, first-class, feature-gated code in FlashInfer and SGLang.

The local implementation lives on `sm120-nvfp4-rebase` in both repos:
- **FlashInfer** (5 commits, `1c2cefc1`→`91e527af`): `MmaMXF4Op` atom + E8M0/32 activation quantizers
  (`fp4_common.py`) + the 3 fused MoE kernels wired for `quant_mode="mxfp4"` / `sf_vec_size=32`, the RT
  runtime-m wrapper parameterized for E8M0, and a **standalone NVFP4 correctness fix** (`num_m_tiles`).
- **SGLang** (2 commits, `1bec556c`→`7b20149a`): a clean `Mxfp4W4A4MoEMethod` (NOT a Marlin hijack) +
  `fp8.py` routing, with the load-time / capture-memory fixes.

The one-line upstream pitch:

> SGLang already runs DeepSeek-V4 on SM120 via **Triton fallback** kernels (PR #24692), and has parallel
> FP4 work for Hopper (SM90 CUTLASS MXFP4, #24816), server Blackwell (DeepGEMM W4A4 MegaMoE, #25052), and
> NVFP4 checkpoints (#25820). This series adds the missing **native SM120 workstation-Blackwell tensor-core
> path: FlashInfer CuTe-DSL MXFP4×MXFP4 W4A4-mx fused MoE** for native DeepSeek-V4-Flash checkpoints,
> replacing the Triton MXFP4 fallback with a fused `MmaMXF4Op` kernel — no re-quant, no calibration, no
> Marlin hijack, no env-var routing, no monkey-patches.

---

## 2. Upstream landscape (fetched 2026-06-07, sgl-project/sglang + flashinfer-ai/flashinfer)

The core work is **not** duplicated upstream. Verified against live PR state:

| Upstream effort | PR | State | Relation to this work |
|---|---|---|---|
| SM120 DeepSeek-V4 enablement (Triton fallback for MoE + MLA) | sglang #24692 | **MERGED** 2026-06-01 | The baseline we build on. Ships `mxfp4_moe_sm120_triton.py` (Triton MXFP4 MoE) + `flash_mla_sm120_triton.py` (Triton sparse decode) + SM120 guards. **We replace the Triton MXFP4 MoE fallback with the fused tensor-core kernel.** |
| `sgl_kernel.flash_mla` in-tree import | sglang #26499 | **MERGED** | FlashMLA kernels now in-tree. **Settles our sparse-decode decision** (see §6). |
| dummy-load `tid2eid` IMA fix | sglang #25892 | **MERGED** | Fixed `--load-format dummy` CUDA illegal-access for `flashinfer_mxfp4` (router `HashTopK.tid2eid`, not MoE weights). **De-risks our dummy-load story** — the known IMA is already handled. |
| SM90 CUTLASS MXFP4 MoE (W4A16) | sglang #24816 | MERGED | Hopper CUTLASS path. Different arch + kernel family. **Naming-collision risk on `flashinfer_mxfp4`** (see §7). |
| DeepGEMM W4A4 MegaMoE | sglang #25052 | MERGED | Server-Blackwell (TMEM/tcgen05). SM120 lacks those. Different hardware. |
| NVFP4 MoE for DeepSeek-V4 | sglang #25820 | **OPEN** | NVFP4-checkpoint path (`"moe_quant_algo": "NVFP4"`). **Routing-collision risk** — keep native-MXFP4 routing separate. |
| FP4 indexer (DSv4) | sglang #26209 / #27059 (SM120, OPEN) | MERGED / OPEN | Indexer, not MoE. #27059 touches `server_args.py` — **a collision file with our PR D**; coordinate, don't bundle. |
| FlashInfer SM90 CUTLASS MXFP4/INT4 | flashinfer #3084 | merged | SM90 CUTLASS, not SM120 CuTe-DSL. |

**Anti-overlap matrix** (use verbatim in PR descriptions):

| Backend | Hardware | Weights | Activations | Engine | Relation |
|---|---|---|---|---|---|
| SM120 Triton fallback (#24692) | SM120 | MXFP4 dequant→BF16 | BF16 | Triton | Existing fallback; **this replaces it with a fused TC path** |
| SM90 CUTLASS MXFP4 (#24816, #3084) | SM90 Hopper | FP4 | BF16/FP8 | CUTLASS | Different arch + kernel family |
| DeepGEMM W4A4 MegaMoE (#25052) | server Blackwell | FP4 | FP4 | DeepGEMM/tcgen05 | Different hardware (no TMEM on SM120) |
| NVFP4 MoE (#25820) | Blackwell NVFP4 ckpt | NVFP4 | NVFP4 | trtllm/cutedsl | Different **checkpoint** quant format |
| **This work** | **SM120 RTX PRO 6000 / 5090** | **native MXFP4 E2M1+E8M0** | **MXFP4 E8M0/32** | **FlashInfer CuTe-DSL `MmaMXF4Op`** | **the missing native workstation-Blackwell W4A4-mx path** |

---

## 3. What gets upstreamed (the only narrative)

> ℹ️ **§3–§6 are the original plan (2026-06-07).** Four things changed during execution (probe is a public
> API; HMMA was kept as a toggle, not dropped; throughput methodology; doc path + PR consolidation). See
> **§10 → "Where the plan diverged from what shipped"** for the authoritative final state, and
> `docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md` for the canonical guide.

### FlashInfer
1. **`num_m_tiles` correctness fix** — `num_m_tiles = tile_m // (16*4)` was copied from the dense kernel's
   4-M-warp `atom_layout=(4,2,1)`; the MoE kernels use `(2,2,1)` (2 M-warps), so the GEMM filled only
   M-tiles {0,1} and zeroed rows 64–127 under >64-rows/expert routing. Fix: `tile_m // (mma_m *
   atom_shape[0])`. **Standalone NVFP4 bug, independent of MXFP4.**
2. **MXF4 common primitives + dense `MmaMXF4Op`** — `fp4_common.py` E8M0/32 quantizers
   (`quantize_block_mxf4`, `quantize_and_pack_32`→2×uint64, `max_abs_32`, `silu_mul_32`/`relu2_32` +
   fused), the dense `MmaMXF4Op` atom branch in `dense_blockscaled_gemm_sm120_b12x.py`, `gemm_base.py`
   `use_nvfp4=False` plumbing. **Has a real public consumer:** `mm_fp4(backend="b12x", use_nvfp4=False)`.
3. **Native SM120 W4A4-mx fused MoE** — `quant_mode="mxfp4"` in static/micro/dynamic kernels (`tile_k=128`,
   E8M0/32 Phase-1 + Phase-2 quant, the dynamic kernel's 3 Phase-1 sub-paths), dispatch threading +
   cache keys, RT wrapper parameterized for `sf_vec_size=32`. All gated on `sf_vec_size==32`.

### SGLang
4. **SM120 `flashinfer_cutedsl` backend selection** — feature-probe (not version string) so `auto` picks
   the fused path when FlashInfer exposes mxfp4; otherwise preserve the merged Triton fallback (#24692).
5. **Clean `Mxfp4W4A4MoEMethod`** — native MXFP4 buffers, gate/up reorder, E8M0→MMA swizzle,
   pre-materialized `_get_weight_views` + capped `_workspace`, `del`+`empty_cache`,
   `apply()` → `launch_sm120_moe(quant_mode="mxfp4")`. Routed by config/shape + feature probe, **no env
   var**.
6. **Docs + benchmark cookbook** — the deploy guide + measured throughput/accuracy/memory tables.

### Explicitly NOT upstreamed (deleted from the story)
NVFP4 re-quant (`Fp8ToNvfp4MoEMethod`), FP8→BF16→NVFP4 conversion, one-shot activation calibration +
sidecar, `input_gs`/`down_input_scale`, the Marlin-class hijack, `SGLANG_FP4_MOE_NVFP4` /
`SGLANG_DSV4_FP4_EXPERTS` / `SGLANG_NVFP4_STATIC_WS_CAP` routing env vars, `sitecustomize.py`,
`sys.path.insert`, PYTHONPATH manipulation, and any tilelang/indexer hacks bundled into MoE work.

---

## 4. PR sequence (STACKED branches — never opened automatically)

Each branch is cut **off the previous PR's branch**, so each PR's diff shows only its own delta. The FlashInfer
stack is rooted on `flashinfer-ai/flashinfer:main`; the SGLang stack on `sgl-project/sglang:main`. SGLang PRs
**D–E depend on FlashInfer PR C's API**, so they can be prepared in parallel but only open once C's branch is
stable.

```
flashinfer-ai/flashinfer:main
        │
        ▼  A   fix: num_m_tiles            (clean cherry-pick of 39e86f19 — VERIFIED conflict-free)
        ▼  B   feat: MXF4 common + dense MmaMXF4Op
        ▼  B½  feat: _StaticMoELaunch runtime-m static wrapper   ← NOT yet upstream; PR C needs it
        ▼  C   feat: SM120 W4A4-mx fused MoE + dispatch + RT(sf_vec_size=32)

sgl-project/sglang:main
        │
        ▼  D   fix: flashinfer_cutedsl selection on SM120   (feature-probes C's API)
        ▼  E   feat: Mxfp4W4A4MoEMethod
        ▼  F   docs + benchmark cookbook

(separate, optional, NEVER blocking: sgl-kernel G — SM120 HMMA sparse-decode perf)
```

Keep each PR small enough that a maintainer can say yes without accepting the whole project. **The human
opens every PR manually** (see Execution Policy at top).

### Consolidated PR table (verified against commit/diff state 2026-06-07)

| PR | Repo | Title | Files | ~Size | Source (local) | Branch-prep method | Depends on |
|----|------|-------|-------|------:|----------------|--------------------|------------|
| **A** | FlashInfer | `fix(sm120-moe): derive M tile count from atom layout` | `moe_{static,micro,dynamic}_kernel.py` + `test_sm120_moe_num_m_tiles.py` | +35/−9 | `39e86f19` | **cherry-pick** (verified conflict-free onto `main`) | — |
| **B** | FlashInfer | `feat(sm120): MXFP4 E8M0/32 quant helpers + MmaMXF4Op dense` | `fp4_common.py` (+127), `dense_blockscaled_gemm_sm120_b12x.py` (+53), `gemm_base.py` (+15) + 2 tests | ~+195 | hunks of `1c2cefc1` | **by-hunk** | A; ⚠️ rebase-check vs upstream `d9b175ac` (NVFP4 4over6, same file) |
| **B½** | FlashInfer | `feat(sm120-moe): _StaticMoELaunch runtime-m static wrapper` | `moe_dispatch.py` (`_StaticMoELaunch` + `_get_static_kernel_rt` + dual-path dispatch) | ~+356 | `906556fb` (Quest-1 base, **not upstream**) | **cherry-pick** (NVFP4-only; pre-dates W4A4) | B |
| **C** | FlashInfer | `feat(sm120-moe): native MXFP4×MXFP4 CuTe-DSL fused MoE` | `moe_{static,micro,dynamic}_kernel.py`, `moe_dispatch.py` (+~81 W4A4 threading), `launch_sm120_moe` + `test_b12x_mxfp4_fused_moe.py` | ~+1100 | hunks of `1c2cefc1`,`b8fdbd02`,`aac5a172`,`91e527af` | **by-hunk** | **A + B + B½** |
| **D** | SGLang | `fix(moe): allow FlashInfer CuTe-DSL MoE selection on SM120` | `server_args.py`, MoE backend-select util + tests | ~small | new (extract from routing) | **author fresh** | FlashInfer **C** API; ⚠️ `server_args.py` collides w/ open #27059 |
| **E** | SGLang | `feat(deepseek-v4): native SM120 MXFP4 W4A4 MoE via FlashInfer CuTe-DSL` | `mxfp4_w4a4_moe.py` (new, +366), `fp8.py` (+32) + test | +398 | `1bec556c`,`7b20149a` | **by-hunk** (drop the HMMA edit) | **C + D**; ⚠️ close dummy-load unknown first |
| **F** | SGLang | `docs(deepseek-v4): native SM120 MXFP4 W4A4 MoE path` | `docs/...` (port deploy guide) | docs | port `docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md` | **author fresh** | E |
| **G** | sgl-kernel | `perf(sgl-kernel): SM120 HMMA FlashMLA backend` | sgl-kernel src + wrapper | — | external `.so` (must port in-tree) | **optional, later** | none; **NEVER blocks A–F** (see §6) |

**The `_StaticMoELaunch` prerequisite (B½) — the one real snag.** Our entire W4A4 stack sits on `906556fb`
("_StaticMoELaunch runtime-m wrapper for NVFP4 static MoE"), which is **not in upstream `main`**. The
`moe_dispatch.py` total delta is ~+429 lines, but **~+356 of that is `_StaticMoELaunch` itself** (the Quest-1
base) and only **~+81 is the W4A4 `quant_mode="mxfp4"` threading**. So PR C cannot apply without it. B½ carries
it as an NVFP4-only infrastructure PR (it pre-dates and is independent of MXFP4 — it's the shape-agnostic
runtime-M static launcher), landing between B and C. If a maintainer would rather take `_StaticMoELaunch` as
part of C, fold B½ into C; default is to keep it separate for reviewability.

---

## 5. Per-PR detail

### PR A — FlashInfer: `fix(sm120-moe): derive M tile count from atom layout`
**Files:** `moe_{static,micro,dynamic}_kernel.py` + `tests/moe/test_sm120_moe_num_m_tiles.py`.
Pure correctness, no MXFP4. **Cherry-pick of `39e86f19` — dry-run VERIFIED conflict-free onto `upstream/main`
(3 files, +35/−9, 0 conflicts).** Independent of `_StaticMoELaunch` (that lives only in `moe_dispatch.py`;
this touches only the 3 kernels' `self.num_m_tiles` line). Test: skewed routing, one expert gets 128/256/768
rows, assert rows 64:128 nonzero vs a BF16/dequant reference; assert NVFP4 random routing unchanged. Validated
locally by `nvfp4_b12x_regression.py` (within=1.0000 after fix). **Lands first — builds reviewer trust,
unblocks nothing downstream from a simple bug.**

### PR B — FlashInfer: `feat(sm120): MXFP4 E8M0/32 quant helpers + MmaMXF4Op dense path`
**Files:** `fp4_common.py`, `dense_blockscaled_gemm_sm120_b12x.py`, `gemm_base.py`, +
`tests/gemm/test_sm120_mxfp4_dense.py`, `tests/cute_dsl/test_mxfp4_quant_helpers.py`.
**Branch-prep: by-hunk** (these files are interleaved with PR-C content in `1c2cefc1`, so cherry-pick can't
separate them). ⚠️ **Rebase-check:** upstream landed `d9b175ac` ("Add CuTe DSL NVFP4 quantization with 4over6
FP16 scoring", #3448) in `fp4_common.py` after our merge-base — likely additive, but verify the hunks apply.
Reviewer promise: *"MXFP4 representation + dense support only; no fused-MoE dispatch change."* Tests:
E8M0 byte-exactness vs numpy; FP4 nibble roundtrip; dense MXFP4 GEMM cos ≥ 0.99999 vs dequant ref; NVFP4
dense unchanged; `MmaMXF4Op` selected only for sf_vec_size=32. Reachable via the public
`mm_fp4(backend="b12x", use_nvfp4=False)`, so it is not a dead standalone.

### PR B½ — FlashInfer: `feat(sm120-moe): _StaticMoELaunch runtime-m static wrapper`
**Files:** `moe_dispatch.py` (`_StaticMoELaunch` class + `_get_static_kernel_rt()` + the dual-path dispatch in
`launch_sm120_static_moe`). **Cherry-pick of `906556fb`** (Quest-1 commit). This is **NVFP4-only and predates
MXFP4** — it removes `m` from the static-kernel compile cache key (the shape-agnostic runtime-M launcher; CUDA
graph → per-M kernel, non-graph → RT wrapper). It is **not in upstream `main`**, and PR C's `moe_dispatch.py`
W4A4 threading is built on top of it, so it must land between B and C (or be folded into C if a maintainer
prefers). Reviewer framing: *"infrastructure: eliminates per-M JIT recompilation for the existing SM120 NVFP4
static MoE; no new quant format."* Tests: a runtime-M cache test (one module reused across M; CUDA-graph path
still uses the per-M kernel).

### PR C — FlashInfer: `feat(sm120-moe): native MXFP4xMXFP4 CuTe-DSL fused MoE`
**Depends on A + B + B½.** **Files:** `moe_{static,micro,dynamic}_kernel.py`, `moe_dispatch.py` (the ~+81 W4A4
`quant_mode="mxfp4"` threading on top of B½'s `_StaticMoELaunch`), public `launch_sm120_moe`, +
`tests/moe/test_b12x_mxfp4_fused_moe.py`. **Branch-prep: by-hunk** (W4A4 hunks span `1c2cefc1`/`b8fdbd02`/
`aac5a172`/`91e527af`). `quant_mode="mxfp4"`, `sf_vec_size=32`, `tile_k=128` (preserves
the FC1-epi-N == FC2-K coupling), E8M0/32 Phase-1/Phase-2 quant, static/micro/dynamic incl. all dynamic
routing sub-paths, `quant_mode`+sf in every cache key, RT wrapper parameterized for E8M0. Reviewer
promise: *"all new behavior gated on `quant_mode=="mxfp4"` / `sf_vec_size==32`; NVFP4 unchanged except
the already-submitted num_m_tiles fix."* Tests: M ∈ {1,8,40,64,65,128,256,768}, top_k ∈ {1,2,6,8}, random
+ skewed routing, micro/static/dynamic backends, cos at the FP4 floor, no NaN/Inf, NVFP4 regression, and
a runtime-M cache test (one RT module reused across M; sf_vec_size/quant_mode force distinct
specializations).
**Open question to decide before opening C:** the RT-wrapper `sf_vec_size=32` extension is a *performance*
change (it does not run inside captured graphs — see §6), so it could split out as a follow-up to keep C
to the correctness-critical kernels. Default: keep in C, but be ready to split if reviewers want C
smaller.

### PR D — SGLang: `fix(moe): allow FlashInfer CuTe-DSL MoE selection on SM120`
**Files:** `server_args.py`, MoE backend-selection util, + selection tests. Feature-**probe** (✏️ **corrected
to the SHIPPED implementation** — the original sketch used `hasattr(flashinfer, "launch_sm120_moe")`, which
predates mxfp4 and false-positives; we instead added a **public** `sm120_moe_supported_quant_modes()` to the
FlashInfer fork and probe that):
```python
# python/sglang/srt/layers/quantization/fp8.py
def _has_flashinfer_sm120_mxfp4_moe() -> bool:
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
            sm120_moe_supported_quant_modes,
        )
    except Exception:
        return False
    return "mxfp4" in sm120_moe_supported_quant_modes()
```
Policy: `auto` + SM120 + native FP4 experts + probe ⇒ `flashinfer_cutedsl`; else keep the merged Triton
fallback (#24692). **Coordinate with #27059** (FP4-indexer-SM120, also edits `server_args.py`) — do not
bundle. No tilelang pin here.

### PR E — SGLang: `feat(deepseek-v4): native SM120 MXFP4 W4A4 MoE via FlashInfer CuTe-DSL`
**Files:** `layers/quantization/mxfp4_w4a4_moe.py`, `layers/quantization/fp8.py`, +
`tests/.../test_mxfp4_w4a4_moe.py`. Local: `1bec556c`+`7b20149a`. `create_weights` native MXFP4 buffers;
`process_weights_after_loading` gate/up reorder + E8M0→`convert_sf_to_mma_layout` +
pre-materialize `_get_weight_views` + capped workspace + `del`/`empty_cache`; `apply` →
`launch_sm120_moe(quant_mode="mxfp4")` with `_weight_views=`/`_workspace=`, ones alphas (self-scale),
`routed_scaling_factor` once, EP/local remap. Routed by config/shape + the §PR-D probe, **no env var**.
Keep SM90-mxfp4 / DeepGEMM / NVFP4 / Triton-fallback paths untouched. **Include a dummy-load test**
(#25892 fixed the router `tid2eid` IMA, but our `process_weights_after_loading` does real swizzles — must
not assume initialized weights; verify capture is safe under `--load-format dummy`). See §6 note: this
is the one untested local unknown — **close it before opening E.**

### PR F — SGLang: `docs(deepseek-v4): native SM120 MXFP4 W4A4 MoE path`
Port `docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md` (the deploy guide) + the throughput/accuracy/memory tables.
Target launch: `python -m sglang.launch_server --model-path DeepSeek-V4-Flash --tp 4 --trust-remote-code
--kv-cache-dtype fp8_e4m3 --max-running-requests 16` — no env vars, no monkey-patches, no PYTHONPATH, no
calibration.

---

## 6. DECISION: the monkey-patched `sparse_decode_fwd` (HMMA `.so`)

**Decision: drop the HMMA monkey-patch from the upstream MoE series entirely. Use upstream's already-merged
Triton SM120 sparse decode. Re-pursue HMMA later (if ever) as a separate, optional sgl-kernel perf PR.**

### Why this is the right call (not a compromise)
- **Our local code is a hand-override of an already-solved upstream path.** `deepseek_v4_backend.py:1053`
  replaces the `if _is_sm120:` branch with `sys.path.insert('/mnt/.../build-docker')` +
  `from deepseek_v4_kernel.ops import sparse_decode_fwd as _hmma_fwd`. Upstream PR **#24692 (MERGED
  2026-06-01)** already ships `flash_mla_sm120_triton.py` — "Triton FlashMLA sparse decode (3.2–5.4×
  vs the FlashInfer fallback)" — as the first-class SM120 path. PR **#26499** moved FlashMLA fully
  in-tree (`sgl_kernel.flash_mla`). **SM120 sparse decode is solved upstream with zero external deps.**
- **The HMMA `.so` is orthogonal to the MoE work.** Our SWE-bench / throughput results (742 tok/s prod-peak;
  588 sustained — §10) were measured *with* the HMMA kernel, but the MoE PR's correctness and throughput do
  not depend on it — the MoE block is a separate stage in the per-token pipeline (attention → MoE).
  Upstreaming MoE on top of the merged Triton
  decode is a clean, self-contained change.
- **The HMMA `.so` is fundamentally un-upstreamable as-is:** external pre-compiled `.so` +
  `sys.path.insert` + a hardcoded `/mnt/hot/...` path + an out-of-tree `deepseek_v4_kernel` package. Even
  packaged into sgl-kernel, it competes with a *merged* Triton kernel that already beats the FlashInfer
  fallback — so it needs a fresh "HMMA vs upstream Triton SM120" benchmark to even justify a PR, which
  we have never run.

### What this means concretely
- **For the MoE upstream series (PRs A–F): the HMMA `.so` does not appear.** On clean upstream + our MoE
  PRs, SM120 sparse decode runs through the merged `flash_mla_sm120_triton.py`. The local
  `deepseek_v4_backend.py` HMMA edit is **a local-only divergence we do not carry upstream.**
- **Optional future PR G** (`perf(sgl-kernel): SM120 HMMA FlashMLA backend`): only if we (a) port the
  HMMA kernel source into sgl-kernel (no `.so`, no `sys.path`, no sitecustomize) behind the existing
  `sgl_kernel.flash_mla` API with a Triton fallback, AND (b) benchmark it beating the merged Triton
  SM120 path. Until both are true, it stays a local optimization. **It never blocks A–F.**
- **Risk if we keep the HMMA dependency in any upstream PR:** instant rejection (external `.so`,
  hardcoded path) + a "why not the merged Triton path?" objection we can't yet answer with data.

---

## 7. Risks & mitigations
1. **"Duplicate FP4 work."** → Lead with the §2 anti-overlap matrix; state SM120 CuTe-DSL W4A4-mx ≠ SM90
   CUTLASS ≠ DeepGEMM ≠ NVFP4-checkpoint.
2. **`flashinfer_mxfp4` naming collision** (#24816 is SM90 W4A16). → Don't reuse that name; keep the
   backend `flashinfer_cutedsl` + explicit `quant_mode="mxfp4"` + SM120 feature probe.
3. **NVFP4 routing collision (#25820, OPEN).** → Keep native-MXFP4 routing separate from
   `"moe_quant_algo": "NVFP4"`; do not upstream FP8→NVFP4 re-quant.
4. **`server_args.py` collision with FP4-indexer-SM120 (#27059, OPEN).** → Minimal selection change;
   coordinate, rebase, don't bundle.
5. **Large MoE diff rejected.** → The A/B/C split exists for exactly this.
6. **Dummy-load capture IMA.** → #25892 fixed the router side; we must verify *our* `apply()` +
   `process_weights_after_loading` are dummy-load-safe (the one open local unknown — close before PR E).
7. **Forks / remotes.** → DONE: both forks synced + branches pushed; FlashInfer `upstream` remote added
   and `main` fetched (§8). PR-A cherry-pick dry-run verified conflict-free.

---

## 8. Pre-flight (ground-truth state, 2026-06-07)

### Remotes / push state — DONE
- **FlashInfer fork `origin`** (`ambientlight/flashinfer`) @ `sm120-nvfp4-rebase` = `91e527af` — **synced,
  all 5 W4A4 commits pushed.** `upstream` (`flashinfer-ai/flashinfer`) **added + `main` fetched** (tip
  `a2870343`). Merge-base with `main` = `0037a9c1`.
- **SGLang fork `origin`** (`ambientlight/sglang`) @ `sm120-nvfp4-rebase` = `7b20149a` — **synced, both W4A4
  commits pushed.** `upstream` (`sgl-project/sglang`) configured.

### Commit-splittability reality (drives branch-prep method per PR)
The B/C split does **not** map to our commits — `1c2cefc1` interleaves PR-B files (`fp4_common.py`, dense,
`gemm_base.py`) and PR-C files (`moe_*`, `moe_dispatch.py`) in one commit. So:
- **PR A** = clean **cherry-pick** of `39e86f19` (**dry-run verified: 0 conflicts onto `main`**).
- **PR B½** = clean **cherry-pick** of `906556fb` (`_StaticMoELaunch`, NVFP4-only, pre-MXFP4).
- **PR B and PR C** = **reconstruct by hunk** from the working tree (not cherry-pickable).
- `b8fdbd02` (micro+dynamic+dispatch) and `aac5a172`+`91e527af` (RT sf_vec_size=32) → fold into C.

### Branch-prep procedure (STACKED — assistant prepares, human opens)
Reminder: **never run `gh pr create` / never open a PR.** Prepare local stacked branches + draft PR bodies;
the human opens each manually.

```bash
# FlashInfer stack (each branch off the PREVIOUS, not off main):
git checkout -b pr/A-num-m-tiles            upstream/main
git cherry-pick 39e86f19                    # verified clean
# (add tests/moe/test_sm120_moe_num_m_tiles.py)

git checkout -b pr/B-mxf4-common-dense      pr/A-num-m-tiles
# reconstruct fp4_common + dense + gemm_base hunks; rebase-check vs d9b175ac

git checkout -b pr/Bhalf-static-rt          pr/B-mxf4-common-dense
git cherry-pick 906556fb                    # _StaticMoELaunch

git checkout -b pr/C-mxfp4-fused-moe        pr/Bhalf-static-rt
# reconstruct the W4A4 hunks (moe_* + moe_dispatch threading + RT sf_vec_size=32)

# SGLang stack (off upstream/main; opens once FlashInfer C is stable):
git checkout -b pr/D-cutedsl-selection      upstream/main
git checkout -b pr/E-mxfp4-w4a4-method      pr/D-cutedsl-selection   # 1bec556c+7b20149a by-hunk, HMMA edit EXCLUDED
git checkout -b pr/F-docs                    pr/E-mxfp4-w4a4-method
```

After preparing each branch: `git range-diff upstream/main..pr/X <our-fork-equivalent>` to confirm only
intended hunks are present and **no HMMA / env-var / sys.path / calibration content leaked in.** When an
earlier PR changes in review, **rebase the whole stack forward** (`git rebase --onto` the updated parent).

### Open items before specific PRs
- **Before PR B:** confirm the `fp4_common.py` hunks apply over upstream `d9b175ac`.
- **Before PR E:** **close the dummy-load unknown** — verify our `Mxfp4W4A4MoEMethod.apply()` +
  `process_weights_after_loading` are safe under `--load-format dummy` (real swizzles assume initialized
  tensors; #25892 only fixed the router `tid2eid`, not MoE weights).

---

## 9. CI / test matrix
- **FlashInfer:** E8M0 byte-exactness; FP4 pack roundtrip; dense MXFP4 cos ≥ 0.99999; MoE static/micro/
  dynamic × random+skewed × M{1,8,40,64,65,128,256,768} × top_k{1,2,6,8}, no NaN/Inf, cos at FP4 floor;
  NVFP4 random unchanged + skewed benefits from num_m_tiles fix; RT compile cache stable across M.
- **SGLang:** backend selection (SM120+probe ⇒ cutedsl; no probe ⇒ Triton fallback; SM90/NVFP4/DeepGEMM
  unchanged); weight load (E2M1 + E8M0, gate/up reorder, swizzle, temp release); runtime (`apply` passes
  `quant_mode="mxfp4"`, `_weight_views` reused, workspace outside capture, scaling applied once,
  **dummy-load capture safe**).
- **Manual E2E:** the §PR-F launch command auto-selects native SM120 MXFP4, no env/patch/PYTHONPATH, no
  CUDA-graph IMA, no per-M JIT storm, ~588 tok/s @16-wide sustained (742 prod-peak — §10).
- **All tests skip cleanly without SM120; no dependency on the benchmark repo; no unconditional pin.**

---

## 10. Completion (2026-06-09)

**Status: COMPLETE — branches prepared, polished, pushed; human-opened drafts live.** The plan's A–F
stacked design was executed and consolidated. Final landed shape (verified against pushed refs):

### FlashInfer (fork `ambientlight/flashinfer`, branches pushed; opened as draft PRs by human)
- **`ambientlight/num-m-tiles`** (`cc871967`) → upstream **draft #3533** — the `num_m_tiles` quadrant-drop
  correctness fix (PR A), + skewed-routing regression test.
- **`ambientlight/mxf4-common-dense`** (`7d78ec26`) — MXFP4 E8M0/32 quant helpers + dense `MmaMXF4Op` (PR B).
- **`ambientlight/mxfp4-fused-moe`** (`440de948`) → upstream **draft #3541** — the native MXFP4×MXFP4 fused
  MoE (PR C), **retitled "dense + fused MoE"** to fold B in, + the public `sm120_moe_supported_quant_modes()`
  capability API (added this session so SGLang feature-probes a public function, not `_normalize_quant_mode`).
- The standalone `_StaticMoELaunch` runtime-m wrapper (planned PR B½) was **deferred** to a perf follow-up
  (`ambientlight/rt-perf`), not on the critical path — the fused-MoE branch uses per-M static kernels.

### SGLang (fork `ambientlight/sglang`, branch pushed; draft prepared, human opens)
- **`feat/sm120-mxfp4-w4a4-moe`** (`e1ca8d96`) — clean branch off `sgl-project/sglang:main` (PRs D+E merged
  into one feature PR): `Mxfp4W4A4MoEMethod` + `mxfp4_sm120_common` swizzle + the **`fp8.py` feature-probe**
  (replaced the `SGLANG_MXFP4_W4A4=1` env var per the "no env var" goal), the **3-way
  `SGLANG_SM120_SPARSE_DECODE` toggle** (hmma|triton|torch), and the **capture-safe indexer routing fix**
  (separate commit). Draft PR body at `quests/pr-bodies/sglang-pr-mxfp4-w4a4.md`.
- All `W4A4-mx` coinage removed from upstream-facing surfaces → `MXFP4 W4A4` / `MXFP4`.

### HMMA (out-of-tree `ambientlight/deepseek-v4-flash-sm120` @ `feat/hmma-tensor-core-sparse-decode`, `dd0a1f4`)
- Kept out-of-tree (PR G never blocked A–F). README leads with the HMMA optimization section; the
  RTX PRO 6000 tuned W8A8 + MoE configs are now **committed** (`dd0a1f4`) so users get them on clone.

### Validated this session (the carryover that seeds Quest 4)
- **E2E bench harness** (`bench/deepseek-v4-flash_W300_TP4_sglang/`) added to the rtx-pro-6000-bench repo:
  the feature-probe selects the MXFP4 W4A4 method with **no env var**, HMMA decode active, tuned configs
  loaded; smoke sweep produces throughput + GPU telemetry (incl. live `sglang:token_usage` KV-cache %) + plots.
- **Open upstream-facing item from §8 still applies:** dummy-load safety of `process_weights_after_loading`
  (real E8M0 swizzles) — left as an unchecked test item in the SGLang draft.
- **NEW GAP → Quest 4:** the SM120 **sparse-prefill** attention path has no SM120 kernel. The stress bench
  (`request-rate=inf`, ≥8 concurrent × 2048-in = 16384 prefill query tokens > the 11673 `_LARGE_INDEXER_
  QUERY_THRESHOLD`) routes to stock `sgl_kernel.sparse_prefill_fwd`, which is **SM90a/SM100f only** →
  `RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures`. Real
  staggered agent traffic (mini-swe-agent @ 16 sessions) never co-batches >11673 fresh prefill tokens, so it
  never tripped — but the synthetic max-throughput sweep does. **Quest 4 builds the SM120 sparse-prefill
  HMMA kernel** (mirroring sparse-decode: drop-in for `flash_mla_sparse_fwd`, out-of-tree).

### Where the plan diverged from what shipped (authoritative — supersedes §3–§6 where they conflict)
The §3–§6 plan was correct *as a plan*; these four things changed during execution. The current source of
truth is **`docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md`**.

1. **Feature-probe is a public API, not `hasattr`.** §5/PR-D sketched
   `hasattr(flashinfer, "launch_sm120_moe") and flashinfer_supports_quant_mode(...)`. **Shipped:**
   `_has_flashinfer_sm120_mxfp4_moe()` → `"mxfp4" in sm120_moe_supported_quant_modes()`. We **added a public
   `sm120_moe_supported_quant_modes()` to the FlashInfer fork** specifically so SGLang probes a stable public
   function (not the private `_normalize_quant_mode`); `launch_sm120_moe` alone predates mxfp4 and would
   false-positive.
2. **HMMA was KEPT, not dropped.** §6 decided to drop the HMMA monkey-patch and rely on upstream Triton.
   **Shipped:** the monkey-patch/`sys.path.insert` is gone, but HMMA lives on as a clean **3-way
   `SGLANG_SM120_SPARSE_DECODE` toggle (hmma | triton | torch, default `triton`)** via
   `is_deepseek_v4_kernel_available()` (importlib probe). It stays **out-of-tree** (the `.so` is not vendored),
   and the in-tree Triton path remains the default fallback — so the upstream-clean argument holds *and* HMMA
   is available. (The original §6 reasoning about un-upstreamability of the `.so` still stands; the toggle is
   the reconciliation.)
3. **Throughput numbers/methodology.** §1/§9/§10 cite **742 tok/s @16w** (prod *instantaneous gen-throughput*
   peak). The deploy doc now reports **`bench_serving` sustained output throughput: 72 / 213 / 351 / 588 tok/s
   @ c1/4/8/16** — a stricter sustained-aggregate metric, not a regression (peak per-run hit 77/249/440/720).
   Both are real; the doc is the canonical figure going forward.
4. **Doc path + PR consolidation.** §5 PR-F referenced `docs/deploy-mxfp4-w4a4-cutedsl.md`; the canonical guide
   is **`docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md`**. The A–F PRs consolidated to **3 FlashInfer
   branches + 1 SGLang feature branch** (B folded into the fused-MoE branch; D+E merged; B½ deferred to
   `rt-perf`). Branches were renamed `pr/A-*` → `ambientlight/*` (feature-descriptive, per the fork-PR review).


---

## 11. Relationship to prior quests

- **Quest 0:** NVFP4 W4A4 on the old fork; discovered the JIT problem; first `_StaticMoELaunch` prototype.
- **Quest 1:** Mainline; fixed JIT cache-key instability; 759 tok/s — but NVFP4 re-quant + calibration (hacky).
- **Quest 2:** Built + validated native MXFP4×MXFP4 (W4A4-mx) — 742 tok/s @16w, zero re-quant/calibration, best local SWE-bench (DONE).
- **Quest 3 (this):** Upstream Quest-2's W4A4-mx as first-class SM120 code, on top of the merged Triton SM120 baseline (#24692), deleting every Quest-1 hack and dropping the HMMA monkey-patch from the MoE story. ✅ **COMPLETE** (branches pushed, drafts live).
- **Quest 4 (next):** Close the SM120 **sparse-prefill** gap surfaced by the E2E throughput bench — build an HMMA tensor-core `flash_mla_sparse_fwd` kernel for SM120 (mirroring the sparse-decode kernel), so high-concurrency / long-prefill batches stop crashing on the SM90a/SM100f-only stock kernel.
