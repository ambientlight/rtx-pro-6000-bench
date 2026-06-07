# Quest 3: Upstream the native MXFP4×MXFP4 (W4A4-mx) SM120 MoE into FlashInfer + SGLang

**Date:** 2026-06-07
**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe)
**Model:** DeepSeek-V4-Flash (291B MoE, 256 experts, top_k=6 — **native MXFP4 checkpoint, used as-is**)
**Status:** Implementation DONE + validated (Quest 2). This quest plans the upstreaming.
**Scope decision:** Upstream **only** the native MXFP4×MXFP4 (W4A4-mx) E2E MoE path. The NVFP4 re-quant +
calibration path and every Quest-1 hack are **deliberately not upstreamed** — they are superseded and only
weaken the upstream argument.

---

## 1. Objective

Land the native MXFP4×MXFP4 fused MoE — **742 tok/s @16-wide** (matches the retired NVFP4 path's 759),
native checkpoint format, **zero re-quant, zero calibration**, and (now validated) **best SWE-bench Lite
accuracy of any local DeepSeek-V4-Flash deployment** — as clean, first-class, feature-gated code in
FlashInfer and SGLang.

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

## 4. PR sequence

```
FlashInfer A: num_m_tiles fix              (standalone correctness — ships first)
        │
        ▼
FlashInfer B: MXF4 common + dense MmaMXF4Op (has public mm_fp4 consumer)
        │
        ▼
FlashInfer C: SM120 W4A4-mx fused MoE + dispatch + RT(sf_vec_size=32)
        │
        ▼
SGLang D: feature-gated flashinfer_cutedsl selection on SM120
        │
        ▼
SGLang E: Mxfp4W4A4MoEMethod
        │
        ▼
SGLang F: docs + benchmark cookbook

(separate, optional, NOT blocking: sgl-kernel G — SM120 HMMA sparse-decode perf)
```

Keep each PR small enough that a maintainer can say yes without accepting the whole project.

---

## 5. Per-PR detail

### PR A — FlashInfer: `fix(sm120-moe): derive M tile count from atom layout`
**Files:** `moe_{static,micro,dynamic}_kernel.py` + `tests/moe/test_sm120_moe_num_m_tiles.py`.
Pure correctness, no MXFP4. Test: skewed routing, one expert gets 128/256/768 rows, assert rows 64:128
nonzero vs a BF16/dequant reference; assert NVFP4 random routing unchanged. Local: `39e86f19`
(validated by `nvfp4_b12x_regression.py`, within=1.0000 after fix). **Lands first — builds reviewer
trust, unblocks nothing downstream from a simple bug.**

### PR B — FlashInfer: `feat(sm120): MXFP4 E8M0/32 quant helpers + MmaMXF4Op dense path`
**Files:** `fp4_common.py`, `dense_blockscaled_gemm_sm120_b12x.py`, `gemm_base.py`, +
`tests/gemm/test_sm120_mxfp4_dense.py`, `tests/cute_dsl/test_mxfp4_quant_helpers.py`.
Reviewer promise: *"MXFP4 representation + dense support only; no fused-MoE dispatch change."* Tests:
E8M0 byte-exactness vs numpy; FP4 nibble roundtrip; dense MXFP4 GEMM cos ≥ 0.99999 vs dequant ref; NVFP4
dense unchanged; `MmaMXF4Op` selected only for sf_vec_size=32. Reachable via the public
`mm_fp4(backend="b12x", use_nvfp4=False)`, so it is not a dead standalone.

### PR C — FlashInfer: `feat(sm120-moe): native MXFP4xMXFP4 CuTe-DSL fused MoE`
**Files:** `moe_{static,micro,dynamic}_kernel.py`, `moe_dispatch.py`, public `launch_sm120_moe`, +
`tests/moe/test_b12x_mxfp4_fused_moe.py`. `quant_mode="mxfp4"`, `sf_vec_size=32`, `tile_k=128` (preserves
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
**Files:** `server_args.py`, MoE backend-selection util, + selection tests. Feature-**probe**:
```python
def has_flashinfer_sm120_mxfp4_moe() -> bool:
    try:
        import flashinfer
    except Exception:
        return False
    return (hasattr(flashinfer, "launch_sm120_moe")
            and flashinfer_supports_quant_mode("mxfp4"))
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
Port `docs/deploy-mxfp4-w4a4-cutedsl.md` (the deploy guide) + the throughput/accuracy/memory tables.
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
- **The HMMA `.so` is orthogonal to the MoE work.** Our SWE-bench / 742-tok/s results were measured *with*
  the HMMA kernel, but the MoE PR's correctness and throughput do not depend on it — the MoE block is a
  separate stage in the per-token pipeline (attention → MoE). Upstreaming MoE on top of the merged Triton
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
7. **Branches not pushed / token stale.** → See §8: the SGLang fork branch is **not pushed yet**.

---

## 8. Pre-flight (ground-truth state, 2026-06-07)

- **FlashInfer fork `origin`** = `https://github.com/ambientlight/flashinfer.git`. `sm120-nvfp4-rebase`
  pushed but **1 commit behind** (`aac5a172`; local tip `91e527af` — the RT comment cleanup). **No
  `upstream` remote configured.**
- **SGLang fork `origin`** = `https://github.com/ambientlight/sglang.git`. `sm120-nvfp4-rebase` is **NOT
  pushed.** `upstream` = `sgl-project/sglang` IS configured.

### Commit-splittability reality (important)
The proposed B/C split does **not** map to our commits — the WIP base `1c2cefc1` interleaves
`fp4_common.py` + `dense_blockscaled_gemm` + `gemm_base.py` (PR B) **and** `moe_static_kernel.py` +
`moe_dispatch.py` (PR C) in **one commit**. So "cherry-pick the dense part" **won't work** — the
upstream-bound B/C branches must be **reconstructed by hunk from the working tree**, not cherry-picked.
(`39e86f19` num_m_tiles is cleanly separable → PR A is a true cherry-pick. `b8fdbd02` is micro+dynamic+
dispatch → folds into C. `aac5a172`+`91e527af` are the RT wrapper → C or split-out.)

### Day-zero
1. Add `upstream` remote to FlashInfer; `git fetch upstream` both repos.
2. Push the SGLang fork branch (token now valid).
3. Reconstruct upstream-bound branches **fresh from `upstream/main`**:
   - FlashInfer: `fix/sm120-moe-num-m-tiles` (cherry-pick `39e86f19`); `feat/sm120-mxfp4-common-dense`
     and `feat/sm120-mxfp4-moe` (re-split `1c2cefc1`/`b8fdbd02`/`aac5a172` **by hunk**).
   - SGLang: `fix/sm120-flashinfer-cutedsl-selection`; `feat/sm120-mxfp4-w4a4-moe` (clean
     `Mxfp4W4A4MoEMethod` only, **HMMA edit excluded**); `docs/sm120-mxfp4-w4a4-moe`.
4. `git range-diff` each upstream branch vs the fork to confirm only intended hunks are included and **no
   HMMA / env-var / sys.path / calibration content leaked in.**
5. **Close the dummy-load unknown locally** (PR E pre-req) before opening E.

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
  CUDA-graph IMA, no per-M JIT storm, ~742 tok/s @16-wide.
- **All tests skip cleanly without SM120; no dependency on the benchmark repo; no unconditional pin.**

---

## 10. Relationship to prior quests
- **Quest 0:** NVFP4 W4A4 on the old fork; discovered the JIT problem; first `_StaticMoELaunch` prototype.
- **Quest 1:** Mainline; fixed JIT cache-key instability; 759 tok/s — but NVFP4 re-quant + calibration (hacky).
- **Quest 2:** Built + validated native MXFP4×MXFP4 (W4A4-mx) — 742 tok/s @16w, zero re-quant/calibration, best local SWE-bench (DONE).
- **Quest 3 (this):** Upstream Quest-2's W4A4-mx as first-class SM120 code, on top of the merged Triton SM120 baseline (#24692), deleting every Quest-1 hack and dropping the HMMA monkey-patch from the MoE story.
