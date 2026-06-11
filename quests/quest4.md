# Quest 4: SM120 sparse-prefill HMMA kernel for DeepSeek-V4-Flash

**Date:** 2026-06-09
**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB each, PCIe), TP=4
**Model:** DeepSeek-V4-Flash (291B MoE, native MXFP4 checkpoint)
**Status:** ✅ **E2E VALIDATED — the crash repro is GREEN.** Kernel correct + race-free, integrated into the
serving venv, and the high-concurrency sweep that originally crashed now runs clean: **3072/3072 requests, 0
failures** across 2K/4K/8K-input × c1–c64, including the c8×2048 (16384-token) trigger. Server logs show **zero**
`only supported on SM90a/SM100f` errors after the fixed launch (all 8 historical crashes predate it). Sweep was
user-killed at 8192in×c16 (partial 16K–64K rows remain); perf hillclimb deferred (~17 TFLOP/s).

> ### Progress log
> - **2026-06-09 — Reference architecture located.** DeepSeek's own sparse-prefill source is vendored at
>   `repos/vllm/.deps/flashmla-src/csrc/{sm90,sm100}/prefill/sparse/` (`KernelTemplate<D_QK,HAVE_TOPK_LENGTH>`,
>   grid = query-token × B_H=64 head-block, B_TOPK=64, online softmax, `is_kv_valid` gather-mask). Params
>   `SparseAttnFwdParams` (params.h:145). Torch oracle `flashmla-src/tests/ref.py::ref_sparse_attn_fwd`.
>   Confirmed: `combined_indices` is **per-query-token**; KV arrives **flat bf16** (sglang pre-dequantises) →
>   no FP8/E8M0 inside the kernel. lse/max_logits in **natural log**; lonely query → out 0 / mx -inf / lse +inf.
> - **Stage 0 DONE.** `tests/reference.py`: `sparse_prefill_reference` + `make_fake_prefill_batch` (faithful to
>   DeepSeek's ref); `tests/test_sparse_prefill.py`. Validated vs an independent dense oracle (cos 0.999997)
>   and the lonely-query edge case.
> - **Stage 1 DONE — scalar SM120 kernel correct.** New: `csrc/common/params.h::SparseAttnPrefillParams`;
>   `csrc/sm120/prefill/{sparse_prefill.h, sparse_prefill_instantiation.cu, sparse_prefill_kernel_scalar_ref.cuh}`;
>   `csrc/api/sparse_prefill.{cpp,h}` (drop-in `flash_mla_sparse_fwd` signature); registered in `api/api.cpp`,
>   `ops.py`, `setup.py`. Built locally (cu130, nvcc 13.1, sm_120a — all 6 TUs compile clean). Correctness vs
>   torch ref: **out cos 0.999998–1.000000** across s_q∈{1,64,1024} × h_q∈{64,128} × topk∈{64,256,2048} +
>   topk_length + attn_sink. `max_logits` bit-exact; `lse` within 1e-2 (bf16-accum logsumexp; sglang discards it).
>   Build note: the repo's `build/` is root-owned (old Docker build) → build with
>   `--build-temp /tmp/... --build-lib /tmp/...`; the venv editable-install pointed at a stale `/tmp/dsv4-e2e`
>   clone, re-pointed to the canonical repo via `pip install -e .`.
> - **NEXT: Stage 2** — HMMA m16n8k16 kernel (`sparse_prefill_kernel.cuh`) mirroring DeepSeek's SM90 arch with
>   the SM120 primitives from `sm120/decode/sparse_decode_kernel.cuh`; build flag `DSV4_PREFILL_USE_HMMA`.
> - **Stage 2 DONE — HMMA kernel correct + race-free.** `csrc/sm120/prefill/sparse_prefill_kernel.cuh`
>   (namespace `prefill_hmma`): KV_CHUNK=64, NUM_WARPS=8, m16n8k16 QK^T + P@V, register-resident O, reuses the
>   decode HMMA helpers. **Three bugs found and fixed this stage:** (1) **`sKvalid` per-row gather mask** —
>   invalid indices (-1 / ≥ s_kv → zeroed sK row) were scored as a *valid* 0, not −inf; added a per-KV-row
>   validity flag in smem (mirroring DeepSeek's `is_kv_valid`) so `max_logits`/`lse` are exact. (2) **Epilogue
>   write-lane** — `max_logits`/`lse` were stored only from `lane_id==0`, but head *h*'s replicated softmax
>   state lives on lanes `4h..4h+3` (`g = lane_id>>2`), so every head but 0/8 got stale memory; moved the store
>   to `warp0 && t==0`. (3) **`ScoreProbUnion` aliasing** — `prob` (bf16) overlaid `score` (f32), so a `prob`
>   write clobbered `score` bytes another warp was still reading (compute-sanitizer **racecheck**: 128 hazards);
>   de-aliased union → struct (+2 KB smem, 86 KB total < 99 KB opt-in, still 1 block/SM) and gated the `prob`
>   write to warp 0. **Result:** full matrix s_q∈{1,64,1024,16384} × h_q∈{64,128} × topk∈{64,256,2048} +
>   topk_length + attn_sink → out cos 0.999997–0.999998, `max_logits`/`lse` **bit-exact**;
>   memcheck/initcheck/racecheck all **0 errors**. `tests/test_sparse_prefill.py` **26 passed** (the s_q=16384
>   cases — the > 11673 production trigger — compare the full-batch kernel against a 1024-row reference slice;
>   the dense reference gather OOMs at 16384×2048×512 ≈ 64 GiB, the kernel streams KV and is fine).
>   Perf baseline ~15–18 TFLOP/s, *flat across shapes* → latency/occupancy-bound (138 regs + 86 KB smem ⇒ 1
>   block/SM ≈ 17 % occupancy; synchronous KV load, no double-buffer; K re-gathered per head-block). The big
>   levers (cp.async double-buffer, B_H head-amortization, split-KV) need a structural rewrite — **deferred**:
>   per the user's call, wire into SGLang first so E2E numbers guide the hillclimb.
> - **Stage 3 code DONE — SGLang toggle + patch hook.** Fork (`feat/sm120-mxfp4-w4a4-moe`): new
>   `python/sglang/srt/layers/attention/flash_mla_sparse_prefill_sm120.py` resolves
>   **`SGLANG_SM120_SPARSE_PREFILL` = {hmma, sglkernel, torch}** once at import (mirrors the decode
>   `flash_mla_sm120.py`); `_forward_prefill_sparse` (`deepseek_v4_backend.py:1194,1266`) now dispatches through
>   it instead of hard-importing the stock `flash_mla_sparse_fwd`. **Safe default verified across a 10-case
>   matrix:** SM120 → `hmma` (or `torch` if the pkg is absent); off-SM120 → `sglkernel`; **`sglkernel` on SM120
>   is always downgraded to `torch`** (warns, never the crashing kernel). The dispatcher's `torch` fallback is
>   a chunked port of the reference (cos 1.0, bit-exact mx/lse). HMMA repo `deepseek_v4_kernel/_patch.py` gains
>   `_patch_sgl_kernel_sparse_prefill()` (overrides `sgl_kernel.flash_mla.flash_mla_sparse_fwd` on SM120,
>   covering the DSA backend's call site for free; no-op off SM120). All files byte-compile; deploy doc +
>   env-var table + pipeline diagram updated with the prefill toggle.
> - **REMAINING: Stage 3 E2E** (user-launched — the env reaps GPU servers across tool calls): re-run the
>   rtx-pro-6000-bench 2K–64K × concurrency sweep (`bench/deepseek-v4-flash_W300_TP4_sglang/`) with
>   `SGLANG_SM120_SPARSE_PREFILL=hmma` and confirm the c8×2048 (16384-token) prefill no longer crashes. Optional
>   follow-up: the deferred perf hillclimb, now guided by in-server numbers.
> - **2026-06-09 (later) — Perf characterized + E2E-readiness audit (§9, §10).** Op-level microbenchmarks:
>   (a) Path A (prefill) vs Path B (decode) — both sparse, identical math (cos 1.0) — are *flat per-token* with
>   Path A ~10 % cheaper (no FP8 unpack), so the > 11673 dispatch is a strict per-token win with no boundary
>   cliff; (b) **sparse vs dense** (the gain that actually matters): sparse is flat in context, dense grows
>   linearly → **1.6× / 3.3× / 6.6× / 12.7× at s_kv = 16K / 32K / 64K / 128K**, gain = `s_kv/topk`. **Audit
>   found two E2E blockers:** serving venv `~/.venvs/dsv4` `.so` lacks `sparse_prefill_fwd` (stale
>   `/tmp/dsv4-e2e` editable); serving sglang is a non-editable copy missing the fork's prefill files. Confirmed
>   the DSv4 path activates via the **in-tree resolver** (no `_patch.install()` needed). **Hardened the op**
>   (`sparse_prefill.cpp`): contiguity guards (q/kv d-stride==1, **indices topk-stride==1**) + attn_sink/
>   topk_length dtype+numel — kernel hard-assumes these; sglang satisfies them but they were unguarded.
>   Rebuilt clean, 26/26 still pass.
> - **2026-06-09 (eve) — E2E reinstall + crash repro GREEN.** Found the real serving venv is `~/.venvs/dsv4-test`
>   (not `dsv4`, which has a pre-existing sgl_kernel/torch-2.12 ABI break). It already had `deepseek_v4_kernel`
>   editable-pointed at the canonical repo (prefill op live); synced the 2 fork sglang files
>   (`flash_mla_sparse_prefill_sm120.py` + the `_forward_prefill_sparse` dispatch) into its non-editable sglang
>   copy; fixed `launch.sh`/`sglang.yaml` to source `dsv4-test` + export `SGLANG_SM120_SPARSE_PREFILL=hmma`.
>   Validated in-venv (resolver→hmma, entry-point cos 0.999998, the exact `nohup.out` crash-site import passes).
>   **Sweep result** (`bench_sweep.log`): 2K/4K/8K-input × c1–c64, **3072/3072 successful, 0 failed**, incl. the
>   c8×2048 = 16384-token batch (the original > 11673 trigger). `nohup.out` forensics: 8 historical
>   `only supported on SM90a/SM100f` crashes ALL predate the fixed 18:01 launch (last crash line 19148 < launch
>   line 19398); **zero** crashes / scheduler exceptions after it; server "fired up and ready". Throughput:
>   2048in 56→472 tok/s (c1→c56), 8192in 45→130 tok/s (c1→c8) — decode-path numbers consistent with the prior
>   baseline; TPOT 16–80 ms. Sweep user-killed at 8192in×c16; 16K–64K-input rows not yet collected.
**Repos:** out-of-tree HMMA kernel repo `ambientlight/deepseek-v4-flash-sm120` (primary); SGLang fork
`ambientlight/sglang @ feat/sm120-mxfp4-w4a4-moe` (integration); rtx-pro-6000-bench (validation).

> **Lineage:** Quest 3 shipped the native MXFP4 W4A4 MoE + HMMA sparse-**decode** path (✅ complete; fork
> branches pushed, drafts live). The E2E throughput bench added in Quest 3 then surfaced the gap this quest
> closes: **sparse-PREFILL has no SM120 kernel.** Mirror exactly how sparse-decode was built.

---

## 1. The failure mode (precisely characterized)

A max-throughput sweep (`vllm bench serve --request-rate inf`, ≥8 concurrent × 2048-input) crashes the
server during **prefill**, not decode, not MoE, not OOM:

```
deepseek_v4_backend.py:1120  if forward_mode.is_extend_without_speculative() and (
                                 q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD   # = 11673
                                 or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()):
deepseek_v4_backend.py:1124      return self._forward_prefill_sparse(...)
deepseek_v4_backend.py:1266          o, _, _ = flash_mla_sparse_fwd(...)
sgl_kernel/flash_mla.py:339              torch.ops.sgl_kernel.sparse_prefill_fwd.default(...)
RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.
```

**Trigger = prefill query-token count, not concurrency.** `_LARGE_INDEXER_QUERY_THRESHOLD = 11673`
(`dsv4/metadata.py:51`). When a single prefill forward pass batches **> 11673 query tokens**, the backend
routes to the sparse-prefill kernel — which stock sgl-kernel compiles **only for SM90a (Hopper) / SM100f
(Blackwell datacenter)**. SM120 has neither.

**Why it never hit in production / mini-swe-agent @ 16 concurrent:** real agent traffic is **staggered**
(LiteLLM conversational sessions; each request waits on the prior response). Individual prefill passes stay
under 11673, so the backend takes the working `_is_sm120` dense path (`flash_mla_with_kvcache_sm120`, line
1134) even for prefill. The **synthetic `request-rate=inf` herd** packs 8 × 2048 = 16384 tokens into one
prefill step → > 11673 → sparse path → crash. Decode (1 token/req) is never affected.

**Ruled out (verified across two launches, mem-fraction 0.80 and 0.67):** not an OOM (capture succeeded,
14.75 / 27.12 GB free), not a graph-bs issue (capture is decode-only; this is prefill), not config.

## 2. Objective

Build an **SM120 HMMA tensor-core sparse-prefill kernel** — a drop-in for
`sgl_kernel.flash_mla.flash_mla_sparse_fwd` — so high-concurrency / long-prefill batches run on SM120
instead of crashing. **Exactly mirror the sparse-decode kernel** (Quest 3's HMMA work): out-of-tree CUDA in
`deepseek-v4-flash-sm120`, scalar reference first then HMMA m16n8k16, op registration, runtime patch +
SGLang toggle. **Initial target: correct CUDA (HMMA), not yet maximally optimized** — same as the decode
kernel's first cut.

## 3. The API contract (what we must match — drop-in)

Stock `flash_mla_sparse_fwd` (`sgl_kernel/flash_mla.py:310`):
```python
def flash_mla_sparse_fwd(
    q: torch.Tensor,          # [s_q, h_q, d_qk] bf16          (s_q = MANY query tokens)
    kv: torch.Tensor,         # [s_kv, h_kv=1, d_qk] bf16      (flat, already dequantized)
    indices: torch.Tensor,    # [s_q, h_kv=1, topk] int32      (-1 or >= s_kv = invalid)
    sm_scale: float,
    d_v: int = 512,           # value dim; 512 only
    attn_sink: Optional[torch.Tensor] = None,   # [h_q] float
    topk_length: Optional[torch.Tensor] = None, # [s_q] int32  per-query valid prefix
) -> (out[s_q, h_q, d_v] bf16, max_logits[s_q, h_q] f32, lse[s_q, h_q] f32)
```

**The decisive simplification vs decode:** sglang's `_forward_prefill_sparse`
(`deepseek_v4_backend.py:1176`) **pre-dequantizes** the paged FP8/E8M0 KV cache into a **flat bf16
workspace** (`SparsePrefillChunkCache`, `dsv4/sparse_prefill_utils.py`) before calling the kernel. So unlike
sparse-decode (which reads packed FP8 pages + E8M0 scales + RoPE directly), **the prefill kernel sees plain
bf16 KV** — no FP8, no E8M0, no paging, no RoPE inverse inside. It's just:

> `out = softmax(q · gather(kv, indices)ᵀ · sm_scale) · gather(kv, indices)[:, :d_v]`, per (query, head),
> with `topk_length` masking and optional `attn_sink` (log-domain mix). **No causal mask** — sparsity is
> entirely expressed by `indices` (invalid = -1 / ≥ s_kv → masked).

Computationally identical to sparse-decode's QK^T·softmax·PV; the differences are: (a) `s_q ≫ 1` (many query
rows, the decode kernel's header explicitly lists `s_q > 1` as "not yet implemented"), (b) flat bf16 KV
instead of packed pages, (c) returns `max_logits` as a second output.

## 4. Design — mirror the sparse-decode kernel exactly

Existing decode layout (the template), in `deepseek-v4-flash-sm120/csrc`:
```
csrc/api/sparse_decode.{cpp,h}                      # torch op: shapes/strides → params, launch
csrc/api/api.cpp                                    # PYBIND11_MODULE m.def("sparse_decode_fwd", ...)
csrc/common/params.h                                # SparseAttnDecodeParams struct
csrc/sm120/decode/sparse_decode.h                   # launch_dsv4_sparse_decode_v32 decl
csrc/sm120/decode/sparse_decode_instantiation.cu    # the .cu TU (picks scalar vs HMMA, launches)
csrc/sm120/decode/sparse_decode_kernel.cuh          # HMMA m16n8k16 kernel (production)
csrc/sm120/decode/sparse_decode_kernel_scalar_ref.cuh  # scalar CUDA-core reference (DSV4_ENABLE_SCALAR_REFERENCE_KERNEL)
csrc/sm120/decode/split_kv_combine.cuh              # split-KV combine
deepseek_v4_kernel/ops.py                           # python wrapper (flat re-export)
deepseek_v4_kernel/_patch.py                        # runtime monkey-patch installer
tests/reference.py, tests/test_sparse_decode.py     # torch ref + pytest
```

**New parallel files for prefill** (same structure):
```
csrc/api/sparse_prefill.{cpp,h}                     # torch op matching flash_mla_sparse_fwd signature
csrc/common/params.h                                # add SparseAttnPrefillParams (or reuse with s_q)
csrc/sm120/prefill/sparse_prefill.h                 # launch decl
csrc/sm120/prefill/sparse_prefill_instantiation.cu
csrc/sm120/prefill/sparse_prefill_kernel.cuh        # HMMA m16n8k16
csrc/sm120/prefill/sparse_prefill_kernel_scalar_ref.cuh  # scalar ref (build + validate first)
```
Register `m.def("sparse_prefill_fwd", ...)` in `api/api.cpp`; export from `ops.py`; add prefill patch hook.

**Kernel sketch (HMMA), per (query-tile, head-group):**
- Grid `(ceil(s_q / Q_TILE), ceil(h_q / HEADS_PER_CTA), [split])`. The decode kernel already parameterizes
  `s_q` in its grid (`(b*s_q, head_blocks, splits)`); for prefill, tile over `s_q` rows so each CTA handles
  a block of query tokens × a head group — reuse `BLOCK_M_HEADS=16`, `KV_CHUNK`, the `mma.sync.aligned.
  m16n8k16.row.col.f32.bf16.bf16.f32` QK^T and P@V paths, and `softmax_and_rescale_reg_o` from decode.
- Load q-tile to smem; stream `kv` rows gathered by `indices` in KV_CHUNK blocks (bf16 direct — no FP8
  unpack); online softmax with `topk_length` mask + `attn_sink`; register-resident O accumulator → out.
- Emit `out`, `lse`, and `max_logits` (the running max — decode tracks it internally; just expose it).
- Reuse `split_kv_combine.cuh` if split-KV is needed for long `s_kv` (defer to V2 like decode did).

**Numerics:** d_qk = d_v = 512 (448 NoPE + 64 RoPE), h_q multiple of 16 (64/TP shard). bf16 throughout;
f32 softmax accumulation. Target cos ≥ 0.999 vs the torch reference (prefill is bf16×bf16, no FP4 floor).

## 5. SGLang integration (two call sites, same API)

Both consume the identical `flash_mla_sparse_fwd` contract → one kernel covers both:
- `deepseek_v4_backend.py:1266` — `DeepseekV4AttnBackend._forward_prefill_sparse` (the DSv4 path we use;
  `attention_backend='dsv4'`).
- `dsa_backend.py:1818` — generic DSA backend (pads h_q to 128, else same call). Not on our DSv4 path but
  covered for free.

**Wiring (mirror the decode toggle):** extend `_patch.py` to also patch
`sgl_kernel.flash_mla.flash_mla_sparse_fwd` when `is_deepseek_v4_kernel_available()` and SM120, **or** add a
3-way `SGLANG_SM120_SPARSE_PREFILL = {hmma, sglkernel, torch}` env toggle in the fork's
`deepseek_v4_backend.py` (parallel to `SGLANG_SM120_SPARSE_DECODE`). Default should be safe: if the HMMA
prefill kernel is absent, **do not** fall to the crashing stock kernel on SM120 — fall to the dense
`flash_mla_with_kvcache_sm120` path or a torch reference. (Decision: prefer the env-toggle in the fork so
it's a clean, reviewable SGLang change matching the decode toggle, with HMMA opt-in.)

**Note on the upstream-clean story:** Quest 3's SGLang PR is MoE-only + the decode toggle. The prefill
kernel is another **out-of-tree** HMMA piece — it does **not** go into the SGLang upstream PR. It rides the
same `SGLANG_SM120_SPARSE_*` toggle pattern, opt-in, `.so` not vendored.

## 6. Validation

1. **✅ DONE — Unit (HMMA repo):** `tests/test_sparse_prefill.py` + torch ref
   `sparse_prefill_reference(...)` in `tests/reference.py`. Parametrized s_q ∈ {1, 64, 1024, 16384}, h_q ∈
   {64, 128}, topk ∈ {64, 256, 2048}, ± `topk_length`, ± `attn_sink` → **26 passed**: out cos
   0.999997–0.999998, `max_logits`/`lse` **bit-exact**. The s_q=16384 cells (the > 11673 trigger) run the
   full-batch kernel and compare a 1024-row reference slice (the dense reference gather OOMs at
   16384×2048×512 ≈ 64 GiB; the kernel streams KV). Skips cleanly off SM120. compute-sanitizer
   memcheck/initcheck/racecheck **0 errors**.
2. **⬜ PENDING — Numerical drop-in (cross-arch parity):** call `deepseek_v4_kernel.ops.sparse_prefill_fwd`
   vs stock `flash_mla_sparse_fwd` **on a Hopper/SM100 box** with identical inputs; assert match. Not yet run
   (no Hopper/SM100 box in this env). Lower priority — the torch reference is the primary oracle and the
   dispatcher's own `torch` fallback is validated bit-exact against it (cos 1.0).
3. **⬜ PENDING (user-launched) — E2E (the original repro):** re-run the crashed sweep
   `bench/deepseek-v4-flash_W300_TP4_sglang/` `--input-lens 2048..65536 --max-concurrency 64,…,64
   --request-rate inf` with `SGLANG_SM120_SPARSE_PREFILL=hmma`. Success = the c8×2048 (16384-token) prefill no
   longer crashes; full 2K–64K × concurrency matrix completes with throughput + telemetry + plots. The env
   reaps GPU servers across tool calls, so this is user-launched.
4. **⬜ PENDING — Regression:** decode path + MoE untouched (the only fork change is the
   `_forward_prefill_sparse` dispatch swap, additive); re-confirm mini-swe-agent staggered traffic on the E2E
   run.

## 7. Risks / open questions — resolution

- **`max_logits` semantics — RESOLVED.** sglang consumes only `out` (`o, _, _ =` at both call sites), so
  `max_logits`/`lse` are best-effort for the server. We nonetheless made them **bit-exact** vs the reference
  (the `sKvalid` invalid→−inf fix + the `warp0 && t==0` epilogue-write-lane fix), at no extra cost. Natural-log
  domain; lonely query → out 0 / mx −inf / lse +inf.
- **`s_kv` size / split-KV — DEFERRED (as planned).** V1 is single-CTA-per-(query, head-block) streaming all
  gathered KV in KV_CHUNK=64 blocks; no `split_kv_combine`. Correct for the tested shapes (topk ≤ 2048).
  Split-KV is part of the deferred perf hillclimb, not correctness.
- **h_kv = 1 (MQA/MLA) — RESOLVED.** `indices`/`kv` carry the flat per-query gather; q has full h_q. The CTA
  grid is `s_q × ceil(h_q/16)`, each CTA owning one query token × a 16-head block — the head-group broadcast is
  implicit. (Note a perf consequence: K is re-gathered per head-block, 8× redundant at h_q=128 — a hillclimb
  target, not a correctness issue.)
- **`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` — RESOLVED.** The fork toggle intercepts inside
  `_forward_prefill_sparse` itself (the dispatch swap), which is downstream of *both* trigger conditions
  (`q.shape[0] > 11673` **and** the env), so it covers the path regardless of which fired.
- **Build — RESOLVED.** Local cu130 build (nvcc 13.1 → sm_120a) links `libcudart.so.13`; root-owned `build/`
  worked around with `--build-temp /tmp/dsv4-prefill-build --build-lib /tmp/dsv4-prefill-lib`. All 6 TUs
  compile clean; ptxas: 138 regs, 0 spills, 86 KB smem.
- **NEW — perf is latency/occupancy-bound (open, deferred).** ~15–18 TFLOP/s flat across shapes; 1 block/SM
  (smem-bound). The structural levers (cp.async double-buffer, B_H head-amortization to kill the redundant K
  re-gather, split-KV) are the follow-up hillclimb, to be guided by the E2E numbers.

## 8. Deliverables (mirrors Quest 3's HMMA artifacts)
- **✅ DONE** — `csrc/sm120/prefill/*` (scalar ref `sparse_prefill_kernel_scalar_ref.cuh` + HMMA
  `sparse_prefill_kernel.cuh` + `sparse_prefill_instantiation.cu` + `sparse_prefill.h`), `csrc/api/
  sparse_prefill.{cpp,h}`, `m.def("sparse_prefill_fwd", …)` in `api/api.cpp`, `ops.py` export, `setup.py`
  source + `-DDSV4_PREFILL_USE_HMMA` flag, `_patch.py` `_patch_sgl_kernel_sparse_prefill()` hook.
- **✅ DONE** — `tests/reference.py` (`sparse_prefill_reference` + `make_fake_prefill_batch`) +
  `tests/test_sparse_prefill.py` (26 passed).
- **✅ DONE** — SGLang fork: `flash_mla_sparse_prefill_sm120.py` (the `SGLANG_SM120_SPARSE_PREFILL` resolver +
  three backends + chunked torch fallback); `_forward_prefill_sparse` dispatches through it so SM120 never
  calls the stock SM90a/SM100f kernel. Safe default verified across a 10-case matrix.
- **✅ DONE** — Deploy-guide update: prefill toggle in the launch block, env-var table, and pipeline diagram,
  next to `SGLANG_SM120_SPARSE_DECODE`. (The `chunked-prefill-size` note is left as-is for now; the >11673 path
  is now *safe*, so dropping the cap is an E2E-time decision, not a correctness requirement.)
- **⬜ PENDING (user-launched)** — rtx-pro-6000-bench: the full 2K–64K sweep completing (the Quest-3 bench that
  exposed the gap, now green) — the §6.3 E2E run.
- **⬜ OPEN (deferred)** — perf hillclimb (cp.async double-buffer, B_H head-amortization, split-KV); kernel is
  correct + race-free at ~17 TFLOP/s, optimization guided by the E2E numbers.

**Not committed yet:** all changes (fork + kernel repo + docs) are in the working tree; commit on request.

## 9. Perf characterization (op-level microbenchmarks, RTX PRO 6000, synthetic KV)

Three comparisons, all at the op level (isolated kernel timing, not E2E). Caveats: synthetic KV; the
"decode-as-prefill" number is a representative proxy (cost is topk-bound per token, ~independent of arena
size); absolute numbers carry the shared ~17 TFLOP/s occupancy wall.

**(a) Path A (prefill kernel) vs Path B (decode kernel) — both sparse, same math.** Verified the two ops
compute identical attention (feed one logical problem in each native format → cos 1.000000). Per-token cost is
**flat in N** for both (no fixed overhead; embarrassingly parallel over tokens), so the 11673 dispatch
boundary has **no perf discontinuity**. Path A is uniformly **~10 % cheaper per token** (it reads
pre-dequantised flat bf16; decode pays an inline FP8/E8M0 + RoPE unpack):

| h_q=128, topk=2048 | N=2048 | N=8192 | N=11673 | N=16384 | N=32768 |
|---|---|---|---|---|---|
| A prefill µs/tok | 29.8 | 29.6 | 29.6 | 29.6 | 29.6 |
| B decode  µs/tok | 30.7 | 32.7 | 32.7 | 32.7 | 32.6 |
| **B/A** | 1.03× | 1.10× | 1.11× | 1.11× | 1.10× |

So routing > 11673 to the prefill kernel is a *strict per-token win*, not just crash-avoidance. The
prefill-path's extra upfront `dequantize_k_cache_paged` is negligible (~0.2 ms/layer at the tested sizes vs
hundreds of ms of attention).

**(b) Sparse vs DENSE — where the sparsity actually pays.** A-vs-B can't show the sparsity gain (both are
sparse). Sweeping *context length* s_kv with a fixed top-k=2048 budget, against a real flash-SDPA dense
baseline (no s_kv² materialisation), s_q=2048, h_q=64:

| context s_kv | sparse µs/tok | dense µs/tok | **dense/sparse** |
|---|---|---|---|
| 8,192 | 15.4 | 12.5 | 0.8× |
| 16,384 | 15.4 | 25.2 | **1.6×** |
| 32,768 | 15.4 | 50.5 | **3.3×** |
| 65,536 | 15.4 | 101.1 | **6.6×** |
| 131,072 | 16.0 | 203.0 | **12.7×** |

Sparse is **flat** (per-token cost capped at the top-k budget, independent of context); dense grows
**linearly** with context. The gain is exactly `s_kv / topk` — crossover at s_kv≈topk, then unbounded. At
DSv4's real million-token / topk≈2048 operating point that's a ~500× asymptotic reduction in attention work.
This is *the reason the kernel exists*: it keeps long-prefill attention flat where dense would explode.

## 10. What's left to actually run E2E in sglang (gap analysis)

The kernel is correct + race-free and the integration code is written, but **two hard deployment blockers**
stand between "unit tests pass" and "server serves the > 11673 path on the HMMA kernel":

1. **🔴 Serving venv `~/.venvs/dsv4` is stale.** Its `deepseek_v4_kernel/cuda*.so` predates the prefill work
   (`hasattr(cuda, "sparse_prefill_fwd")` → **False**) and the editable install points at an old
   `/tmp/dsv4-e2e/...` clone, not the canonical repo where the kernel was built. **Fix:** rebuild/reinstall the
   package into `~/.venvs/dsv4` from the canonical repo (`pip install -e .` or copy the freshly-built `.so`),
   then confirm `from deepseek_v4_kernel.ops import sparse_prefill_fwd`.
2. **🔴 Serving sglang is a non-editable copy.** `…/site-packages/sglang/` is a plain directory (installed
   Jun 8), so the fork's new `flash_mla_sparse_prefill_sm120.py` and the edited `deepseek_v4_backend.py` are
   **not present** in the running server (`flash_mla_sparse_fwd_sm120` grep → 0 hits). The decode toggle only
   works because it was installed *before* this work. **Fix:** reinstall sglang from the fork into the serving
   venv (or sync the two changed files into site-packages).

**Dispatch mechanism (confirmed):** the DSv4 backend activates the kernel through the **in-tree resolver**
(`_forward_prefill_sparse` → `flash_mla_sparse_fwd_sm120` → `deepseek_v4_kernel.ops`), exactly like the decode
toggle — **no `_patch.install()` needed** on this path. The `_patch.py` hook only matters for the
`dsa_backend` / stock-sglang case (covered for free, but not on the critical path here). So once (1)+(2) are
reinstalled, `SGLANG_SM120_SPARSE_PREFILL=hmma` is the only runtime knob.

**Hardening done this session:** added op-level `TORCH_CHECK` guards (`sparse_prefill.cpp`) for the
contiguity the kernel hard-assumes — q d_qk stride==1, kv d_qk stride==1, **indices topk stride==1** (the
kernel reads `indices_base[off+tok]` directly), plus attn_sink/topk_length dtype+numel. sglang's current
`combined_indices` satisfies these, but unguarded they'd silently read garbage if a caller passed a
non-contiguous topk dim. Rebuilt clean, 26/26 tests still pass.

**Remaining validation (in priority order):**
- **Build-into-serving-venv smoke (🔴 gating):** after (1)+(2), launch the server with
  `SGLANG_SM120_SPARSE_PREFILL=hmma`, send one > 11673-token prefill, confirm no crash + sane output.
- **Real-tensor parity (🟡):** capture an actual `_forward_prefill_sparse` call's
  `(q_flat, workspace, combined_indices, combined_lens, attn_sink)` and diff kernel vs the torch reference on
  *live* shapes/strides (the unit tests use synthetic contiguous tensors).
- **Cross-arch parity (🟡, no box in this env):** our op vs stock `flash_mla_sparse_fwd` on a Hopper/SM100
  box, identical inputs.
- **E2E sweep (🟢 user-launched):** the §6.3 2K–64K × concurrency repro, now green.
- **Commit (🟢):** nothing is committed; both repos are working-tree only. Commit the kernel repo
  (`csrc/sm120/prefill/*`, `csrc/api/sparse_prefill.*`, op reg, `_patch.py`, `ops.py`, `setup.py`, tests) and
  the fork (`flash_mla_sparse_prefill_sm120.py` + backend dispatch) before/with the E2E run.

## 11. Relationship to prior quests
- **Quest 2:** native MXFP4 W4A4 MoE (shipped).
- **Quest 3:** upstreamed the MoE + HMMA sparse-**decode** (complete; branches/drafts live). Its E2E bench
  surfaced this prefill gap.
- **Quest 4 (this):** the missing SM120 sparse-**prefill** kernel — same HMMA approach, same out-of-tree
  repo, same toggle pattern — to make the full throughput sweep (and any >11673-token prefill batch) run on
  SM120 instead of crashing on the SM90a/SM100f-only stock kernel.
