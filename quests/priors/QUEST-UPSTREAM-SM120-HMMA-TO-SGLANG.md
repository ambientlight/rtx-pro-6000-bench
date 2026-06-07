# Quest: Upstream SM120 HMMA Kernels to SGLang

**Status**: Planning  
**Created**: 2026-05-24  
**Depends on**: `feat/hmma-tensor-core-sparse-decode` branch on `ambientlight/deepseek-v4-flash-sm120`  
**Target**: PR to `sgl-project/sglang`

---

## Goal

Eliminate the monkey-patch (`sitecustomize.py` + `PYTHONPATH=/dsv4`) by upstreaming our SM120 HMMA sparse decode kernel directly into SGLang. Any user with an RTX PRO 6000 / RTX 5090 (SM120) should be able to run DeepSeek-V4-Flash out of the box with `--attention-backend compressed`.

---

## Current State (Monkey-Patch)

```
sitecustomize.py
  → deepseek_v4_kernel/_patch.py:install()
    → replaces flash_mla.flash_mla_with_kvcache at module level
    → gates on: is_sparse AND is_sm120 AND is_fp8_kvcache AND bf16_query
    → routes to our CUDA extension (dsv4_kernel.sm120.launch_dsv4_sparse_decode_v32)
    → falls back to original flash_mla for non-SM120 / non-sparse
```

This works but requires mounting a build directory and setting `PYTHONPATH`. Not suitable for production or other SM120 users.

---

## SGLang Attention Dispatch Architecture

### Layer 1: Backend Selection
`attention_registry.py` → selects `DeepSeekV4RadixAttentionBackend` for compressed attention.

### Layer 2: Compressed Decode
`deepseek_v4_backend_radix.py` → `forward_decode()` → calls C4 indexer for sparse topk → calls `flash_mla_with_kvcache_entrypoint()`.

### Layer 3: Kernel Entrypoint
`debug_flash_mla_adapter.py`:
```python
def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    assert backend == "kernel"
    import flash_mla
    return flash_mla.flash_mla_with_kvcache(**kwargs)  # SM90 WGMMA only
```

### Layer 4: Indexer Kernel
`compressed/indexer.py` → `fp8_paged_mqa_logits` dispatch:
- `SGLANG_OPT_USE_TILELANG_INDEXER` → tilelang kernel (SM90)
- `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` → PyTorch fallback (what we use on SM120)
- Default → `deep_gemm.fp8_paged_mqa_logits` (SM90/SM100)

---

## PR Scope

### Files to Create

| File | Contents |
|---|---|
| `sgl_kernel/csrc/sm120/decode/sparse_decode_kernel.cuh` | HMMA tensor-core sparse decode kernel (KV64, 8 warps, reg-O, split-KV) |
| `sgl_kernel/csrc/sm120/decode/split_kv_combine.cuh` | Split-KV online-softmax combine kernel |
| `sgl_kernel/csrc/sm120/decode/sparse_decode_instantiation.cu` | CUDA translation unit + launch function |
| `sgl_kernel/csrc/sm120/decode/sparse_decode.h` | Public header (params struct, launch declaration) |
| `sgl_kernel/csrc/sm120/common/defines.h` | SM120 constants, macros |
| `sgl_kernel/csrc/sm120/common/params.h` | `SparseAttnDecodeParams` struct |
| `sgl_kernel/csrc/sm120/common/cutlass_shim.h` | Minimal cutlass bf16 type shim |
| `sgl_kernel/csrc/sm120/api/sparse_decode.cpp` | C++ API: adaptive split-KV logic, tensor validation, launch |
| `sgl_kernel/python/sgl_kernel/sm120_sparse_decode.py` | Python binding (torch extension) |

### Files to Modify

| File | Change | Difficulty |
|---|---|---|
| `debug_flash_mla_adapter.py` | Add arch dispatch: `if SM120 and sparse → sm120 kernel; else → flash_mla` | Easy (~10 lines) |
| `compressed/indexer.py` ~L376 | Add SM120 branch for indexer kernel (use torch fallback, or port tilelang) | Easy |
| `compressed/indexer.py` ~L391 | Fix `seq_lens` 2D→1D squeeze bug (our patch already does this) | Easy (1 line) |
| `sgl_kernel/CMakeLists.txt` | Add SM120 CUDA sources with `-gencode arch=compute_120,code=sm_120a` | Medium |
| `sgl_kernel/setup.py` or `pyproject.toml` | Add SM120 compilation target | Medium |
| `deepseek_v4_backend_radix.py` | Cache arch check at init time, pass to entrypoint | Easy |

### Files to Add (Tests)

| File | Contents |
|---|---|
| `tests/test_sm120_sparse_decode.py` | Correctness: compare SM120 HMMA output vs torch reference on random inputs |
| `tests/test_sm120_split_kv.py` | Correctness: split-KV combine produces same output as single-CTA |
| `benchmarks/sm120/bench_sparse_decode.py` | Performance: scalar vs HMMA vs split-KV at various context lengths |

---

## Implementation Steps

### Phase 1: Kernel Package Integration
1. [ ] Fork `sgl-project/sglang`, create branch `feat/sm120-hmma-sparse-decode`
2. [ ] Copy kernel sources from `ambientlight/deepseek-v4-flash-sm120/csrc/sm120/` into `sgl_kernel/csrc/sm120/`
3. [ ] Add CMake rules for SM120 compilation (`-gencode arch=compute_120,code=sm_120a`)
4. [ ] Add Python binding in `sgl_kernel/python/sgl_kernel/sm120_sparse_decode.py`
5. [ ] Verify `sgl_kernel` builds with SM120 target on our machine

### Phase 2: Dispatch Integration
6. [ ] Modify `debug_flash_mla_adapter.py` to add SM120 arch dispatch
7. [ ] Fix `seq_lens` squeeze bug in `compressed/indexer.py`
8. [ ] Add SM120 indexer fallback selection (auto-detect, use torch path)
9. [ ] Cache device capability at backend init, not per-call

### Phase 3: Testing
10. [ ] Write correctness tests (random inputs, compare to torch reference)
11. [ ] Write split-KV correctness test (n_splits=1 vs n_splits=32)
12. [ ] Run existing SGLang DeepSeek-V4 tests on SM120
13. [ ] Benchmark: HMMA vs scalar, with/without split-KV

### Phase 4: PR & Review
14. [ ] Clean up kernel code (remove debug prints, trace macros)
15. [ ] Write PR description with benchmark results
16. [ ] Address reviewer feedback (likely: build system, coding style, test coverage)

---

## Hard Parts & Risks

### 1. sgl_kernel Build System
SGLang's `sgl_kernel` is pre-compiled in Docker images. Adding SM120 means either:
- **Fat binary** with all SM targets (increases image size ~50MB)
- **JIT compile** at first launch (slow startup, ~2 min)
- **Separate wheel** `sgl_kernel-sm120` (fragmented packaging)

**Recommendation**: Fat binary. SM120 kernels are small (~100KB compiled). The Docker image is already 15GB+.

### 2. PTX Inline Assembly
Our kernel uses raw PTX `mma.sync.aligned.m16n8k16` via inline asm. SGLang might prefer CUTLASS or higher-level abstraction. But:
- CUTLASS 3.x targets SM90 WGMMA, not SM120 HMMA
- CUTLASS 2.x `mma.sync` wrappers exist but are unmaintained
- Raw PTX is the only reliable path for SM120 HMMA today

**Recommendation**: Ship raw PTX with thorough comments. It's what NVIDIA's own examples do for m16n8k16.

### 3. flash_mla Ownership
`flash_mla` is `deepseek-ai/FlashMLA` on GitHub — the actual sparse decode kernel package. SGLang Docker images build it from source (`git clone https://github.com/deepseek-ai/FlashMLA.git`). The repo looks stale (no commits for 4 months) but that's because V4-Flash reuses the same MLA architecture as V3 — the kernels were already complete for SM90/SM100.

**Two upstream paths:**
- **(a)** PR to `deepseek-ai/FlashMLA` to add SM120 HMMA — **cleanest**. Adds `sparse_decode_fwd` SM120 implementation alongside existing SM90 WGMMA. All SGLang users with SM120 automatically benefit without any SGLang changes. But repo may not be actively reviewing external PRs.
- **(b)** PR to `sgl-project/sglang` to bypass flash_mla for SM120, route to sgl_kernel — **more reliable**. Precedent: SGLang already has `triton_attention` as an alternative to FlashInfer. Doesn't depend on DeepSeek team reviewing.
- **(c)** Both: FlashMLA PR for the kernel, SGLang PR for the dispatch + fallback.

**Recommendation**: Try (a) first since it's the right layer. Fall back to (b) if DeepSeek team is unresponsive.

### 4. Indexer Kernel on SM120
The C4 indexer uses `deep_gemm` (SM90+) or `tilelang` (SM90). Neither works on SM120. We use the PyTorch fallback (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`). This is functional but slower.

**For the initial PR**: Use torch fallback on SM120 (auto-detected). Writing an SM120-native indexer is a separate follow-up.

### 5. Reviewer Expectations
SGLang is actively maintained with high code quality standards. Expect:
- Style conformance (black, isort, type hints)
- Comprehensive docstrings
- CI must pass (may need SM120 runner or skip-if-not-SM120)
- Benchmark data in PR description

---

## Dispatch Code (Target State)

```python
# debug_flash_mla_adapter.py (or renamed to flash_mla_dispatch.py)

_sm120_available = None

def _is_sm120():
    global _sm120_available
    if _sm120_available is None:
        major, minor = torch.cuda.get_device_capability()
        _sm120_available = (major == 12 and minor == 0)
    return _sm120_available

def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    assert backend == "kernel"
    
    is_sparse = kwargs.get("indices") is not None
    
    if _is_sm120() and is_sparse:
        from sgl_kernel.sm120_sparse_decode import sparse_decode_fwd
        return sparse_decode_fwd(**kwargs)
    else:
        import flash_mla
        return flash_mla.flash_mla_with_kvcache(**kwargs)
```

---

## Source Files Reference

All kernel sources are in `ambientlight/deepseek-v4-flash-sm120` branch `feat/hmma-tensor-core-sparse-decode`:

| File | Lines | What |
|---|---|---|
| `csrc/sm120/decode/sparse_decode_kernel.cuh` | ~600 | Production HMMA kernel |
| `csrc/sm120/decode/split_kv_combine.cuh` | ~120 | Split-KV combine |
| `csrc/sm120/decode/sparse_decode_instantiation.cu` | ~65 | Launch entry point |
| `csrc/sm120/decode/sparse_decode.h` | ~30 | Header |
| `csrc/sm120/common/defines.h` | ~40 | Constants |
| `csrc/sm120/common/params.h` | ~80 | Params struct |
| `csrc/api/sparse_decode.cpp` | ~200 | Adaptive split-KV + validation |
| `csrc/ops.py` | ~50 | Python binding |
| `_patch.py` | ~160 | Monkey-patch (reference for dispatch logic) |

---

## Estimated Effort

| Phase | Time | Notes |
|---|---|---|
| Phase 1 (kernel integration) | 1 day | Mostly copy + build system |
| Phase 2 (dispatch) | 0.5 day | Small code changes |
| Phase 3 (tests) | 1 day | Correctness + benchmarks |
| Phase 4 (PR + review) | 1-2 weeks | Review cycles with SGLang maintainers |
| **Total** | ~2-3 days coding + review time | |

---

## File Locations

| What | Path |
|---|---|
| Our kernel sources | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/csrc/sm120/` |
| Our API layer | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/csrc/api/` |
| Our monkey-patch | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker/_patch.py` |
| SGLang attention backends | (in Docker) `/workspace/sglang/python/sglang/srt/layers/attention/` |
| SGLang compressed backend | (in Docker) `/workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` |
| SGLang flash_mla adapter | (in Docker) `/workspace/sglang/python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` |
| SGLang indexer | (in Docker) `/workspace/sglang/python/sglang/srt/layers/attention/compressed/indexer.py` |
| sgl_kernel build | (in Docker) `/workspace/sglang/sgl-kernel/` |
| Tuned FP8 configs | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/tuned-configs-final/` |
| This quest | `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/QUEST-UPSTREAM-SM120-HMMA-TO-SGLANG.md` |
