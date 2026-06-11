# Quest: Run Official DeepSeek-V4-Flash FP4+FP8 Checkpoint on SM120

**Status**: FP4 + HMMA kernel working! 45.5 tok/s decode (vs 67 FP8). 25 GB less VRAM, 14 GB headroom, no OOM. Production-viable for stability-critical workloads.  
**Created**: 2026-05-26  
**Hardware**: 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96GB each, PCIe)  
**Model**: `deepseek-ai/DeepSeek-V4-Flash` (official FP4+FP8 mixed checkpoint)  
**Docker image built**: `sglang-dsv4-sm120:latest` (CUDA 13.0.1, FlashInfer 0.6.11, SGLang latest)

---

## Why This Matters

The official FP4+FP8 checkpoint uses FP4 for MoE expert weights, saving ~24 GB per GPU:

| Metric | FP8 Repack (current) | FP4+FP8 Official | Delta |
|---|---|---|---|
| Model weights per GPU | 69.86 GB | **45.66 GB** | **-24.2 GB** |
| Available after weights | 23.60 GB | **48.40 GB** | **+24.8 GB** |
| KV pool (0.85 fraction) | 644K full + 64K SWA | **2.27M full + 227K SWA** | **3.5× larger** |
| Headroom for transient allocs | 3.54 GB | **~14.5 GB** | **4× more** |

This would **eliminate all OOM crashes** (tilelang 5.34 GB, flash-MLA 1.92 GB both fit trivially in 14.5 GB headroom) and allow much higher concurrency.

---

## What Works

1. ✅ Docker image `sglang-dsv4-sm120:latest` built from source with CUDA 13.0.1 + FlashInfer 0.6.11 + SGLang latest + SM120 in TORCH_CUDA_ARCH_LIST
2. ✅ Official checkpoint downloads (149 GB vs 274 GB FP8 repack)
3. ✅ Model loads and weights recognized as FP4+FP8
4. ✅ FP4 expert weight shuffling works: `Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel`
5. ✅ KV cache pool allocates correctly with massive headroom: `avail mem=48.40 GB`
6. ✅ Pool sizes: `full=2272512, swa=227072` (3.5× larger than FP8 config)
7. ✅ FlashInfer has SM120 CUTLASS FP4 MoE module: `gen_cutlass_fused_moe_sm120_module` with `-DENABLE_FP4 -gencode=sm_120f`
8. ✅ SM120 has native FP4 block-scaled MMA via `mma.sync.aligned...kind::mxf4.block_scale` (PTX ISA 9.2)
9. ✅ No Expert Parallelism needed (TP=4, EP=1, all 256 experts replicated)
10. ✅ `triton_kernels` package available in container (NVIDIA's proprietary Triton MoE library)

---

## Current Blocker

### `deep_gemm` SF Layout Transform Crash

```
File "deepseek_v4.py", line 1472, in _setup_fp8_wo_a_scales
    attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
File "deep_gemm/__init__.py", line 245, in transform_sf_into_required_layout
    return _C.transform_sf_into_required_layout(sf, mn, k, recipe_a, recipe_b, recipe_c, ...)
tvm.error.InternalError: Assertion error (/deepgemm/csrc/apis/layout.hpp:59): Unknown SF transformation
```

**Where**: `deepseek_v4.py:post_load_weights()` → `_setup_fp8_wo_a_scales()` → `deep_gemm.transform_sf_into_required_layout()`

**What it does**: After loading the non-expert FP8 weights, SGLang calls `deep_gemm` to transform the scale factor tensors into the layout required for FP8 GEMM execution. The function maps `(recipe_a, recipe_b, recipe_c)` to a specific layout transformation.

**Why it fails**: The official checkpoint's FP8 scale factors use a format/recipe that `deep_gemm` on SM120 doesn't recognize. The `deep_gemm` library has recipes for SM90/SM100 but SM120 may need different recipes, or the specific combination of recipes from the official checkpoint isn't in deep_gemm's lookup table.

**Key detail**: This crash happens during **weight loading**, not during inference. Even with `--fp8-gemm-backend triton`, deep_gemm is still called to transform the scale layout. The Triton FP8 GEMM backend bypasses deep_gemm for the actual matmul, but the weight post-processing still uses it.

**Affected weights**: `attn.wo_a.weight_scale_inv` — the FP8 scale factors for the attention output projection (non-expert weight). Every layer's `wo_a` hits this.

---

## Previous Attempts

### Attempt 1: Old Docker image (SGLang 0.5.10rc0) + FP4 checkpoint

| Backend | Result | Error |
|---|---|---|
| `--moe-runner-backend triton` (auto) | ❌ Hidden size mismatch | Triton expects unpacked FP8, gets FP4 packed uint8 |
| `--moe-runner-backend flashinfer_mxfp4` | ❌ SM100 cubin crash | Routes to `trtllm_fp4_block_scale_moe` → SM100 cubins |
| `--moe-runner-backend flashinfer_cutlass` | ❌ Missing `.runner` attribute | `Fp8MoEMethod` doesn't have runner interface for this backend |
| `--moe-runner-backend flashinfer_cutedsl` | ❌ Missing `.runner` attribute | Same — old SGLang doesn't wire FP4→CuteDSL |

**Root cause**: SGLang 0.5.10rc0 doesn't have the SM120+MXFP4 routing code that exists in 0.5.12+.

### Attempt 2: In-place pip upgrade to SGLang 0.5.12

| Approach | Result | Error |
|---|---|---|
| `pip install sglang==0.5.12.post1 --no-deps` | ❌ FlashInfer import crash | `flashinfer.comm.cuda_ipc` needs newer FlashInfer |
| `pip install sglang[all]==0.5.12.post1` | ❌ `libnvrtc.so.13` missing | deep_gemm needs CUDA 13.x, container has 12.9 |

**Root cause**: Can't upgrade SGLang without also upgrading CUDA toolkit + FlashInfer.

### Attempt 3: Full Docker rebuild (CUDA 13.0.1 + SGLang latest + FlashInfer 0.6.11)

| Step | Result |
|---|---|
| Docker build | ✅ Success — `sglang-dsv4-sm120:latest` |
| Weight loading | ✅ FP4 experts shuffle correctly |
| Scale layout transform | ❌ **`deep_gemm` Unknown SF transformation** |

**Root cause**: deep_gemm's `transform_sf_into_required_layout` doesn't handle the scale format in the official checkpoint on SM120.

---

## Hypothesized Fixes (Ordered by Likelihood)

### Fix 1: Patch `deepseek_v4.py` to Skip deep_gemm Transform for Triton Backend

The simplest fix. When `fp8_gemm_backend=triton`, the scale factors don't need deep_gemm's layout transformation — Triton handles raw scales directly.

**File**: `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py`  
**Function**: `_setup_fp8_wo_a_scales()` (line ~1472)  
**Change**: Guard `transform_sf_into_required_layout()` call behind `if fp8_gemm_backend != 'triton'`

```python
# Before:
attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(...)

# After:
if get_fp8_gemm_runner_backend().is_deep_gemm():
    attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(...)
# else: leave scales in their original layout for Triton
```

**Risk**: Triton FP8 GEMM may expect a specific scale layout. Need to verify.  
**Effort**: 5 minutes to patch + test.

### Fix 2: Update deep_gemm Version

The `deep_gemm` in the Docker image may be too old for SM120 SF recipes. Check if a newer version has SM120 recipe support.

**Check**: `pip show deep-gemm` in the container, compare with latest PyPI/GitHub version.  
**Effort**: 10 minutes.

### Fix 3: Add SM120 Recipe to deep_gemm

If the recipe is genuinely missing, add it to `deep_gemm/csrc/apis/layout.hpp` line 59. The layout.hpp file has a switch/if-else on `(recipe_a, recipe_b, recipe_c)` combinations.

**Check**: Inspect the assertion and find which recipe combination is missing.  
**Effort**: 30 minutes if the layout logic is straightforward.

### Fix 4: Use `--json-model-override-args` to Change Scale Format

The official checkpoint uses `"scale_fmt": "ue8m0"`. The FP8 repack may use a different format. Can we override the scale format during loading?

**Check**: Compare `config.json` between official and FP8 repack.  
**Effort**: 15 minutes.

### Fix 5: Convert Official Checkpoint's FP8 Scales to Match FP8 Repack Format

Pre-process the checkpoint to convert the non-expert FP8 scales from the official format to the format our stack understands.

**Effort**: 1-2 hours.

---

## Architecture Context from Deep Research

### SM120 FP4 Instruction Path (Verified)
- PTX ISA 9.2 documents `mma.sync.aligned.m16n8k64.kind::mxf4.block_scale` on `sm_120a`
- This is **different** from SM100's `tcgen05` FP4 path
- SM120 cubins cannot run SM100 FP4 kernels and vice versa
- FlashInfer's `gen_cutlass_fused_moe_sm120_module` targets `sm_120f` correctly

### SGLang SM120+MXFP4 Routing (In Latest SGLang)
```python
# server_args.py (SGLang 0.5.12+)
elif is_sm120_supported() and is_mxfp4_quant_format:
    self.moe_runner_backend = "triton_kernel"
```
Uses `triton_kernel` (NVIDIA's proprietary `triton_kernels` package) for FP4 MoE dispatch on SM120.

### MoE Backend Options Available
| Backend | FP4 Support | SM120 Support | Status |
|---|---|---|---|
| `triton_kernel` | ✅ (NVIDIA triton_kernels) | ✅ (auto-selected) | Available, untested |
| `flashinfer_mxfp4` | ✅ | ❌ (routes to SM100 cubins) | Broken on SM120 |
| `flashinfer_cutlass` | ✅ (via CUTLASS SM120) | Partial | Missing runner wiring |
| `flashinfer_cutedsl` | ✅ (via CuteDSL b12x) | ✅ (FlashInfer 0.6.9+) | Available in latest |

---

## Files Inventory

### Docker Image
- **Image**: `sglang-dsv4-sm120:latest` (built 2026-05-26)
- **Base**: `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`
- **SGLang**: latest main branch
- **FlashInfer**: 0.6.11.post1
- **CUDA arch**: `9.0;10.0;10.3;12.0`
- **Build Dockerfile**: `/mnt/hot/ambientlight/repos/sglang-latest/docker/Dockerfile` (patched with 12.0 in arch list)

### Model Checkpoints
- **Official FP4+FP8**: `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash/` (149 GB, 46 safetensors)
- **FP8 repack** (working): `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8/` (274 GB, 46 safetensors)

### Key Source Files to Inspect/Patch
- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py` — `_setup_fp8_wo_a_scales()` line ~1472
- `/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py` — `Fp8MoEMethod`
- `/sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4.py` — `DeepSeekMxfp4MoEMethod` + `_patch_sm120_mxfp4_min_warps()`
- `/usr/local/lib/python3.12/dist-packages/deep_gemm/csrc/apis/layout.hpp` line 59 — SF transform assertion
- `/sgl-workspace/sglang/python/sglang/srt/server_args.py` line ~2073 — SM120+MXFP4 auto-detection

### SM120 HMMA Decode Kernel Patch
- **Patch dir**: `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker/`
- Mounted as `-v .../build-docker:/dsv4:ro -e PYTHONPATH=/dsv4`
- Works with both FP4 and FP8 checkpoints (attention kernel is independent of expert quantization)

---

## Reproduction Commands

### Build Docker Image (already done)
```bash
cd /mnt/hot/ambientlight/repos/sglang-latest
# Patch: add 12.0 to arch list in docker/Dockerfile
sed -i "s/'9.0;10.0;10.3'/'9.0;10.0;10.3;12.0'/g" docker/Dockerfile
docker build -f docker/Dockerfile -t sglang-dsv4-sm120:latest \
  --build-arg CUDA_VERSION=13.0.1 \
  --build-arg BUILD_AND_DOWNLOAD_PARALLEL=64 \
  --build-arg USE_LATEST_SGLANG=1 .
```

### Launch (hits the blocker)
```bash
docker run -d --name sglang-dsv4-fp4-sm120 \
  --gpus all --privileged --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v /mnt/hot/ambientlight/models/DeepSeek-V4-Flash:/workspace/model:ro \
  -v /mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker:/dsv4:ro \
  -e PYTHONPATH=/dsv4 \
  -e SGLANG_DSV4_FP4_EXPERTS=1 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_THINKING=1 -e SGLANG_REASONING_EFFORT=max \
  -e NCCL_PROTO=LL -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=8 -e NCCL_NTHREADS=512 \
  sglang-dsv4-sm120:latest \
  bash -c "pip install distro -q && python3 -m sglang.launch_server \
    --model-path /workspace/model --host 0.0.0.0 --port 8000 \
    --served-model-name deepseek-v4-flash --trust-remote-code \
    --tensor-parallel-size 4 --context-length 262144 \
    --mem-fraction-static 0.85 --max-running-requests 8 \
    --kv-cache-dtype fp8_e4m3 --attention-backend compressed \
    --fp8-gemm-backend triton \
    --chunked-prefill-size 32768 --page-size 256 --cuda-graph-max-bs 32 \
    --enable-return-routed-experts --disable-custom-all-reduce \
    --disable-shared-experts-fusion"
```

### Debug the Blocker
```bash
# 1. Inspect deep_gemm version and recipe table
docker exec sglang-dsv4-fp4-sm120 python3 -c "import deep_gemm; print(deep_gemm.__version__)"
docker exec sglang-dsv4-fp4-sm120 python3 -c "
from deep_gemm import get_supported_recipes
print(get_supported_recipes())  # if this API exists
"

# 2. Check what recipe the official checkpoint expects
docker exec sglang-dsv4-fp4-sm120 python3 -c "
import torch, safetensors.torch
# Load one scale tensor and inspect its shape/dtype
t = safetensors.torch.load_file('/workspace/model/model-00001-of-00046.safetensors')
for k,v in t.items():
    if 'scale' in k and 'wo_a' in k:
        print(k, v.shape, v.dtype)
        break
"

# 3. Compare with FP8 repack's scales
# ...
```

---

## Next Steps (In Order)

1. **Debug the SF recipe**: Inspect what `(recipe_a, recipe_b, recipe_c)` the official checkpoint produces and why deep_gemm rejects it
2. **Try Fix 1**: Patch `deepseek_v4.py` to skip deep_gemm transform when using Triton FP8 backend
3. **If Fix 1 works**: Test FP4 MoE dispatch (the SM120 `triton_kernel` path or CuteDSL path)
4. **If Fix 1 doesn't work**: Try Fix 2 (update deep_gemm) or Fix 3 (add SM120 recipe)
5. **Benchmark**: Compare FP4+FP8 vs FP8-only on decode throughput, TTFT, and OOM stability

---

## Fallback

If native FP4 proves too complex, the deep research identified a **FP4→FP8 conversion path**:
- DeepSeek's own inference README supports `--expert-dtype fp8` to convert FP4 experts to FP8 at load time
- This would use the official checkpoint structure but dequant experts to FP8
- Memory savings would be smaller (~10 GB instead of ~24 GB) but still meaningful
- Uses the proven FP8 MoE Triton path

---

### Benchmark Results (2026-05-26)

FP4 path from PR #24692 works but is dramatically slower than our optimized FP8 stack:

**Prefill (TTFT):**

| Context | FP8 + HMMA + Tuned | FP4 + PR Marlin/Triton | Delta |
|---:|---:|---:|---:|
| 256 | **287ms** | 2,183ms | 7.6× slower |
| 1K | **332ms** | 5,253ms | 15.8× slower |
| 4K | **991ms** | 19,260ms | 19.4× slower |
| 8K | **1,363ms** | 26,137ms | 19.2× slower |

**Decode (Steady-State ITL):**

| Context | FP8 + HMMA + Tuned | FP4 + PR Marlin/Triton | Delta |
|---:|---:|---:|---:|
| 256 | **67.6 tok/s** | 21.9 tok/s | 3.1× slower |
| 1K | **60.8 tok/s** | 21.6 tok/s | 2.8× slower |
| 4K | **53.7 tok/s** | 21.7 tok/s | 2.5× slower |
| 8K | **51.9 tok/s** | 21.5 tok/s | 2.4× slower |

**Root cause of slowness:**
- MoE backend: `marlin` (software FP4→BF16 dequant, no tensor core acceleration)
- Attention: PR's Triton FlashMLA (no HMMA tensor cores, no split-KV)
- No tuned kernel configs (all "sub-optimal" warnings)
- No CUDA graph optimization for decode

**Stability:** OOMs at 16K context — FlashMLA tries to allocate 16.42 GiB internal buffers. Even with 14 GB headroom, not enough.

**Conclusion:** FP4 gives 25 GB memory savings but at a steep 3-19× performance cost with current software fallback kernels. The ideal configuration would be FP4 weights + SM120 native FP4 tensor-core MoE kernels (`mma.sync.aligned.kind::mxf4.block_scale`) + our HMMA attention kernel — but that combination doesn't exist yet. **FP8 repack + our optimized HMMA+tuned stack remains the production choice.**

### Benchmark: FP4 + HMMA Kernel (2026-05-26)

Docker image `sglang-fp4-hmma-sm120:latest` with:
- PR #24692 SGLang (FP4 weight loading, Marlin MoE, SM120 dispatch)
- Our HMMA attention kernel (patched into `flash_mla_sm120.py`)
- Tuned W8A8 FP8 configs for non-expert linear layers
- CUDA graphs enabled, all tilelang disabled

**Decode (Steady-State ITL):**

| Context | FP8+HMMA+Tuned | FP4+HMMA | FP4+Triton | FP4 no graphs |
|---:|---:|---:|---:|---:|
| 256 | **67.6 tok/s** | **45.5 tok/s** | 27.0 tok/s | 4.8 tok/s |
| 1K | **60.8 tok/s** | **40.0 tok/s** | 26.9 tok/s | 4.8 tok/s |
| 4K | **53.7 tok/s** | **36.7 tok/s** | 26.6 tok/s | 4.9 tok/s |
| 8K | **51.9 tok/s** | **37.6 tok/s** | 26.6 tok/s | 4.9 tok/s |

**HMMA kernel contribution**: 1.7× speedup over PR's Triton FlashMLA (45 vs 27 tok/s at 256 ctx)

**FP4+HMMA vs FP8+HMMA**: 67% of FP8 speed. The remaining gap is Marlin MoE (software FP4 dequant) vs our tuned Triton FP8 MoE.

**Memory advantage**: 25 GB less VRAM, 14 GB headroom, 235K SWA pool — no OOM crashes

**Production assessment**: 45.5 tok/s single-stream with no OOM risk is viable for SWE-bench and coding workloads where stability > peak speed.

### Config that works**
```bash
# PR image: sglang-dsv4-sm120-fp4:latest
# Key flags:
SGLANG_OPT_FP8_WO_A_GEMM=0           # Skip deep_gemm SF transform
SGLANG_OPT_USE_TILELANG_INDEXER=0     # Disable tilelang indexer
SGLANG_OPT_USE_TILELANG_SWA_PREPARE=0 # Disable all tilelang
SGLANG_OPT_USE_TILELANG_MHC_PRE=0
SGLANG_OPT_USE_TILELANG_MHC_POST=0
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1  # Torch fallback (SM120 variant is graph-safe)
# tilelang stub fix: ln -sf .../libcudart.so .../tilelang/lib/libcudart_stub.so
```

**Why FP4 is 2.5× slower than FP8+HMMA:**
1. No HMMA attention kernel — PR uses Triton FlashMLA sparse decode
2. Marlin MoE — software FP4 dequant instead of tensor-core MMA
3. No tuned FP8 GEMM configs for non-expert layers

**Path to FP4 matching FP8 speed:**
1. Port our HMMA attention kernel to the PR Docker image
2. Replace Marlin MoE with our native FP4 tensor-core GEMM (41.3 TFLOPS)
3. Mount tuned W8A8 configs for non-expert linear layers

### Docker Images Built

| Image | Base | SGLang | Use |
|---|---|---|---|
| `sglang-dsv4-sm120:latest` | CUDA 13.0.1 | latest main | General SM120 testing |
| `sglang-dsv4-sm120-fp4:latest` | CUDA 13.0.1 | PR #24692 branch | FP4 testing (slow MoE) |
| `lmsysorg/sglang:deepseek-v4-blackwell` | CUDA 12.9 | 0.5.10rc0 | **Production** (FP8+HMMA) |

### Upstream PR Discovery (2026-05-26)

Found **PR #24692** (`AliceChenyy/sglang:sm120-dsv4-rebase`) — "feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference". This PR adds:
- `mxfp4_moe_sm120_triton.py` — Triton fused MXFP4 dequant + GEMM for MoE experts (4.1× vs PyTorch per-GEMM)
- `flash_mla_sm120_triton.py` — Triton FlashMLA sparse decode for SM120
- CUDA graph capture support
- Proper TP sharding for FP4 packed weights on SM120

Also found PR #24303 by a different author with similar goals.

Building Docker image from PR #24692 branch instead of patching ourselves. This is the most robust path since the PR author has tested the full SM120 FP4 stack end-to-end.

**Build command:**
```bash
cd /mnt/hot/ambientlight/repos/sglang-sm120-pr24692
sed -i "s/'9.0;10.0;10.3'/'9.0;10.0;10.3;12.0'/g" docker/Dockerfile
docker build -f docker/Dockerfile -t sglang-dsv4-sm120-fp4:latest \
  --build-arg CUDA_VERSION=13.0.1 \
  --build-arg BUILD_AND_DOWNLOAD_PARALLEL=64 \
  --build-arg BRANCH_TYPE=local .
```

### All Blockers Found and Resolved

| # | Blocker | Status | Fix |
|---|---|---|---|
| 1 | `deep_gemm` SF layout crash | ✅ Solved | `SGLANG_OPT_FP8_WO_A_GEMM=0` |
| 2 | SM120 HMMA kernel for CUDA 13 | ✅ Solved | Rebuilt `.so` with CUDA 13.0.1 |
| 3 | `tilelang libcudart_stub.so` | ✅ Solved | `ln -sf .../libcudart.so .../tilelang/lib/libcudart_stub.so` |
| 4 | Scale TP shard_dim flipped | ✅ Solved | `if "scale" not in weight_name` guard |
| 5 | Weight TP narrow out of range | ✅ **Solved by PR** | PR #24692 has proper SM120 MXFP4 MoE weight loading |

---

## Related Quests
- `QUEST-DEEPSEEK-V4-FLASH-HILLCLIMB.md` — main optimization quest (FP8 path)
- `QUEST-UPSTREAM-SM120-HMMA-TO-SGLANG.md` — upstreaming HMMA attention kernel
- `GUIDE-INFERENCE-TUNING-SM120.md` — production tuning guide (FP8 config)
- `deep_research_sm120_fp4_moe_gpt54.md` / `gpt55` — deep research on SM120 FP4 MoE paths
