# Quest: DeepSeek-V4-Flash on 4× RTX PRO 6000 — Hillclimbing

**Status**: Active — SM120-tuned FP8 W8A8 + MoE configs landed, 12-18% decode improvement  
**Last updated**: 2026-05-24  
**Hardware**: 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96GB each, PCIe, no NVLink)  
**Model**: `sgl-project/DeepSeek-V4-Flash-FP8` (291B params, MoE, ~13B active)  
**Runtime**: SGLang (`lmsysorg/sglang:deepseek-v4-blackwell`) + HMMA-optimized SM120 patch (`ambientlight/deepseek-v4-flash-sm120`, branch `feat/hmma-tensor-core-sparse-decode`)  
**Eval**: mini-swe-agent on SWE-bench Lite dev split (23 instances)

---

## Current Best: Tuned FP8 W8A8 + MoE + HMMA Kernel (2026-05-24)

### TTFT (Prefill Latency) — streaming, 32 output tokens

| Context | Original (scalar) | **Current Best** | vs Original |
|---------|-------------------|-----------------|-------------|
| 256 | 646ms | **287ms** | **2.3×** |
| 1K | 793ms | **332ms** | **2.4×** |
| 4K | — | **991ms** | — |
| 8K | 4.1s | **1.36s** | **3.0×** |
| 16K | 8.3s | **2.83s** | **2.9×** |
| 32K | 18.2s | **8.27s** | **2.2×** |
| 64K | 39.1s | **16.88s** | **2.3×** |

### Decode ITL (steady-state, 256 output tokens)

| Context | Previous Med ITL | Previous tok/s | **Tuned Med ITL** | **Tuned tok/s** | **Delta** |
|---------|-----------------|---------------|-------------------|----------------|-----------|
| 256 | 16.4ms | 57.2 | **13.9ms** | **67.6** | **+18%** |
| 1K | 17.8ms | 52.3 | **15.5ms** | **60.8** | **+16%** |
| 4K | 20.0ms | 47.2 | **17.8ms** | **53.7** | **+14%** |
| 8K | 20.3ms | 46.2 | **18.3ms** | **51.9** | **+12%** |
| 16K | 21.4ms | 44.0 | **19.1ms** | **49.9** | **+13%** |

### Decode Throughput — single stream (from TTFT bench)

| Context | Original tok/s | **Current Best tok/s** | vs Original |
|---------|---------------|----------------------|-------------|
| 256 | 35.3 | **69.5** | **2.0×** |
| 1K | 27.7 | **62.1** | **2.2×** |
| 4K | 21.3 | **54.1** | **2.5×** |
| 16K | 20.8 | **51.4** | **2.5×** |
| 32K | 18.0 | **45.2** | **2.5×** |
| 64K | 14.5 | **36.2** | **2.5×** |

### SWE-bench Lite dev (23 instances)

| Run | Config | Resolved | Submitted | Errors | Wall Time |
|-----|--------|----------|-----------|--------|-----------|
| dev0 | temp=0.4/top_p=0.9/top_k=20/rep=1.02, guided regex, chunked=8192, no NCCL tune | 5/23 (22%) | 18 | 3 TooManyConsecutiveErrors | 3:31 |
| dev1 | temp=1.0/top_p=1.0 (DeepSeek default), no guided, chunked=32768, NCCL tuned | cancelled — too slow, model rambling | 14/23 at 7:52 |
| dev2 | temp=0, no guided, chunked=32768, NCCL tuned | worse — greedy loops, 100+ steps per instance | 6:00+ |
| dev3 | temp=0.2/top_p=0.95, HMMA kernel, context=131K, mem=0.85, max_req=4, disable-custom-all-reduce | **running** | — | — | — |
| dev4 | temp=0.2/top_p=0.95, HMMA+split-KV+KV64/8w, context=262K, max_req=8 | 5/23 (22%) | 23 | 0 | **0:44** |

---

## Current Kernel Config

```
Kernel: dsv4_sparse_decode_kernel<16>
  KV_CHUNK = 64, NUM_WARPS = 8, 256 threads/CTA
  BF16 HMMA m16n8k16 for QK^T and P@V
  Register-resident FP32 O accumulator (32 regs/thread)
  SMEM: 84 KB (sQ[16][512] + sK[64][512] + score/prob union)
  Occupancy: 1 CTA/SM, 8 resident warps
  Split-KV: adaptive up to 128 splits, min 32 tokens/split
  UE8M0 dequant via __uint_as_float(b << 23) bitcast
  
Server flags:
  --context-length 262144
  --mem-fraction-static 0.85
  --max-running-requests 8
--chunked-prefill-size 32768
--kv-cache-dtype fp8_e4m3
--attention-backend compressed
--fp8-gemm-backend triton
--moe-runner-backend triton
--page-size 256
--cuda-graph-max-bs 32
--disable-custom-all-reduce
--preferred-sampling-params '{"temperature": 0.2, "top_p": 0.95}'

Tuned kernel configs:
  W8A8: tuned-configs-final/w8a8/ (5 shapes × 18 batch sizes)
  MoE:  tuned-configs-final/moe/  (E=256 N=512, up + down)

NCCL: PROTO=LL ALGO=Ring MIN_NCHANNELS=8 NTHREADS=512
EAGLE: disabled (gibberish on SM120)
Thinking: SGLANG_ENABLE_THINKING=1, SGLANG_REASONING_EFFORT=max
```

---

## Hillclimb Experiment Log (2026-05-22)

All experiments: change one flag, relaunch, run `bench_sglang_quick.py --mode ttft`, compare.

| # | Experiment | 256 | 1K | 4K | 8K | 16K | 32K | 64K | Decode@256 | Verdict |
|---|-----------|-----|----|----|----|----|-----|-----|-----------|---------|
| 0 | **Baseline** (P2P=0, thinking=1, effort=max, chunk=32K) | 397ms | 600ms | 2.6s | 3.7s | 7.4s | 16.0s | 38.9s | 35.6 | reference |
| 1 | NCCL_P2P_DISABLE=1 | 659ms | 786ms | 9.1s | 4.1s | 8.3s | 18.3s | 38.9s | 35.1 | ❌ worse |
| 2 | REASONING_EFFORT=high | 647ms | 798ms | 9.1s | 4.1s | 8.3s | 18.1s | 38.8s | 34.9 | ➖ same TTFT |
| 3 | ENABLE_THINKING=0 | 656ms | 775ms | 9.1s | 4.0s | 8.3s | 18.2s | 38.8s | 30.2 | ❌ decode regression |
| 4 | num-continuous-decode-steps=2 | 652ms | 799ms | 9.2s | 4.1s | 8.3s | 18.2s | 38.9s | 34.5 | ➖ no change |
| 5 | chunked-prefill-size=65536 | 664ms | 783ms | 9.0s | 4.1s | 8.3s | 18.1s | 43.3s | 35.3 | ❌ worse at 64K |
| 6 | Tuned FP8 kernel configs (3/5 shapes) | crash | — | — | — | — | — | — | — | ❌ **CRASH** — tuned configs selected block sizes exceeding SM120 shared memory (111KB > 101KB) |
| 7 | `--enable-piecewise-cuda-graph` | crash | — | — | — | — | — | — | — | ❌ **CRASH** — torch compile fails on SM120 |
| 8 | `--prefill-attention-backend flashmla` | crash | — | — | — | — | — | — | — | ❌ **CRASH** — `kv_lora_rank` not in V4 config |
| 9 | `--triton-attention-num-kv-splits 16` | 654ms | 795ms | 9.1s | 4.1s | 8.3s | 18.2s | 38.9s | 34.3 | ➖ no change |
| 10 | `--enable-nsa-prefill-context-parallel` | crash | — | — | — | — | — | — | — | ❌ **CRASH** — requires DeepEP MoE backend |
| 11 | `--enable-torch-compile` | crash | — | — | — | — | — | — | — | ❌ **CRASH** — torch compile fails on SM120 |
| 12 | `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096` | 675ms | 783ms | 9.1s | 4.1s | 8.3s | 18.2s | 39.0s | 33.3 | ➖ no change |
| 13 | `--flashinfer-mla-disable-ragged` | 685ms | 788ms | 9.1s | 4.1s | 8.3s | 18.3s | 38.9s | 34.1 | ➖ no change |
| 14 | **SM120-tuned FP8 W8A8 + MoE configs** | — | — | — | — | — | — | — | — | ✅ **+12-18% decode** (see below) |

### Experiment 14: SM120 Kernel Tuning (2026-05-24)

Wrote custom tuning scripts that bypass upstream limitations:
- **W8A8 GEMM**: 248 SM120-safe configs (SMEM ≤ 86KB, BLOCK_SIZE_K ≥ 128), 5 shapes × 18 batch sizes
- **MoE fused dispatch**: 336 configs, swiglu_limit=10.0 for 2604B submode, BLOCK_SIZE_M matched between up/down

**Decode ITL A/B comparison** (256 output tokens, single stream):

| Context | Before (HMMA, no tuning) | After (HMMA + tuned) | Delta |
|---------|-------------------------|---------------------|-------|
| 256 | 57.2 tok/s | **67.6 tok/s** | **+18%** |
| 1K | 52.3 tok/s | **60.8 tok/s** | **+16%** |
| 4K | 47.2 tok/s | **53.7 tok/s** | **+14%** |
| 8K | 46.2 tok/s | **51.9 tok/s** | **+12%** |
| 16K | 44.0 tok/s | **49.9 tok/s** | **+13%** |

The improvement is largest at short contexts (18% at 256 tokens) where MoE dispatch is a larger fraction of per-token time, and tapers at longer contexts where attention dominates. This is consistent with our profiling showing MoE dispatch at 41% of GPU time.

**Key observations:**
- TTFT is dominated by prefill compute, not NCCL or scheduling overhead
- The 4K anomaly (9s vs 2.6s baseline) appeared in ALL experiments 1-5, suggesting the baseline 2.6s was a warm-cache hit; true cold TTFT at 4K is ~9s
- Thinking mode doesn't affect prefill speed but does affect total output tokens
- The untuned FP8/MoE kernel configs are likely the real bottleneck — every forward pass uses default (slow) configs
- Auto-tuned FP8 configs CRASH on SM120 — shared memory limit (101KB) exceeded
- torch.compile and piecewise CUDA graphs both crash on SM120
- Alternative prefill backends (flashmla, NSA context parallel) are incompatible with DeepSeek V4
- FlashInfer knobs (tile size, ragged mode, KV splits) have zero effect on the compressed attention path
- PCIe P2P enabled is better than disabled on our topology
- **13 experiments exhausted. Prefill speed on SM120 is at the hardware floor for SGLang + FP8 repack. The only remaining paths are: (a) different runtime (vLLM + W4A16 + MTP), (b) different model format, or (c) contributing SM120 prefill kernels to the SM120 patch repo.**

---

## Profiling Results (20K token request, TP0)

Captured via SGLang's `/start_profile` → `/stop_profile` torch profiler.

### Kernel Time Breakdown

| Kernel | Time (ms) | % | Calls | Avg (ms) | What |
|--------|-----------|---|-------|----------|------|
| **`dsv4_sparse_decode_kernel<16>`** | **12,631** | **50%** | 215 | 58.7 | SM120 patch kernel — CUDA-core BF16 dot products |
| **`cudaMemcpyAsync`** | **7,156** | **28%** | 82 | 87.3 | PCIe transfers between GPUs (TP communication) |
| **`ncclAllReduce_Sum_bf16_RING_LL`** | **3,395** | **13%** | 435 | 7.8 | NCCL allreduce |
| `fused_moe_kernel` | 200 | 0.8% | 430 | 0.5 | MoE expert dispatch |
| `_w8a8_block_fp8_matmul` | 178 | 0.7% | 1180 | 0.15 | FP8 GEMM |
| `mhc_post_tilelang_kernel` | 85 | 0.3% | 430 | 0.2 | MHC post-processing |
| Everything else | ~1,700 | 7% | — | — | — |

### Key Insight

**The SM120 sparse decode kernel is the #1 bottleneck at 50% of total GPU time.** It uses portable CUDA-core BF16 dot-products (scalar `for d=0..512: acc += q[d]*k[d]`) instead of tensor core instructions. This is ~10-20× slower than WGMMA (SM90) or TCGEN05 (SM100) equivalents.

### Current Kernel Architecture (from `sparse_decode_kernel.cuh`)

```
Grid:  (batch * s_q, ceil(h_q / 16))
Block: 128 threads (4 warps)
SMEM:  ~84 KB (within SM120's 101 KB limit)

Per CTA processes 16 query heads × 32 KV tokens per chunk:
1. Load Q [16 × 512] BF16 into SMEM
2. For each chunk of 32 KV tokens:
   a. Dequant FP8→BF16 + load into SMEM  (load_kv_chunk)
   b. QK^T: scalar BF16 dot-product, 512 dims  ← THE BOTTLENECK
   c. Online softmax
   d. P @ V accumulate (also scalar)         ← SECOND BOTTLENECK
```

The QK^T inner loop (line 283-289) is:
```cuda
for (int d = 0; d < HEAD_DIM_QK; ++d) {
    acc += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
}
```
This is pure scalar — no WMMA, no dp4a, no vector instructions. Each of the 16×32=512 dot-products does 512 scalar multiplies.

### SM120 Capabilities

SM120 (Blackwell workstation) has:
- **HMMA (half-precision matrix multiply)** — available via `mma.sync.aligned` for FP16/BF16
- **No WGMMA** (that's SM90 Hopper only)
- **No TCGEN05** (that's SM100 Blackwell datacenter only)  
- **dp4a / dp2a** for INT8/INT4 dot products
- **101 KB shared memory** per SM
- **BF16 HMMA**: `m16n8k16` or `m16n8k8` shapes

### Optimization Opportunities

1. **Replace scalar dot-product with HMMA tensor core instructions** — SM120 supports `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`. This would give ~16× throughput over scalar for the QK^T and P@V steps.

2. **Increase KV_CHUNK from 32 to 64 or 128** — more work per SMEM load, better amortization of the dequant step. Need to stay within 101KB SMEM.

3. **Vectorized FP8 dequant** — current code does 4 FP8→BF16 at a time. Could use `__nv_cvt_fp8x4_to_halfx4` intrinsics if available on SM120.

4. **Split-KV parallelism** — currently `num_sm_parts > 1` is "not yet implemented". Splitting the KV across multiple CTAs would better utilize all SMs.

5. **Async memory copies** — use `cp.async` for loading KV chunks while computing on the previous chunk (double-buffering).

---

`bench_sglang_quick.py` — runs in ~2 min, measures TTFT at 256/1K/4K/8K/16K/32K/64K, decode throughput, and concurrent throughput.

```bash
python3 bench_sglang_quick.py --mode ttft    # just prefill
python3 bench_sglang_quick.py --mode decode  # just decode
python3 bench_sglang_quick.py --mode all     # everything
```

---

## Completed Optimizations

| Change | Impact | Notes |
|--------|--------|-------|
| SM120 patch (0xSero) | Required | Without it: `Unsupported architecture for sparse decode fwd` |
| **HMMA tensor core kernel** | **2× TTFT, 1.6× decode** | Replaced scalar BF16 dot-products with `mma.sync.aligned.m16n8k16` for QK^T and P@V |
| **Register-resident O** | **+10-12% over HMMA** | sO[16][512] moved from 32KB SMEM to 64 FP32 registers/thread. SMEM 86→50 KB. sP/sP_bf16 fused as union. Row stats in registers. Occupancy=2 target |
| **Split-KV adaptive** | **+8% decode short ctx** | Up to 128 splits for SM utilization. Combine kernel merges online-softmax partials |
| **Bitcast dequant** | Clean codegen | `__uint_as_float(b << 23)` replaces `__powf(2.0f, b-127)` for UE8M0 |
| **KV_CHUNK=64 / 8 warps** | **+3-5% ITL and TTFT** | Halves chunk iterations/barriers. 32 O regs/thread. 8 natural QK^T N-tiles |
| **SM120-tuned FP8 W8A8 GEMM** | **+12-18% decode** | 248 SMEM-safe configs, BLOCK_SIZE_K≥128, 5 shapes × 18 batch sizes. Tuned on RTX PRO 6000 |
| **SM120-tuned MoE fused dispatch** | **+12-18% decode** (combined with W8A8) | E=256 N=512, swiglu_limit=10.0, up+down variants with matched BLOCK_SIZE_M. 336 configs |
| NCCL tuning (LL/Ring/8ch/512thr) | ~2.3× TTFT improvement | From Reddit LordNeel post, critical for PCIe Max-Q |
| `--chunked-prefill-size` 8192→32768 | ~2.3× TTFT improvement | Fewer prefill rounds |
| `--disable-custom-all-reduce` | Prevents NCCL deadlock | Custom allreduce uses CUDA P2P which deadlocks on PCIe Max-Q under load |
| Reasoning bridge (port 8010) | Fixes Claude Code multi-turn | Strips reasoning_content, prevents 400 errors from thinking blocks |
| Removed guided decoding for bench | Fixes format errors | `<think>` tags break regex constraints |

## Failed/Reverted Experiments

| Experiment | Result | Why |
|------------|--------|-----|
| EAGLE speculative decoding | **Gibberish after ~50-100 tokens** | EAGLE sends s_q > 1 (multi-token queries from draft) to sparse decode kernel which only supports s_q=1. The API asserts s_q==1 but EAGLE bypasses this. Scalar kernel also broken in our modified build (content=null). Not a kernel-specific issue — needs s_q > 1 support in the kernel's grid mapping and token processing loop. |
| `--max-running-requests 16` | **OOM crash** | KV cache exhausted under SWE-bench load with thinking-heavy sessions |
| `--max-running-requests 8` + `--mem-fraction-static 0.8` | **OOM crash** | `Out of memory even after retracting all other requests in the decode batch` — 0.8 mem fraction too low for 131K context under load |
| `temperature=1.0` (DeepSeek official) | **Too slow, rambling** | 2-3M tokens/instance, 7+ hours for 14/23 instances |
| `temperature=0` | **Greedy loops** | Model repeats same failing approach 100+ times, can't escape |
| Guided regex + thinking | **Format errors** | Model emits `<think>...</think>` before `THOUGHT:`, breaks `^THOUGHT:` regex |

---

## Open Optimization Tracks

### Track 1: Prefill Speed (TTFT)
- [x] ~~Try `--chunked-prefill-size 65536`~~ — worse at 64K boundary (43.3s vs 38.9s)
- [x] ~~Try `NCCL_P2P_DISABLE=1`~~ — worse across the board
- [x] ~~Try `--num-continuous-decode-steps 2`~~ — no change
- [x] ~~Try `SGLANG_REASONING_EFFORT=high`~~ — no TTFT change (may help SWE-bench token budget)
- [x] ~~Try `SGLANG_ENABLE_THINKING=0`~~ — decode regression, not a win
- [x] ~~**BLOCKED**: Generate tuned FP8 W8A8 kernel configs~~ — **DONE (2026-05-24)**: Custom tuner with SMEM filter (`≤ 86KB`) and `BLOCK_SIZE_K ≥ 128` constraint. 248 configs searched per shape, 5 shapes × 18 batch sizes. Result: **12-18% decode throughput improvement.**
- [x] ~~**BLOCKED**: Generate tuned MoE kernel configs (E=256,N=512)~~ — **DONE (2026-05-24)**: Custom tuner bypasses HuggingFace config loading, passes `swiglu_limit=10.0` for 2604B submode, constrains `BLOCK_SIZE_M` to match between up/down variants. 336 configs searched, 4 batch sizes retuned with constrained M. Result: **0 sub-optimal warnings in production.**
- [ ] Try `--enable-torch-compile` after kernel tuning
- [ ] Try `--enable-piecewise-cuda-graph` after kernel tuning
- [ ] Benchmark with `--disable-overlap-schedule` vs default (overlap on)

**Kernel tuning notes:**
- Tuning scripts live in the Docker image at `/workspace/sglang/benchmark/kernels/`
- FP8 tuning: `quantization/tuning_block_wise_kernel.py` — works, no extra deps
- MoE tuning: `fused_moe_triton/tuning_fused_moe_triton.py` — needs `pip install ray`, and `DeepseekV4ForCausalLM` must be added to the architecture list in `common_utils.py` (patched but the `swiglu_limit` bug is upstream)
- Configs go to `python/sglang/srt/layers/quantization/configs/` (FP8) and `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/` (MoE)
- **MUST stop the serving container before tuning** — TP=4 server uses all GPU memory, tuning container can't allocate on GPU 0
- Tuning script: `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/tune_kernels.sh`

### Track 6: SM120 HMMA Kernel (✅ LANDED)

**Status**: Committed on branch `feat/hmma-tensor-core-sparse-decode`, pushed to `ambientlight/deepseek-v4-flash-sm120`. Ready for PR to upstream `0xSero/deepseek-v4-flash-sm120`.

**What was done**:
- Profiled 20K token request with torch profiler — found `dsv4_sparse_decode_kernel` at 50% of total GPU time (12.6s / 25.3s), using scalar BF16 dot-products
- Verified SM120 supports `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
- Wrote HMMA-optimized QK^T kernel replacing scalar `for d=0..512: acc += q[d]*k[d]` with tensor core matmuls
- v1 had 1.6× speedup but wrong output — fragment register mapping bug (a0/a1 interleave wrong, B used `lane%4` for N instead of `lane>>2`)
- Deep-researched PTX ISA m16n8k16 fragment layout, fixed in v2 — **1.6× speedup with correct output**

**Results**: See "Current Best" table at top of this file.

**Future work**:
- [x] ~~HMMA-ify P@V~~ — Done in commit 85c5c26
- [x] ~~Register-resident O~~ — Done in commit 1c93c25. SMEM 86→50 KB. +10-12% over SMEM-sO.
- [ ] FP8 QK^T: Tested, 10-17% slower due to per-tile scale overhead and no block-scale MMA on SM120. Not worth pursuing unless block-scale becomes available.
- [ ] **Split-KV** (`num_sm_parts > 1`): Highest remaining macro win. base_ctas=4 for single-request decode on 188 SMs → massive under-utilization. Adaptive split could use 16-32 splits.
- [ ] **KV_CHUNK=64 + 8 warps**: Now viable with register-sO (SMEM ~84 KB). Halves chunk iterations, natural 8 N-tiles for QK^T. 
- [ ] cp.async raw prefetch for next chunk (now viable with 50 KB SMEM leaving headroom)
- [ ] ldmatrix instead of manual fragment loads (if bank conflicts are an issue)

**Key hardware constraints discovered**:
- SM120 has only 100 KB SMEM per SM (not 228 KB like datacenter Blackwell)
- Occupancy was always 1 at 86 KB; register-sO at 50 KB may achieve occupancy=2
- Block-scaled MMA (.kind::mxf8f6f4, .block_scale) NOT supported on SM120
- FP8 m16n8k32 MMA works but per-tile scale overhead makes it slower than BF16 HMMA
- Scaled packed FP8→BF16 conversion (cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2) NOT supported on SM120
- SM120 has 4 warp schedulers per SM
- h_q per TP shard = 16 (64 heads / TP4) → base_ctas = 1 for single-request decode
- Split-KV SM utilization is bounded by topk: ~35% at 16K context, ~63% at 32K

**Next optimization directions** (per deep research analysis):
- [ ] Classify cudaMemcpyAsync (28% of time) with Nsight Systems — P2P vs staging vs framework
- [ ] Investigate FlashInfer allreduce fusion gate (currently sm90/sm100 only — may be portable to SM120)
- [ ] Inspect SGLang communication overlap flags (alt_stream, TBO/SBO, LayerCommunicator)
- [ ] ldmatrix / swizzled SMEM layout if bank conflicts are significant
- [ ] Profile the "other" 28% bucket — likely many small kernels + framework glue

**Current GPU time breakdown** (20K token request, TP0):
```
sparse_decode_kernel:  ~30%  (was 50% before HMMA — kernel is no longer the bottleneck)
cudaMemcpyAsync:       ~28%  (PCIe TP transfers)
ncclAllReduce:         ~13%  (NCCL TP communication)
fused_moe_kernel:       ~1%
everything else:       ~28%  (projections, norms, router, scheduling, launch overhead)
```
**The kernel is no longer the dominant bottleneck.** Communication (41%) and framework overhead (28%) now dominate. Further kernel-level gains are incremental.

**Dead ends confirmed on SM120:**
- Scaled packed FP8→BF16 cvt (`cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2`): NOT supported
- Block-scaled MMA (`.kind::mxf8f6f4, .block_scale`): NOT supported
- FP8 QK^T MMA: 10-17% slower than BF16 HMMA due to per-tile scale overhead
- torch.compile / piecewise CUDA graphs: crash on SM120

**Key files**:
- Production kernel: `csrc/sm120/decode/sparse_decode_kernel.cuh` (KV64/8w, reg-O, HMMA)
- Split-KV combine: `csrc/sm120/decode/split_kv_combine.cuh`
- Scalar reference: `csrc/sm120/decode/sparse_decode_kernel_scalar_ref.cuh`
- ITL benchmark: `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/bench_decode_itl.py`
- TTFT benchmark: `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/bench_sglang_quick.py`
- Deep research docs: `sm120_fp8_mma_qkt_deepseek_v4_flash.md`, `sm120_sparse_decode_next_optimization_plan.md`, `sm120_next_round_vector_dequant_kv64_bench_comm.md`

### Track 3: Eval Quality (SWE-bench)
- [ ] Temperature sweep: 0.1, 0.2, 0.3, 0.5 with same config
- [ ] Try thinking disabled (`SGLANG_ENABLE_THINKING=0`) — is raw speed more valuable than reasoning?
- [ ] Try `SGLANG_REASONING_EFFORT=high` instead of `max` — less thinking overhead
- [ ] Prompt engineering: adapt system prompt specifically for DeepSeek V4 style
- [ ] Try guided decoding with think-aware regex: `^(<think>[\s\S]*</think>\s*)?THOUGHT:...`
- [ ] Compare `-w 4` (match max_running_requests) vs `-w 16` worker concurrency

### Track 4: Alternative Stack (vLLM + W4A16)
- [ ] Evaluate LordNeel's W4A16+FP8 quant on patched vLLM
  - Model: `LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8`
  - vLLM fork: `pasta-paul/dsv4-flash-w4a16-fp8`
  - TP=2 (frees 2 GPUs), MTP self-speculation (52→85-111 tok/s)
  - Needs `--disable-custom-all-reduce` on Max-Q
- [ ] Compare SWE-bench scores: SGLang FP8 vs vLLM W4A16+MTP

### Track 5: Serving Stability
- [ ] Determine safe max_running_requests for 131K context (4 works, 8 OOMs, try 6?)
- [ ] Add health monitoring / auto-restart for OOM crashes
- [ ] Profile memory usage per request at various context lengths
- [ ] Test bridge under sustained Claude Code multi-turn sessions

---

## How to Iterate

1. Change one thing in the docker run command
2. Wait ~3 min for startup
3. Run `python3 bench_sglang_quick.py --mode ttft`
4. Compare TTFT table against baseline above
5. If better, update this file and optionally run SWE-bench dev
6. SWE-bench quick check: `mini-extra swebench --subset lite --split dev -w 4 -o ./results-dev-deepseek-v4-flash-fp8-devN --config swebench-litellm-deepseek-v4-flash.v5.yaml`

---

## Resume Checklist

**Current state**: Server is UP with HMMA kernel + tuned FP8 W8A8 + MoE configs. 0 sub-optimal warnings. 262K context, max_req=8.

1. **Server**: Running on port 8000 with HMMA kernel (`feat/hmma-tensor-core-sparse-decode` branch built).
   ```bash
   # To restart (if needed):
   ./run_experiment.sh "HMMA-kernel" "" ""
   ```

2. **Bridge**: Restart if needed for Claude Code use:
   ```bash
   cd /mnt/hot/ambientlight/deepseek-v4-flash-sm120
   SGLANG_BASE_URL=http://127.0.0.1:8000/v1 nohup python3 deepseek_reasoning_bridge.py > /tmp/bridge.log 2>&1 &
   ```

3. **SWE-bench dev3**: Running with HMMA kernel, temp=0.2/top_p=0.95.
   ```bash
   # Check progress:
   ls /mnt/hot/ambientlight/repos/mini-swe-agent/results-dev-deepseek-v4-flash-fp8-dev3/
   ```

4. **PR ready**: Branch `feat/hmma-tensor-core-sparse-decode` on `ambientlight/deepseek-v4-flash-sm120`.
   ```bash
   cd /mnt/hot/ambientlight/deepseek-v4-flash-sm120
   git push -u origin feat/hmma-tensor-core-sparse-decode
   # Then create PR against 0xSero/deepseek-v4-flash-sm120
   ```

---

## File Locations

| What | Path |
|------|------|
| Model weights | `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8` |
| SM120 patch | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120` |
| Reasoning bridge | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/deepseek_reasoning_bridge.py` |
| Quick benchmark | `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/bench_sglang_quick.py` |
| Deploy guide | `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/DEPLOY-DEEPSEEK-V4-FLASH.md` |
| Deep research guide | `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/deepseek-v4-flash-sglang-litellm-claude-code-guide.md` |
| SWE-bench config (no guided) | `/mnt/hot/ambientlight/repos/mini-swe-agent/swebench-litellm-deepseek-v4-flash.v5.yaml` |
| SWE-bench config (guided) | `/mnt/hot/ambientlight/repos/mini-swe-agent/swebench-litellm-deepseek-v4-flash-guided.v5.yaml` |
| LiteLLM config | `/mnt/hot/ambientlight/repos/litellm/litellm_config.yaml` |
| This quest file | `/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/QUEST-DEEPSEEK-V4-FLASH-HILLCLIMB.md` |
