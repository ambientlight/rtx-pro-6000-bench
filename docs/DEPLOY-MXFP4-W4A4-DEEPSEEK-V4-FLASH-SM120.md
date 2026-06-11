# DeepSeek-V4-Flash — Native MXFP4 W4A4 on SM120

Recipe for serving [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) with **native MXFP4×MXFP4 (W4A4)** fused MoE and custom **HMMA tensor-core sparse-attention** kernels — on **4× RTX PRO 6000 Blackwell
(SM120, TP=4)**. This wires together three forks — [flashinfer](https://github.com/ambientlight/flashinfer/tree/ambientlight/mxfp4-fused-moe) (MXFP4 kernels), [sglang](https://github.com/ambientlight/sglang/tree/feat/sm120-mxfp4-w4a4-moe) (serving), and custom [sparse_decode_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/decode/sparse_decode_kernel.cuh) + [sparse_prefill_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/prefill/sparse_prefill_kernel.cuh) HMMA kernels from [deepseek-v4-flash-sm120](https://github.com/ambientlight/deepseek-v4-flash-sm120) as a drop-in replacement to DSv4 stock FlashMLA kernels unavailable for SM120.

**Benchmark** ([`bench/deepseek-v4-flash_W300_TP4_sglang/`](https://github.com/ambientlight/rtx-pro-6000-bench/tree/main/bench/deepseek-v4-flash_W300_TP4_sglang)):
single-stream decode scales to the **full 1M context** — 64K / 128K / 256K / 512K / 1M-token prompts sustain
41 / 31 / 21 / 12 / 7 tok/s (TTFT 0.5 / 0.8 / 1.6 / 3.3 / 6.7 s), the 1M prompt peaking at 95% VRAM
([`…chunk8192.log`](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/bench/deepseek-v4-flash_W300_TP4_sglang/bench_sweep_single.chunk8192.log)).

Concurrency sweep at W300/TP4 ([`bench_sweep.log`](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/bench/deepseek-v4-flash_W300_TP4_sglang/bench_sweep.log)): **8576/8576 requests, 0 failed over ~27 h continuous load**, best sustained 756 tok/s @ 2K (c64). See
[Performance](#performance).

---

## Specs

| Component | Spec |
|---|---|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 4x96GB) |
| Interconnect | PCIe (no NVLink) — NCCL tuned accordingly |
| RAM | 512 GB DDR5 ECC |

## Software stack

| Package | Version | Notes |
|---|---|---|
| Python | 3.12.3 | |
| PyTorch | 2.11.0+cu130 | CUDA 13.0 |
| Triton | 3.6.0 | |
| flashinfer-python | 0.6.13 (fork [`ambientlight/mxfp4-fused-moe`](https://github.com/ambientlight/flashinfer/tree/ambientlight/mxfp4-fused-moe), over 0.6.11.post3 base) | MXFP4 kernels. Mismatches cubin → needs `FLASHINFER_DISABLE_VERSION_CHECK=1` |
| flashinfer-cubin | 0.6.12 | pulled by sglang's pin; the fork's python tolerates it |
| transformers | 5.8.1 | needs `deepseek_v4` |
| sgl-kernel | 0.4.3 | pulled by sglang; SM120 path uses no new API |
| sglang | fork [`feat/sm120-mxfp4-w4a4-moe`](https://github.com/ambientlight/sglang/tree/feat/sm120-mxfp4-w4a4-moe) | MXFP4 W4A4 method + feature-probe + decode & prefill toggles |
| deepseek_v4_kernel (HMMA) | fork [`feat/hmma-tensor-core-sparse-decode`](https://github.com/ambientlight/deepseek-v4-flash-sm120/tree/feat/hmma-tensor-core-sparse-decode) | custom [sparse_decode_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/decode/sparse_decode_kernel.cuh) + [sparse_prefill_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/prefill/sparse_prefill_kernel.cuh) |

---

## Installation

Needs CUDA 12.8+ on `PATH` (host `nvcc` builds the HMMA kernel for `sm_120a`; this box has 13.1).

```bash
python3.12 -m venv ~/.venvs/dsv4 && source ~/.venvs/dsv4/bin/activate
pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install "flashinfer-python==0.6.11.post3" "flashinfer-cubin==0.6.11.post3"
pip install --no-deps --force-reinstall "git+https://github.com/ambientlight/flashinfer.git@ambientlight/mxfp4-fused-moe"
pip install --no-build-isolation "transformers==5.8.1" \
  "git+https://github.com/ambientlight/sglang.git@feat/sm120-mxfp4-w4a4-moe#subdirectory=python"
pip install --no-deps --force-reinstall "git+https://github.com/ambientlight/flashinfer.git@ambientlight/mxfp4-fused-moe"
git clone -b feat/hmma-tensor-core-sparse-decode https://github.com/ambientlight/deepseek-v4-flash-sm120.git
pip install -e deepseek-v4-flash-sm120 --no-deps --no-build-isolation

# RTX PRO 6000 tuned W8A8 + MoE kernel configs (from the HMMA repo)
# Without these sglang falls back to default tiles (the "Using default W8A8 ... sub-optimal" warning).
SGL=$(python -c "import os,sglang;print(os.path.dirname(sglang.__file__))")
cp deepseek-v4-flash-sm120/tuned-configs/w8a8/*.json "$SGL/srt/layers/quantization/configs/"
cp deepseek-v4-flash-sm120/tuned-configs/moe/*.json  "$SGL/srt/layers/moe/moe_runner/triton_utils/configs/"

# Model download
# huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./DeepSeek-V4-Flash --local-dir-use-symlinks False

# verify (flashinfer-cubin stays 0.6.12 vs fork 0.6.13, so bypass the version guard)
export FLASHINFER_DISABLE_VERSION_CHECK=1
python -c "from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import sm120_moe_supported_quant_modes as f; assert 'mxfp4' in f(); print('flashinfer mxfp4 OK')"
python -c "from sglang.srt.layers.quantization.mxfp4_w4a4_moe import Mxfp4W4A4MoEMethod; print('sglang method OK')"
python -c "from deepseek_v4_kernel.ops import sparse_decode_fwd, sparse_prefill_fwd; print('hmma decode + prefill kernels OK')"
```

### Additions

Three forks, all gated SM120-only:

- **FlashInfer** [ambientlight/mxfp4-fused-moe](https://github.com/ambientlight/flashinfer/tree/ambientlight/mxfp4-fused-moe) ([#3541](https://github.com/flashinfer-ai/flashinfer/pull/3541), draft) — CuTe-DSL fused-SwiGLU `MmaMXF4Op` MXFP4 MoE kernels + the `sm120_moe_supported_quant_modes()` capability probe.
- **SGLang** [feat/sm120-mxfp4-w4a4-moe](https://github.com/ambientlight/sglang/tree/feat/sm120-mxfp4-w4a4-moe) — `Mxfp4W4A4MoEMethod` (+ shared E8M0 swizzle), the `fp8.py` feature-probe that auto-selects it, the `SGLANG_SM120_SPARSE_DECODE` / `SGLANG_SM120_SPARSE_PREFILL` attention toggles, and the capture-safe indexer routing.
- [sparse_decode_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/decode/sparse_decode_kernel.cuh) + [sparse_prefill_kernel.cuh](https://github.com/ambientlight/deepseek-v4-flash-sm120/blob/feat/hmma-tensor-core-sparse-decode/csrc/sm120/prefill/sparse_prefill_kernel.cuh) HMMA kernels from [OxSero/deepseek-v4-flash-sm120 fork](https://github.com/ambientlight/deepseek-v4-flash-sm120) that was built against during the hillclimb. Both use warp-level `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. (SM120 has no `wgmma` (SM90) or `tcgen05` (SM100)).

---

## Launch

A ready launch script lives at `repos/sglang/debug/launch-mxfp4-w4a4-e2e-test.sh`. The essentials:

```bash
source ~/.venvs/dsv4/bin/activate
# The HMMA kernel is a pip-installed package (Installation step), so no PYTHONPATH is
# needed. The version guard is required (flashinfer-cubin 0.6.12 vs fork 0.6.13).

# --- selection ---
export SGLANG_SM120_SPARSE_DECODE=hmma           # sparse attention, default path (decode + prefill <= 11673 tok)
export SGLANG_SM120_SPARSE_PREFILL=hmma           # sparse attention, large-batch path (prefill > 11673 tok)
#   MoE method auto-selects via the FlashInfer feature-probe — NO SGLANG_MXFP4_W4A4 env var.

# SM120+DeepseekV4 auto-sets FP8_WO_A_GEMM, USE_TOPK_V2, TILELANG_MHC_PRE,
# DEEPGEMM_HC_PRENORM, FP8_PAGED_MQA_LOGITS_TORCH at startup — no need to export them.
export SGLANG_OPT_USE_TILELANG_INDEXER=1        # default off; the fast SM120 indexer
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0             # no SM120 DeepGEMM recipe
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0    # breaks CUDA-graph capture on SM120
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0         # SM120 dtype mismatch

# --- version guards ---
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NCCL (PCIe, no NVLink) ---
export NCCL_PROTO=LL NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=8 NCCL_NTHREADS=512
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
  --model-path ./DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --host 0.0.0.0 --port 8000 \
  --context-length 1048576 --mem-fraction-static 0.80 \
  --max-running-requests 128 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend triton \
  --chunked-prefill-size 16384 --page-size 256 \
  --cuda-graph-max-bs 128 --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --disable-custom-all-reduce --disable-shared-experts-fusion \
  --dsa-topk-backend torch \
  --watchdog-timeout 3600 --log-level info
```

Startup ~5 min (weight load + CUDA-graph capture across the bs 1…128 ladder).

---

## How the three repos select each other at runtime

```
 sglang.launch_server  (Layer 2)
      |
      +-- MoE experts (DeepSeek-V4-Flash, native MXFP4)
      |     fp8.py get_quant_method:
      |       is_fp4_experts && is_sm120_supported() && _has_flashinfer_sm120_mxfp4_moe()
      |         -> Mxfp4W4A4MoEMethod  ── apply() ─┐
      |                                            v
      |     FlashInfer fork: launch_sm120_moe(quant_mode="mxfp4")   [LAYER 1]
      |       CuTe-DSL fused SwiGLU MmaMXF4Op, E8M0 self-scaling
      |
      +-- Sparse attention (per-token top-k; prefill and decode are one op)
      |     default — decode + prefill <= 11673 tok (reads paged FP8):
      |       flash_mla_sm120.py, SGLANG_SM120_SPARSE_DECODE=hmma
      |         -> deepseek_v4_kernel.ops.sparse_decode_fwd (.so)  [LAYER 3, our custom]
      |     large-batch — prefill > 11673 tok (reads pre-staged flat bf16):
      |       flash_mla_sparse_prefill_sm120.py, SGLANG_SM120_SPARSE_PREFILL=hmma
      |         -> deepseek_v4_kernel.ops.sparse_prefill_fwd (.so)  [LAYER 3, our custom]
      |
      +-- Indexer (tilelang FP8 paged-MQA-logits)
            indexer.py: is_sm120_supported() -> capture-safe dsv4/ kernel
```

### MoE — FlashInfer feature-probe

`fp8.py` selects `Mxfp4W4A4MoEMethod` only when all three hold:

```python
self.is_fp4_experts                      # native MXFP4 checkpoint
and is_sm120_supported()                 # RTX PRO 6000 / SM120
and _has_flashinfer_sm120_mxfp4_moe()    # "mxfp4" in sm120_moe_supported_quant_modes()
```

The probe queries FlashInfer's **public capability API**, not a version string. On a stock
FlashInfer (no fork) the set lacks `mxfp4`, the probe is False, and SGLang silently uses the
Triton MoE fallback. Installing Layer 1 flips it True — that is the entire activation
mechanism. Confirm with the install block's L1 probe one-liner.

---

## Environment variable reference

| Variable | Value | Why |
|---|---|---|
| `SGLANG_SM120_SPARSE_DECODE` | `hmma` | Sparse-attention default path (decode + prefill ≤ 11673 tok) |
| `SGLANG_SM120_SPARSE_PREFILL` | `hmma` | Sparse-attention large-batch path (prefill > 11673 tok). Required on SM120 — the stock kernel is SM90a/SM100f-only and raises here |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | fork flashinfer-python 0.6.13 vs cubin 0.6.12 |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` | `1` | only if the venv's sgl-kernel lags the branch's request; our path uses no new API |
| `SGLANG_OPT_USE_TILELANG_INDEXER` | `1` | Fast FP8 paged-MQA-logits indexer; SM120 routing fix sends it to the capture-safe `dsv4/` kernel |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `0` | No SM120 DeepGEMM recipe |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | `0` | Multi-stream breaks CUDA-graph capture on SM120 |
| `SGLANG_OPT_USE_FUSED_HASH_TOPK` | `0` | SM120 dtype mismatch |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Leaves CUDA-graph-capture headroom; avoids fragmentation OOM |
| `NCCL_PROTO`/`NCCL_ALGO`/`NCCL_MIN_NCHANNELS`/`NCCL_NTHREADS` | `LL`/`Ring`/`8`/`512` | PCIe allreduce tuning (no NVLink) |

---

## Performance

Full 2K–64K × concurrency sweep, W300 / TP4, output 1024, 128 prompts/run — **67 cells, 8576 / 8576
requests succeeded, 0 failed** over ~27 h of continuous load. All numbers below are sustained
(run-duration mean `output_throughput`), from
[`bench_sweep.log`](../bench/deepseek-v4-flash_W300_TP4_sglang/bench_sweep.log)
in [`../bench/deepseek-v4-flash_W300_TP4_sglang/`](../bench/deepseek-v4-flash_W300_TP4_sglang).

Best sustained output throughput per input length (and the telemetry at that cell):

| Input | Sustained tok/s | @ conc | Mean power | KV peak |
|---|---:|:---:|---:|---:|
| 2,048 | **756** | c64 | 1,009 W | 80% |
| 4,096 | **444** | c32 | 1,009 W | 69% |
| 8,192 | **263** | c40 | 984 W | 100% |
| 16,384 | **163** | c104 | 931 W | 100% |
| 32,768 | **79** | c40 | 914 W | 100% |
| 65,536 | **40** | c16 | 919 W | 100% |

Single-user (c1) by prompt length — TTFT scales with prefill cost; TPOT stays ~17–24 ms (decode is the
prompt-independent sparse top-k), so single-stream throughput tapers as KV grows:

| Input | TTFT p50 | TPOT p50 | Output tok/s |
|---|---:|---:|---:|
| 2,048 | 604 ms | 16.8 ms | 57.6 |
| 8,192 | 2,653 ms | 17.5 ms | 49.8 |
| 16,384 | 5,381 ms | 18.5 ms | 42.2 |
| 32,768 | 11,199 ms | 20.4 ms | 32.0 |
| 65,536 | 23,914 ms | 24.2 ms | 21.0 |

Long-context throughput is gated by sparse-attention gather + KV-cache capacity (the ≥ 8K rows saturate KV
at 100%, so they peak at low concurrency and queue beyond it — graceful backpressure, not failure).
Per-input-length charts + per-cell telemetry in the benchmark folder (`plots/`, `telemetry_*.png`); reproduce
with the matrix sweep in the [repo README](https://github.com/ambientlight/rtx-pro-6000-bench).

### Long-context scaling to 1M (single stream)

A separate single-concurrency sweep (config:
[`sglang-single.yaml`](../bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml) — full **1,048,576**
context, `mem-fraction-static 0.80`, `chunked-prefill-size 8192`, `max-running-requests 1`; launch with
[`launch-single.sh`](../bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh)) doubles the prompt
2K → **1,047,552** (= 1M − 1024, filling the native context exactly), one prompt per length, output 1024.
**All 10 lengths completed, including the full 1M prompt.** Numbers from
[`bench_sweep_single.chunk8192.log`](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/bench/deepseek-v4-flash_W300_TP4_sglang/bench_sweep_single.chunk8192.log):

| Input | TTFT (prefill) | TPOT (decode) | Output tok/s | Peak VRAM/GPU | KV |
|------:|:--------------:|:-------------:|:------------:|:-------------:|:--:|
| 2,048 | 0.19 s | 16.8 ms | 58.8 | 85% | 1% |
| 8,192 | 0.23 s | 17.5 ms | 56.3 | 85% | 2% |
| 32,768 | 0.32 s | 20.4 ms | 48.3 | 87% | 7% |
| 131,072 | 0.85 s | 31.8 ms | 30.7 | 87% | 7% |
| 262,144 | 1.60 s | 46.9 ms | 20.7 | 88% | 12% |
| 524,288 | 3.29 s | 77.3 ms | 12.4 | 90% | 23% |
| **1,047,552** | **6.73 s** | **137.6 ms** | **6.9** | **95%** | **47%** |

TTFT scales **sub-linearly** (0.19 → 6.73 s for 512× the tokens — the sparse-prefill kernel) and TPOT ~8×
(16.8 → 137.6 ms, the per-step sparse gather over a growing KV workspace). The 1M prompt peaks at **95%
VRAM/GPU** (364 GB of 4×96 GB) — and notably **KV is only 47%**: the binding constraint at extreme context is
the C4 indexer's transient per-chunk logits buffer, not the KV pool. That is why `chunked-prefill-size` and
`mem-fraction-static` (not just context length) gate how far you scale: a larger chunk doubles that buffer.
**0.80 / chunk 8192 is the config that runs both this 1M single-stream sweep and the full concurrency sweep
above healthily**.

---

## End-to-end pipeline (MXFP4 W4A4 decode, 1 token)

Blocks marked `*` differ from the FP8 path. Only the **MoE FFN expert GEMM** is MXFP4 W4A4; For a single decode token the **default sparse-attention path** runs (selected by
`SGLANG_SM120_SPARSE_DECODE`; `hmma` shown). The **large-batch path** below it (`SGLANG_SM120_SPARSE_PREFILL`)
is the *same attention op* with KV pre-staged to flat bf16 — it fires only on the extend path when a prefill
batch exceeds 11673 query tokens, never for a decode token.

```
TOKEN IN
    |
    v
+-- EMBEDDING                                       [Torch] ------------+
| VocabParallelEmbedding -> embed + repeat for HC                       |
+-----------------------------------------------------------------------+
    |
    v
+-- PER LAYER x61 ------------------------------------------------------+
|                                                                       |
|  +-- HC_PRE (attention) ------------------------------------------+   |
|  | deep_gemm tf32 prenorm GEMM            [DeepGEMM]              |   |
|  | mhc_pre fused Sinkhorn+RMSNorm         [TileLang]              |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- MQA ATTENTION -----------------------------------------------+   |
|  | Q proj: wq_a FP8 GEMM                  [Triton FP8]            |   |
|  | Q norm + RoPE fused                    [Triton]                |   |
|  | Q proj: wq_b FP8 GEMM                  [Triton FP8]            |   |
|  | KV proj + cache write                  [Triton]                |   |
|  |                                                                |   |
|  | C4 Indexer:                                                    |   |
|  |   compressor gate GEMM                 [Triton]                |   |
|  |   fused_q_indexer_rope_hadamard_quant  [Triton]                |   |
|  |   fp8_paged_mqa_logits (SM120: dsv4/)  [TileLang]              |   |
|  |   topk_transform_512                   [Triton]                |   |
|  |                                                                |   |
|  | SPARSE ATTENTION — per-token top-k gather                      |   |
|  | (no causal mask,   prefill and decode are the SAME op;         |   |
|  |                    routed by query-batch size, not "phase"):   |   |
|  |                                                                |   |
|  | Default path — decode AND prefill <= 11673 tok                 |   |
|  |   (paged-FP8 KV read directly; SGLANG_SM120_SPARSE_DECODE):    |   |
|  |   hmma  -> sparse_decode_fwd      [HMMA custom .so] *          |   |
|  |                                                                |   |
|  | Large-batch path — prefill > 11673 tok (same math, KV pre-     |   |
|  |   staged to flat bf16; SGLANG_SM120_SPARSE_PREFILL):           |   |
|  |   hmma  -> sparse_prefill_fwd     [HMMA custom .so] *          |   |
|  |                                                                |   |
|  | Output:                                                        |   |
|  |   wo_a FP8 einsum                      [DeepGEMM]              |   |
|  |   wo_b FP8 GEMM + AllReduce            [Triton+NCCL]           |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- HC_POST (attention) -----------------------------------------+   |
|  | mhc_post fused combine+residual        [TileLang]              |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- HC_PRE (FFN) ------------------------------------------------+   |
|  | (same as attention HC_PRE)       [DeepGEMM+TileLang]           |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- MoE FFN (43 MoE layers)  * ----------------------------------+   |
|  | Router gate GEMM + topk                [Triton]                |   |
|  |                                                                |   |
|  | * MXFP4xMXFP4 Fused MoE (FlashInfer CuTe-DSL SM120):           |   |
|  |    weights: E2M1 int8 + E8M0/32 (loaded as-is)                 |   |
|  |    MmaMXF4Op -> mma.kind::mxf4 .scale_vec::2X .ue8m0           |   |
|  |                                                                |   |
|  |  +-- DECODE (captured CUDA graph replay) ------------------+   |   |
|  |  | bs 1/2/4 (routed<=40)  -> MICRO  (per-M, fixed)         |   |   |
|  |  | bs 8/16  (routed<=640) -> STATIC (per-M, fixed)         |   |   |
|  |  |   Phase-1: quantize x -> MXFP4 (E8M0/32 self-sc)        |   |   |
|  |  |   FC1 (w3|w1 gate/up) GEMM                              |   |   |
|  |  |   SiLU(gate)*up + Phase-2 requant -> MXFP4              |   |   |
|  |  |   FC2 (w2 down) GEMM           [CuTe-DSL JIT]           |   |   |
|  |  +---------------------------------------------------------+   |   |
|  |  +-- EAGER / PREFILL (non-graph) --------------------------+   |   |
|  |  | routed<=640 -> STATIC per-M (MoEStaticKernel)           |   |   |
|  |  | routed >640 -> DYNAMIC W4A4 (M-independent RT)          |   |   |
|  |  +---------------------------------------------------------+   |   |
|  |                                                                |   |
|  | TP AllReduce                           [NCCL LL/Ring]          |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- Dense FFN (3+15 layers) -------------------------------------+   |
|  | (Same as FP8: Triton FP8 fused MoE runner)                     |   |
|  +----------------------------------------------------------------+   |
|     |                                                                 |
|     v                                                                 |
|  +-- HC_POST (FFN) -----------------------------------------------+   |
|  | mhc_post fused combine+residual        [TileLang]              |   |
|  +----------------------------------------------------------------+   |
+-----------------------------------------------------------------------+
    |
    v
+-- LM HEAD ------------------------------------------------------------+
| fused_hc_head (weighted sum + RMSNorm) [Triton]                       |
| lm_head FP8 GEMM -> logits             [Triton FP8]                   |
+-----------------------------------------------------------------------+
    |
    v
+-- SAMPLING -----------------------------------------------------------+
| top_k_top_p_sampling_from_probs        [FlashInfer AOT]               |
+-----------------------------------------------------------------------+
    |
    v
TOKEN OUT
```
