# MiniMax-M3 — Native MXFP4 W4A4 on SM120

Recipe for serving **MiniMax-M3** (428B MoE / 23B active, MiniMax Sparse Attention) with **native
MXFP4×MXFP4 (W4A4)** fused MoE + a **native MXFP8 weight-only linear** path — on **4× RTX PRO 6000 Blackwell
(SM120, TP=4)**. It carries the [DeepSeek-V4-Flash native-MXFP4 recipe](./DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md)
to M3, whose checkpoint is **compressed-tensors `mixed-precision`** (routed experts MXFP4, everything else
MXFP8) and whose **clamped SwiGLU-OAI** activation is the one MoE-epilogue piece DSV4 (plain SiLU) didn't need.

This wires together two forks plus the stock sglang M3 stack:
- [flashinfer](https://github.com/ambientlight/flashinfer) (native SM120 MXFP4 MoE + the **swigluoai** clamp
  epilogue added here) — injected via `PYTHONPATH`.
- the experimental **sglang `dev-cu13-minimax-m3` image** (M3 model def + a real SM120 **Triton** MSA
  attention backend) with the MXFP4-MoE bridge, the MXFP8-linear scheme, the compressed-tensors routing, and
  the checkpoint loader fix added on top.

Unlike the DSV4 recipe, **M3 does NOT use a custom HMMA attention kernel** — its MiniMax Sparse Attention runs
on the stock sglang SM120 Triton kernels (block-sparse, two-step indexer→main). Only the **MoE** and the
**MXFP8 linears** are custom on this stack.

---

## Specs

| Component | Spec |
|---|---|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 4×96 GB) |
| Interconnect | PCIe (no NVLink) — P2P over PHB/NODE host bridges; NCCL tuned accordingly |
| Host driver | 595.80 (staged at `/mnt/hot/minimax-m3-nvhost`) |
| Sandbox | bubblewrap (`bwrap`) over the `dev-cu13-minimax-m3` image rootfs |

## Software stack

| Package | Version | Notes |
|---|---|---|
| CUDA | 13.0 (image) / host `nvcc` ≥ 12.8 | SM120 = `sm_120a` |
| PyTorch | image build (cu130) | |
| flashinfer-python | repo `/mnt/hot/ambientlight/repos/flashinfer` (labeled 0.6.13, **not on PyPI**) | the image's baked 0.6.12 **lacks** native mxfp4; inject the repo via `PYTHONPATH`. Needs `FLASHINFER_DISABLE_VERSION_CHECK=1` |
| sglang | `dev-cu13-minimax-m3` image rootfs (`/mnt/hot/minimax-m3-rootfs/sgl-workspace/sglang`) | M3 model def + SM120 Triton MSA backend + the edits below |
| Model | `olka-fi/MiniMax-M3-MXFP4` (compressed-tensors mixed-precision) at `/mnt/hot/ambientlight/models/minimax-m3-mxfp4` | |

### Model shape (from `config.json`)

| | Value |
|---|---|
| Layers | 60 (first **3 dense**, last **57 MoE** — `moe_layer_freq=[0,0,0,1×57]`) |
| Hidden / Inter | 6144 / 3072 |
| Attention | GQA, `num_attention_heads=64`, `num_key_value_heads=4`, `head_dim=128` |
| Routed experts | 128, `top_k=4`, `routed_scaling_factor=2.0`, `scoring_func=sigmoid` |
| Shared experts | 1 (`shared_intermediate_size=3072`) — fusion must be **OFF** (E stays 128) |
| Activation | clamped **SwiGLU-OAI**: `swiglu_alpha=1.702`, `swiglu_limit=7.0`, beta=1.0 |
| Sparse attn | block-sparse, `sparse_block_size=128`, `sparse_topk_blocks=16`, `sparse_index_dim=128`, `sparse_num_index_heads=4`, `score_type=max` |
| Vocab | 200064 |

### What is MXFP4 vs MXFP8 vs BF16 (from `quantization_config`)

`compressed-tensors`, `format=mixed-precision`, two config groups + an ignore list:

| Group | Targets | Precision |
|---|---|---|
| `group_0` | `re:.*\.experts\.\d+\..*` — the **128 routed experts** | **MXFP4** (num_bits=4 float, group-32, no input-act → weight-only) |
| `group_1` | self_attn q/k/v/o + index_q/k, dense MLP gate/up/down, shared-expert gate/up/down (+ the fused `qkv_proj`/`gate_up_proj`) | **MXFP8** (num_bits=8 float, group-32, no input-act → weight-only) |
| `ignore` | `lm_head`, `embed_tokens`, `block_sparse_moe.gate` (router), all `vision_tower.*` | **BF16** (unquantized) |

So only the routed-experts `FusedMoE` is W4A4; everything else dense is MXFP8 weight-only; router/lm_head/embeds
are full precision.

---

## Quick start (Docker)

The unified image [`ambientlight/sglang-sm120-mxfp4`](https://hub.docker.com/r/ambientlight/sglang-sm120-mxfp4)
bakes the flashinfer MXFP4-MoE + swigluoai fork and the sglang M3 stack (MXFP8 linears, compressed-tensors
routing, the loader fix) for `sm_120a`. Mount the checkpoint at `/model` and serve with `MODEL=m3`:

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -e MODEL=m3 \
  -v /mnt/hot/ambientlight/models/minimax-m3-mxfp4:/model:ro \
  ambientlight/sglang-sm120-mxfp4:latest        # OpenAI API on :8000, ~8 min to /v1/models
```

- The image runs in a real container (native `/sys`), so the optimized PCIe NCCL config (see the [environment variable reference](#environment-variable-reference)) applies without the bwrap `--ro-bind /sys` workaround — it's baked into the entrypoint.
- Server flags (`CONTEXT_LENGTH`, `MEM_FRACTION_STATIC` default 0.95, `KV_CACHE_DTYPE=fp8_e4m3`, reasoning/tool-call parsers, `SGLANG_M3_FORCE_STATIC`) are baked into the entrypoint and overridable via `-e`.
- Get the weights: `huggingface-cli download olka-fi/MiniMax-M3-MXFP4 --local-dir /mnt/hot/ambientlight/models/minimax-m3-mxfp4`

Build it yourself from [`docker/sm120-unified/`](../docker/sm120-unified/) (`./build.sh`).

---

## How the stack selects each path at runtime

```
 sglang.launch_server  (--quantization compressed-tensors, mixed-precision)
      |
      +-- MoE experts (128 routed, group_0 = MXFP4)
      |     compressed_tensors.py :: get_moe_scheme
      |       _is_mxfp4_weight_only(num_bits=4,float,group-32,no-input-act)   # reordered BEFORE wNa16
      |       and not SGLANG_M3_FORCE_MARLIN
      |       and _flashinfer_has_native_mxfp4()    # "mxfp4" in sm120_moe_supported_quant_modes()
      |         -> Mxfp4W4A4MoEScheme -> Mxfp4W4A4MoEMethod.apply() ─┐
      |                                                              v
      |     FlashInfer (repo): launch_sm120_moe(quant_mode="mxfp4", activation="swigluoai",
      |                          swiglu_alpha=1.702, swiglu_limit=7.0, swiglu_beta=1.0)
      |       CuTe-DSL fused SwiGLU-OAI MmaMXF4Op, uint8 E8M0 self-scaling, [w3,w1] gate/up
      |
      +-- Dense / attention / shared-expert linears (group_1 = MXFP8)
      |     compressed_tensors.py :: _get_scheme_from_parts
      |       _is_mxfp8_weight_only(num_bits=8,float,group-32,no-input-act)   # reordered BEFORE wNa16
      |         -> CompressedTensorsW8A8Mxfp8.apply_weights()
      |             -> mxfp8_native.dot_scaled_mxfp8_blockscaled_linear (uint8 E8M0 verbatim)
      |
      +-- MiniMax Sparse Attention (block-sparse, top-16 of 128-token blocks)
      |     minimax_sparse_backend.py: use_msa = msa_available() ...   # FALSE on SM120
      |       (fmha_sm100 is SM100-only -> M3 runs the stock SM120 TRITON 2-step path)
      |     minimax_sparse_decode():
      |       (1) flash_decode_with_topk_idx   # lightning indexer: select top-16 KV blocks
      |       (2) flash_decode_with_gqa_share_sparse   # GQA main attention over selected blocks
      |
      +-- Router gate / lm_head / embeds  -> BF16 (ignore list)
```

### MoE feature-probe

`get_moe_scheme` selects `Mxfp4W4A4MoEScheme` only when the experts are MXFP4 weight-only **and**
flashinfer's **public capability API** reports `mxfp4` (`_flashinfer_has_native_mxfp4()` →
`"mxfp4" in sm120_moe_supported_quant_modes()`). On the image's stock 0.6.12 the set lacks `mxfp4`, the probe
is False, and it falls back to Marlin W4A16. Injecting the repo flashinfer on `PYTHONPATH` flips it True — that
is the activation mechanism. `SGLANG_M3_FORCE_MARLIN=1` forces the (slow) reference path for A/B.

---

## Environment variable reference

| Variable | Value | Why |
|---|---|---|
| `PYTHONPATH` | `<repo_flashinfer>:/sgl-workspace/sglang/python` | inject native-mxfp4 flashinfer over the image's 0.6.12 |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | repo flashinfer (0.6.13) vs image cubin (0.6.12) |
| `FLASHINFER_WORKSPACE_BASE` | `/mnt/hot/minimax-m3-ficache` | writable CuteDSL JIT cache (image FS is RO) |
| `NCCL_SHM_DISABLE` | `0` | **SHM ON** → legacy-IPC P2P all-reduce (45 µs/12 KB). `=1` forces TCP socket-loopback (277 µs, 6× slower) |
| `NCCL_CUMEM_ENABLE` | `0` | legacy IPC path; cuMem fabric handles hit CUDA-801 under bwrap |
| `NCCL_SOCKET_IFNAME` | `lo` | single-host TP=4 loopback; else NCCL auto-picks a dead host `veth*`/`docker0` → `bind failed` crash + 24-min capture |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | `0` / `1` | P2P on (PHB/NODE bridges), no InfiniBand |
| (bind) `/sys` | ro-bind | NCCL PCIe-topology detection → direct P2P (`maxBw 1.2→48`) |
| `SGLANG_M3_FORCE_STATIC` | `1` | route large-prefill batches to the smem-safe **static** MoE kernel (the dynamic kernel overflows SM120 smem at k=6144) |
| `SGLANG_M3_FORCE_MARLIN` | `1` (A/B only) | revert MoE to slow Marlin W4A16 reference |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA-graph-capture headroom; avoids fragmentation OOM |
| `--skip-server-warmup` | flag | else the warmup forward's first-decode JIT trips the watchdog |
| `--watchdog-timeout` | `1200` | absorbs the one-time first-decode JIT |
| `--disable-shared-experts-fusion` | flag | keep E = 128 (the MXFP4 scheme asserts it) |

---

## End-to-end pipeline (MXFP4 W4A4 decode, 1 token)

Blocks marked `*` are the custom SM120 paths added for this stack. Only the **routed-expert FFN GEMM** is MXFP4
W4A4; the dense/attention/shared-expert linears are MXFP8 weight-only; sparse attention is stock SM120 Triton.

```
TOKEN IN
    |
    v
+-- EMBEDDING                                          [Torch, BF16] -----+
| VocabParallelEmbedding (embed_tokens, ignore-list = unquantized)        |
+-------------------------------------------------------------------------+
    |
    v
+-- PER LAYER x60  (layers 0-2 DENSE, layers 3-59 MoE) -------------------+
|                                                                         |
|  +-- INPUT RMSNorm                         [Triton]                  |  |
|  |                                                                   |  |
|  +-- MINIMAX SPARSE ATTENTION (GQA: 64 q-heads, 4 kv-heads, d=128) --+  |
|  |  qkv_proj (fused q/k/v)   MXFP8 weight-only  * dot_scaled         |  |
|  |  q/k RMSNorm + RoPE (theta=5e6)             [Triton]              |  |
|  |  KV-cache write (paged, bf16)               [Triton]              |  |
|  |                                                                   |  |
|  |  Lightning INDEXER (index_q/k_proj MXFP8 *, sparse_index_dim=128):|  |
|  |    flash_decode_with_topk_idx  -> select top-16 of the 128-tok    |  |
|  |      KV blocks per query        [Triton, SM120; split-KV chunks]  |  |
|  |                                                                   |  |
|  |  MAIN SPARSE ATTENTION (block-sparse, no MSA fmha on SM120):      |  |
|  |    flash_decode_with_gqa_share_sparse  over the selected blocks   |  |
|  |      [Triton, SM120]  (score -> attn -> merge)                    |  |
|  |                                                                   |  |
|  |  o_proj   MXFP8 weight-only  * dot_scaled  + TP AllReduce  [NCCL] |  |
|  +-------------------------------------------------------------------+  |
|     |                                                                   |
|     v                                                                   |
|  +-- POST-ATTN RMSNorm + residual           [Triton]                 |  |
|     |                                                                   |
|     v                                                                   |
|  +-- FFN ------------------------------------------------------------+  |
|  |  layers 0-2 : DENSE MLP (gate/up/down)  MXFP8 weight-only *        |  |
|  |  layers 3-59: MoE (128 routed + 1 shared) --------------------+    |  |
|  |    Router gate GEMM (BF16) + sigmoid top-4   [Triton]         |    |  |
|  |      routed_scaling_factor=2.0 baked into topk_weights        |    |  |
|  |                                                               |    |  |
|  |    SHARED expert (gate/up/down)  MXFP8 weight-only *           |    |  |
|  |                                                               |    |  |
|  |  * ROUTED experts: MXFP4 x MXFP4 Fused MoE (FlashInfer SM120): |    |  |
|  |     weights E2M1 int8 + E8M0/32 (loaded verbatim, [w3,w1])     |    |  |
|  |     MmaMXF4Op -> mma.kind::mxf4 .scale_vec::2X .ue8m0          |    |  |
|  |                                                               |    |  |
|  |   +-- DECODE (captured CUDA-graph replay) -----------------+  |    |  |
|  |   | routed_rows = bs*top_k(4):                             |  |    |  |
|  |   |  <=40  -> MICRO  (bs<=10; single_token fast path @bs=1)|  |    |  |
|  |   |  <=640 -> STATIC (bs 12..32)                           |  |    |  |
|  |   |   P1: quantize x -> MXFP4 (E8M0/32 self-scale)         |  |    |  |
|  |   |   FC1 (w3|w1) GEMM                                     |  |    |  |
|  |   |   * SwiGLU-OAI: g=fmin(g,7); u=clamp(u,+-7);           |  |    |  |
|  |   |       g*sigmoid(1.702*g)*(u+1)  + P2 requant -> MXFP4  |  |    |  |
|  |   |   FC2 (w2 down) GEMM            [CuTe-DSL JIT]         |  |    |  |
|  |   +--------------------------------------------------------+  |    |  |
|  |   +-- EAGER / PREFILL (chunked over M, SGLANG_M3_FORCE_STATIC)+ |    |  |
|  |   | static per-M sub-launches (<= ws_cap rows each)        |  |    |  |
|  |   +--------------------------------------------------------+  |    |  |
|  |                                                               |    |  |
|  |    down_proj / w2 + TP AllReduce            [NCCL]            |    |  |
|  +---------------------------------------------------------------+    |  |
|     |                                                                   |
|     v                                                                   |
|  +-- POST-FFN RMSNorm + residual             [Triton]                 |  |
+-------------------------------------------------------------------------+
    |
    v
+-- LM HEAD  (BF16, ignore list) ----------------------------------------+
| final RMSNorm -> lm_head GEMM -> logits     [Torch/Triton]              |
+------------------------------------------------------------------------+
    |
    v
+-- SAMPLING -----------------------------------------------------------+
| reasoning-parser minimax-m3 / tool-call-parser minimax-m3 downstream  |
+----------------------------------------------------------------------+
    |
    v
TOKEN OUT
```

---

## Performance

All numbers TP=4, SM120, 300 W per GPU (stock, uncapped), cuda-graph decode, fp8 KV + split-K MXFP8 linears.

### Decode throughput (short prompt, steady-state)

The native path batches concurrent decode where Marlin serializes. Decode-only aggregate tok/s
(`scripts/decode_only_sweep.py`):

| conc | 1 | 2 | 4 | 16 | 32 |
|---|---|---|---|---|---|
| **agg tok/s** | **79** | 131 | 231 | 701 | 975 |

conc-1 already beats the Marlin W4A16 baseline (24.6 tok/s); it scales monotonically to ~975 tok/s at conc-32.
Three accuracy-neutral wins got it here: the NCCL transport fix (`NCCL_SHM_DISABLE=0`+`NCCL_CUMEM_ENABLE=0`+
`/sys`, ~6× all-reduce), fp8 KV (no decode cost, large KV pool), and split-K MXFP8 linears (+25 % at conc-1).

### Long-context scaling to 1M (conc-1)

Single-stream, input 2K → ~1.04M, decode steady-state (`scripts/longprompt_sweep.py`; ctx 1048576,
chunk 8192, mem-fraction 0.95 — fits a 1,069,580-token fp8 KV pool). Compared against the
[DeepSeek-V4-Flash single-seq sweep](./DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md#long-context-scaling-to-1m-single-stream)
— **same hardware, same 300 W cap**, so the difference is purely architectural (M3 GQA block-sparse vs DSV4 MLA):

| Input | M3 TTFT | M3 prefill tok/s | **M3 decode tok/s** | DSV4 decode tok/s | M3/DSV4 |
|---|---:|---:|---:|---:|:---:|
| 2,048 | 0.7 s¹ | — ¹ | **76.4** | 59.8 | 1.3× |
| 32,768 | 0.4 s | 79,300 | **72.6** | 52.9 | 1.4× |
| 65,536 | 0.5 s | 129,500 | **69.4** | 47.1 | 1.5× |
| 131,072 | 0.9 s | 139,800 | **65.8** | 38.7 | 1.7× |
| 262,144 | 1.9 s | 134,900 | **59.3** | 28.6 | 2.1× |
| 524,288 | 4.2 s | 125,700 | **50.1** | 18.7 | 2.7× |
| **~1.04M** | **7.7 s** | **136,800** | **39.2** | **11.1** | **3.5×** |

¹ 2K is prefill-trivial (TTFT is fixed launch overhead, not compute), so its "prefill tok/s" isn't meaningful.
**These are fresh runs on the dynamic-M stack**: the earlier long-context sweep charged a per-chunk CuteDSL
recompile storm to TTFT (1M prefill took **513 s @ ~2,028 tok/s**); compiling the SM120 MoE kernel with a
symbolic token dim (`FLASHINFER_B12X_STATIC_DYNAMIC_M`, default on) collapses that to **7.7 s @ 136,800 tok/s —
a 67× prefill TTFT win**, with **decode unchanged** (the fix is purely prefill-side). DSV4 column = its **best**
long-ctx config (split-KV indexer).

**M3 decode degrades only −49 % over a 512× context increase (76.4 → 39.2 tok/s); DSV4 drops −81 % even with
its split-KV indexer (59.8 → 11.1).** The gap widens with length — 1.3× at 2K to 3.5× at 1M. Driver: M3's
sparse-attention decode scans only the top-16 KV blocks (cost ~context-independent), the indexer auto-shards its
logits scan across up to 256 CTAs at conc-1, fp8 KV keeps KV bandwidth low, and split-K keeps the MXFP8 linears
off the critical path. Post-dynamic-M, prefill sustains ~130 k tok/s all the way out, so the full 1M prompt is
processed in **7.7 s** (was ~8.5 min pre-fix). The full 1M single sequence fits at mem-fraction 0.95 (1M run
holds 370 GB VRAM, KV 98 %; ~1,083 W mean / 1,196 W peak).

### Agentic eval — SWE-bench Verified

**374 / 500 = 74.8%** (SWE-bench Verified, full split), via
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) v2.4.2 driving the local server with **native
tool-calling**. Sampling per the M3 model card (thinking-on): `temperature 1.0, top_p 0.95, top_k 40,
frequency_penalty 0.1`, `max_tokens 65536`, `step_limit 250`. Full config, trajectories, and reproduce guide:
[`evals/swebench-verified/minimax-m3-2026-06-25/`](../evals/swebench-verified/minimax-m3-2026-06-25/README.md).

Two harness-side fixes were needed to get a clean number (neither touches the model/server):

| Fix | What | Effect |
|---|---|---|
| **Submit nudge** | step-aware reminder in `observation_template` (soft at step 150, hard "submit now" at 200, via the in-scope `n_model_calls`/`step_limit` vars) | turn-limit (`LimitsExceeded`) failures **43 → 6**; +17 resolved; **−13 % wall-clock** (the eliminated 250-turn runs were the most expensive) |
| **Command timeout 60 → 600 s** | per-command bash timeout (matches DSV4 thinking) | slow matplotlib/sympy/django test suites no longer killed mid-run |

A **temp 0.2** variant (to match DSV4) was tried and abandoned — low temperature removes the sampling noise
the agent needs to escape failed-approach loops (a single trajectory hit 250 turns repeating
`git checkout`→rewrite→test 76×), confirming the model card's temp 1.0 is load-bearing for agentic recovery.

> Throughput numbers above are at temp-1.0 short-prompt/long-context sweeps; the SWE-bench run used the same
> server (dynamic-M MoE JIT, fp8 KV, split-K MXFP8, NCCL transport fix all ON).

---

## Correctness & quality

> Throughput is measured (see [Performance](#performance)); agentic-eval is **SWE-bench Verified 374/500 =
> 74.8%** (see [Agentic eval](#agentic-eval--swe-bench-verified)).

| Check | Result |
|---|---|
| MoE numerical parity (`spikes/m3_mxfp4_real.py`, real weights, dual-golden) | swigluoai **cos 0.984** vs BF16 golden; `[w3,w1]` swap pinned (swap-off 0.891); clamp bit-identical in fp32 |
| MXFP8 linear (`spikes/m3_mxfp8_real.py`, `m3_mxfp8_tp.py`) | `dot_scaled` vs BF16 **cos 0.9996**, correct under TP=4 col+row sharding |
| Sparse attention (in-server dumps) | prefill + decode `o` vs dense softmax **cos 1.00000** on all 4 ranks |
| Full-server correctness (eager + cuda-graph) | "capital of Japan?"→**Tokyo**; "17+25"→**42**; greedy **deterministic** |
| Quality (`spikes/m3_gsm8k_live.py`, GSM8K temp=0) | **40/41 = 97.6%**, 0 truncations |
| Agentic eval (mini-swe-agent v2.4.2, SWE-bench Verified) | **374/500 = 74.8%** (temp 1.0, native tool-calling; ±2 across re-scores) |
| Load-completeness guard (`spikes/m3_load_guard_sim.py`) | fires on the original `weight_packed` silent-skip; silent on the fix |

---

## Relationship to the DSV4 recipe

| | DeepSeek-V4-Flash | MiniMax-M3 (this) |
|---|---|---|
| MoE | native MXFP4 W4A4, SiLU | native MXFP4 W4A4, **clamped SwiGLU-OAI** epilogue (added) |
| Dense/attn linears | FP8 (DeepGEMM/Triton) | **MXFP8 weight-only** (`dot_scaled`, scheme added) |
| Checkpoint | `wX.weight`/`wX.scale` (clean) | compressed-tensors mixed-precision, `weight_packed`/`weight_scale` (**loader fix**) |
| Attention | MLA, per-token sparse, **custom HMMA** decode/prefill `.so` | GQA, block-sparse, **stock SM120 Triton** (2-step indexer→main); no custom kernel |
| Shared infra | `launch_sm120_moe` bridge, repo flashinfer, PCIe NCCL tuning | same bridge + same NCCL transport fix |
