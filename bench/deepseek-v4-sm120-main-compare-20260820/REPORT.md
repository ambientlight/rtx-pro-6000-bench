# DeepSeek-V4-Flash-0731: SGLang-main comparison — 2026-08-20

**PASS:** native SGLang main improved output throughput, TTFT, TPOT and
end-to-end latency in all nine matched cells against the previous custom stack.
Both images completed 16/16 requests in every cell.

Docker image: [`ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a`](https://hub.docker.com/r/ambientlight/sglang-sm120-mxfp4?tag=2026.08.0-cu130-sm120a).

## Setup

- 4 × RTX PRO 6000 Blackwell Max-Q (SM120), TP4, 300 W/GPU; DeepSeek-V4-Flash-0731.
- Control: SGLang `3e5151a1309fbe1759436a643071f2dd54d5fe28`, custom
  FlashMLA/HMMA attention and FlashInfer `eb68b4d865a497afd5d26ce02e60c4ab5075361e` W4A4 MoE.
- Candidate: SGLang `b03ac355e795b3a86b26b8732c47c0965fd71bbc`, native SM120
  attention and FlashInfer 0.6.17 W4A8 MoE (MXFP4 weights, MXFP8 activations).
- Per cell: 16 deterministic random prompts; 8,192 or 65,536 input tokens,
  1,024 output tokens; unlimited request rate at the concurrency shown below.
- Shared settings: 1M context, FP8 E4M3 KV, 8,192-token prefill chunks, page size
  256, 16 maximum running requests, CUDA graphs 1/2/4/8/16, custom all-reduce off.

## Results

Values are **control → candidate**. Output throughput is output tokens divided
by total run time (prefill + decode); latencies are means. C = concurrency.

| Input | C | Output tok/s | TTFT s | TPOT ms | E2E s |
|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 46.94 → 82.65 | 3.809 → 1.383 | 17.60 → 10.76 | 21.816 → 12.390 |
| 8K | 2 | 73.79 → 158.62 | 7.250 → 0.414 | 20.04 → 12.22 | 27.754 → 12.911 |
| 8K | 4 | 113.32 → 297.00 | 12.151 → 0.427 | 23.45 → 13.06 | 36.145 → 13.790 |
| 8K | 8 | 158.00 → 500.35 | 20.628 → 0.615 | 30.52 → 15.40 | 51.847 → 16.370 |
| 8K | 16 | 143.83 → 784.22 | 33.467 → 1.138 | 54.67 → 19.30 | 89.397 → 20.887 |
| 64K | 1 | 18.38 → 47.19 | 33.603 → 10.444 | 21.61 → 11.00 | 55.712 → 21.701 |
| 64K | 2 | 22.47 → 148.95 | 50.559 → 0.644 | 39.66 → 12.80 | 91.134 → 13.739 |
| 64K | 4 | 25.80 → 256.65 | 85.447 → 0.890 | 71.64 → 14.67 | 158.731 → 15.897 |
| 64K | 8 | 28.02 → 432.78 | 145.129 → 1.709 | 143.90 → 16.58 | 292.341 → 18.675 |

Seed-0 prompts were reused across sequential cells, so later concurrency cells
include prefix-cache reuse rather than cold prefill. An independent 8K/C1
post-restart repeat still beat the control: 73.34 output tok/s, 1.389 s TTFT,
12.29 ms TPOT and 13.962 s E2E.

The FP4-indexer, HMMA-hybrid and W4A4-replay probes did not improve overall
throughput over native main. W4A4 had lower TPOT (9.86 vs 10.76 ms) but worse
TTFT (2.978 vs 1.383 s) and output throughput (78.35 vs 82.65 tok/s).

## Artifacts

Raw JSON, telemetry and plots are preserved under the conventional model/power/TP
directory, with the original API model alias retained in filenames:

- [Control](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/current/) and [candidate](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/main-b03ac355/).
- [Cold recheck](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/main-b03ac355-cold-recheck/), [FP4 indexer](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/main-b03ac355-fp4-indexer-fix/), [HMMA hybrid](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/main-b03ac355-hmma-hybrid/) and [W4A4 replay](../deepseek-v4-flash-0731_W300_TP4_sglang/sglang-main-compare-20260820/main-b03ac355-w4a4-replay/).
- [Release image](../../docker/sglang-main-sm120/Dockerfile.release), [build script](../../docker/sglang-main-sm120/build-release.sh) and [image documentation](../../docker/sglang-main-sm120/DOCKERHUB.md). Use the benchmark settings above when reproducing; deployment defaults may differ.
