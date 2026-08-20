# DeepSeek-V4 SM120 SGLang-main qualification — 2026-08-20

## Decision

**PASS.** SGLang main at benchmark cutoff commit
`b03ac355e795b3a86b26b8732c47c0965fd71bbc` beats the deployed custom stack in
all nine matched cells. No measured primary metric regressed: aggregate output
throughput increased, while mean TTFT, mean TPOT, and mean end-to-end latency
all decreased.

The benchmark-qualified image has been cut locally as:

```text
ambientlight/sglang-sm120-mxfp4:2026.08.20-main-b03ac355-cu130
sha256:72f0692d5fcdf9dc42e167323c631cb24f828ee0339bf1237534b196f0e5eabc
```

It has deliberately **not** been promoted to `latest`, pushed, or deployed.
The original `dsv4` and `qwen3-embed` services were restored after testing.

## What main replaces

The deployed control is SGLang
`3e5151a1309fbe1759436a643071f2dd54d5fe28` plus a custom SM120 FlashMLA/HMMA
attention path and FlashInfer
`eb68b4d865a497afd5d26ce02e60c4ab5075361e` W4A4 fused MoE path.

The qualified main image uses upstream paths on both sides of the model:

- Attention: `DeepseekV4AttnBackend` with main's SM120 FlashMLA implementation.
- Experts: `Mxfp4FlashinferCutlassMoEMethod` from FlashInfer 0.6.17. The
  checkpoint weights remain MXFP4, but SM120 activations are MXFP8: this is
  W4A8, not the custom W4A4 path.
- Runtime: PyTorch 2.13, CUDA 13.0.3, `sglang-kernel` 0.4.6.post1,
  TileLang 0.1.12, and `sgl-deep-ep` 0.1.2.

The remote `main` head advanced to
`7f8f030000b628ea2cb033e7457a13dd0ac80f99` after the image was built. The
three intervening commits only alter diffusion code/tests; there is no diff in
`python/sglang/srt`, `python/sglang/kernels`, or `python/sglang/launch_server.py`.
Thus `b03ac355` remains the latest DeepSeek/text-generation runtime at this
qualification cutoff while providing an immutable image revision.

## Benchmark setup

- Hardware: 4 × NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120), TP=4, 300 W/GPU.
- Model: `DeepSeek-V4-Flash-0731`.
- Workload per cell: 16 requests, random deterministic 8,192- or 65,536-token
  input, 1,024 output tokens, unlimited request rate.
- Concurrency: 1/2/4/8 for both lengths, plus 16 for 8K.
- Common serving shape: 1M context, FP8 E4M3 KV cache, 8,192 chunked prefill,
  page size 256, maximum 16 running requests, CUDA graphs at 1/2/4/8/16, and
  custom all-reduce disabled.
- Success: 16/16 requests and zero failures in every cell for both images.

Values below are `deployed current -> exact main`. Output is aggregate output
tokens/s; TTFT and E2E are means in seconds; TPOT is the mean in milliseconds.

| Input | C | Output tok/s | TTFT s | TPOT ms | E2E s |
|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 46.94 -> 82.65 | 3.809 -> 1.383 | 17.60 -> 10.76 | 21.816 -> 12.390 |
| 8K | 2 | 73.79 -> 158.62 | 7.250 -> 0.414 | 20.04 -> 12.22 | 27.754 -> 12.911 |
| 8K | 4 | 113.32 -> 297.00 | 12.151 -> 0.427 | 23.45 -> 13.06 | 36.145 -> 13.790 |
| 8K | 8 | 158.00 -> 500.35 | 20.628 -> 0.615 | 30.52 -> 15.40 | 51.847 -> 16.370 |
| 8K | 16 | 143.83 -> 784.22 | 33.467 -> 1.138 | 54.67 -> 19.30 | 89.397 -> 20.887 |
| 64K | 1 | 18.38 -> 47.19 | 33.603 -> 10.444 | 21.61 -> 11.00 | 55.712 -> 21.701 |
| 64K | 2 | 22.47 -> 148.95 | 50.559 -> 0.644 | 39.66 -> 12.80 | 91.134 -> 13.739 |
| 64K | 4 | 25.80 -> 256.65 | 85.447 -> 0.890 | 71.64 -> 14.67 | 158.731 -> 15.897 |
| 64K | 8 | 28.02 -> 432.78 | 145.129 -> 1.709 | 143.90 -> 16.58 | 292.341 -> 18.675 |

The independently useful cold/concurrency-1 comparisons are decisive without
depending on cross-cell prefix-cache reuse:

- 8K C1 first matrix run: output +76.1%, TTFT -63.7%, TPOT -38.9%, E2E -43.2%.
- 8K C1 post-restart repeat: 73.34 output tok/s, 1.389 s TTFT, 12.29 ms
  TPOT, and 13.962 s E2E; it still beats current on every metric.
- 64K C1: output +156.7%, TTFT -68.9%, TPOT -49.1%, E2E -61.0%.

### Cache caveat

Each side ran deterministic seed-0 prompts sequentially, so later concurrency
cells can reuse prefixes left in the unified radix cache. Main benefits much
more strongly from that reuse. This is a real warm-cache behavior and the
procedure is matched across images, but it should not be mistaken for cold
prefill. The C1 results and the 8K post-restart repeat establish the no-regression
decision independently.

## Replay experiments

The existing patch stack was split into hypothesis-driven probes on top of the
same main image:

| 8K / C1 variant | Output tok/s | TTFT s | TPOT ms | E2E s | Decision |
|---|---:|---:|---:|---:|---|
| Native main W4A8 | 82.65 | 1.383 | 10.76 | 12.390 | Winner |
| FP4 indexer guard/fix | 81.65 | 1.510 | 10.78 | 12.541 | Reject |
| Custom HMMA attention + main MoE | 54.28 | 2.254 | 16.24 | 18.864 | Reject |
| Main attention + replayed W4A4 MoE | 78.35 | 2.978 | 9.86 | 13.069 | Reject overall |

The W4A4 replay retains a decode-only advantage (lower TPOT) but loses too much
prefill/TTFT and total throughput against native main for this workload. It
still beats the deployed image overall, but there is no reason to carry that
compatibility stack into this release. A strongly decode-dominated workload
should be gated separately before removing W4A4 from that use case.

## Upstream support timeline

| Date (UTC) | Milestone | Consequence for this deployment |
|---|---|---|
| 2026-05-08 | [SGLang DeepSeek-V4 base support #23882](https://github.com/sgl-project/sglang/pull/23882) merged. | Model/runtime integration begins. |
| 2026-06-01 | [SM120 support #24692](https://github.com/sgl-project/sglang/pull/24692) merged; it shipped in [v0.5.13 on June 13](https://github.com/sgl-project/sglang/releases/tag/v0.5.13). | First functional RTX/SM120 support, initially with fallback paths. |
| 2026-06-19 | [Marlin SM120 MXFP4 MoE #28231](https://github.com/sgl-project/sglang/pull/28231) merged. | First upstream SM120 MXFP4 expert acceleration. |
| 2026-07-18 | [FlashInfer SM120 MXFP4 MoE + TP2/TP4 #30272](https://github.com/sgl-project/sglang/pull/30272) merged. | Native W4A8 FlashInfer expert path used by this candidate. |
| 2026-07-24 | [FP4 indexer #27059](https://github.com/sgl-project/sglang/pull/27059) merged. | Indexer support exists, but its current SM120 backend selection is not viable here and the guarded probe did not win. |
| 2026-08-03 | [SM120 FlashMLA page-split optimization #32320](https://github.com/sgl-project/sglang/pull/32320) merged; [SGLang v0.5.17](https://github.com/sgl-project/sglang/releases/tag/v0.5.17) followed on August 8. | Attention/prefill path is now fast enough to replace the hand-patched HMMA route on this host. |
| 2026-08-10 | FlashInfer's native SM120 [MXFP4×MXFP4 W4A4 fused MoE #4290](https://github.com/flashinfer-ai/flashinfer/pull/4290) merged. | The upstream kernel now exists. |
| 2026-08-11/12 | [FlashInfer v0.6.17](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17) was released, then [SGLang #33997](https://github.com/sgl-project/sglang/pull/33997) pinned it. | v0.6.17 does **not** contain #4290; exact SGLang main therefore still selects W4A8 on SM120. |

### Timeline conclusion

Functional and performance-qualified DeepSeek-V4 support for SM120/MXFP4
weights is available **now** on SGLang main through upstream attention plus
FlashInfer W4A8 MoE. Fully upstream CUDA W4A4 is not end-to-end available in
SGLang yet: FlashInfer #4290 is only on FlashInfer main, no released FlashInfer
version contains it, and no merged SGLang adapter/selector consumes it.

There is no committed ETA for that last W4A4 integration step. The older custom
[SGLang #28025](https://github.com/sgl-project/sglang/pull/28025) and the broader
[#29927 stack](https://github.com/sgl-project/sglang/pull/29927) remain open.
The exact 4× RTX PRO prefill gap is also tracked in open
[issue #33422](https://github.com/sgl-project/sglang/issues/33422). The practical
watchpoint is therefore: first FlashInfer release containing #4290, then a
merged SGLang selection path, followed by the same performance gate.

## Release and reproduction artifacts

- Qualified image definition: `../../docker/sglang-main-sm120/Dockerfile.release`
- Release build: `../../docker/sglang-main-sm120/build-release.sh`
- Exact-main control definition: `../../docker/sglang-main-sm120/Dockerfile`
- Native-main entrypoint: `../../docker/sglang-main-sm120/entrypoint-dsv4.sh`
- Rejected experimental patches/images remain recorded under
  `../../docker/sglang-main-sm120/` and their raw benchmark directories here.
- Complete current and main JSON, telemetry, and plots are under `current/` and
  `main-b03ac355/`.

The default chat request leaks the reasoning trace into `content` with
`reasoning_content=null` on both the deployed image and main, so this is not a
main regression. Supplying `chat_template_kwargs: {"thinking": true}` separates
the fields correctly on the deployed service; callers relying on separate
reasoning should keep sending that explicit option.
