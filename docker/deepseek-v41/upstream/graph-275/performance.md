# DeepSeek serving and Pearl mining at 275 W per GPU

The graph-enabled prototype kept short single-stream decode near 115 tokens/s while producing verifiable work from inference. Useful decode recovered **0.00067%** of the four-card direct miner’s rate. Prefill recovered **0.031%**. On identical arrivals at the same share difficulty, the joined arm completed **0.30–0.53% less total mining work** over 80.5 seconds. The interval reflects the hardware counter sampling boundaries. The arms ran once in sequence, so that difference also includes run-to-run variation.

## Matched serving and mining

The control drains all four warm K780 workers before admission, runs the checkpoint’s native projection, then releases the workers after all four serving ranks fence. The joined arm uses the same workers and admission rules while mining the real layer-zero attention output projection during prefill and six-row DSPARK target verification. Both ran the same full checkpoint and resolved server configuration, including TP4/EP4, all eight full decode CUDA-graph batch sizes, native eager prefill, and the same KV capacity of 3,745,024. Both ran from the staged serving image with patched SGLang sources. All four native startup checks passed on both. [Control settings](measurements/switch-profile.json) · [Joined settings](measurements/joined-profile.json) · [Source and hardware pins](measurement-setup.json)

Four uncached requests each had 2,048 input and 256 generated tokens. The same four prompt hashes were sent at 20-second intervals. The measurement includes the idle interval following the final request. Medians below describe the four requests. Native prefill is the server’s timing. Time to first token and decode throughput are client measurements. The last two rows average the entire schedule, including active serving. The pure direct fleet completed **1,233.67 TMAC/s** in a separate six-second calibration. [Direct miner calibration](direct-calibration.json)

| Metric | Fast switching | Joined, proof qualification target | Joined, same target as direct |
|---|---:|---:|---:|
| Native prefill, tokens/s | 9,769 | 9,669 | 9,615 |
| Time to first token, ms | 220.0 | 223.3 | 224.8 |
| Decode, tokens/s | 114.90 | 115.48 | 115.21 |
| p99 interval between token-bearing stream events, ms | 21.22 | 21.71 | 21.32 |
| Measured target-graph replays per GPU | 437 | 436 | 433 |
| Replays with independently scanned tickets per GPU | 0 | 292 | 290 |
| Effective mining during prefill, % of pure direct | 0 | 0.03130% | 0.03116% |
| Effective mining during decode, % of pure direct | 0 | 0.000669% | 0.000670% |
| Total mining including idle, TMAC/s | 1,040.44–1,041.64 | 1,038.84–1,040.05 | 1,036.16–1,037.35 |
| Total mining including idle, % of pure direct | 84.34–84.43% | 84.21–84.31% | 83.99–84.09% |

In the equal-target run the direct workers could mine for 70.74 seconds, versus 70.69 seconds under fast switching. The joined projection added **416.58 billion MAC-equivalents across four requests**. At the measured direct rate, that represents about **0.338 ms of direct mining in total**, or 84 μs per request. Variations in direct mining while the requests were idle outweighed this contribution. The fixed one-second-gap trial yielded 109.02 versus 110.76 decode tokens/s and 220.3 versus 217.9 ms to first token. Its elapsed horizon varies with response length. [Equal-target comparison](measurements/compare-equal-target-fixed-period.json) · [Qualification-target comparison](measurements/compare-fixed-period.json) · [One-second-gap comparison](measurements/compare-fixed-gap.json)

## Switching latency

These 24 short requests per arm had 256 input and 16 output tokens, separated by 250 ms after each stream closed. The direct workers entered every request already mining. Each worker remained held with a stationary hardware slice cursor throughout every decode. The broker polling timestamp includes ingress and observation time. A worker reports GPU entry by polling its device-start flag. The observed maxima describe this sample.

| Time across four GPUs, ms | Switch median | Switch p95 | Switch max | Joined median | Joined p95 | Joined max |
|---|---:|---:|---:|---:|---:|---:|
| Request send to observed miner drain | 6.07 | 6.94 | 7.40 | 6.25 | 7.47 | 7.84 |
| Response stream close to all miners back on GPU | 20.00 | 21.25 | 21.59 | 20.09 | 20.83 | 21.42 |
| First worker release to all miners back on GPU | 0.261 | 0.519 | 0.532 | 0.258 | 0.523 | 0.705 |

Median time to first token on these short requests rose from **135.0 to 148.1 ms**. For the fast-switch arm, all four GPUs executed 122 target graphs over the burst. The requests counted 98 target verifications in their returned results. That difference is exactly one extra replay per request. The scheduler can launch its next speculative step before discovering that the previous result completed the request. In the longer paired control traces, requests with this extra replay took about 20 ms to resume mining, while requests without it resumed in about 1–2 ms. A targeted scheduling experiment could process an in-flight result early when a lone request is within one speculative block of its token cap, preserving all four GPU fences. [Fast-switch handoffs](measurements/switch-handoff-burst-fleet.json) · [Joined handoffs](measurements/joined-handoff-burst-fleet.json) · [Settled graph counters](measurements/compare-handoff-burst.json)

## Historical prompt replays

Each row is one uncached historical Responses request per arm. The 32K request generated the full 8,192 tokens. The longer prefills generated 512 each. The first 32K time to first token varied substantially between launches. These single observations do not isolate a 32K prefill speedup. The 200K joined result was **10.7% slower** on decode. The completions were sampled independently, so the difference also includes the effect of the generated text on speculative acceptance. [Paired historical measurements](measurements/compare-handbook.json)

| Actual input / output tokens | Switch TTFT, s | Joined TTFT, s | Switch decode, tok/s | Joined decode, tok/s | Earlier 280 W stock graph / eager Pearl decode, tok/s |
|---|---:|---:|---:|---:|---:|
| 32,779 / 8,192 | 7.41 | 3.58 | 112.78 | 112.22 | 111.41 / 10.51 |
| 131,387 / 512 | 13.70 | 13.81 | 112.07 | 107.47 | 107.58 / 10.59 |
| 200,305 / 512 | 21.20 | 21.42 | 117.72 | 105.08 | 103.17 / 10.38 |

Across the three joined historical requests, measured decode work was **0.000625–0.000638%** of pure mining. Prefill was **0.0311–0.0323%**. The companion comparison includes the September historical runs, the prior incumbent, prompt hashes, context-specific effective prefill, and per-rank accounting for the final odd-sized chunk. Short 2K traces had no increments in the hardware or software thermal counters. Across the longer runs GPU 2 accumulated 1.62 s of software thermal slowdown near the 131K/200K control boundary and 2.36 s near the same joined boundary. GPU 0 and GPU 2 reached 92°C. None of the four hardware thermal-slowdown counters increased. [Thermal counter changes](measurements/thermal-events.json) · [Historical comparison](measurements/compare-handbook.json)

## How much work can the layer contribute?

Each GPU owns 5,120 × 2,048 multiply-accumulates per real projection row. Across four ranks one real row covers 41,943,040 MACs. A single-stream DSPARK target pass with six real rows covers up to 251,658,240 MACs. Some target passes take the plain calibrated path when the bounded proof queue is full. On the equal-target run those counts were 290 mined and 143 bypassed per rank. Even with every one of those 433 passes mined at the measured wall time, this layer would yield about **0.0010% of pure mining during decode**. More coverage would require mining additional compatible model operations or more real rows per unit time. [Per-rank graph and ticket accounting](measurements/joined-equal-target-fixed-period.json) · [Graph build and fixture](README.md)

Four-rank decode graph counters and independently scanned tickets were reconciled. Prefill efficiency uses request send to first token and decode uses first to last token. A final speculative replay can retire after the stream closes. Scanned model-origin work includes those tail passes. All standalone work comes from completed direct GPU tiles. Both are expressed in MAC-equivalents under the Pearl ticket target scaling. [Measurement definitions](measurements/README.md)

The local verifier accepted **976 unique inference proofs** over qualification, warm-ups and benchmarks. On the one-second-gap trace it verified 16 unique six-row decode proofs and 16 unique 2,048-row prefill proofs, with all four model shards represented in each phase. The qualification target was 65,536 times easier than the direct fleet target. The equal-target trace scanned real tickets and produced zero winning inference proofs. These measurements establish a functioning proof path on public inputs. They measure no F2Pool credit or coin payout. The joined operator also requantizes one FP8 checkpoint layer to INT7. All four 256-token greedy completions differed from their paired control on each fixed-period trace. Of the 24 short completions, 21 matched exactly. Production quality and proof disclosure for private prompts require separate evaluation. [Local verification](measurements/local-proof-summary.json) · [Greedy output comparisons](measurements/compare-equal-target-fixed-period.json)

The original `dsv41` service on port 8000 passed a fresh arithmetic request and its native arithmetic, structured-output, tool and vision startup checks after benchmarking. Its effective KV capacity returned to 4,121,856, and all four boards retained the 275 W cap. The paid F2Pool supervisor was already failed before these tests and remained failed afterward. All temporary benchmark services and the recovery timer were stopped. [Restoration receipt](measurements/restoration.json)
