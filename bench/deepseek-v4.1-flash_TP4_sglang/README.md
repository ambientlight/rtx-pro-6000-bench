# DeepSeek-V4.1-Flash · TP4 / EP4 · SGLang

Long-context and cache experiments on four RTX PRO 6000 Blackwell Max-Q GPUs.
The long-call baseline records a 250 W/GPU limit; the other reports do not
independently record that limit, so this collection has no common `W{watt}` tag.
Each report preserves its measured deployment profile and results.

| Date (UTC) | Experiment |
|---|---|
| 2026-09-15 | [32k / 131k / 200k long-call TPS baseline](BENCHMARK-LONG-TPS-2026-09-15.md) |
| 2026-09-15 | [400k prefill, 0.85 static memory](BENCHMARK-FULL-PREFILL-2026-09-15.md) |
| 2026-09-16 | [400k prefill with RAM Engram offload](BENCHMARK-FULL-PREFILL-RAM-2026-09-16.md) |
| 2026-09-16 | [524k prefill, 0.90 static memory](BENCHMARK-FULL-PREFILL-V2-090-2026-09-16.md) |
| 2026-09-16 | [524k replay after closing the game](BENCHMARK-FULL-PREFILL-V2-NO-GAME-2026-09-16.md) |
| 2026-09-16 | [524k replay with non-expandable allocator](BENCHMARK-FULL-PREFILL-V2-NONEXPANDABLE-2026-09-16.md) |
| 2026-09-16 | [524k prefill on restored 0.85 profile](BENCHMARK-FULL-PREFILL-RESTORED-085-2026-09-16.md) |
| 2026-09-18 | [64 × 256k two-tier cache pressure](CACHE-PRESSURE-64X256K-2026-09-18.md) |
| 2026-09-20–25 | [L16: 111-hour real-traffic totals and 67/33 RAM cache](L16-REAL-TRAFFIC-2026-09-25.md) |

Deployment files and operational guides remain in
[`docker/deepseek-v41/`](../../docker/deepseek-v41/README.md). Reproduction commands
in the reports run from the repository root.

Raw requests, SSE streams, wire captures and runtime evidence remain outside
Git at the private paths recorded in each report; this directory contains only
the published benchmark reports.
