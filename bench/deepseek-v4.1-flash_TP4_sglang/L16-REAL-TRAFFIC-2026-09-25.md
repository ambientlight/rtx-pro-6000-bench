# DeepSeek-V4.1-Flash: L16 real-traffic totals

L16 ran from **2026-09-20 11:09:41 UTC to 2026-09-25 02:16:33 UTC**
(**111 h 06 m 52 s**). The run ended when host RAM pressure caused the kernel
to OOM-kill a scheduler; these totals stop before the recovery launch.
Container: `59eb1ccc2927`; capture: `capture-20260920T110944Z-1959598`.

## Profile and cache changes

TP4 / EP4, 524,288-token context, 0.87 static memory fraction, eight running
requests, 2,048-token prefill chunks, prefill/decode interval 4, DSPARK block 5,
RAM Engram offload, and a **256,000,000,000-byte host-total HiCache budget**.
Its tensor payload was split **67% SWA / 33% FULL KV by bytes**; allocator
allowances and page rounding remain within the total budget.

Our [request-boundary retention policy](../../docker/deepseek-v41/SWA-RETENTION-2026-09-18.md)
evicts ordinary intermediate SWA checkpoints before completed-request windows.
Keeping the resumption window lets an otherwise valid FULL prefix be reused
after GPU eviction. The preference is bounded: boundary windows can still be
evicted under pressure. The [67/33 allocation](../../docker/deepseek-v41/HICACHE-SPLIT-67-L15-2026-09-20.md)
provides more SWA capacity without increasing the host-total budget.

## Final measurements

| Metric | L16 |
|---|---:|
| Inference API arrivals | 70,584 |
| Completed engine requests | 70,572 |
| Input / output tokens | 8,744,078,901 / 53,519,666 |
| Average full prompt | 123,903 tokens |
| Mean per-request decode | 97.9 tok/s |
| Mean scheduler batch generation | 216.9 tok/s |
| Mean scheduler prefill input | 2,552 tok/s |
| Input served from GPU L1 | 8,236,351,232 tokens · 94.19% |
| Input served from L1 + L2 | 8,553,065,472 tokens · 97.82% |
| Input served from RAM L2 | 316,714,240 tokens · 3.62% of all input |
| Logged non-health completions with any cache reuse | 67,474 / 70,477 (95.74%) |
| Logged non-health completions with RAM L2 reuse | 3,025 / 70,477 (4.29%) |

## RAM cost and eviction balance

The practical cost was **256 GB of host RAM for 3.62 additional percentage
points of observed input-cache coverage**: RAM L2 served 316,714,240 input
tokens on top of GPU L1's 94.19%, for 97.82% combined. Percentages are rounded
independently. This is input-token reuse, not a measured throughput gain or
an A/B estimate of the 67/33 split's effect alone.

The split balanced capacity pressure sufficiently for **both host pools to
reach 100% occupancy**, but did not make eviction symmetric. The final monitor
recorded 18,527,744 boundary SWA slots evicted under SWA pressure while FULL
remained, versus 10,156,544 boundary slots evicted alongside FULL removal.
At the final census, 1,746,176 FULL host tokens (13.89%) lacked resumable SWA
coverage. Thus 67/33 was the practical workload-tuned allocation, not a
guarantee that the associated FULL and SWA always survive or evict together.

## Sources and definitions

### Comparison population

The [README comparison](../../README.md#deepseek-v4-flash-0731-vs-deepseek-v41-flash)
keeps the original forum table's **logged non-health completion** scope for
generation, prompt, token, latency and cache rows. Its DSV4-0731 column is
unchanged: retained August 24–September 15 telemetry across 16 documented
launches. The V4.1 column includes only L16, not the recovery or no-swap launches.
Tests and four internal tool-recovery generations remain in the L16 sample.

This differs slightly from the broader cumulative engine counters used in the
final-measurements table above:

| Metric | Logged non-health completions (README) | Cumulative engine counters |
|---|---:|---:|
| Completed generations | 70,477 | 70,572 |
| Input tokens | 8,734,391,441 | 8,744,078,901 |
| Output tokens | 53,386,522 | 53,519,666 |
| Average full prompt | 123,933 | 123,903 |
| Cached input tokens | 8,543,849,984 | 8,553,065,472 |
| RAM L2 input tokens | 315,928,576 | 316,714,240 |
| Combined / RAM L2 input reuse | 97.82% / 3.62% | 97.82% / 3.62% |

The comparison's reported reasoning total is **33,351,924 tokens**, from the
same 70,477 completion records. Its 4,700 health-check generations count logged
HEALTH_CHECK completions, not all HTTP health probes. Median, p95, p99 and
maximum prompt sizes, latency quantiles and zero-hit counts use that same
non-health population. Quantiles follow the original report: exact median,
then sorted index `min(floor(N*p), N-1)` for p95/p99.

For L16's HTTP rows, final Prometheus counters are more complete than native
access logs: they include **23 Chat Completions HTTP 400s** missing from those
logs. API arrivals total 70,584; response counters record 70,556 inference
HTTP 200s and 23 HTTP 400s. Across all routes, they record 117,009 HTTP 200s,
four HTTP 503s (health probes), and 23 HTTP 400s. This gives **99.977% all-route
2xx / 99.967% inference-route 2xx**. Requests still in flight at the final scrape
can separate arrival and response counts. HTTP success counts headers, not
complete SSE streams; the host OOM ending L16 remains part of its outcome.

The 111.03-hour request-activity window includes idle gaps between the first
and last completion; it is not GPU-busy time. This is an operational comparison
of different observation windows, not a controlled A/B performance test.

### Counter summary above

Final cumulative Prometheus counters supply API arrivals, engine completions,
token totals, average prompt size and token-weighted cache shares. Arrival
counts exclude health/count-token/metrics routes but include inference startup
checks. Engine completions can include internal recovery generations, so they
are not a one-to-one count of HTTP requests.

Throughput comes from native logs: 70,472 valid per-request decode-rate samples,
162,578 scheduler generation samples and 141,798 scheduler prefill samples.
These are arithmetic sample means, not total tokens divided by 111 hours.
Output includes reasoning. Cache percentages measure input/prefix reuse, not
cached responses; RAM L2 is included in the L1 + L2 total. L2 token attribution
is not a physical transfer-byte measurement.

Frozen private evidence: `/mnt/hot/dsv41_state/latest-deployments-20260925-FFnygIQt/`:
`L16-final-metrics.json`, `comparison.json`, `report.md` and
`launch-20260920T110941Z-59eb1ccc2927.json`. These contain the final counter
samples and the matching launch's aggregate analysis; no raw prompts or
responses are copied into this repository.

The [Level1Techs post](https://forum.level1techs.com/p/4119105) describes the earlier
September 18 deployment snapshot, not these L16 totals.
