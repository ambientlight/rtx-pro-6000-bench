# Near-full-window prefill after RAM offload — 2026-09-16 UTC

The requested repeat **passed**: 409397 uncached input tokens, correct `42`
response, valid Responses stream, no fatal error or restart. RAM offloading
did **not** eliminate GPU allocator pressure: this isolated run produced
**156 recoverable allocation-retry warning lines**.

## Matched request and deployment

Exactly one synthetic inference request replayed the previous test's saved
payload: repeated reference notes, temperature 0, thinking disabled, output
budget 64, streaming `/v1/responses`, no tools. Tokenization reconfirmed exactly
409397 input tokens before inference. No cache flush, deployment change or
second synthetic inference was performed. The endpoint was idle for 20 seconds
before preflight and idle again immediately before sending the inference.

Container `dsv41`, image
`sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca`;
RAM offload, separate NVMe cache disabled. Context 409600, memory fraction 0.85,
TP4/EP4, prefill chunks 2048, concurrency 8, DSPARK block 5, total KV capacity
2272512. Serving settings and container identity were unchanged by the test.

## Comparison

| Measurement | Earlier NVMe/64 GiB cache | Current RAM mode |
|---|---:|---:|
| Input tokens | 409397 | 409397 |
| Prefix-cache hits | 0 | 0 |
| Output | `42`, 2 tokens | `42`, 2 tokens |
| Server prefill interval | 75.4163 s | 73.6361 s |
| Effective prefill rate | 5428.50 tokens/s | 5559.73 tokens/s |
| Server end-to-end | 76.1706 s | 74.2422 s |
| Client first-output latency | Not recorded | 75.8793 s |
| Client wall time | 78.4100 s | 76.2324 s |
| Request queue time | 0.0042 s | 0.0007 s |
| Recoverable allocator retry warning lines | 109 | 156 |
| Retry lines: GPU 0 / 1 / 2 / 3 | 26 / 31 / 26 / 26 | 37 / 45 / 37 / 37 |
| Fatal OOM / API failure / retraction / restart | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Sampled peak GPU memory per card | Not recorded here | 94.626 GiB |
| Scheduler-worker storage reads | Not recorded | 6.699 MiB |
| Scheduler-worker major faults | Not recorded | 11 |
| Other client request received during test | 1 | 0 |
| Capture packet drops added | 0 | 0 |

The prefill interval is the native `prefill_finished_time - forward_entry_time`,
including chunk scheduling and allocator overhead, not isolated GPU kernel time.
Token counts agree between tokenization, API usage and native completion data.
The two-token output is not a meaningful decode-TPS test. GPU memory was sampled
every five seconds, so the true transient peak may be higher.

The measured prefill rate is 2.42% higher than the earlier single run. This is
not a statistically established RAM-offload speedup: the previous run had a
mostly cached client waiting behind the probe, and allocator/cache warm states
also differ. In that earlier run all retry warnings preceded the other client's
first forward pass. The current run had no other native request IDs or observed
load contention.

## Memory assessment

All 156 retry lines belong to this probe's native-log window, not earlier
startup or validation. First warning: `01:27:24.699791062Z`; last:
`01:28:26.960308596Z`. The largest failed allocation was 3338665984 bytes
(~3.11 GiB); the lowest reported free GPU memory at a retry was 43253760 bytes
(41.25 MiB). These are per-rank warning lines, not failed requests or necessarily
156 distinct synchronized events. Recovery succeeded and the response completed.

Host available memory remained about 273.7 GiB. Full-context prefill is functional,
but moving Engram backing to host RAM does not remove the observed GPU allocator
pressure. No allocator snapshots were collected, so the exact temporary tensor
and contribution of fragmentation remain unproven.

Worker I/O counters increased by 7024640 bytes and 11 major faults across the
request window. Those counters cover all reads by the four schedulers, not
exclusively Engram. They do not establish disk-free operation; the separate
post-rollout residency probes already demonstrated disk faults on sampled
Engram weight pages. This test does not repair or fully characterize that issue.

## Collector safety finding

Fourteen observer samples reported zero running/waiting requests while active
prefill tokens were nonzero. The pinned load-inquirer counts the running decode
batch separately from its chunked-prefill request. Request counts alone are
therefore not a safe idle/drain condition.

The client-side guard now requires active, used, total, and remaining-prefill
token counters to be zero, in addition to request and queue counts. Missing or
invalid token counters fail closed. All fourteen saved samples are correctly
rejected as idle by the updated guard; three new regressions bring the V4.1
suite to 46 tests, plus 10 passing capture tests. The earlier RAM rollout's five
final idle snapshots were rechecked with the stricter guard: all token counters
were zero. No SGLang/image/kernel change or restart was made for this guard fix.

## Limits and evidence

This is one repetitive synthetic prefill and a two-token verification answer,
not semantic long-context retrieval, tool calling, long-output endurance,
agent-workload responsiveness or eight-way concurrency qualification.

- Private evidence: `/mnt/hot/dsv41_state/prefill-400k-ram-CAD2p72m/`.
- `run_probe.py` records the replay procedure; **running it sends inference**.
  `analyze_probe.py` reanalyzes saved evidence without sending inference.
- `result.json`, `native-finished.json`, `native-warnings.json`,
  `allocation-retries.json`, `observer.jsonl`, `timeline.jsonl`, `response.sse`,
  and worker I/O snapshots retain the measurements.
- Response ID: `resp_f4ee0901b3ee4912bde40f5c920de103`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260916T011159Z-2340443`.
- The backend remained healthy and idle after the test; capture continued with
  zero packet drops. No synthetic workload was left running.
