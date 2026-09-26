# Near-full-window V4.1 prefill — 2026-09-15

The requested full-400k test **completed correctly**, but the restored profile
is **not allocation-retry-free**. It also delayed a mostly cached client request
for roughly 69 seconds. No deployment configuration was changed for this test.

## Profile and scope

- Container `dsv41`, image ID
  `sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca`.
- Context 409600, effective maximum input 409594, static memory fraction 0.85.
- TP4/EP4, prefill chunks 2048, concurrency 8, DSPARK block 5; KV capacity
  2272512 tokens. `prefill_decode_interval=0` remains unchanged.
- Exactly one synthetic Responses request, target 409400 input tokens, output
  budget 64, temperature 0, thinking disabled. Repeated reference notes end with
  an instruction to answer `42`. The server was idle at the preflight check.
- This tests prefill capacity and basic completion, not semantic long-context
  accuracy, tool calling, long-output endurance, or eight-way concurrency.

## Result

| Measurement | Result |
|---|---:|
| Actual input | 409397 tokens, 99.9504% of configured context |
| Prefix-cache hits | 0 tokens |
| Output | `42`, 2 tokens, natural stop |
| HTTP / terminal stream | 200 / valid `response.completed` |
| Server prefill interval | 75.4163 s |
| Effective server prefill rate | 5428.50 input tokens/s |
| Server end-to-end time | 76.1706 s |
| Client request wall time | 78.4100 s |
| Test request queue time | 0.0042 s |
| New allocator OOM/retry warning lines | 109 |
| Warning lines by GPU 0 / 1 / 2 / 3 | 26 / 31 / 26 / 26 |
| Fatal OOM, traceback, deleted-state errors | 0 in the test window |
| Request retractions / container restarts | 0 / 0 |
| Capture kernel packet drops added | 0 |

Server prefill time is the recorded first-forward-entry to prefill-finished
interval, including chunk scheduling and allocator overhead; it is not isolated
GPU kernel timing. The client did not record per-event TTFT. Decode throughput
from only two generated tokens is not a useful long-output TPS benchmark.

The warning count starts from the recorded native-log byte offset before this
probe and ends at its native completion. The deployment already had 37 warning
lines before the probe; those are excluded. The 109 lines are aggregated across
four ranks, not 109 failed requests or necessarily 109 distinct global events.
The largest failed allocation was 3321888768 bytes (about 3.09 GiB); the lowest
reported free GPU memory at a retry was 68419584 bytes (about 65.25 MiB).

## Overlapping client and scheduling

Another request arrived six seconds after the probe's native receipt. Its
227863-token prompt had 227584 prefix-cache hits: only **279 uncached tokens**.
It nevertheless spent **69.1811 seconds queued**, then completed 90 output
tokens successfully in 71.7021 seconds end to end.

| Event | UTC |
|---|---|
| Probe first forward entry | 23:20:22.754977 |
| Other request native receipt | 23:20:28.500108 |
| First allocator retry warning | 23:20:58.547745 |
| Last allocator retry warning | 23:21:36.845022 |
| Other request first forward entry | 23:21:37.949599 |
| Probe prefill finished | 23:21:38.171275 |

This was not a perfectly isolated benchmark. However, **every retry warning
preceded the other request's first forward pass**, the last by 1.1046 seconds.
The other client was mostly waiting, not performing another large uncached
prefill during the warnings. The evidence therefore points to working-memory
pressure in the long prefill, rather than concurrent forward computation being
necessary to trigger the warnings. Allocator snapshots were not collected, so
the exact tensor and whether fragmentation contributed remain unproven.

## Assessment

The restored profile can process an effectively full context and return a valid
response, including with a client waiting behind it. It does not establish
memory-clean operation: lowering the configured context did not eliminate
allocator retries. The nearly 70-second wait for an almost entirely cached
request is also a material interactive/agent-workload latency issue.

The next investigations should separate temporary prefill memory/allocator
behavior from request admission and scheduling fairness. A smaller-chunk test
would be a controlled experiment, not a proven fix. No such tuning or additional
synthetic inference was performed in this run. The backend remained healthy,
and capture continued afterward.

## Evidence

- Private artifacts:
  `/mnt/hot/dsv41_state/prefill-400k-085-S6HHJkQU/`.
  `before.json`, `result.json`, `native-finished.json`, `native-warnings.json`,
  `allocation-retries.json`, plus raw request/response artifacts under `runs/`.
- Probe response ID: `resp_b04aa394991f44d3b1e667794c948546`.
- Overlapping request ID: `5f866c0d387247018fe8d617c2834ce7`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260915T231145Z-1536990`.
- Local timing semantics verified in the pinned source's
  `python/sglang/srt/observability/req_time_stats.py`: first forward and prefill
  completion timestamps, converted into the native output metadata.
