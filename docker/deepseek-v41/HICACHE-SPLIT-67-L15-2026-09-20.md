# L15: restore L13's 67% SWA / 33% FULL host-cache split

The user selected 67/33 after observing better RAM-cache coverage in L13 and
authorized recreation **only after all traffic drains plus one continuous
minute idle**. This is a configuration-only rollout, not a new model image or
cache-policy change. No extra throughput or cache-pressure workload is sent.

## Status and evidence

The watcher runs independently of the interactive session under lingering user
systemd. Its status file is authoritative; staging is not startup completion.

- Service: `dsv41-cache-split-relaunch@cache-split-rollout-l15-tPDqwg7c.service`.
- State: `/mnt/hot/dsv41_state/cache-split-rollout-l15-tPDqwg7c/`.
- Current rollout state: `status.json` in that directory.
- Drain observations: `drain.jsonl`.
- Previous L14 profile and Python bytes: `rollback.compose.json`, `rollback-src/`.
- Frozen candidate: `candidate.compose.json`, `candidate-src/`.
- On success: `complete.json`, `allocation.json`, `metrics.after.json`,
  `launch-spec.json`, capture receipt and private startup/identity evidence.

The capture and passive monitor continue following container generations.
`launch-spec.json` tags the new launch **L15** without changing the public model
alias, image version or container name.

## Unchanged serving contract

The candidate is compared against the complete archived L13 profile at
`/mnt/hot/dsv41_state/cache-attribution-rollout-rg2rbn1u/candidate.compose.json`.
Python snapshot locations may differ; their SHA-256 hashes must match.

| Setting | L15 |
|---|---|
| Image | `ambientlight/dsv41-sm120:cache-attribution-v1` |
| Immutable image | `sha256:fcd6ae8574240c6f0df3c6277b477234328a028007ef7e97acdcb9ed1db1349c` |
| Host-total budget | 256,000,000,000 bytes, including allocator allowances |
| SWA / FULL tensor payload | 67% / 33%, subject to whole-page rounding |
| Expected FULL host capacity | 12,570,368 logical tokens, as in L13 |
| Expected SWA host capacity | 1,656,064 slots per rank, as in L13 |
| Context / chunked prefill / decode interval | 524288 / 2048 / 4 |
| GPU allocation | 0.87 static, automatic KV, expandable segments disabled |
| Parallelism / concurrency | TP4 / EP4 / 8 |
| Engram | Full RAM offload; separate row cache disabled |
| Endpoint | `dsv41`, port 8000, alias `deepseek-v4-flash`, no backend key |
| Restart policy / stop grace | `unless-stopped` / 90 seconds |

Exact host capacities are checked against all four startup allocation records
and live metrics. Automatic GPU capacity can vary with available GPU memory;
the configuration and host-budget policy remain unchanged.

## Drain and safety guards

The watcher polls fresh `/v1/loads` and `/metrics` every five seconds. Running
requests, queues, in-flight prefill tokens, HTTP streams or changed inference
arrival counters reset the full 60-second timer. Missing, repeated or stale
scheduler data cannot qualify as idle. Health/telemetry probes do not count as
inference traffic. A final fresh observation is made immediately before
recreation, preserving the existing Docker stop grace period.

Before starting, it rechecks the original container identity/restart count,
immutable image, staged-file hashes and projected RAM headroom (at least
24 GiB after crediting only verified old cache buffers). It records one attempt
and never retries the recreation or rolls back automatically after an error.
The old L14 capture and rollback evidence are preserved.

Verification requires built-in startup readiness, Docker health, unchanged
effective flags and environment, four-rank allocation and the requested split,
frozen runtime hashes, capture rotation, active passive monitoring and enabled
attribution with matching exported capacities. Only normal built-in arithmetic,
JSON/tool and vision startup checks run. Long-run reliability is not established
by those startup checks.

## Offline validation

All **75 rollout/load/monitor tests passed** before arming, including stale/busy
load handling, brief arrivals between polls, no-repeat guards, allocation
rounding, L15's fraction-only transition and exact archived-L13 source/profile
comparison. The historical 50/50 helper default is unchanged; this explicit
profile is selected using `arm --profile l15-67`.

To inspect without affecting serving:

```bash
systemctl --user status dsv41-cache-split-relaunch@cache-split-rollout-l15-tPDqwg7c.service
jq . /mnt/hot/dsv41_state/cache-split-rollout-l15-tPDqwg7c/status.json
```

To cancel **before** the state becomes `relaunching`, stop that watcher service;
this does not stop `dsv41`. Compose remains staged at 0.67 for the next
explicitly authorized recreation. A plain `docker restart` retains the existing
container environment and does not apply a changed Compose profile.
