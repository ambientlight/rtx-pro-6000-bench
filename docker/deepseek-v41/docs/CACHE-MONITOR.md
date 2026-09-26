# Passive FULL/SWA RAM-cache monitoring

[`dsv41-cache-monitor.service`](../../../systemd/dsv41-cache-monitor.service)
samples the selected `dsv41` deployment approximately every 30 seconds. User linger
allows it to run without an interactive login. It resumes collection when the
container comes back after a restart. It sends **only GET `/v1/loads` and
`/metrics`**, plus read-only Docker inspection and host-memory reads.

There is no inference traffic, cache flush, auto-tuning, model restart, external
notification, or model-source change. Warning observations are saved locally
and printed to the service journal for subsequent review.

## What it records

- FULL and SWA host occupancy/capacity independently, using rank 0 to avoid
  multiplying logical capacity by TP4.
- Cumulative total SWA and request-boundary SWA evictions, plus interval deltas.
  Boundary slots are a **subset** of total evicted SWA slots.
- Ordinary SWA eviction as total minus boundary eviction.
- Request-level device/host cached-token counters and separate FULL/SWA
  host-backup/load-back counters. Transfer-token counters explicitly retain
  their aggregate-across-ranks scope; they are not unique request-token counts.
- Running/waiting requests, active tokens, HTTP activity and available RAM.
- Container identity, sample timestamps, counter resets and telemetry outages.
- On the cache-attribution image: separate pressure causes
  and FULL co-removal for total/boundary SWA evictions; unique host FULL coverage
  by usable SWA endpoints; marked/unmarked SWA residency; SWA-limited lookup
  counters. Old or unsupported images report these fields as **null**.

Missing required gauges make a sample unavailable, not zero. Optional
lazy-exported hit/transfer counters remain null until observed; their first
appearance establishes a delta baseline. A new launch gets a separate directory
and cannot produce negative eviction deltas against the previous launch.

## Attribution, residency and legacy warning heuristics

The cache-attribution image exports six disjoint cause/disposition cells for each eviction
counter. The monitor selects rank 0 and requires their sums to equal the legacy
totals. Trigger (`swa_pressure`, `full_pressure`, `other`) and disposition
(`swa_only`, `with_full`) are independent: SWA pressure can evict FULL too.
New flags distinguish `boundary_swa_pressure_without_full_removal` from
`boundary_eviction_alongside_full`. Neither is a count of failed requests.

The bounded rank-0 census runs at most once per minute. Its valid timestamp,
completion flag, duration and node count travel with the totals. The monitor
publishes `full_swa_uncovered_fraction_of_tree_host_tokens` only for a complete
snapshot aged 0–150 seconds, using its own tree-residency denominator. Failed
or stale scans report that derived fraction as null. Complete census values
must conserve `tree = covered + SWA-uncovered + FULL-chain-gap`.

See [source definitions and rollout](history/CACHE-ATTRIBUTION-2026-09-19.md). Lookup
loss is counted per lookup, including retries, not per completed request.

| Condition | Recorded flag |
|---|---|
| SWA occupancy >=90%, FULL occupancy <80% | `swa_pressure_with_full_headroom` |
| Boundary eviction delta >0, FULL occupancy <80% | `boundary_eviction_with_full_headroom` |
| Boundary eviction delta >0, FULL occupancy >=80% | `boundary_eviction_observed` |
| Retention disabled or a counter decreases | Separate retention/reset flags |

The 80%/90% cutoffs are operational review thresholds, not proven optimal ratios.
Ordinary checkpoint churn can be healthy even at 100% SWA occupancy. The goal
is useful reusable prefixes, not identical occupancy percentages.

On the old image, these aggregate counters **cannot establish that a particular associated FULL
prefix survived its SWA window, or that a later request missed because of it**.
Boundary counters also include FULL-driven eviction, and occupancy is measured
at observation time, not at the instant of each eviction. Such warnings identify
intervals to correlate with request captures; precise victim attribution requires
the cache-attribution TreeCore instrumentation. The passive collector itself does
not inspect or modify the live tree.

## Data and operation

Root: `/mnt/hot/dsv41_state/cache-monitor/` (private permissions).

- `status.json`: monitoring/unavailable status and current flags.
- `latest.json`: last valid sample, including interval analysis.
- `current/summary.json`: current launch's summary and atomic resume checkpoint.
- `current/samples-YYYY-MM-DD.jsonl`: daily-rotated sample files, retained.
- `current/events-YYYY-MM-DD.jsonl`: warning transitions and boundary-eviction
  intervals; created only when events occur.
- Older `launch-<UTC-start>-<container-id>/` directories remain preserved.

No raw prompts, outputs, credentials or full Docker inspection are recorded.
Outages preserve the previous valid sample with its original timestamp; consult
`status.json` before treating `latest.json` as current. Interval deltas spanning
an outage include the elapsed interval, not an instantaneous eviction rate.

```bash
systemctl --user status dsv41-cache-monitor.service
journalctl --user -u dsv41-cache-monitor.service
jq . /mnt/hot/dsv41_state/cache-monitor/current/summary.json
```

Source: [dsv41_cache_monitor.py](../../../scripts/dsv41_cache_monitor.py).
The retired local tests (`scripts/test_dsv41_cache_monitor.py` at Git revision
`dc74842`) covered
missing/invalid metrics, TP/model filtering, eviction semantics, resets,
source-only reads, launch rotation, resuming after an interrupted write,
cause-counter reconciliation, census conservation, and unavailable/stale data.
