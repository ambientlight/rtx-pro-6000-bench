# Full-window prefill after closing Starfield — 2026-09-16 UTC

**Passed, with significant recoverable allocator pressure.** Exactly the saved
524177-token request that previously crashed was replayed after the user closed
Starfield. It returned `42` correctly; no fatal/API error, retraction, container
restart or packet drop occurred. There were **210 allocation-retry warning lines**.

## Results

| Measurement | Previous, game running | Repeat, game stopped |
|---|---:|---:|
| Context / memory fraction / prefill chunk | 524288 / 0.90 / 2048 | Unchanged |
| Input tokens | 524177 | 524177 |
| Cached input tokens | No completed usage; prefill logs showed 0 | 0 |
| Output | None | `42`, 2 tokens |
| Server prefill interval | Did not complete | 100.2216 s |
| Uncached prefill rate | Not available | 5230.18 tokens/s |
| Server end-to-end | No completion | 101.0281 s |
| Client first text | None | 103.2933 s |
| Client stream wall time | 159.52 s to failed EOF | 103.8062 s |
| CUDA allocation-retry warning lines | 173 before crash | 210, recovered |
| Warning lines: GPU 0 / 1 / 2 / 3 | 27 / 92 / 27 / 27 | 51 / 57 / 51 / 51 |
| Fatal OOM / new restart | 1 / 1 | 0 / 0 |
| Other non-health inference requests | 0 | 0 |
| KV pool capacity | 1059840 | 1107712 |

Inference started at **06:22:19 UTC** and validation finished at **06:24:05 UTC**.
Native and Responses usage agree on input/output counts, final status is
`completed`, and the stream matches the final text with contiguous sequence
numbers. The request bytes and SHA-256 match the saved failed request exactly.
Temperature 0, thinking disabled, no tools, a 64-token output cap and a stable
20-second idle window match the earlier procedure. No cache flush was used.

Prefill throughput uses the native interval from `forward_entry_time` to
`prefill_finished_time`: it includes chunk scheduling and allocation overhead,
not isolated GPU kernel time. The two-token output does not qualify decode TPS.

## Assessment and limits

Closing the game was followed by a successful full-window request, but this is
**not memory-pressure-free operation**. Failed allocations before recovery
reached 4261412864 bytes (~3.97 GiB); the smallest device-free reading at a retry
was 11862016 bytes (~11.31 MiB). Sampled GPU peaks were 95215 / 96862 / 96903 /
96903 MiB. Warning counts are per-rank log lines, not failed requests; snapshots
every five seconds can miss transient peaks. No observer errors occurred.

The image, process start, fraction, context and other settings were unchanged
for this repeat. **There was no restart after closing the game**, so the KV pool
stayed at 1107712 slots, allocated by the preceding automatic recovery while the
game was still running. The failed run had 1059840 slots before that recovery.
Thus this is not a perfectly controlled same-process comparison with the crash,
and it does **not** qualify a future restart at 0.90 with all GPU memory free:
that startup could allocate a larger cache and consume some of the freed headroom.

This qualifies one isolated, repeated-reference prefill with a short answer—not
semantic retrieval, long-output endurance, mixed traffic or eight full-window
requests. Worker counters added 335872 read bytes (0.3203 MiB), with zero major
faults. These include all worker reads, not just Engram; the earlier RAM residency
limitation is not resolved by this measurement.

The backend remained healthy and idle. Restart count stayed at **1** (the prior
crash), all serving settings are unchanged, and capture remained active with zero
drops. No test workload was left running.

## Evidence

- Private evidence: `/mnt/hot/dsv41_state/prefill-524k-v2-no-game-MbU2UEX0/`.
- `run_probe.py` reuses the earlier measurement helpers, reads the exact saved
  payload and refuses to replay after it starts.
- `result.json`, `before.json`, `native-finished.json`, `native-warnings.json`,
  `observer.jsonl`, `timeline.jsonl`, `response.sse` and worker-I/O snapshots
  preserve the measurements.
- Response ID: `resp_6acb763d5551475d93a76f176e58d514`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260916T060736Z-3122440`.
- Previous failure: [game-running test](BENCHMARK-FULL-PREFILL-V2-090-2026-09-16.md).
