# 64 × 256k two-tier cache experiment — September 18, 2026 UTC

Status: **complete**. Ran 03:16:39–04:13:35 UTC (56.93 minutes) on container
`10a2a24ce3d8`; the test service exited successfully with status 0.
This is a user-requested experiment on the active 256 GB RAM HiCache profile;
no serving settings, deployment image or restart policy were changed, and no
explicit cache flush was performed.

Private evidence: `/mnt/hot/dsv41_state/cache-pressure-64x256k-bjNludyH/`.
Runner: `scripts/dsv41_cache_pressure.py`.
Unit: `dsv41-cache-pressure-64x256k-bjNludyH.service` (user systemd).

## Results

The model and serving process remained stable, but the older exact replays
did not benefit from the RAM cache. The recent-prefix control succeeded via
an SWA-only restore; a FULL-KV restore from RAM is **not yet demonstrated**.

| Measurement | Result |
|---|---:|
| Distinct cold prompts | 64 × exactly 256,000 input tokens |
| Cold input total | 16,384,000 tokens |
| Additional exact replays | 8 |
| All-request input / output totals | 18,432,000 / 216 tokens |
| Correct, complete Responses streams | 72 / 72 |
| API errors / wrong answers / deployment restarts | 0 / 0 / 0 |
| Cold TTFT mean / p95 | 43.770 / 43.965 seconds |
| Cold TTFT range | 43.333–44.652 seconds |
| Input tokens divided by mean cold TTFT | 5,849 tokens/second |
| Older replay cache misses | 7 / 7 |
| Recent replay cached tokens / TTFT | 255,744 (99.9%) / 1.285 seconds |
| End-of-run host FULL occupancy | 16,381,952 / 32,699,648 tokens (50.10%) |
| Available system RAM, start → finish | 40.46 → 37.95 GiB |
| Minimum sampled available system RAM | 37.74 GiB |
| Swap | None |

The rate above includes request handling and time to first token, not just GPU
prefill execution. All requests were sequential and uncontended according to
HTTP admission counters. This tests long prefill with three-token answers,
not concurrent load or long decoding. Cache buffers were allocated before
the experiment: logical occupancy growth must not be treated as incremental
system RAM allocation. FULL occupancy is not the occupancy of the SWA pool.

| Replayed case (zero-based) | After this many cold prompts | Cached tokens | TTFT, seconds |
|---|---:|---:|---:|
| 0 | 16 | 0 | 43.391 |
| 8 | 32 | 0 | 43.624 |
| 16 | 48 | 0 | 43.423 |
| 0 | 64 | 0 | 43.658 |
| 24 | 64 | 0 | 43.460 |
| 40 | 64 | 0 | 43.596 |
| 48 | 64 | 0 | 43.498 |
| 63 | 64 | 255,744 | 1.285 |

Case 63 was about 34× faster than mean cold TTFT. Its scheduler counter
classified 255,744 tokens as `host_hit`, but the per-pool transfer counter
increased by only **1,024 SWA slots across four ranks** (one 256-slot page per
rank), with no FULL-KV load-back. FULL KV was still on GPU; restoring the
matching SWA state made the prefix usable. Scheduler host-hit tokens are
therefore not synonymous with FULL-KV tokens copied from host memory.

Final checks at 04:17 UTC found the same healthy container, restart count 0,
zero running/queued requests, and an active packet capture with zero kernel
drops. The test-scoped ten-second metrics archive is closed and complete;
the deployment's ordinary wire capture continues.

## Workload and guards

- 64 distinct deterministic-random English-word reference prompts, each
  calibrated through `/v1/tokenize` to exactly **256,000 input tokens**.
- Unique identities at the beginning prevent sharing a complete cache page.
- `/v1/responses`, disabled thinking, temperature zero, maximum 16 output tokens;
  each response must return its case-specific six-digit verification code.
- One request at a time, with idle checks before admission. Each actual API
  usage count must match calibration; full prompts and streams are retained.
- Eight exact replays: case 0 after 16 cold calls; case 8 after 32; case 16
  after 48; then cases 0, 24, 40, 48 and 63 after all 64. Replays distinguish
  retained GPU prefixes from host restoration, including a recent-prefix control.
- Full Prometheus metrics, loads, memory availability and pressure are archived
  every ten seconds to `telemetry.jsonl.gz`; `latest-telemetry.json` is readable
  while the job runs. Before/after snapshots accompany every request.
- Stops on wrong/incomplete output, API errors, deployment change/restart,
  repeated monitoring loss, two samples below 24 GiB available RAM, or one below
  16 GiB. New admission requires at least 24 GiB. No forced cache flush/eviction,
  model restart or synthetic retries; maximum service lifetime four hours.
- HTTP admission counters mark any other inference during a request so its
  aggregate metric delta is not falsely attributed to this test alone.

33 offline harness/stream-parser tests passed before launch. All 72 live
requests then passed exact input-token, answer and Responses-stream validation.

## Reading the evidence

### Exact-input checks and the reuse gap

After 16 distinct 256k prompts, the exact replay of case 0 returned the right
answer but had **zero device/host hits, zero host load-back, and 43.39-second
TTFT** (cold was 44.65 seconds). RAM FULL occupancy was about 4.1M tokens.
The two HTTP request files are identical, and captured native `input_ids`
arrays are also identical: 256000 tokens with SHA-256
`a5ccab71247f1db3042630a53abe7fad048843f5cc782f4a11242220af8dc55b`.
Cache salt, extra key, session and LoRA IDs were null in both; neither request
was retracted. `first-replay-identity-audit.json` saves those comparisons.

The final native-log audit checked **all eight replays**: each had exactly the
same 256,000 native input token IDs and cache namespace fields as its original,
and zero retractions. There were no parse errors in those selected native
records. Native cached-token counts agreed with API usage for every replay.

These are genuine reuse misses, not prompt or cache-namespace changes.
SWA-state retention is the leading candidate explanation: the deployed
`swa_component.py:create_match_validator` and
`unified_tree_core.py:_match_prefix_helper` require a valid corresponding SWA
window as well as FULL KV. Host SWA tombstones can lose their SWA state while
FULL KV remains allocated. Thus the half-empty FULL host pool does not
establish that older prefixes are reusable. The successful recent SWA restore
also argues against a universal exact-length or API-rendering failure.

The host has 254,976 SWA slots per rank, but per-request live host-SWA occupancy
and eviction reasons are not exposed in the captured metrics. Exact eviction
or matching behavior remains unproven. The next diagnostic should trace that
state before changing capacity or policy; increasing FULL RAM capacity alone
is not justified by this result. No production fix, forced cache flush, or
additional inference was performed after completing this experiment.

```bash
systemctl --user status dsv41-cache-pressure-64x256k-bjNludyH.service
jq . /mnt/hot/dsv41_state/cache-pressure-64x256k-bjNludyH/status.json
jq . /mnt/hot/dsv41_state/cache-pressure-64x256k-bjNludyH/results.json
```

`results.json` contains per-request usage, TTFT, metrics deltas, final occupancy,
available RAM and an uncontended flag. `prompts/` holds the exact reusable input
and calibration history. The standard request/response packet capture remains
active under `/mnt/hot/dsv41_dumps/current/`. Preserve the experiment directory
for exact reproduction and offline analysis; the runner refuses to overwrite
an existing experiment or automatically replay it.

## Important metric interpretation caveat

During this experiment we found a byte-accounting mismatch in the running
SGLang source. `DeepSeekV4PagedHostPool` sets `size_per_token = item_bytes`,
where `item_bytes` describes one **page row of one layer**. The hybrid cache
controller's `_transfer_num_bytes` multiplies this by the number of **token
slots**, without converting page size and layer count. Thus the exposed
`hicache_backup_bytes_total` / `load_back_bytes_total` values are not reliable
physical-transfer-byte measurements for these paged pools. There is no single
universal correction factor because pool layer counts differ.

The container and local source hashes match for both affected files. The
actual copy path separately converts token slots to page indices and supplies
layer counts, so this finding alone does not establish a data-copy defect.
The bounded host allocation helper uses actual tensor shapes/bytes, not this
metric's scalar. No production patch was made while the test was running.

The earlier reported 13.96 GB backup figure was this raw counter, **not a
validated physical transfer volume**. Assess reuse primarily using per-tier
token hits, cache occupancy, request correctness and replay TTFT. Any byte-rate
report must correct and validate the paged-pool accounting first. Scheduler
hit/occupancy metrics are selected at TP rank 0; unlabelled operation counters
may aggregate all four ranks and must not be read as unique conversation tokens.
