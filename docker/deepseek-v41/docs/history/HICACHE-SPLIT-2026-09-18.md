# RAM HiCache payload split: 23.6% SWA / 76.4% FULL

Historical record; see the [cache guide](../HICACHE.md) for the supported configuration.

## Scope and configuration

User-approved RAM-only rebalance, deployed September 18, 2026. Keeps the existing
`diagnostics-v1` image and all inference, API, CUDA/NCCL, Engram and capture
settings. The deployment label becomes `dsv41-v2-hicache-split-v1`.

```yaml
DSV41_HICACHE_HOST_BUDGET_BYTES: "256000000000"
DSV41_HICACHE_SWA_FRACTION: "0.236"
```

The fraction applies to **tensor payload bytes**, not token slots or the entire
256 GB including metadata. The planner keeps slot-index allowances and the
fixed 1 GiB reserve inside the same host-total cap. No GPU-memory setting is
changed. This is a capacity ratio, **not a five-boundary-per-conversation cap**.

Absent the new variable, existing GPU-proportional host sizing is unchanged.
The explicit fraction decouples RAM allocation from variations in automatic
GPU cache sizing. Invalid fractions, a fraction without a host budget, and
budgets unable to hold one device-cache equivalent fail closed.

For FULL page count F and requested fraction q:

```text
SWA_pages = floor(F * FULL_page_bytes * q / (SWA_page_bytes * (1-q)))
```

The existing byte-bounded search selects F while including both pools' metadata.
Whole-page rounding makes the effective split approximate. FULL and SWA still
have independent pools; the change does not add dynamic borrowing or alter the
soft request-boundary retention policy.

## Verified allocation

| Item | Previous diagnostics launch | New live allocation |
|---|---:|---:|
| FULL RAM logical token capacity | 32,562,944 | 28,234,496 |
| SWA RAM slots per rank | 264,704 | 565,760 |
| FULL payload across TP4 | 212.64 GB | 184.37 GB |
| SWA payload across TP4 | 26.63 GB | 56.93 GB |
| SWA fraction of tensor payload | 11.13% | 23.5919% |
| Planned total including allowances | 256.00 GB | 255.98 GB |

Logical capacity is not multiplied by four; byte totals above cover all four
ranks. The previous SWA share was about 10.4% of the entire 256 GB budget,
or 11.13% of tensor payload. These are different denominators.

The new split roughly doubles SWA capacity and reduces FULL capacity by 13.3%.
It is a workload-tuning experiment, not proof that future boundary evictions
or cache misses are eliminated.

## Tests and rollout

Completed **23:08:29 UTC (16:08 PDT)** after **63.642 seconds** of continuous
idle. Container `dc03b4e129b6` started at **23:02:46 UTC**, is healthy, and has
zero unexpected restarts. All four ranks agree on the table above. Actual
initial payload/index buffers total **245,833,653,760 bytes**; planned total
including allowances is **255,978,642,944 bytes**, within the 256 GB cap.
Available RAM at verification was **41,697,734,656 bytes (38.8 GiB)**.
Device FULL/SWA capacities remain **3,211,520 / 26,112**, unchanged from the
previous launch. The model image, API contract and inference flags are unchanged.

Built-in arithmetic, structured JSON, tool round-trip and vision checks passed.
Prometheus confirms both new host capacities and zero SWA/boundary evictions at
qualification; the latter is a fresh-cache observation, not a workload result.
Capture followed to `/mnt/hot/dsv41_dumps/capture-20260918T230247Z-2231712`.

- 48 CPU tests passed using the production image's dependencies with the new
  allocator overrides. Includes actual assembler wiring on mock-sized pools,
  packed DSPARK accounting, 1,000 randomized default/explicit sizing cases,
  startup validation and the existing SWA-retention regressions.
- 99 deployment/helper tests passed, including source freezing, change-scope
  guards, invalid/busy telemetry, brief requests between polls, and no automatic
  repeat after an attempted rollout.
- No long-context, throughput or GPU pressure workload was requested or sent.

The one-shot user-systemd watcher waited for 60 continuous idle seconds. It
checks `/v1/loads`, active inference HTTP streams, and arrival counters, and
resets on activity, invalid telemetry or insufficient projected RAM. Sources,
configuration and container identity are frozen and checked before recreation.
Normal built-in startup smoke tests ran after the restart. The job completed
with exit code 0 and is now inactive; it will not recreate the deployment again.

Service: `dsv41-cache-split-relaunch-rPapt8Dr.service`.

Private evidence: `/mnt/hot/dsv41_state/cache-split-rollout-rPapt8Dr/`.
`status.json` tracks rollout progress; `drain.jsonl` records the idle window.
`candidate.compose.json` and `candidate-src/` freeze the tested deployment.
`rollback.compose.json` and `rollback-src/` preserve the previous configuration
and actual Python source bytes, so rollback does not depend on edited binds.
There is no automatic rollback or extra recreation attempt.

## Source

SGLang branch: `/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay`.
Changed runtime files: `deployment/boot.py`,
`python/sglang/srt/mem_cache/hybrid_cache/dsv41_host_budget.py`, and
`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`.
These use the deployment's existing read-only override mechanism; no new image
was built. Startup verifies the frozen mounted-source hashes, all four actual
pool allocations, serving flags, capture heartbeat and restart policy.

See the branch's `deployment/HICACHE_SWA_RETENTION.md` for retention semantics,
byte accounting, mathematical sizing, and the distinction between intermediate
and request-boundary SWA checkpoints.
