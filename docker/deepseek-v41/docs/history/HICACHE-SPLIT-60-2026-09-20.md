# 60% SWA / 40% FULL RAM-cache payload rollout

Historical record; see the [cache guide](../HICACHE.md) for the supported configuration.

The user explicitly requested an immediate restart, replacing the 67/33 split.
The idle gate was bypassed; the existing 90-second stop grace period was kept.
No image, runtime-source, API, model, GPU or retention-policy change accompanies
this rebalance. The host-total plan remains **256,000,000,000 bytes**.

## Launch identity

- Container: `83e8386860468bb0b2e87070a431bc0a5f9d3e188f358fb9aed06fbd8a12548a`.
- Started: **2026-09-20 00:05:12 UTC** / **September 19, 17:05:12 PDT**.
- Image: `ambientlight/dsv41-sm120:cache-attribution-v1`.
- Immutable image: `sha256:fcd6ae8574240c6f0df3c6277b477234328a028007ef7e97acdcb9ed1db1349c`.
- State/rollback: `/mnt/hot/dsv41_state/cache-split-60-rollout-0o3aksh6/`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260920T000516Z-750212`.

## Capacity plan

| Setting | Previous 67/33 | Requested 60/40 |
|---|---:|---:|
| `DSV41_HICACHE_SWA_FRACTION` | 0.67 | 0.6 |
| FULL host token capacity | 12,570,368 | 15,161,856 |
| SWA host slots per rank | 1,656,064 | 1,475,840 |
| FULL payload, all ranks | 82,084,503,040 bytes | 99,006,919,680 bytes |
| SWA payload, all ranks | 166,633,159,680 bytes | 148,499,020,800 bytes |
| Plan including allowances | 255,984,163,328 bytes | 255,999,053,824 bytes |

The allocator's whole-page plan gives **59.998164% SWA payload**. The fraction
is a share of tensor bytes, not token slots. Metadata allowances stay inside
the same host-total budget. The plan was verified against startup allocation on
all four ranks and live exported capacities. Initial buffers total
**249,979,064,320 bytes**; the larger plan includes metadata allowances.

## Verification status

Startup and live verification passed at **2026-09-20 00:11:07 UTC** / September
19, 17:11:07 PDT, with **zero unexpected restarts**. Built-in arithmetic,
structured JSON, tool round-trip and vision checks passed. Effective serving
flags, all four host allocations, runtime fingerprints, NCCL diagnostic FIFOs,
capture rotation and enabled/completed nonempty census were verified.

**120 deployment/monitor tests and 17 allocator tests** passed. Candidate and
rollback profiles use immutable image identity and frozen Python bind sources;
source hashes are unchanged. Only the cache-fraction environment value differs.
No additional throughput or cache-pressure test was sent. These startup checks
do not establish long-run reliability or an improvement over the previous split.

Device capacity remains **4,062,720 FULL tokens / 26,112 SWA slots per rank**.
Available system RAM at verification was **35.1 GiB**. Context 524288, static
fraction 0.87, TP4/EP4, concurrency 8, prefill/decode interval 4, RAM Engram,
public model name, port 8000, no backend key and `unless-stopped` are unchanged.

Passive monitoring continues at
`/mnt/hot/dsv41_state/cache-monitor/launch-20260920T000512Z-83e838686046/`.
The rollout directory contains `verified.json`, `allocation.json`,
`metrics.after.json`, the exact candidate and the previous-profile rollback.

## Previous-launch health evidence

Before any recreation, container `17a496e4fce5` was running but **unhealthy**,
with **30 consecutive health-check failures** and **zero container restarts**.
The retained probe records say `Health check exceeded timeout (10s)`.
This is a pre-existing condition, not an error introduced by the 60/40 split.
Its cause has not been diagnosed in this configuration-only rollout.

`before.container.private.json`, `before.monitor.json`, and the previous
allocation are preserved in the rollout directory. The original capture stays
at `/mnt/hot/dsv41_dumps/capture-20260919T072952Z-3156801`. No old logs, traces,
captures or model data were deleted.
