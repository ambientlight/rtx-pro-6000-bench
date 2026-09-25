# Original rollout: cache eviction attribution and 67/33 host payload

Historical 67/33 launch record. The same telemetry image is now deployed at
[60/40](HICACHE-SPLIT-60-2026-09-20.md); the capacities below describe the
original September 19 launch, not the current allocation.

Initially staged without a restart; subsequently deployed after the user
explicitly authorized relaunch. Container `17a496e4fce5` started September 19
at **07:29:47 UTC**, following **60.8 seconds idle**. Startup and live telemetry
were verified at **07:35:17 UTC**, with zero unexpected restarts. Only the
built-in startup smokes ran; no additional load/cache-pressure test was sent.
The previous 50/50 profile and frozen Python sources are preserved for rollback.

## What the image adds

`ambientlight/dsv41-sm120:cache-attribution-v1` layers four runtime files on
the immutable diagnostics parent. Existing eviction and matching decisions,
API fixes, model kernels, NCCL execution, GPU pool sizing and Engram offload
are unchanged. Source/test snapshots and immutable image identity are retained
in the build receipt referenced by `baseline.lock.json`.

| Measurement | What it distinguishes |
|---|---|
| SWA eviction trigger | SWA pressure, FULL pressure, other cleanup. |
| SWA eviction disposition | SWA-only removal vs actual FULL host removal on the same node in the same operation. |
| Boundary subset | Same breakdown for marked SWA slots; sums reconcile with the existing boundary total. |
| Host FULL coverage | Unique tree-attached host FULL tokens covered by any usable FULL+SWA prefix, or uncovered because SWA is missing. |
| Missing FULL ancestors | Reported separately, never attributed to SWA. |
| Prefix-lookup loss | Available FULL prefix tokens denied specifically by SWA, including earlier-boundary fallback. Counts lookups, not unique requests. |

An interior missing SWA copy does not strand history if a usable descendant
endpoint covers it. GPU SWA counts as available even when its RAM copy is
missing. Shared prefixes count once in the residency union. These distinctions
are tested against the real CPU TreeCore.

The rank-0 census runs at most once per minute with a 250 ms work budget and
200,000-node limit. It checks only metadata/slot-array lengths, never GPU data.
Incomplete scans preserve old complete values with their original timestamp
and `complete=0`; the monitor never reports them as current or as zero loss.
This synchronous scan can briefly delay an iteration, so its duration is
exported and a CPU metadata-only cost check is retained alongside test logs.

See the full source-side reference:
`/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay/deployment/HICACHE_CACHE_ATTRIBUTION.md`.

## Verified configuration

| Setting | Original 67/33 profile |
|---|---|
| Deployment label | `dsv41-v2-cache-attribution-v1` |
| Image | `ambientlight/dsv41-sm120:cache-attribution-v1` |
| `DSV41_HICACHE_SWA_FRACTION` | `0.67` (67% SWA / 33% FULL tensor payload) |
| Host budget | 256,000,000,000 bytes, across all four ranks and including allowances |
| Verified FULL capacity | 12,570,368 logical tokens |
| Verified SWA capacity | 1,656,064 slots per rank |
| Plan, including allowances | 255,984,163,328 bytes |
| Actual initial buffers, all ranks | 250,781,915,648 bytes |
| Inference settings | Unchanged: 524288 context, 0.87 static fraction, automatic GPU KV, TP4/EP4, concurrency 8, RAM Engram |

Allocation was verified on all four ranks and against live metrics; the actual
SWA share is 66.9969% after rounding. This trades roughly one-third of FULL capacity for 35.6%
more SWA capacity. It is not a promise that boundary evictions disappear.

The normal **authorized** `./launch_all.sh dsv41` path recreates from Compose.
A plain `docker restart dsv41` retains the existing container's old image and
environment. The historical 50/50 idle-relaunch helper remains intentionally
restricted to that old image-preserving rollout and cannot deploy this image.

## Reproducibility and checks

`scripts/dsv41_build_cache_attribution_image.py` freezes the scoped dirty
sources, verifies their committed baselines against the pinned parent, builds
without network dependency changes, compares API/kernel/diagnostic fingerprints,
preserves parent layers and runs CPU tests against the actual built bytes with
`--runtime runc` and `NVIDIA_VISIBLE_DEVICES=void`. It never starts inference.

The passive monitor understands both old and new metric schemas. Old images
yield **null**, not zero, for unavailable attribution fields. The live collector
now records enabled diagnostics and completed census snapshots under
`/mnt/hot/dsv41_state/cache-monitor/launch-20260919T072947Z-17a496e4fce5/`.

Live checks passed: enabled/completed census, six-cell reconciliation against
each legacy eviction total, residency conservation, four-rank capacities,
unchanged serving flags, frozen source hashes and four NCCL control FIFOs.
The initial census was empty, before startup smoke traffic. The **07:37:12 UTC**
census then visited **55 nodes in 0.302 ms**, reporting **35,584 host FULL tokens,
all covered**, with no SWA-uncovered or FULL-chain-gap tokens. Both eviction
totals were still zero. The passive follow-up snapshot is preserved as
`nonempty-census.after.json` in the rollout directory. This verifies live
nonempty-tree measurement, not improved retention under pressure. Real-traffic
eviction and hit-rate behavior still need observation. No GPU crash was injected.

The prior container's SIGTERM/SIGQUIT cleanup at 07:28 UTC was induced by this
authorized recreation, not an additional spontaneous NCCL failure.

Rollout evidence and rollback profile:
`/mnt/hot/dsv41_state/cache-attribution-rollout-rg2rbn1u/`.
Capture: `/mnt/hot/dsv41_dumps/capture-20260919T072952Z-3156801`.

### Built-image evidence

Receipt: `/mnt/hot/dsv41_state/cache-attribution-image-4xc2df37/receipt.json`.
Image ID: `sha256:fcd6ae8574240c6f0df3c6277b477234328a028007ef7e97acdcb9ed1db1349c`.
All **40 built-image attribution/retention regressions**, **120 deployment/monitor
tests**, and **17 host-budget tests** passed (177 total). The CPU metadata census completed at 1k / 10k / 50k / 80k
nodes in **2.6 / 26.7 / 151.3 / 245.9 ms**, respectively. These are measured
metadata-scan costs, not model throughput or guarantees for every live tree.
`cpu-tests.log` and `census-cost.log` are alongside the receipt.
