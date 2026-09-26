# Scheduled 50/50 SWA/FULL RAM HiCache payload split

Historical record; see the [cache guide](../HICACHE.md) for the supported configuration.

User-approved September 18 PDT / September 19 UTC. This is a configuration-only
follow-up to [the 23.6% rollout](HICACHE-SPLIT-2026-09-18.md).

```yaml
DSV41_HICACHE_HOST_BUDGET_BYTES: "256000000000"
DSV41_HICACHE_SWA_FRACTION: "0.5"
```

Only this fraction and the deployment label (`dsv41-v2-hicache-split-v2`)
change. Image `ambientlight/dsv41-sm120:diagnostics-v1`, all four runtime
Python bind contents, GPU settings, APIs, Engram mode, diagnostics, port and
restart policy are unchanged. Canonical source hashes were compared with both
the actual live mounts and files inside the container before freezing rollback.

## Expected allocation; verify against allocation.json after launch

| Item | Previous q=0.236 | Planned q=0.5 |
|---|---:|---:|
| FULL logical token capacity | 28,234,496 | 18,819,840 |
| SWA slots per rank | 565,760 | 1,221,120 |
| FULL tensor bytes, host total | 184,371,258,880 | 122,893,555,200 |
| SWA tensor bytes, host total | 56,926,771,200 | 122,869,094,400 |
| Total plan including metadata allowances | 255,978,642,944 | 255,987,142,144 |

The share is tensor payload, not slots or the entire budget including metadata.
Whole-page rounding predicts 49.995% SWA. Logical capacities are not four
independent pools to add. No new request-boundary quota, pinning, dynamic
borrowing or TTL is introduced. GPU KV remains automatically sized.

The 20:50 PDT pre-rollout snapshot had 16,532,992 FULL tokens (58.6% of the old
pool) and a full SWA pool, with 1,748,736 cumulative retained-boundary slots
evicted. That FULL working set would occupy 87.8% of the smaller pool. The
relaunch empties both caches; success must be judged after traffic warms them,
not from zero post-restart eviction counters. The existing passive monitor
follows the new container and retains the previous launch history.

## One-shot scheduling and evidence

Private state directory:
`/mnt/hot/dsv41_state/cache-split-rollout-Jl8Mru7O/`.

User-systemd service:
`dsv41-cache-split-relaunch@cache-split-rollout-Jl8Mru7O.service`.

Armed and started **2026-09-19 03:59:27 UTC (September 18, 20:59:27 PDT)**.
All **114 deployment/helper tests and 16 host-budget tests passed**, including
the 23.6%-to-50% scope guard, rejection of the old fraction, rounded production
capacities, unchanged serving contract, idle/arrival handling and no repeat
after an attempt. The real allocator reproduced the expected capacities and
byte cap without allocating model memory. Compose and the service unit validate.

The watcher polls `/v1/loads` and HTTP inference counters every five seconds.
Active/queued work, active HTTP streams, or arrivals between polls reset the
60-second idle window. Invalid/stale telemetry cannot qualify as idle. The
existing projected-free-RAM floor is 24 GiB. Frozen configuration, image and
source identities are checked before a single recreation attempt. A changed
deployment or source cancels the attempt rather than applying stale intent.

The user manager already has linger enabled, so this job does not depend on
the interactive session. `Restart=no`; no automatic repeated rollout or
rollback. A machine reboot is not a reason to retry against a changed container.

`status.json` is authoritative: `waiting_for_idle`, `waiting_for_ram`,
`relaunching`, `startup_checks_pending`, `complete`, or `failed`.
`drain.jsonl` records the actual idle interval. `rollback.compose.json` and
`rollback-src/` preserve the previous profile; `candidate.compose.json` and
`candidate-src/` freeze the next one. `manifest.json` records identities/hashes.

After the normal built-in arithmetic/JSON/tool/vision startup checks, the
watcher verifies four-rank cache allocation, runtime source hashes, API flags,
restart count and capture following. It writes `complete.json` only after
verification succeeds. There is no extra throughput, long-prefill or eviction
test, and no external push notification.

```bash
systemctl --user status dsv41-cache-split-relaunch@cache-split-rollout-Jl8Mru7O.service
jq . /mnt/hot/dsv41_state/cache-split-rollout-Jl8Mru7O/status.json
```
