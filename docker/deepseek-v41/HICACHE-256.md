# 256 GB RAM HiCache expansion — September 18, 2026 UTC

User-approved expansion from the qualified 192 GB profile, after another
continuous **60-second idle** window. Container `10a2a24ce3d8` started at
**02:52:20 UTC**, passed built-in startup checks at **02:58:05 UTC**, and remains
healthy with no unexpected restarts. The one-shot rollout service exited 0.

| Verified setting / allocation | Value |
|---|---:|
| Host-total allocation budget, all four ranks | 256,000,000,000 bytes (238.42 GiB) |
| KV payload, all ranks | 239,184,386,560 bytes |
| Payload plus initial free-slot arrays, all ranks | 244,424,489,472 bytes |
| Planned payload, slot allowance and fixed reserve | 255,978,437,120 bytes |
| Host logical FULL capacity | 32,699,648 tokens |
| Host SWA capacity | 254,976 token slots per rank |
| Device FULL / SWA capacity | 3,345,408 / 26,112 token slots per rank |
| Available system RAM at 02:59:32 UTC | 39.86 GiB |
| Remaining RAM after allowing the rest of the cache budget | Approximately 29.1 GiB |
| Swap | None |
| Unexpected container restarts | 0 |

FULL capacity represents one TP-sharded logical cache, not four additive
conversation-token pools. The budget is decimal GB and includes sizing
allowances; it is **not a hard limit on total process RSS**. Dynamic request
state and other applications still need the remaining host headroom.

## Configuration and safety

`compose.api.yaml` sets `DSV41_HICACHE_HOST_BUDGET_BYTES=256000000000`.
The boot validator now accepts up to that value and still permits 192 GB for
rollback. The same tested host-pool sizing code and native copy kernels are
used. Runtime is the unchanged `dsv41-v2` image plus read-only boot and two
host-cache source mounts; no model, API, kernel or scheduler implementation
changed in this expansion.

The watcher credits only the old cache's **verified actual allocations** when
projecting available RAM after replacement, not the entire old 192 GB budget.
It requires 24 GiB projected spare RAM, matching the boot preflight floor and
the user's explicitly approved tighter profile. This is a startup guard, not
ongoing protection against other processes consuming RAM. Tests cover the old
192 GB allocation, the 256 GB target, insufficient RAM, and unchanged settings.

Unchanged: context 524288, fraction 0.87, automatic GPU KV, prefill chunk 2048,
prefill/decode interval 4, concurrency 8, TP4/EP4, DSPARK block 5, RAM Engram,
expandable segments disabled, crash diagnostics, model alias
`deepseek-v4-flash`, port 8000, no backend key and `unless-stopped`.
HiCache remains RAM-only (`cache`, `write_through`, `kernel`, `page_first`),
with no NVMe/storage tier and no persistence across restarts.

## End-to-end checks

**89 offline tests passed:** 66 deployment/drain/API-helper tests plus 23 boot
and real-assembler sizing tests in the pinned image without GPU access.

**11 bounded live inference requests passed** in addition to unchanged startup
smokes:

- Seven API requests: streamed strict tool calls and tool-result round trips
  on Responses and Messages, omitted-thinking signature replay, and native
  image input. Validators check stream ordering, completion, argument assembly
  and absence of raw DeepSeek tool protocol.
- Four cache requests: a unique approximately 5k-token prompt and its exact
  repeat through each API, each returning the correct strict fixture tool call.

| Endpoint | Actual input tokens | Warm device-hit tokens | RAM write-through observed on cold request |
|---|---:|---:|---|
| `/v1/responses` | 4,951 | 4,864 | Yes |
| `/v1/messages` | 4,934 | 4,864 | Yes |

The cache test checks HTTP admission counters to exclude other inference from
metric attribution. Prefill counters are page-rounded (256 uncached plus 4864
device-hit tokens for each warm request); API usage gives the unpadded input
count. Both cold requests increased host-backup byte counters. Both warm
requests had zero host-hit and storage-hit tokens: **this proves warm GPU
prefix reuse and RAM write-through, not restore after GPU eviction**. No cache
flush, forced eviction, long-context load test or throughput benchmark was run.
Sustained concurrent stability and RAM-restore correctness remain unqualified.

### Subsequent 64 × 256k cache-pressure experiment

The [completed long-prefill experiment](CACHE-PRESSURE-64X256K-2026-09-18.md)
ran 03:16–04:13 UTC: all 64 cold calls and eight exact replays returned correct
answers without a restart. Seven older replays missed; the recent replay reused
255,744 tokens in 1.285 seconds via SWA-only restoration, with FULL KV still on
GPU. Host FULL occupancy reached 50.10%, and available RAM stayed above
37.74 GiB. FULL-KV restoration after GPU eviction remains unqualified, and
SWA retention is the leading candidate for the older-prefix reuse gap.
That report also documents a confirmed paged-pool **byte-metric accounting
error**: raw backup/load-back byte counters must not be treated as physical
transfer volumes. No serving changes were made during the experiment.

To repeat the bounded check on an idle service:

```bash
python3 scripts/dsv41_live_acceptance.py --suite cache
```

## Evidence, capture and rollback

Private rollout directory:
`/mnt/hot/dsv41_state/deferred-hicache-256-VuxS5Php/`.
It includes the previous 192 GB resolved Compose, old verified allocation,
source/config identities, drain samples, startup logs, exact new allocations,
effective flags and private E2E request/response artifacts under `acceptance/`.
The previous rollout is documented in [HICACHE.md](HICACHE.md).

Service: `dsv41-hicache-relaunch@deferred-hicache-256-VuxS5Php.service`.
It completed successfully and will not repeat the relaunch. Notifications
were attempted under the notify skill but the endpoint timed out; push
delivery did not succeed.

Capture followed to
`/mnt/hot/dsv41_dumps/capture-20260918T025223Z-760974`.
At 02:59:32 UTC its heartbeat was fresh, packet drops were zero, and native
logging had refreshed successfully after six expected startup-time failures.
The API was idle and healthy, with no observed fatal scheduler/CUDA errors.
The tighter RAM budget warrants monitoring; rollback remains an explicit
operator action using the saved 192 GB Compose after draining new work.
