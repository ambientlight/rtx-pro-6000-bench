# V4.1 integration: pinned SM120 baseline

## Current: preserved L16 runtime with persistent no-swap policy

As of September 25, the live container is `9137db3cfdc7`, started at
12:35:58 UTC. It runs the binary-safe log forwarder, 256 GB RAM HiCache with
67% SWA / 33% FULL payload, and `dsv41.slice` with `MemorySwapMax=0`.
Inference settings and the image remain unchanged. See the
[release and restore guide](RELEASE-2026-09-25.md) and [no-swap policy](NO-SWAP.md).
The older sections below are historical deployment records.

## Historical: L16 binary-safe, supervised launcher logs

Before this rollout, L15 was live at **67% SWA / 33% FULL**, following a restart at September 20,
08:50:41 UTC. Its earlier run stalled when the launcher's strict UTF-8 log
reader died, leaving an unread pipe. The tested L16 candidate forwards binary
log chunks with streaming key redaction and supervises reader failure.
The model image, inference/API settings and cache allocation are unchanged.

The one-shot watcher waited for **60 continuous idle seconds**; L16 first
started September 20 at **11:09:41 UTC**. See
[L16 qualification and rollback](LOG-FORWARDING-2026-09-20.md).

## Previous: L15 restored L13's 67% SWA / 33% FULL split

The L15 rollout completed at **05:42:43 UTC September 20**, after 63.6 seconds
idle. Its first container start was 05:35:49 UTC. It matched archived L13's
complete profile and runtime hashes. The same container restarted at 08:50:41
UTC after the logging stall; this is not an uninterrupted stability result.
See [L15 original rollout](HICACHE-SPLIT-67-L15-2026-09-20.md).

## Previous: 60% SWA / 40% FULL RAM payload

The user requested an immediate restart at `DSV41_HICACHE_SWA_FRACTION=0.6`.
Container `83e838686046` started **September 20, 00:05:12 UTC** (September 19,
17:05:12 PDT). Startup, allocation and telemetry verification passed at
**00:11:07 UTC** with zero restarts. Host capacities are **15,161,856 FULL tokens /
1,475,840 SWA slots per rank**. Image, frozen runtime sources, GPU/inference
settings and 256 GB host budget are unchanged. Capture and monitoring are active.
See [60/40 rollout and prior health evidence](HICACHE-SPLIT-60-2026-09-20.md).

## Previous: eviction attribution and 67% SWA / 33% FULL RAM payload

This previous profile used `DSV41_HICACHE_SWA_FRACTION=0.67`, with the same 256 GB
host-total budget, and the `cache-attribution-v1` image. New telemetry separates
eviction triggers from FULL co-removal and measures host FULL residency not
covered by any usable SWA endpoint. See [attribution and deployment](CACHE-ATTRIBUTION-2026-09-19.md).

**Deployed September 19, 07:29:47 UTC**, container `17a496e4fce5`, after
60.8 seconds idle. Startup, allocation and live telemetry checks passed at
07:35:17 UTC with zero restarts. Capacities are **12,570,368 FULL tokens /
1,656,064 SWA slots per rank**. Capture and the passive monitor followed the
new launch. No extra load test was sent. GPU, API, Engram, NCCL and retention
policies are unchanged; actual GPU capacity remains automatically sized.

## Previous rollout: 50% SWA / 50% FULL RAM payload

Completed September 19 at **04:31:08 UTC**, container `33a99cf71062`, started
04:24:24 UTC. Its capacities are **18,819,840 FULL tokens / 1,221,120 SWA
slots per rank**, with the same `diagnostics-v1` runtime and 256 GB budget.
See [50/50 rollout receipts](HICACHE-SPLIT-50-2026-09-19.md). The old one-shot
watcher completed; it is not armed to deploy the new attribution image.

## Previous rollout: 23.6% SWA / 76.4% FULL RAM-cache payload

`dsv41-v2-hicache-split-v1` is healthy as of **23:08 UTC September 18**. It uses
the unchanged `diagnostics-v1` image with three tested, read-only runtime
overrides. After 63.6 continuous idle seconds, the RAM cache was rebalanced to
**565,760 SWA slots / 28,234,496 FULL tokens** within the same **256 GB** budget.
Metadata allowances remain inside that budget, outside the payload split.
GPU pool sizing and inference/API/diagnostic settings are unchanged. All 48
allocator/retention tests, 99 helper tests and built-in startup checks passed;
capture followed, with zero unexpected restarts. No throughput or GPU pressure
test was sent. See [RAM split rollout](HICACHE-SPLIT-2026-09-18.md).

A [passive cache monitor](CACHE-MONITOR.md) now records FULL/SWA occupancy,
boundary evictions and RAM restores every 30 seconds across container launches.
It flags possible SWA/FULL imbalance for review; it does not change the model
or prove per-prefix eviction ordering from aggregate counters.

## Previous: diagnostics-v1

The running container uses **diagnostics-v1**, layered on Responses-prefix-v1
and the SWA retention fixes below. It was recreated after 63.6 continuous idle
seconds and passed built-in startup checks at **17:49 UTC September 18**.
The new diagnostic settings and all four NCCL control FIFOs were verified;
capture followed the new container with zero packet drops and no restart.
It changes fatal capture/recovery only, not NCCL execution, API behavior or
cache-sizing settings. No extra load test or real GPU fault was injected.
See [diagnostics](DIAGNOSTICS.md) and [deployment status](STATUS.md) for exact
bounds, provenance, current automatic pool sizes and validation scope.

## Earlier qualification: SWA-retention layer on the 256 GB RAM HiCache

`ambientlight/dsv41-sm120:swa-retention-v1` is deployed as `dsv41` after a
60-second idle window. Built-in startup checks and the seven API acceptance
requests passed at 05:19 UTC September 18. The cache now favors completed-request
SWA windows over intermediate prefill checkpoints, while retaining bounded
fallback eviction and transfer locks. Context 524288, fraction 0.87, interval 4,
the 256 GB host budget and the existing API/kernel/Engram stack are unchanged.
Live pressure/replay validation passed: after sixteen 256k prompts, the oldest
replay reused 255,744 tokens in 1.54 s TTFT through real FULL+SWA RAM restore.
Four header-recall replays across Responses/Messages also restored both pools
correctly. No boundary windows were lost under host-SWA pressure, and there
were no API failures or unexpected restarts. Sustained concurrency remains
unqualified; recoverable allocator retries still occur. See the
[SWA retention report](SWA-RETENTION-2026-09-18.md) for current results and rollback.

## Previous: 256 GB RAM HiCache after one minute idle

The [256 GB expansion](HICACHE-256.md) completed at **02:58:05 UTC September
18** after 60 continuous idle seconds. The host-total allocation budget is
256 GB, with 239.18 GB KV payload / 244.42 GB including initial slot arrays,
32,699,648 logical FULL tokens and 254,976 SWA slots per rank. About 39.9 GiB
system RAM remains available (about 29.1 GiB after the remaining cache-budget
allowance); this is a tighter-headroom profile, not a process-RSS limit.
Startup checks, 89 offline tests and 11 bounded API/cache requests passed.
Both APIs reused 4864 prefix tokens and cold requests wrote to RAM HiCache.
Restore after GPU eviction and sustained performance remain unqualified.
No NVMe KV tier, cache flush or throughput benchmark was enabled/run.
Context, fraction 0.87, interval 4 and concurrency 8 remain unchanged; read-only
boot and sizing modules overlay the same v2 image. Capture followed with zero
drops and no unexpected restart. The one-shot job exited 0; push delivery failed.
The [original 192 GB rollout](HICACHE.md) and rollback configuration are preserved.

## Promoted endpoint

**Pre-HiCache baseline configuration:** **prefill/decode interval 4, 0.87 static fraction,
automatic KV sizing, expandable segments disabled**, with persistent
[CUDA/NCCL crash diagnostics](DIAGNOSTICS.md) on the unchanged v2 image plus
the read-only boot-script override described below. Restarted at **19:08:36 UTC
September 17**, ready at **19:12:22 UTC** with all built-in startup checks passing,
no CUDA allocation-retry warnings and no unexpected restart. The effective main KV pool is **3,002,624
tokens**; the independently capped SWA pool remains **26,112 tokens**, with a
32-prefix-tail sizing allowance. No load or throughput test was run for this
profile. `STATUS.md` records evidence and validation scope.

### Active: prefill/decode interval 4

`compose.api.yaml` now sets `PREFILL_DECODE_INTERVAL=4`. A read-only mount of
`sglang-dsv41-production-overlay/deployment/boot.py` forwards it as
`--prefill-decode-interval 4`; the unchanged image's original boot does not read
this variable. The standalone launcher retains upstream's default of 0 when
the variable is absent and rejects invalid or negative values. The interval-only
rollout originally overrode just boot; HiCache now also overrides host-cache
sizing. Model, API, kernel and scheduler implementations remain the same image
bytes.

This allows four decode scheduling rounds between prefill batches when decode
work is available. It does not enable mixed chunks or two-batch overlap.
Context, memory fraction, KV sizing, prefill chunk size and concurrency are
unchanged. This is startup qualification, not throughput qualification.

**Confirmed active September 17 at 19:12 UTC:** after a continuous 20-second
idle window and an immediate idle recheck, `./launch_all.sh dsv41` recreated
the container. `/server_info` reports interval 4; arithmetic, structured JSON,
tool call/result and vision startup checks passed. All 24 offline deployment
tests passed. No additional inference or throughput test was run. Automatic KV
sizing changed from 2,980,608 to 3,002,624 slots with this fresh startup; its
configuration did not change. Capture followed the new container automatically.

Future changes to Compose environment or mounts still require a Compose
recreation, not a plain `docker restart`. The now-created container already has
interval 4 and retains it across ordinary or automatic restarts.

The preceding **0.85** restored-v1 profile was initially verified
with built-in startup smokes and read-only configuration/health/capture checks.
Startup passed at **07:30:56 UTC September 16**; the endpoint is healthy with zero
restarts and an automatically sized **2259712-token KV pool**. Capture is active.
The user subsequently requested a [full-window prefill](BENCHMARK-FULL-PREFILL-RESTORED-085-2026-09-16.md):
524177 uncached input tokens passed at **09:12 UTC**, with the correct answer,
100.39 s server prefill (~5221 tokens/s), 272 recoverable allocation-retry warning
lines, and no fatal/API error or restart. No serving settings changed for the test.
See `STATUS.md` for identity, validation scope and evidence.

The previous 0.90 / 1107712-capped profile passed startup, 14 API requests and
the [exact 524177-token replay](BENCHMARK-FULL-PREFILL-V2-NONEXPANDABLE-2026-09-16.md),
with 208 recoverable allocator retries. That benchmark is historical evidence,
not a test of the newly restored profile. Both expandable-segments variants
failed startup tool correctness; see
[failure details](EXPANDABLE-STARTUP-FAILURE-2026-09-16.md).

The initial v2/0.83 attempt failed KV allocation with external GPU usage. The
user then approved **0.90**; see `STATUS.md` for the current rollout outcome.

The user-approved production profile is **dsv41-v2**, in `compose.api.yaml`: container **dsv41**,
port **8000** with the same host-interface binding as V12, public model name
**deepseek-v4-flash**, context **524288** (half native), and restart **unless-stopped**.
Backend API-key authentication is explicitly disabled to preserve V12 client
compatibility. Keep this endpoint on the same trusted network as the former
V12 backend; LiteLLM's separate frontend authentication is unchanged.

Only V4.1 is enabled for GPU inference. The stopped `dsv4`, `dsv41-api`, and
`qwen3-embed` containers remain intact. `launch_all.sh` now starts LiteLLM plus
`dsv41`, never the embedding model; its old `dsv4` target aliases the new service.

```bash
# Version the already-qualified runtime without rebuilding or changing code.
docker tag sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca ambientlight/dsv41-sm120:dsv41-v2
docker compose -f docker/deepseek-v41/compose.api.yaml up -d --no-build deepseek
python3 scripts/dsv41_live_acceptance.py --suite api
```

The default acceptance/benchmark clients now target port 8000 and the legacy
model name without a key. `--base-url`, `--model`, and optional `--key-file`
remain available for an authenticated canary. Production state is private under
`/mnt/hot/dsv41_state/production`. Pre-cutover state and metadata are preserved
under `/mnt/hot/dsv41_state/pre-promotion-20260915T224125Z`; the original canary's
state remains under `api-candidate`, unchanged.

The production profile is **524288 context and 0.87 memory fraction**.
Compose explicitly sets `PYTORCH_ALLOC_CONF=expandable_segments:False` and
`MAX_TOTAL_TOKENS=""`: the KV pool is automatically sized from available GPU
memory at startup, matching v1's configuration rather than guaranteeing its
historical exact capacity. Both expandable-segments variants failed startup
correctness and are not qualified for serving.
This supersedes the unsuccessful 0.83 attempt; context is unchanged.
The initial rollout excluded throughput and long-prefill tests, as then requested.
The subsequently requested [524k prefill](BENCHMARK-FULL-PREFILL-V2-090-2026-09-16.md)
failed with a fatal GPU 1 OOM during attention-indexer score padding, while
Starfield was active. Docker automatically restarted the unchanged container.
Startup/API health does not qualify this shared-GPU profile for full-window use.
After the user stopped Starfield, the [exact 524177-token replay](BENCHMARK-FULL-PREFILL-V2-NO-GAME-2026-09-16.md)
passed in 100.22 s server prefill (~5230 tokens/s), with 210 recoverable allocator
retry lines and no fatal/API error or new restart. No serving restart or setting
change was made for that repeat; the existing 1107712-slot KV pool was preserved.
That earlier test did not qualify a fresh no-game startup that could size a
larger cache. The subsequent allocator-off control explicitly fixed capacity at
1107712 and passed after a fresh startup. The user then requested the current
0.85 / automatic-KV profile without a load test during relaunch, followed later
by the separately requested successful full-window prefill reported above.
Long-output/concurrent full-window stability remains unqualified.
The older 0.82 experiment remains inactive.
TP4/EP4, eight request slots, 2048-token prefill,
and DSPARK block 5 are unchanged. Engram offloading is configured as
`OFFLOAD_MODE=ram`, `DSV41_CACHE_GIB=0`: the adapter preloads and locks the two
backing shards (~189.1 GiB of shared physical host RAM), without a separate
NVMe row cache. This is CPU RAM offload, not GPU-resident Engram storage;
selected rows still transfer from host RAM to the GPUs. The backend replaces
explicit direct-I/O row fetches with reads from the locked file mappings.
**Residency caveat:** post-startup checks found about 0.25% of backing pages
nonresident, and isolated reads of sampled missing Engram pages caused disk I/O.
RAM mode is active and API tests pass, but strictly disk-free operation is not
yet established; see `STATUS.md`. The `dsv41-v2` image tag aliases the same
immutable runtime image as `sero-v12-api-524k-legacy`; no model, API, or kernel
code was rebuilt. The container label records the release version, while
Compose explicitly controls context and RAM offload. Do not infer these
deployment settings from the image tag when running it outside Compose.
The original canary's long-output TPS runs had no allocator retries, but earlier
short-output long-prefill checks did have recoverable retries; this profile
is not a claim of universal OOM-warning-free operation. Earlier 524288-context
probes also showed allocator pressure; the larger limit is not qualification
of eight simultaneous full-window requests.
The dsv41-v1 isolated 519989-input-token repeat completed correctly in 103.509 s
(100.269 s server prefill, 5185.93 input tokens/s), with 270 recoverable CUDA
allocation-retry warning lines and no fatal/API error or restart. See `STATUS.md`
for exact validation scope and rollback evidence.
The public alias does not select
the encoder: the model config and explicit parser flags still select V4.1.
The durable V12 thinking key remains mounted, allowing old opaque thinking
history to be replayed under the same public alias. These signatures bind to
the alias, not cryptographically to the checkpoint revision.

The promotion was explicitly approved after bounded qualification; a completed
eight-hour V4.1 production soak is not claimed. Any historical LiteLLM `[1m]`
route labels still reach this backend but do not override its 524288-token
limit.

## Original baseline provenance

The hardware reference is the complete 0xSero recipe at `45f538a18569420721e00e37353f9a7e1af7e5da`.
See `baseline.lock.json` for exact image, model, source-layer and V12 rollback pins.
The base image's Git revision remains unknown. Do not substitute a newer SGLang
tree or kernel dependency set during baseline reproduction.

`compose.baseline.yaml` preserves the upstream default inference profile and
adapts only local paths, image/container identity and explicit manual restart.
It serves an authenticated, localhost-only API at port 8010. The generated key
is stored privately at `/mnt/hot/dsv41_state/sero-baseline/api-key`; do not commit
it or print it in shared logs. This baseline does not yet have our V12 API fixes.

## Stages

1. Preserve `dsv4` and `qwen3-embed` inspection, image IDs, mounts and capture position
   with `python3 scripts/dsv41_preserve_deployment.py`. Artifacts are private under
   `/mnt/hot/dsv41_state/rollback-*`; the script does not stop containers.
2. Build the untouched pinned recipe as `ambientlight/dsv41-sm120:sero-45f538a-baseline`.
   Download the exact model revision into the locked model directory, then run the
   recipe's `prepare` command to verify every file against publisher metadata.
3. Stop only `dsv4` and `qwen3-embed` once staging is complete. Keep both containers,
   images, model data, the V12 thinking key, and previous capture files intact.
4. Start the baseline with Docker Compose and capture its traffic. Archive supplied
   smoke outputs, startup logs, effective flags, image ID and resource measurements.
5. Add a separate, reviewed API overlay. Run V12 regressions adapted for spaced
   V4.1 DSML, then endpoint, cancellation and long-workload tests. No production
   routing change before this qualification.
6. Qualify hardware/runtime optimizations independently. Use a manual-workload
   canary before promotion; no unattended synthetic soak is part of this plan.

## Rollback

First stop `dsv41`, then explicitly start the preserved `dsv4` container and
verify port 8000 and the existing `dsv4-wire-dump.service`. Leave embeddings
stopped unless separately requested. Do not remove/recreate preserved containers
or delete their volumes. The original baseline and canary have no automatic
restart policy; the promoted `dsv41` uses `unless-stopped`.

The original V4.1 baseline/canary used port 8010. The promoted container replaces
V12 on port 8000; port 8001 remains offline with the embedding model stopped.

## Reproducible API candidate

The API source is a separate image-pinned worktree at
`/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay`. Its
`DSV41_OVERLAY.md` records provenance and the port scope. There is no SGLang Git
revision label in the pinned image; the import commit and source-layer digest
make that uncertainty explicit.

Before GPU qualification:

```bash
python3 scripts/dsv41_encoder_reference_check.py
bash scripts/dsv41_api_regressions.sh deployment/test_boot.py
python3 scripts/dsv41_build_api_image.py
```

The build helper requires a clean, committed API worktree and the exact locally
built baseline image ID. It checks all thirteen copied runtime files against the
source, and checks that attention, Engram, scheduler and tokenizer-manager files
still match the baseline. No dependency install is performed. CPU test evidence
and image receipts are private under `/mnt/hot/dsv41_state/`.

After baseline smoke acceptance, stop (do not delete) `dsv41-baseline`, prepare
the private `/mnt/hot/dsv41_state/api-candidate` directory and its API key, and
start `compose.api.yaml`. Repoint the capture service's container/key settings
to `dsv41-api` and the candidate key before its startup. Preserve baseline smoke
and capture artifacts. The candidate uses the same hardware flags and cache
volume, but separate state and the V12 durable thinking key (read-only).

```bash
python3 scripts/dsv41_live_acceptance.py --suite api
python3 scripts/dsv41_live_acceptance.py --suite long
python3 scripts/dsv41_live_acceptance.py --suite cancellation
python3 scripts/dsv41_live_acceptance.py --suite mixed
```

The long suite is three bounded requests at roughly 32k, 131k and 200k input
tokens with 64-token output budgets; it is not a soak. The cancellation suite
closes each API during decoding and checks scheduler drain. The mixed suite
overlaps exactly one Responses decode with one long Messages tool request and
records decode gaps. Longer manual workload qualification remains useful but
was not completed before the user's explicit promotion approval.
The API suite includes both tool-result round trips, omitted-thinking signature
replay, and a native image through Responses (seven bounded inference requests).

Load observers and cancellation/promotion checks use `/v1/loads`, including its
`loads` envelope, separate running/waiting counts and additional scheduler queues.
Idle checks also require active/used/pending-prefill token counts to be zero:
this runtime can report zero running/waiting requests during chunked prefill.
Missing, malformed or stale scheduler snapshots fail closed. Historical TPS
observer artifacts retain their list shape and `num_reqs` alias. Before a planned
restart, this read-only helper waits for 20 seconds of fresh, fully idle snapshots:

```bash
python3 scripts/dsv41_load.py --log-file /path/to/private/new-drain-log.jsonl
```

The helper does not stop admission or restart the server itself. The rollout
must follow immediately after drain; new requests can still arrive between
the last snapshot and shutdown. This is not an admission fence.
The production Compose grants `IPC_LOCK` and
unlimited `memlock`; RAM mode requires at least backing-file size plus 24 GiB
of host `MemAvailable` at startup. Keep these settings if recreating the container.

The candidate defaults only missing sampling fields to temperature 1, top-p .95
and a 32768 output budget; explicit caller limits win. The Responses adapter
retains its existing two-token reserve. Named reasoning `high` still means 50
under the pinned publisher revision; newer publisher tier mappings are deferred.
Function tools are in scope. Upstream custom/freeform Responses tools are not
part of this initial port.

## Capture

`dsv41-wire-dump.service` is a user-systemd service outside Docker. It waits for
the selected container and runs the recorder inside its network namespace.
V4.1 uses opt-in `interface=any`, covering both published traffic and the
recipe's in-container loopback smoke requests. A real Docker fixture verified
both directions without duplicated loopback requests. PCAP and native logs are
under `/mnt/hot/dsv41_dumps`; the wire config also records the image ID, container
ID, container init PID and recorder PID. These are sensitive private artifacts.
The recorder attaches a classic BPF IPv4/TCP-port filter before its receive
queue; this prevents unrelated NCCL/ZMQ startup traffic from crowding out HTTP
when capturing all interfaces. User-space filtering still restricts endpoint
addresses and removes the duplicate loopback direction. Startup capture before
this filter was applied had socket drops and must not be treated as lossless.
The existing PCAP analyzer accepts `--port 8000` for production (`8010` for the
preserved canary captures). Use `--server-ip 127.0.0.1`
for in-container smoke traffic, and the container IP from `wire/config.json`
for host-originated requests; report the two disjoint streams separately.

The existing V12 capture unit is left running and resumes on V12 rollback.
The V4.1 unit follows container `dsv41` on port 8000 without an API key; it remains
enabled and waits through container stops/reboots. Starting a new container
instance produces a new private capture folder automatically.
Weights live directly in `~/models/DeepSeek-V4.1-Flash/`, not a deployment-state
or temporary folder.
