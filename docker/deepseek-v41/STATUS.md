# Integration status — 2026-09-25 UTC

## Current: L16 plus persistent no-swap policy, preserved release

Container `9137db3cfdc7` started September 25 at **12:35:58 UTC** and passed
startup checks at **12:43:09 UTC**. The live image remains
`ambientlight/dsv41-sm120:cache-attribution-v1`, with four frozen Python
bind mounts. The 67/33 split, 256 GB HiCache, RAM Engram offload and all GPU/API
settings are unchanged. The enabled host slice now enforces `MemorySwapMax=0`.
See [release preservation and restore](RELEASE-2026-09-25.md) and
[the exact no-swap policy](NO-SWAP.md). Earlier entries below are historical.

## Historical staging: L16, launcher log-pipe stall fix after one minute idle

The background service
`dsv41-log-forwarding-relaunch@log-forwarding-rollout-XHq1vglm.service` was armed.
It preserves the current L15 run until 60 continuous idle seconds. Only the
tested boot launcher and deployment label change; the image, cache allocator,
67/33 fraction, 256 GB budget and all API/GPU settings remain unchanged.

Qualification: **1,203 API/launcher tests plus 622 subtests**, **31 frozen-boot
tests** in a GPU/network-disabled container, and **84 rollout/monitor tests**
passed. The raw-byte forwarder retains cross-chunk key redaction; reader failure
triggers bounded process-group shutdown rather than leaving an unread live pipe.

Authoritative status: `/mnt/hot/dsv41_state/log-forwarding-rollout-XHq1vglm/status.json`.
See [L16 evidence and rollback](LOG-FORWARDING-2026-09-20.md). This scheduled
record predates activation. L16 subsequently started September 20 at 11:09:41
UTC as `59eb1ccc2927`; its status receipt reports completion.

## Current before L16: L15 at 67/33, restarted after logging stall

L15 originally started **05:35:49 UTC September 20** and completed startup
verification at **05:42:43 UTC**, after 63.6 continuous idle seconds. Container
`b4d6d80f5c4f` used the same image/runtime as L13. At **08:06:24 UTC**, strict
UTF-8 decoding killed `/opt/dsv41/boot.py`'s log-reader thread; the API stopped
responding. A stop operation began 08:49:10; Docker force-killed it after 90
seconds. The same container restarted at **08:50:41**, with startup complete
around **08:55:38 UTC**. It is healthy, but this was not an uninterrupted run.

Host capacities remain **12,570,368 FULL tokens / 1,656,064 SWA slots/rank**.
No SWA eviction or uncovered FULL was observed before the stall. Evidence:
`/mnt/hot/dsv41_dumps/capture-20260920T053551Z-1374102`;
current pre-L16 capture: `capture-20260920T085045Z-1715667`.

## Previous: 60/40 RAM payload split

User-authorized immediate recreation, without the idle gate. Container
`83e838686046` started **September 20, 00:05:12 UTC** (September 19, 17:05:12
PDT). Startup and live telemetry were verified at **00:11:07 UTC** with zero
restarts. The image/runtime are unchanged; only `DSV41_HICACHE_SWA_FRACTION`
changes from `0.67` to `0.6`. Verified host capacities: **15,161,856 FULL tokens /
1,475,840 SWA slots/rank**, inside the unchanged **256 GB** plan.

Actual initial buffers total **249.98 GB**; plan including allowances is
**256.00 GB** (255,999,053,824 bytes, below the budget). Available RAM at
verification: **35.1 GiB**. Device FULL/SWA remains **4,062,720 / 26,112**.
Built-in arithmetic, JSON, tool and vision checks, frozen source hashes,
four-rank allocation, diagnostics, capture and live census passed. All 120
deployment/monitor and 17 allocator tests passed. No extra load test was sent.

The prior container had **30 consecutive 10-second health-check timeouts**
before this restart, without a container restart. That evidence is preserved;
its cause is not established. See [60/40 rollout](HICACHE-SPLIT-60-2026-09-20.md).

## Previous: cache attribution and 67/33 RAM payload split

User-authorized recreation completed after **60.8 seconds idle**. Container
`17a496e4fce5` started **September 19 at 07:29:47 UTC (00:29:47 PDT)**;
startup and live telemetry were verified at **07:35:17 UTC** with zero restarts.
Profile: `dsv41-v2-cache-attribution-v1`, fraction `0.67`, unchanged 256 GB budget.

Verified capacities: **12,570,368 FULL tokens / 1,656,064 SWA slots per rank**.
Whole-page rounding gives **66.9969% SWA payload**. Actual initial buffers total
**250.78 GB**; the complete plan including allowances is **255.98 GB**.
Available system RAM at verification: **34.6 GiB**.

Arithmetic, structured JSON, tool round-trip and vision startup checks passed.
New cause/disposition counters reconcile with legacy totals; the enabled
rank-0 census completes and its residency partition reconciles. No added
cache-pressure or throughput test was sent; improved hit rates are not yet
qualified. GPU sizing remains automatic: this launch obtained **4,062,720 FULL
device tokens / 26,112 SWA slots**, with unchanged 0.87 static, 524288 context,
TP4/EP4, concurrency 8, interval 4 and RAM Engram settings.

Capture: `/mnt/hot/dsv41_dumps/capture-20260919T072952Z-3156801`.
Frozen candidate, previous-profile rollback and verification receipts:
`/mnt/hot/dsv41_state/cache-attribution-rollout-rg2rbn1u/`.
See [attribution definitions and rollout evidence](CACHE-ATTRIBUTION-2026-09-19.md).

## Previous rollout: 50/50 RAM payload split

Completed **September 19, 04:31:08 UTC**, after 63.4 continuous idle seconds.
Container `33a99cf71062` started at 04:24:24 UTC, with **18,819,840 FULL tokens /
1,221,120 SWA slots per rank**. `diagnostics-v1`, runtime Python overrides,
GPU configuration, API settings and 256 GB host budget were unchanged.
Frozen receipts: `/mnt/hot/dsv41_state/cache-split-rollout-Jl8Mru7O/`.
See [50/50 rollout](HICACHE-SPLIT-50-2026-09-19.md).

## Previous: 23.6% SWA / 76.4% FULL RAM-cache payload split

Completed **September 18, 23:08:29 UTC (16:08 PDT)** after **63.6 continuous
idle seconds**. Container `dc03b4e129b6`, started 23:02:46 UTC, is healthy with
zero unexpected restarts. The one-shot watcher exited successfully.

The same `diagnostics-v1` image now uses the tested host allocator/assembler
and launcher overrides with `DSV41_HICACHE_SWA_FRACTION=0.236`. Whole-page
rounding gives **23.5919% SWA payload**. The **256 GB host-total budget** still
includes metadata allowances. Capacities are **28,234,496 FULL tokens** and
**565,760 SWA slots/rank**, versus the previous 32,562,944 / 264,704.
Actual initial buffers total **245.83 GB**; the plan including allowances is
**255.98 GB**. Available RAM at verification was **38.8 GiB**.

Device FULL/SWA remains **3,211,520 / 26,112**. The GPU memory fraction,
context, concurrency, DSPARK, Engram mode, API behavior and diagnostic image
are unchanged. All **48 CPU allocator/retention tests**, **99 helper tests**,
and built-in arithmetic/JSON/tool/vision startup checks passed. No additional
long-context, throughput or GPU cache-pressure workload was sent.

Capture: `/mnt/hot/dsv41_dumps/capture-20260918T230247Z-2231712`.
Frozen sources, prior-source rollback and live verification:
`/mnt/hot/dsv41_state/cache-split-rollout-rPapt8Dr/`.
See [RAM split details](HICACHE-SPLIT-2026-09-18.md).

Passive follow-up: enabled `dsv41-cache-monitor.service` samples every 30 seconds
without model changes or inference traffic. It follows launches and records
aggregate pressure/eviction warnings under `/mnt/hot/dsv41_state/cache-monitor/`.
Twelve dedicated monitoring tests pass. See [monitoring scope](CACHE-MONITOR.md)
for the distinction between ordinary churn, boundary-loss candidates and exact
per-prefix attribution, which is not available from these aggregate gauges.

## Previous: diagnostic hardening deployed after one minute idle

The approved **`diagnostics-v1`** rollout completed September 18 at **17:49:36
UTC (10:49 PDT)**. Container `8db682f6f4e1` started at **17:44:13 UTC**, after
**63.6 continuous idle seconds** verified against fresh scheduler snapshots,
HTTP active streams and arrival counters. The one-shot user-systemd job exited
successfully. Built-in arithmetic, structured JSON, tool round-trip and vision
checks passed; zero unexpected restarts. No extra load test was sent.

The context, RAM/HiCache, 0.87 memory fraction, API and NCCL execution settings
are unchanged. Crash collectors are bounded, CUDA/NCCL evidence is requested
before stack inspection, and PyTorch's exception-path trace dump is enabled.
All four current-run NCCL control FIFOs exist; the effective settings and four
runtime-file hashes match the tested image. All 26 built-image crash-path tests
and the expanded **93 deployment/helper tests** pass.
The full CPU API regression rerun passed 1,172 tests plus 409 subtests with
the candidate's dependencies and the working-tree overlay. Evidence:
`/mnt/hot/dsv41_state/api-tests-UtwnSLi0/`.
See [DIAGNOSTICS.md](DIAGNOSTICS.md); source receipt:
`/mnt/hot/dsv41_state/diagnostics-image-ywucbvdg/receipt.json`.

Automatic device FULL / SWA capacities on this launch are **3,211,520 /
26,112** slots. Host FULL / SWA capacities are **32,562,944 / 264,704**;
actual initial host buffers total **244,489,082,368 bytes**, within the unchanged
256 GB budget. Available system RAM was **39.54 GiB** at qualification.
Automatic sizing was not pinned to the previous launch's pool size.

Capture followed to `/mnt/hot/dsv41_dumps/capture-20260918T174417Z-1886888`,
with fresh heartbeat and zero packet drops. Private rollout, drain history,
rollback configuration and qualification evidence:
`/mnt/hot/dsv41_state/diagnostics-rollout-c1w07czi/`.

The previous Responses-prefix container `d2818c9bc2c3` had restart count 1 after
a CUDA launch timeout. This rollout improves diagnostics/recovery; it is **not
a demonstrated fix for that initiating GPU stall**. No real CUDA failure was
injected, so failed-hardware capture remains unqualified.

## Previous: bounded SWA host retention — September 18 UTC

The reviewed `ambientlight/dsv41-sm120:swa-retention-v1` image is running as
`dsv41`, container `6f3e836ee3f8`. Started 05:13:03 UTC after 60 seconds idle;
built-in checks passed at 05:19:09 UTC, followed by all seven API acceptance
requests. There were zero unexpected restarts at startup qualification.

Soft request-boundary SWA retention, incremental backup and host-lock split
fixes are enabled. The new separate host-SWA metrics are exported. The API,
kernel and Engram stack is unchanged; HiCache stays at 256 GB total, context
524288, fraction 0.87, interval 4, TP4/EP4 and concurrency 8. Actual host FULL /
SWA capacities are 32,721,664 / 253,440, with device FULL / SWA capacities
3,367,936 / 26,112. Cache TTL remains disabled.

1,202 regressions + 377 subtests, 87 deployment/helper tests and all 16
built-image SWA tests passed. GPU pressure/replay validation also passed:
16 × 256k cold prompts, an oldest-prefix replay with 255,744 cached tokens and
1.54 s TTFT, and four correct header-recall replays across both APIs. All five
replays demonstrably restored FULL KV and SWA from RAM. 139,776 intermediate
SWA slots were reclaimed with zero boundary-slot evictions. No API failure,
fatal CUDA/NCCL error or unexpected restart occurred; recoverable allocation
failures still appeared in debug logs. Minimum available RAM was 39.23 GiB.
The deployment is healthy and idle. Sustained concurrency is not yet qualified.
See [SWA-RETENTION-2026-09-18.md](SWA-RETENTION-2026-09-18.md) for upstream
decisions, safety boundaries, exact provenance and qualification results.
Private rollout/rollback: `/mnt/hot/dsv41_state/swa-retention-rollout-grf55j8h/`.

## Previous: 256 GB host-total RAM HiCache — September 18 UTC

Expanded after another **60-second idle** window. Container `10a2a24ce3d8`
started at **02:52:20 UTC**, ready **02:58:05 UTC**, healthy with zero restarts.
All ranks agree on **239.18 GB KV payload / 244.42 GB actual initial buffers**
inside the 256 GB allocation budget. Host logical FULL capacity is 32,699,648
tokens; host SWA is 254,976 slots per rank. Device FULL/SWA remains
3,345,408 / 26,112. Available RAM was **39.86 GiB** at 02:59:32 UTC; allowing
the remaining cache-budget metadata reserve leaves approximately 29.1 GiB.
There is no swap. No other serving setting changed.

**89 offline tests and 11 bounded live inference requests passed.** Responses
and Messages strict streaming tool/result flows, thinking signatures and
vision passed. Both repeated approximately 5k-token prompts reused 4864 tokens;
cold requests produced RAM write-through activity. No GPU eviction was forced,
so RAM restore after eviction is not yet qualified. No long-context stress or
throughput benchmark was run. See [HICACHE-256.md](HICACHE-256.md).

Evidence: `/mnt/hot/dsv41_state/deferred-hicache-256-VuxS5Php/`.
The corresponding one-shot user service completed with exit 0.
Capture: `/mnt/hot/dsv41_dumps/capture-20260918T025223Z-760974`, fresh heartbeat,
native logging active and zero packet drops. Push notification delivery failed.
The saved 192 GB rollback is retained. This expansion reduces spare RAM and
should be monitored under real concurrent traffic.

## Previous: 192 GB host-total RAM HiCache — September 18 UTC

After a confirmed continuous **60-second idle** window, container
`2cfa1628cb23` started at **02:07:23 UTC** and passed normal startup checks at
**02:12:58 UTC**. The four ranks agree on 179.15 GB payload / 183.08 GB with
initial slot arrays, inside the 192 GB host-total budget. Effective flags were
verified; all 68 offline tests passed. No extra synthetic inference was sent.
There were zero container restarts; capture followed to
`capture-20260918T020728Z-700991` with zero observed packet drops.
See [HICACHE.md](HICACHE.md) for exact sizes, tests and qualification limits.
The deferred rollout's live status is
`/mnt/hot/dsv41_state/deferred-hicache-BEJbBSuX/status.json` and its user service is
`dsv41-hicache-relaunch@deferred-hicache-BEJbBSuX.service`.
The one-shot job completed successfully. Push delivery failed because the
notification endpoint was unreachable. Real RAM-cache hits/restore correctness
and performance have not yet been qualified beyond built-in startup checks.

## Previous state: prefill/decode interval 4 on the 0.87 diagnostic profile

The user approved applying the staged interval setting once traffic drained.
One request arrived during the drain check; it completed before a continuous
20-second idle window and an immediate pre-restart idle recheck. The normal
`./launch_all.sh dsv41` path then recreated only the V4.1 service.

| Item | Verified value |
|---|---|
| Container | `dsv41`, ID prefix `a645387e534d` |
| Image | Unchanged `ambientlight/dsv41-sm120:dsv41-v2` |
| Boot override | Read-only `sglang-dsv41-production-overlay/deployment/boot.py`; container/source hashes match |
| Started / boot ready | September 17, 19:08:36 / 19:12:22 UTC (12:08:36 / 12:12:22 PDT) |
| Effective prefill/decode interval | **4**, confirmed by `/server_info` |
| Unchanged serving settings | Context 524288, static fraction 0.87, prefill chunk 2048, concurrency 8, TP4/EP4, DSPARK block 5 |
| Other unchanged settings | RAM Engram offload, expandable segments disabled, mixed chunks disabled |
| Main KV capacity | **3,002,624 tokens**, automatically sized (previously 2,980,608) |
| SWA capacity | **26,112 token slots per rank**, cap mode, `prefix_tails=32` |
| Endpoint / auth / restart | `:8000`, `deepseek-v4-flash`, no backend key, `unless-stopped` |
| Startup checks | Arithmetic, structured JSON, tool call/result and vision passed |
| Startup CUDA allocation-retry warnings / fatal errors / unexpected restarts | **0 / 0 / 0** observed |
| Offline checks | **24 passed** (9 boot tests, 15 configuration/diagnostics tests) |

The live interval changed from 0 to 4. A selected effective-serving-configuration
comparison confirmed the other settings above stayed unchanged. Main KV capacity
changed because the same automatic sizing ran again at startup; no explicit cap
or memory-fraction change was made. The API, model, scheduler implementation and
kernel files remain from the same immutable image. No extra inference, throughput
or long-prefill test was run beyond the normal startup/health checks. This does
not establish a measured decode-latency improvement under concurrent traffic.

Capture followed the new container to
`/mnt/hot/dsv41_dumps/capture-20260917T190839Z-215608`, with zero observed packet
drops. Native logging refresh failed four times while the API was unavailable
during startup, then succeeded once the server was up; these were not inference
failures. Persistent CUDA/NCCL diagnostics for this start are under
`/mnt/hot/dsv41_dumps/diagnostics/run-20260917T190837Z-a645387e534d-w7x8h52x`.

Private pre-restart metadata, drain snapshots, launcher output, startup logs,
effective flags, smoke results and post-restart capture status:
`/mnt/hot/dsv41_state/relaunch-interval4-53BjUWsS/`.

## Previous state: 0.87 static fraction with persistent CUDA/NCCL diagnostics

The user requested a restart at **0.87** with additional diagnostics for the
observed CUDA launch-timeout/NCCL crash, and an exact SWA-capacity report.
The unchanged dsv41-v2 image was recreated after 20 seconds of fresh, fully
idle `/v1/loads` snapshots plus an immediate pre-restart idle check.

| Item | Verified value |
|---|---|
| Container | `dsv41`, ID prefix `48d048730cd1` |
| Image | `ambientlight/dsv41-sm120:dsv41-v2`, same immutable image bytes |
| Started | 2026-09-17 06:17:27 UTC / September 16 23:17:27 PDT |
| Boot checks complete | 2026-09-17 06:21:12 UTC / September 16 23:21:12 PDT |
| Static fraction | **0.87** |
| Context | 524288 |
| Main KV capacity | **2,980,608 tokens**, automatic; no explicit cap |
| SWA capacity | **26,112 token slots**, cap mode, on every TP rank |
| SWA sizing allowance | `prefix_tails=32`; 13,824 request-budget tokens + 12,288 cache-headroom tokens |
| Sliding window / radix page | 128 / 256 tokens |
| Other serving settings | TP4/EP4, concurrency 8, prefill chunk 2048, RAM Engram offload |
| Allocator | `expandable_segments:False` |
| Endpoint / auth / restart | `:8000`, `deepseek-v4-flash`, no backend key, `unless-stopped` |
| Startup outcome | Arithmetic, structured JSON, tool call/result and vision passed |
| Startup allocation-retry warnings / unexpected restarts | **0 / 0** |
| Offline tests | **53 passed**, including diagnostics path/permissions/boot delegation |

SWA capacity is a shared logical token budget, not four additive pools of unique
conversation tokens. Increasing the static fraction did not increase it. The
API's normalized `swa_full_tokens_ratio=0.8` and `swa_prefix_tails=null` do not
describe the resolved cap; startup explicitly logs `mode=cap`, `swa_tokens=26112`
and `prefix_tails=32`. The main KV capacity also depends on GPU availability at
startup, so its change is not a controlled measurement of the fraction alone.

The [diagnostics configuration](DIAGNOSTICS.md) enables CUDA error logging,
GPU core generation on exception and SGLang's crash trigger, NCCL INFO logs,
and an 8192-entry PyTorch flight recorder with C++ stacks and timeout dumping.
`SYS_PTRACE` makes the existing py-spy crash snapshots possible; a native stack
dump of TP0 succeeded. No kernel/API source was modified or rebuilt. A small
read-only-mounted wrapper assigns unique per-start paths before invoking the
original model boot. Bulk GPU global memory and CPU core dumps are excluded;
no CUDA launch blocking or additional collective timing events are enabled.

Persistent diagnostic directory for this start:
`/mnt/hot/dsv41_dumps/diagnostics/run-20260917T061728Z-48d048730cd1-o9ey0mmg`.
CUDA dump pipes and PyTorch trace-trigger pipes were present for all four
workers; NCCL files were being written. Non-fatal PyTorch trace exports then
succeeded on **all four ranks**, each with **1161 collective entries and stack
frames** (447860 bytes per file). The server stayed healthy with no restart.
CUDA dumps were not deliberately triggered. An actual crash may still prevent
a complete dump.

Request/response capture followed the new container automatically:
`/mnt/hot/dsv41_dumps/capture-20260917T061731Z-3539988`, with native logging active
and zero observed packet drops. No load test, long-prefill test or additional
inference request was issued beyond unchanged built-in startup/health probes.
This is startup qualification, not proof of long-context/concurrent stability.

Rollback configuration, previous container inspection/state, drain evidence,
startup logs, effective server settings and verification are preserved privately:
`/mnt/hot/dsv41_state/restart-087-cuda-XadPMrWs/`.

## Previous state: v1 memory profile restored on the unchanged v2 image

The user requested **0.85 static memory fraction and automatic KV sizing**, with
**no load or throughput tests**. Compose now sets `MEMORY_FRACTION=0.85`, clears
`MAX_TOTAL_TOKENS`, and retains `PYTORCH_ALLOC_CONF=expandable_segments:False`.
Context remains 524288, chunked prefill 2048, concurrency 8, TP4/EP4, DSPARK block
5 and RAM Engram offload. The image bytes/tag, container alias, API contract and
restart policy are unchanged. A normalized comparison against the preserved v1
Compose confirms matching serving settings; only the v2 release identity and
explicitly disabled allocator flag differ from the v1 file.

All **49 V4.1 tests + 10 capture tests** passed. Following 20 seconds of fresh,
continuously idle scheduler snapshots, the container was recreated at
**07:27:11 UTC**. Built-in arithmetic, structured JSON, tool call/result and vision
checks passed; the boot-ready marker appeared at **07:30:56 UTC**. The runtime
reports **0.85**, `max_total_tokens=null` (no explicit cap), **2259712 effective
KV slots**, and maximum input **524282**. The launch command omits
`--max-total-tokens`. Automatic sizing is 12800 slots below the historical v1
pool (2272512); matching configuration does not fix startup GPU availability.

The backend is healthy with **zero restarts and zero startup allocation-retry
warnings**. Capture followed the replacement container to
`/mnt/hot/dsv41_dumps/capture-20260916T072715Z-3288697`, with native logging active
and zero kernel packet drops. Old V12, canary and embedding containers remain
stopped. **No additional inference requests, API acceptance suite, long-prefill,
throughput or synthetic-load test was run during that relaunch**. A later,
separately requested full-window prefill is recorded below.
Private preservation, drain and verification evidence:
`/mnt/hot/dsv41_state/restore-v1-memory-085-zDNsfbGu/`.

### Subsequently requested full-window prefill — passed at 09:12 UTC

The user then requested another full half-native-context prefill. The
[exact saved 524177-token request](BENCHMARK-FULL-PREFILL-RESTORED-085-2026-09-16.md)
was replayed once at **09:10:38 UTC**, after 20 continuous idle seconds. The
container, image, settings and cache pool were unchanged; no restart or cache
flush was performed. Starfield was absent and no other non-health inference
overlapped the probe.

Validation finished at **09:12:24 UTC**: **524177 uncached input tokens**, correct
`42` answer (two output tokens), **100.3928 s server prefill / 5221.26 tokens/s**,
and **103.9238 s client wall time**. There were **272 recoverable allocation-retry
warning lines** (GPU 0/1/2/3: 65/77/65/65), with no fatal/API error, retraction,
deleted-request-state warning, restart or packet drop. The backend remained
healthy and idle. No ongoing test workload was left running.

Performance is essentially unchanged from the 0.90/capped test (5229.05 tokens/s),
while warning lines increased from 208 to 272. The KV pool is now **2.04 times
larger** (2259712 versus 1107712), so this is not a memory-fraction-only comparison.
This qualifies one full-window short-answer request, not long-output endurance,
semantic retrieval, concurrent full-window requests or co-running the game.
Private evidence: `/mnt/hot/dsv41_state/prefill-524k-restored-085-ZmKGG3uU/`.

## Previous state: v2 / 0.90 capped — non-expandable allocator restored and tested

At user direction, the failed expandable-segments deployment was preserved and
recreated at **07:03:46 UTC** with `PYTORCH_ALLOC_CONF=expandable_segments:False`.
The **0.90 fraction and explicit 1107712-token KV cap are unchanged** from the
second failed experiment below. The image, model, kernels, API overlay, context
524288, RAM offload and remaining serving configuration are identical; a
normalized Compose comparison permits only the allocator-setting change.
All **49 V4.1 tests + 10 capture tests** passed. The boot-ready marker appeared
at **07:07:16 UTC**, including the tool smoke that failed in both expandable
experiments: this time it returned the correct tool call in **49 output tokens**.
All **14 live API requests** (seven before and seven after the full prefill)
passed Responses/Messages tool streaming and continuation, thinking replay,
and vision checks.

The [exact full-window replay](BENCHMARK-FULL-PREFILL-V2-NONEXPANDABLE-2026-09-16.md)
completed at **07:10:27 UTC**: **524177 uncached input tokens**, correct `42`
(two output tokens), **100.2432 s server prefill / 5229.05 tokens/s**, and
**103.5697 s client wall time**. There were **208 recoverable allocator-retry
warning lines** (GPU 0/1/2/3: 50/58/50/50), versus 210 in the earlier successful
no-game run. This is effectively the same performance and allocator pressure,
not a meaningful retry improvement. No fatal/API error, retraction, restart,
other non-health inference during the prefill, or packet drop occurred.

At **07:10:56 UTC**, the backend was healthy and idle with **zero restarts**;
capture remains active at
`/mnt/hot/dsv41_dumps/capture-20260916T070348Z-3227082`. The old V12 and embedding
containers remain stopped, and no synthetic workload is left running.
Private evidence: `/mnt/hot/dsv41_state/nonexpandable-090-kvcap-HZT2NRcA/`.

Disabling expandable segments while preserving the capped pool restored startup
tool correctness. This strongly implicates an interaction with the allocator
mode in this pinned runtime; it does **not** identify a specific faulty kernel,
CUDA graph or allocator implementation. The explicit cap also qualifies this
fresh no-game startup, unlike the earlier uncapped configuration. One short-
answer full-window request does not qualify long-output endurance, semantic
retrieval, concurrent full-window requests, or running the game alongside it.
The RAM-residency caveat below is unchanged.

## Earlier expandable-segments startup failures

The user requested expandable segments, a no-game relaunch at 0.90 and another
full-window prefill. The only Compose serving change was
`PYTORCH_ALLOC_CONF=expandable_segments:True`; all 49 V4.1 and 10 capture tests
passed. The new container started at **06:36:19 UTC**, with **5329152 KV slots**.
Arithmetic and structured JSON passed, but the 289-token startup tool request
generated 32768 tokens of repetitive prose and no tool call. The boot assertion
failed at **06:42:54 UTC**; no boot-ready marker or full-window prefill followed.

After one automatic restart, `dsv41` was intentionally **stopped at 06:43:59 UTC**
to prevent a restart loop. That first attempt changed both allocator mode and
startup-sized cache capacity, so it did not isolate the causal factor.

The user then approved retrying with an explicit **1107712-token KV cap**.
The capped expandable deployment started at **06:49:13 UTC**. Arithmetic and
structured JSON again passed, but the same 289-token tool request generated
32768 output tokens (16383 repetitions of `.0`) and no tool call. It ended at
**06:54:55 UTC**, followed by the boot assertion and one automatic restart.
The container was explicitly stopped at **06:55:39 UTC**. Neither expandable
attempt reached the full-window test. No allocator-retry or fatal OOM was
observed in either startup. Matching the old KV capacity did not fix the
correctness failure; the exact underlying mechanism is not yet proven.
Capped evidence: `/mnt/hot/dsv41_state/expandable-090-kvcap-xOxT9HDJ/`.

See [failure details](EXPANDABLE-STARTUP-FAILURE-2026-09-16.md). Private evidence
and prior-config rollback: `/mnt/hot/dsv41_state/expandable-090-YXkuqtGz/`.

## Previous successful state: v2 / 0.90 without expandable segments

After the user stopped Starfield, the [exact full-window replay](BENCHMARK-FULL-PREFILL-V2-NO-GAME-2026-09-16.md)
**passed at 06:24:05 UTC**: **524177 uncached input tokens**, correct `42` answer,
**100.2216 s server prefill / 5230.18 tokens/s**, and **103.8062 s client wall time**.
There were **210 recoverable allocation-retry warning lines**, but no fatal/API
error, retraction, new restart, other non-health inference traffic or packet
drop. Stream ordering, assembled output and native/API token usage agree. The
backend remained healthy and idle, with restart count still 1 from the prior
crash. Evidence: `/mnt/hot/dsv41_state/prefill-524k-v2-no-game-MbU2UEX0/`.

No restart or configuration change was performed after closing the game; the
KV pool remains **1107712** slots from the earlier automatic recovery. This
qualifies one isolated short-answer prefill, not allocation-retry-free operation,
long output or concurrent full-window calls. A later restart without the game
could size a larger cache at the same 0.90 fraction and is not qualified by this
test. The earlier RAM-residency limitation is unchanged.

### Previous full-window attempt with Starfield running

The subsequently requested [full-window prefill](BENCHMARK-FULL-PREFILL-V2-090-2026-09-16.md)
**failed at 06:06:08 UTC**: 524177 input tokens, zero output, 173 allocator-retry
warning lines followed by a fatal GPU 1 OOM. The attention indexer's candidate
score padding needed 1 GiB with only 808.94 MiB device-free. Last scheduled
progress was 387072 tokens; 137105 remained pending. Starfield was active on
GPU 1. There was no other non-health inference traffic and no capture packet
drop. HTTP 200 was followed by an unterminated Responses stream, not success.

Docker automatically restarted the unchanged container at **06:07:36 UTC**.
It was ready at **06:11:56 UTC** and verified healthy/idle at **06:12:44 UTC**,
with fresh built-in startup smokes, one total restart and native capture active
with zero packet drops. The unchanged 0.90 setting now sizes **1107712 KV slots**
against current GPU availability; maximum input remains **524282**. No manual
restart, configuration change or repeated probe was performed. This shared-GPU
profile **with the game running was not qualified for full-window prefill**. Private evidence:
`/mnt/hot/dsv41_state/prefill-524k-v2-090-VbCEm8U8/`.

### Initial 0.90 startup validation

After the failed 0.83 attempt below, the user approved **0.90 memory fraction**.
Container `dsv41` was recreated at **05:46:34 UTC September 16** and ready at
**05:51:54 UTC** (22:51:54 PDT September 15). The configured **524288-token
context** is preserved: effective maximum input is **524282**, with **1059840
KV slots**. The smaller KV pool than the earlier v1 deployment reflects the
changed startup GPU availability; Starfield remained running and was not touched.
The fraction controls startup cache sizing, not a hard runtime memory cap or
isolation from later growth in another GPU application's allocations.

The runtime image bytes, model, API overlay/parsers, kernels, TP4/EP4, 2048-token
prefill chunks, concurrency 8, DSPARK block 5, RAM Engram offload, port 8000,
`deepseek-v4-flash` alias, no-backend-key access and `unless-stopped` policy are
unchanged. Old V12 and embedding containers remain stopped.

All **48 V4.1 configuration/harness tests + 10 capture tests**, built-in startup
smokes, and **seven small live API requests** passed. The latter cover strict
streamed tools and result continuations on Responses and Messages, omitted-thinking
signature replay, and Responses vision. At **05:52:50 UTC**, the container was
healthy with zero restarts, allocator-retry warnings, fatal OOMs, tracebacks or
deleted-request-state warnings in the new capture. The scheduler was idle.
**No throughput or long-prefill tests were run during that initial rollout**, as
then requested. The later explicitly requested full-window test failed as above.
Neither check resolves the earlier RAM-residency limitation.

Initial rollout capture was
`/mnt/hot/dsv41_dumps/capture-20260916T054637Z-3042052`, with native level-3
logging active and zero kernel packet drops. Five logging refresh attempts
failed before API readiness; recording was already active. Private rollout,
test, API and verification evidence, plus the preserved v1 rollback Compose:
`/mnt/hot/dsv41_state/dsv41-v2-090-XLf9e22N/`. The initial verification receipt
is also at `/mnt/hot/dsv41_state/production/dsv41-v2-verification.json`.

## Earlier v2/0.83 attempt — failed under concurrent GPU use

The user requested `dsv41-v2` with **0.83 memory fraction**, preserving 524288
context and all other serving settings, and explicitly excluded throughput
tests. After a verified 20-second idle window, container `dsv41` was recreated
at **05:36:41 UTC September 16**. Image bytes are unchanged; only the release
tag/label and configured memory fraction differ. All **48 configuration/harness
tests and 10 capture tests** passed before relaunch.

At **05:40:25 UTC**, startup failed while sizing KV cache. Starfield was using
about **6.5 GiB on GPU 1**, leaving approximately 11.28 GiB free there after
the main and DSpark loads. This runtime bases its non-static reserve on the
minimum available GPU memory sampled before loading, not the card's nominal
capacity. At 0.83, the reserve exceeded post-weight free memory. SGLang reported
minimum viable fraction **0.8694** and suggested above **0.870**; that is a
positive-KV-budget threshold, not a qualified context-capacity target.

Docker automatically restarted once. The failed container was then intentionally
stopped at **05:41:30 UTC** to prevent repeated reloads. Restart policy remains
`unless-stopped`; the game was not touched. The inference endpoint remained
offline until the user approved the 0.90 relaunch above.
No throughput, long-prefill, or post-start API tests were run in this failed
attempt; no usable 524k capacity was established at 0.83.

Both failed-start capture folders are preserved:

- `/mnt/hot/dsv41_dumps/capture-20260916T053645Z-3026323`
- `/mnt/hot/dsv41_dumps/capture-20260916T054051Z-3038735`

Private state, test logs and the exact v1 rollback Compose are under
`/mnt/hot/dsv41_state/dsv41-v2-rollout-fESHEjZf/`. The v1 rollback is also subject
to current GPU occupancy; it was not automatically relaunched.

## Previous release: dsv41-v1 — half-native context

**Previous deployment:** `dsv41-v1` ran in container **`dsv41`**, recreated
at **03:57:29 UTC September 16** (20:57:29 PDT September 15), after 20 continuous
seconds of confirmed idle scheduler state. Startup readiness was confirmed at
**04:01:25 UTC**. Context is **524288**, half the native 1048576-token window;
effective maximum input is **524282** and KV capacity remains **2272512** slots.
The release tag `ambientlight/dsv41-sm120:dsv41-v1` references the same immutable
runtime image as before; no model, API/parser, or kernel code changed.

RAM Engram offload (`OFFLOAD_MODE=ram`, `DSV41_CACHE_GIB=0`), memory fraction
0.85, 2048-token prefill chunks, concurrency 8, TP4/EP4, DSPARK block 5, port
8000, public model `deepseek-v4-flash`, no-backend-key access, and
`unless-stopped` are preserved. Old V12 and embedding containers remain stopped.

Validation: **48 V4.1 tests + 10 capture tests**, all five built-in startup
inference checks, and **seven live Responses/Messages inference requests**
passed. The live checks cover strict streamed tools and result continuations
on both APIs, omitted-thinking signature replay, and Responses vision.

The isolated near-limit prefill finished at **04:05:19 UTC**: **519989 uncached
input tokens**, correct `42` answer (two output tokens), **100.269 s server
prefill / 5185.93 tokens/s**, and **103.509 s client wall time**. SSE ordering,
terminal status, and native/API token accounting agree. There were no fatal/API
errors, retractions, deleted-request-state warnings, restarts, or packet drops.
However, there were **270 recoverable CUDA allocation-retry warning lines**
across the four ranks; sampled GPU usage peaked at **97191 MiB**. This qualifies
one near-limit, short-answer request, **not** memory-pressure-free operation,
long-output quality, semantic retrieval, or eight concurrent full-window calls.
The earlier RAM-residency limitation below has not been resolved or requalified.

Capture followed the replacement container automatically:
`/mnt/hot/dsv41_dumps/capture-20260916T035730Z-2783908`.
Native level-3 logging is active. Private drain, before/after, API, prefill, and
rollback evidence: `/mnt/hot/dsv41_state/dsv41-v1-rollout-yPAD7HKt/`.
`rollback-compose.yaml` there preserves the previous 409600-context RAM profile;
using it requires a fresh queue drain and an explicitly requested relaunch.
No ongoing synthetic workload was left running.

## Previous 409600-context RAM deployment (01:11 UTC)

Previously, `dsv41` was recreated at **01:11:54 UTC September 16**
(18:11:54 PDT September 15) with **`OFFLOAD_MODE=ram`** and **`DSV41_CACHE_GIB=0`**.
The drain gate observed 20 continuous seconds with no running requests or
queued work before recreation. Startup readiness was confirmed at **01:18:04 UTC**;
all five built-in inference smokes passed. The same image, API overlay, GPU flags,
409600 context, 0.85 memory fraction, port/model alias, no-key access, and
`unless-stopped` policy are preserved. Effective KV capacity remains **2272512**.

The subsequent [full-400k RAM-mode repeat](BENCHMARK-FULL-PREFILL-RAM-2026-09-16.md)
completed at **01:28:28 UTC**: **409397 uncached input tokens**, correct `42`,
**73.636 s server prefill / 5559.73 tokens/s**, and **156 recoverable allocator
retry warning lines**. This run was isolated, with no API/fatal error or restart;
RAM offload did not solve GPU allocator pressure. Scheduler-worker storage
reads totaled 6.699 MiB and 11 major faults during the test. No serving settings
were changed. The test also exposed a client idle-check gap during chunked
prefill; the guard now checks active and pending token counts as well as request
and queue counts. All **46 V4.1 tests plus 10 capture tests** pass. The earlier
rollout's idle window was revalidated with the stricter check and was truly idle.

Before that full-prefill repeat, all **43 V4.1 configuration/collector tests**, **10 packet-capture tests**,
**seven live API requests** and **two live disconnect/drain checks** passed.
The small CPU RAM row-store fixture also passed all three cache-budget cases
(36252 checked row lookups, zero explicit disk reads). The updated observer and
idle check were exercised live against `/v1/loads`, with no observer errors.
No old `/get_load` access, allocator retry, fatal OOM, traceback, deleted-request
state warning, or container restart was observed in that initial rollout validation.

**Important limitation:** RAM mode is active, but strictly disk-free residency
is **not fully achieved/verified**. All eight Engram mappings (two files on four
ranks) cover ~189.1265 GiB of shared backing and carry the locked VMA flag.
Nevertheless, post-startup page-cache checks measured **99.5701%** and
**99.9247%** residency: ~0.25% of pages (~0.48 GiB) were still absent. Eight
read-only probes of missing pages, all inside Engram weight tensors, each caused
a major fault and storage read. Worker counters also increased by ~3.57 MiB
during the two cancellation probes; that aggregate is not attributed exclusively
to Engram. Host available RAM was about **274 GiB**, so the preflight capacity
check was not marginal. Do not treat a successful `mlock()` call, RAM-mode log,
or global `Mlocked` counter alone as proof of zero disk I/O. The underlying
pinning/residency cause remains unproven; no speculative image/kernel workaround
was applied. See the private residency evidence below for reproducible offsets.

Capture followed the new container automatically, with zero packet drops and
native level-3 logging active. Six logging-configuration attempts failed before
the API became ready; packet recording was already active during that period.
Private rollback, drain, test, memory, and deployment evidence:
`/mnt/hot/dsv41_state/ram-offload-kRktvpVH/`.

## Earlier context restoration (September 15)

Earlier, at the user's request, **dsv41** was restarted at **23:11:40 UTC** with the
original **409600-token context and 0.85 memory fraction**. At **23:15 UTC** it
was **healthy**, with all five built-in startup inference checks passed, zero
allocator retries since this restart, and no active/queued requests. It retains
**port 8000**, served model **deepseek-v4-flash**,
restart **unless-stopped**, and **no backend API key**, matching the V12 access contract.
It is the only enabled GPU inference backend. `dsv4`, `dsv41-api`, and
`qwen3-embed` remain stopped and intact; normal startup no longer launches the
embedding model. No synthetic workload is left running.

The subsequently requested [near-full-window prefill test](BENCHMARK-FULL-PREFILL-2026-09-15.md)
completed at 23:21 UTC: **409397 uncached input tokens**, correct `42` response,
**75.42 s server prefill / 78.41 s client wall time**, and **109 new recoverable
allocator retry warning lines**. No fatal OOM, API stream failure, retraction or
restart occurred. A mostly cached client request (only 279 uncached tokens)
waited **69.18 s**; all retry warnings preceded its first forward pass. Thus the
restored profile is functional but is not allocation-retry-free or qualified
for low-latency concurrent agent traffic.

The earlier 524288-context, 0.85-fraction single-request 499,989-token input test returned `42` correctly in
98.88 seconds, but produced **247 recoverable CUDA allocator OOM/retry warning
lines** (GPU0/1/2/3: 60/67/60/60). There was no terminal OOM, traceback, deleted
TokenizerManager state, or restart. This establishes bounded 500k-input success,
not memory-pressure-free operation or eight simultaneous full-context requests.

At 22:25 UTC, the earlier canary's [long-call TPS baseline](BENCHMARK-LONG-TPS-2026-09-15.md)
also completed: three sequential 32k/131k/200k-context requests generated 22,471
tokens at 73.1–83.3 decode tokens/s. No new allocator retries or API stream errors
were observed in those requests. Those measurements used the original 409600-context
canary profile; they do not qualify concurrent-agent throughput.

## Current deployment identity

| Component | State / identity |
|---|---|
| Weight destination | `/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash` (`~/models/DeepSeek-V4.1-Flash`) |
| Model revision | `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Pinned recipe checkout | `../deepseek-v4.1-flash-4x-rtx-pro-6000`, clean at `45f538a18569420721e00e37353f9a7e1af7e5da` |
| Baseline image | `ambientlight/dsv41-sm120:sero-45f538a-baseline` |
| Baseline image ID | `sha256:e930a2bd721c583d9aae78fd28f0cbb90141a09213fa2169a1afed0d075fd691` |
| API source | `../sglang-dsv41-production-overlay`, clean at `47d89e66dd14b60aaa6daa2c09197e3761f5da86` |
| Configured image | `ambientlight/dsv41-sm120:dsv41-v2`; healthy with restored v1 memory settings, restart count 0 |
| Active image ID | `sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca` |
| Container ID | `67f5f11c1a1b65dc8724745ef400e48529969861a1ac7e634566967780a4341d` |
| Configured profile | Context 524288; memory fraction 0.85; automatic KV sizing; prefill chunks 2048; concurrency 8 |
| Engram offload | RAM mode; separate NVMe row cache disabled; residency limitation documented above |
| Allocator setting | `PYTORCH_ALLOC_CONF=expandable_segments:False`; unchanged for this relaunch |
| Effective capacity | 2259712 automatically sized KV slots; maximum input 524282; no explicit KV cap |
| Production routing | Port 8000, same host-interface binding and no-key access as V12; LiteLLM config unchanged |
| Capture | Enabled/active `dsv41-wire-dump.service`, following container `dsv41` and port 8000 |
| Latest capture | `/mnt/hot/dsv41_dumps/capture-20260916T072715Z-3288697`; recorder and native logging active |
| API access | `http://127.0.0.1:8000/v1`, model `deepseek-v4-flash`, no backend key |
| Production state | `/mnt/hot/dsv41_state/production` |

## Earlier memory tuning and restoration

Only runtime configuration changed; the image, API overlay, parsers, kernels,
model weights, endpoint identity and restart policy were preserved. The image's
historical `524k-legacy` tag does not override the current Compose context.

| Context | Memory fraction | Observed result |
|---:|---:|---|
| 409600 | 0.85 | Original 32k/131k/200k long-output TPS tests had zero retries. Separate earlier short-output long-prefill checks had 37 recoverable retry warning lines. |
| 524288 | 0.85 | 499989-input request completed correctly in 98.88 s, with 247 retry warning lines. |
| 524288 | 0.83 | 519989-input probe deliberately interrupted when the user requested 0.82. There were 207 retry warning lines after probe start; 29 preceded a separate 225878-input client request, so overlap was not the sole cause. Not a completed or isolated latency result. |
| 524288 | 0.82 | KV capacity fell to 480768, limiting actual input to 480762 despite the larger configured/advertised context. No 520k synthetic request was submitted. Real client traffic resumed during startup. |
| 409600 | 0.85 | Restored and startup checks passed at 23:15 UTC. The later requested 409397-input test completed correctly with 109 new retry warning lines and a 69.18 s wait for a mostly cached companion request; see the full-prefill report. |

Private experiment evidence and pre-change state are retained under:

- `/mnt/hot/dsv41_state/memory-083-520k-3AxHPizi/`
- `/mnt/hot/dsv41_state/memory-082-520k-BIaMZFTP/`
- `/mnt/hot/dsv41_state/restore-400k-085-qM2GhUTJ/`

At that restoration, the acceptance and TPS clients defaulted to context limit
409600. All 28 configuration, harness-boundary and token-accounting tests
passed; a 520k target was rejected under that limit before inference. For the
current dsv41-v2 release, the default is 524288; an explicit 409600 rollback
limit still rejects a 520k target before inference.

Restoration verification: HTTP health 200; `/v1/models` reports 409600;
effective maximum input is 409594; no crash/restart or deleted-request-state
errors. The fresh capture matches the new container and has zero packet drops;
native logging is active after three pre-readiness configuration attempts.
Original V12, canary, and embedding containers remain stopped. Evidence:
`/mnt/hot/dsv41_state/restore-400k-085-qM2GhUTJ/verification.json`.

## Earlier 524288-context promotion verification (22:50 UTC)

- Full CPU regressions: **1170 passed, 373 subtests passed**, including seven
  launcher-policy tests. Evidence: `/mnt/hot/dsv41_state/api-tests-BNV9L8PI/`.
- Seven additional configuration/stubbed-launcher tests passed. They check the
  legacy endpoint, restart policy, capture settings, and that neither normal
  startup nor the old `dsv4` launcher alias revives V12 or embeddings.
- Image receipt: `/mnt/hot/dsv41_state/api-image-20260915T224257Z/receipt.json`.
  Only the boot wrapper changed in the runtime image; API/parser file hashes
  match the previous candidate, and kernel/lifecycle files match the baseline.
- All five original startup smokes passed using `deepseek-v4-flash`, including
  structured output, an actual tool round trip and full-budget vision.
- Seven unauthenticated live API requests passed: strict streamed tool calls
  and result continuations on both APIs, omitted-thinking signature replay,
  and Responses vision. Evidence:
  `/mnt/hot/dsv41_state/acceptance/api-20260915T224658Z/`.
- Existing authenticated LiteLLM frontend `/v1/messages` route returned HTTP
  200, model `deepseek-v4-flash`, answer `42`, and `end_turn`, without editing
  its configuration or credentials.
- 499,989-input-token Responses request: HTTP 200, two output tokens, natural
  stop, no prefix-cache hits. Evidence:
  `/mnt/hot/dsv41_state/acceptance/long-20260915T224735Z/`.
- `/v1/models` advertises `deepseek-v4-flash` with `max_model_len=524288`.
  Native V4.1 tool/reasoning parsers remain selected despite the legacy alias.
- Final health HTTP 200, no active/queued requests, zero container restarts,
  capture active with zero kernel packet drops. Native logging had three
  expected configuration attempts before startup was ready; packet capture
  was already active. The first five smoke responses are preserved by their
  boot artifacts and PCAP; native level-3 logging covers the nine later calls.
- Canary rollback snapshot:
  `/mnt/hot/dsv41_state/pre-promotion-20260915T224125Z/`. Its
  `capture-canary.service` records the pre-reload unit configuration. Original
  canary container/state and V12/embedding containers were not deleted.

## Earlier baseline/canary checks

- All 88 publisher files, including all 48 weight shards, hash verified at the
  pinned model revision. The original recipe's `prepare` exited zero. Receipt:
  `/mnt/hot/dsv41_state/sero-baseline/verification.json`.
- 0xSero CPU row-store tests: all three cache budgets passed, each checking
  12,084 returned rows, including concurrency/eviction parity.
- Combined source API/parser/lifecycle/launcher regressions in the pinned image:
  **1,165 passed, 373 subtests passed**, including the native-ID fix. Evidence:
  `/mnt/hot/dsv41_state/api-tests-MsqtNzGP/`.
- All five pinned publisher prompt goldens match exactly.
- Built API image: required/auto/named structural grammars compiled and matched
  valid real-tokenizer output; invalid strict arguments rejected.
- Build helper verified the thirteen copied source files and launcher hashes;
  attention, Engram adapter, scheduler and tokenizer-manager hashes match the
  baseline. Receipt:
  Latest corrected-image receipt:
  `/mnt/hot/dsv41_state/api-image-20260915T211332Z/receipt.json`.
- Capture filter/authentication/provenance tests: 10 passed; capture analyzer
  regressions: 7 passed. Real Docker fixture captured external and loopback
  requests exactly once with zero kernel drops.
- Hardware startup exposed receive-queue drops in unfiltered all-interface
  capture. A libpcap-derived classic BPF TCP-port filter now runs before that
  queue, tested against the real Linux socket-filter implementation. Recorder
  restarted at 20:58 UTC before inference smokes. Pre-filter capture is retained
  but is not assumed lossless; no model restart was needed for the capture fix.
- Live-check SSE validators: 4 passed (these validate the test harness only).
- Latest private pre-stop rollback snapshot:
  `/mnt/hot/dsv41_state/rollback-20260915T205215Z/`.
- Graceful stop: both preserved V12 and embedding containers exited zero; all
  four GPUs released model memory before baseline launch.
- Untouched 0xSero GPU baseline: all original smokes passed at 21:00:07 UTC.
  Arithmetic `42`; structured `{"answer":42}`; automatic tool call with
  `{"key":"alpha"}` and value-42 continuation; native vision correctly reported
  a red circle left of a blue square with exactly 1024 image tokens. Original
  response artifacts and launch flags are under `/mnt/hot/dsv41_state/sero-baseline/`.
  Post-filter capture: `/mnt/hot/dsv41_dumps/capture-20260915T205843Z-830461`.
  An additional bounded host-origin arithmetic probe also returned `42`.
- First API image (`789e5ddb…`) also passed all original smokes, but its first
  `/v1/responses` request failed before generation: the inherited adapter chose
  an empty text field on a vision-capable native-ID encoder. Evidence:
  `/mnt/hot/dsv41_state/acceptance/api-20260915T210435Z/` and capture
  `/mnt/hot/dsv41_dumps/capture-20260915T210155Z-849705`.
  Fix `da53ffc` shares Chat/Responses prompt selection and preserves image data.
  New focused tests pass all eight full/stream, text/image, plain/tool cases;
  the same cases fail against the original image (negative control container
  `dsv41-conversion-negative-control`). Full suite rerun passed.
- First API container preserved as `dsv41-api-pre-native-prompt-fix`; its image
  remains referenced. State snapshot:
  `/mnt/hot/dsv41_state/api-before-native-prompt-fix-20260915/`.
  Baseline and first API containers returned 247 following the explicitly
  requested stop (launcher reports its terminated server child's status), not
  an unprompted inference crash. Both had passed their smoke suite before stop.
- Corrected live API suite passed all seven requests: strict streamed function
  tools and stateless/result continuation on both endpoints; omitted-thinking
  signature replay; full-budget native Responses vision (1044 total input
  tokens). Evidence: `/mnt/hot/dsv41_state/acceptance/api-20260915T211959Z/`.
  Earlier `api-20260915T211914Z` also returned valid responses for all seven;
  its small 64px image correctly used fewer image tokens than the harness's
  mistaken full-budget assertion. The rerun used the recipe's 3024x588 size.
- Both live mid-decode disconnects drained the scheduler; probes plus health
  confirmation took about 2.01 seconds each. Evidence:
  `/mnt/hot/dsv41_state/acceptance/cancellation-20260915T212003Z/`.
  These are intentional client-aborted streams, not server failure cases.
- All three long Responses requests returned exactly `42`, with two output
  tokens and valid terminal streams. No prefix-cache hits were reported.
  Evidence: `/mnt/hot/dsv41_state/acceptance/long-20260915T212033Z/`.

  | Measured input tokens | Request wall time | Result |
  |---:|---:|---|
  | 32,757 | 5.37 s | Passed |
  | 131,061 | 20.42 s | Passed |
  | 199,989 | 32.08 s | Passed |

- Mixed qualification passed: a 66,313-token Messages tool request overlapped
  a Responses decode. Tool arguments and streams were valid; the decode reached
  its deliberately small output limit. Largest observed decode gap: **10.12 s**.
  Evidence: `/mnt/hot/dsv41_state/acceptance/mixed-20260915T212135Z/`.
- Corrected-image wire review: 11 Responses and 10 Messages requests, all HTTP
  200. Eighteen completed normally, one reached its requested output budget, and
  two were intentionally disconnected. The two analyzer integrity findings
  match the exact IDs of the cancellation probes, not unexplained truncation.
  Zero API error events, raw protocol leakage, detected output degeneration,
  HTTP/TCP parse gaps, or packet drops. Reports:
  `/mnt/hot/dsv41_state/api-candidate/wire-{responses,messages}-qualified.json`.
  Native logs contain no traceback, terminal OutOfMemoryError, parser failure,
  or deleted-TokenizerManager-state error in this corrected deployment.

## Original canary qualification caveats (historical profile)

- Long prefills produced **37 CUDA allocator OOM/retry warning lines** across
  the four ranks (9/10/9/9). They recovered and all requests completed, but this
  is memory pressure worth investigating; do not describe the run as OOM-warning
  free. Kernel/backend/allocator settings remain the pinned recipe's settings.
- Concurrent long prefill paused decode for 10.12 seconds. This run establishes
  bounded correctness, not production latency or eight-way long-request capacity.
- The configured context is 409,600; largest tested input here was 199,989.
- Only declared function tools are qualified. Newer custom/freeform Responses
  tools remain outside this port. No claim of full newer-client tool compatibility.
- Candidate is localhost-only, restart policy `no`, with production routes left
  unchanged. Old endpoints 8000 and 8001 are stopped. The original containers,
  images, models, durable thinking key and captures remain available for rollback.

## Remaining qualification work

1. Longer manual-workload capture review, including memory retries and
   long-prefill pauses. The user approved production promotion without the
   originally planned eight-hour canary; no unattended synthetic soak was run.
2. A dedicated rollback rehearsal remains unperformed. The promotion decision
   and subsequent restoration of the original context profile were explicitly
   approved by the user.

Baseline GPU smokes do not establish API-overlay or long-workload reliability.
No new upstream kernel stack,
publisher reasoning-tier update, RAM-offload optimization or custom/freeform
Responses API backport is included in this initial candidate.
