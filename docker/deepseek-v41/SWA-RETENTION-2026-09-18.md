# SWA host retention: investigation, patch and qualification

## Executive summary

- The older replay misses are consistent with a **component-retention mismatch**, not a TTL expiry: FULL KV survived in RAM, while eviction could discard its required SWA resumption window. The real Python cache-tree regression reproduces this miss with legacy eviction and retains all 64 prefixes with the new policy.
- Added an opt-in, bounded **request-boundary-first host SWA eviction policy**. Intermediate prefill checkpoints are reclaimed before completed-request windows. Retained windows remain evictable; there is no extra RAM pool, permanent pin or relaxed cache-match validation.
- Selectively backported current upstream SWA incremental backup and split/host-lock bookkeeping. Did **not** transplant the old, closed/unmerged trim-on-eviction proposal or upgrade the hardware-qualified SM120 stack wholesale.
- CPU qualification passed: **1,202 tests + 377 subtests**, **87 deployment/helper tests**, and the **16 dedicated SWA tests against the built image**. Live rollout/qualification results are recorded below.
- Live qualification passed: sixteen 256k cold requests, the oldest 256k replay,
  and four header-recall replays across Responses/Messages. The long replay
  reused **255,744 tokens (99.9%) in 1.54 s TTFT**, with real FULL+SWA RAM
  restoration. There were no boundary-window evictions or unexpected restarts.

## Scope and method

Research and local inspection: September 18, 2026 UTC (September 17 local).
Reviewed the deployed container, the production overlay and the prior 64 × 256k
capture, then upstream PR discussions, merge records, patches and current source.
Primary upstream records all belong to SGLang; they are not independent
corroboration of one another. The local pressure regression is a separate
experiment, but is not a CUDA correctness or model-accuracy test.

Scope: Python UnifiedTreeCore, FULL+SWA, RAM-only HiCache in `cache`/
`write_through` mode. No API format, sampling, attention kernel, Engram,
allocator, context, static memory fraction or budget change. Rust, L3/NVMe,
cross-process persistence and the complete branching-point scheduler feature
are outside this patch's deployment qualification.

## Detailed findings

### What failed locally

The [previous live experiment](CACHE-PRESSURE-64X256K-2026-09-18.md) completed
64 unique 256,000-token requests and eight exact replays correctly, but seven
older replays had zero usable cached tokens. The newest replay reused 255,744
tokens via SWA-only RAM restore; FULL KV was still on GPU. At completion,
FULL host occupancy was only 16,381,952 / 32,699,648 tokens (50.10%).

The captured per-request SWA backups were 24,064 slots/rank (94 pages).
The host pool has only 254,976 SWA slots/rank: approximately **10.6 such
checkpoint footprints**, versus approximately 128 complete 256k FULL prefixes.
These are token-slot capacities, not physical byte-transfer estimates.
Exact replay does not help if the corresponding SWA page was evicted.

This is not proof of every historical node-level eviction: the earlier capture
did not record the host-SWA LRU contents. It is a supported mechanism, confirmed
by source inspection and reproduced in a bounded test using the actual tree
implementation. Genuine prefix differences still cause legitimate cache misses.

### Upstream status and integration choices

| Work | Status at review | Relevance and action |
|---|---|---|
| [#26907: retain SWA suffix during eviction](https://github.com/sgl-project/sglang/pull/26907) | Closed, **not merged**, July 13 | Discussion moved away from increasingly complex trim-on-evict splitting toward insertion-time checkpoints. Not cherry-picked. Our policy works with existing nodes and changes host-victim preference only. |
| [#27009: alternative SWA keep-window work](https://github.com/sgl-project/sglang/pull/27009) | Closed, not merged | Not treated as an available upstream fix. |
| [#27210: cap SWA compute-lock scope](https://github.com/sgl-project/sglang/pull/27210) | Open, unmerged | Not transplanted wholesale. No change to request compute-lock span in this patch. |
| [#27444: pin host buffers through async H→D](https://github.com/sgl-project/sglang/pull/27444) | Merged June 13 | Its essential asynchronous host-pinning behavior was already in the deployed source. Preserved. |
| [#34565: SWA branching-point caching](https://github.com/sgl-project/sglang/pull/34565) | Merged September 5 | Our pinned component still returned false for incremental backup. Selected current window-scoped backup plumbing, not the complete branch scheduler change. |
| [#37584: Rust branch-caching follow-up](https://github.com/sgl-project/sglang/pull/37584) | Merged September 12 | Current Python collector includes node-local buffer-mode behavior. Preserved that distinction; no Rust feature enabled. |
| [#38138: SWA host lock on split](https://github.com/sgl-project/sglang/pull/38138) and [#38835: host-lock boundary fix](https://github.com/sgl-project/sglang/pull/38835) | Merged September 7 / 11 | Carried forward host lock counts and a **single** host UUID-boundary migration; locked split fragments must not enter the host eviction LRU. |
| [#38482: auxiliary LRU recency on split](https://github.com/sgl-project/sglang/pull/38482) | Merged September 12 | Not included in this bounded backport. The retention policy introduces no new splits. |
| [#38926: V4.1 HiCache L2/L3 tests](https://github.com/sgl-project/sglang/pull/38926) | Open, unmerged | Relevant qualification direction, but its GB300 results do not establish correctness on this SM120 deployment. |

The current [upstream SWA component](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/swa.py)
collects missing device-to-host SWA copies within a sliding window even when
FULL KV is already backed up. The local implementation follows this contract.
It deliberately does not copy the additional replay-lookbehind page in the
same asynchronous transfer: the existing device transfer lock covers only one
window. The old [SWA reuse tracking issue #26577](https://github.com/sgl-project/sglang/issues/26577)
also discusses FULL/SWA capacity imbalance; its closure is not evidence that
our deployed host retention behavior was fixed.

### Local policy and safeguards

Runtime flag: `SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES=1`.
Default in the source remains off; the candidate explicitly selects Python
TreeCore and enables the flag.

On a non-chunked insertion, mark the existing nodes covering one sliding
window plus one page at the request boundary. With window 128 and page size
256, this normally covers two pages. The extra page accommodates exact replay's
held-back final token. Markers are a soft preference, not a guarantee.

Under SWA host pressure:

1. Reclaim ordinary intermediate checkpoints first.
2. Reclaim boundary windows if still necessary.
3. Preserve existing session priority, asynchronous IO locks, atomic host-leaf
   eviction, and cascading cleanup throughout.

No node splits or device allocations occur in the new eviction pass. Markers
are conservatively copied when existing operations split a node; coarse
existing nodes may retain more than the minimum until fallback eviction.
The host budget remains **256,000,000,000 bytes total across TP4**, not 256 GiB.

New Prometheus metrics, per scheduler/rank:

| Metric suffix (`sglang:` prefix) | Meaning |
|---|---|
| `hicache_swa_host_used_tokens` | Current allocated SWA host slots |
| `hicache_swa_host_total_tokens` | Fixed SWA host-slot capacity |
| `hicache_swa_host_evicted_tokens` | Cumulative reclaimed SWA host slots since cache construction |
| `hicache_swa_host_boundary_evicted_tokens` | Cumulative reclaimed marked boundary slots, including FULL-driven eviction |
| `hicache_swa_host_retention_enabled` | Whether the policy is active |

The eviction values are snapshot gauges, not Prometheus Counters. For logical
slot capacity use TP0, not the sum across identical TP shards. Existing
`load_back_tokens_total{pool="kv"}` and `{pool="swa"}` distinguish FULL from
SWA restoration; these counters aggregate ranks in the current export.
The earlier paged-pool byte-accounting defect remains separate: do not infer
physical traffic volumes from raw backup/load-back byte counters.

### Qualification and packaging

Dedicated CPU tests exercise the deployed Python tree, FULL and SWA components,
bounded host allocation and payload-index alignment. Coverage includes:

- 64 × 256k tree footprints with 24,064 SWA slots each and 254,976 host slots;
  every FULL device prefix is evicted, then all 64 exact replays retain a
  255,744-token host hit with correct transfer indices.
- Legacy eviction reproducing an old-prefix miss despite surviving FULL KV.
- Real stepped insertion, prefill versus finished-request boundaries,
  divergent branches, incremental repair without a redundant FULL copy.
- Retention fallback under exhausted capacity, session priority, pending-copy
  and missing-window barriers, host locks through splitting and release.
- Buffer-only compatibility, larger multi-page windows, unsupported backend
  rejection, bounded transfer-lock coverage, real Prometheus export.

Full regression evidence:
`/mnt/hot/dsv41_state/api-tests-pUlmvL5h/` — 1,202 tests and 377 subtests passed.
An earlier test attempt used the production image's missing mounted signature
key and an incomplete metrics fixture; neither was a production failure. The
normal regression baseline and corrected fixture passed. The built image's
own installed cache modules separately passed all 16 retention tests.

Candidate: `ambientlight/dsv41-sm120:swa-retention-v1`.
Image ID: `sha256:2d811ff02b68885294bfb8da4a75ec357ccb62bc7d95ff83c690b5575f40a5cc`.
Immutable parent: `sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca`.
Frozen source/test snapshots, hashes and unchanged-runtime verification:
`/mnt/hot/dsv41_state/swa-retention-image-8yswbpcd/`.

The candidate packages exactly five runtime files. Existing boot and host-budget
read-only mounts remain unchanged. API parsers/serving, scheduler main loop,
tokenizer manager, SM120 attention kernel and Engram adapter hashes match the
parent. Existing unrelated worktree edits were neither committed nor discarded.
The normal release builder retains its clean/committed-source requirement; the
separate candidate builder explicitly records its dirty-source snapshot.

### Live rollout and results

Approved rollout evidence:
`/mnt/hot/dsv41_state/swa-retention-rollout-grf55j8h/`.
The rollback profile and old container identity were preserved before restart.
Container `6f3e836ee3f8` started at **05:13:03 UTC**, with built-in arithmetic,
structured output, tool round-trip and vision checks passing at **05:19:09 UTC**.
Seven additional live API requests passed: strict streaming tool/result flows
on Responses and Messages, opaque-thinking signature replay and native image
input. Capture followed to
`/mnt/hot/dsv41_dumps/capture-20260918T051306Z-956043`.

| Current measured capacity / configuration | Value |
|---|---:|
| Host-total HiCache budget | 256,000,000,000 bytes (unchanged) |
| Actual initial host buffers / KV payload | 244,417,175,040 / 239,173,598,720 bytes |
| FULL host capacity | 32,721,664 logical tokens |
| SWA host capacity | 253,440 slots/rank (990 pages) |
| FULL device capacity | 3,367,936 logical tokens |
| SWA device capacity | 26,112 slots/rank |
| Context / static fraction / prefill interval | 524288 / 0.87 / 4 |
| TTL | Disabled (`null`) |

Small capacity differences from the previous startup come from unchanged
automatic GPU sizing and proportional host-pool sizing, not a larger budget.
Persistent Compose exactly matches the deployed resolved profile. Only the
candidate image, retention environment and version label differ from rollback.

GPU qualification completed by **05:37:43 UTC**. The pressure run processed
4,352,000 input tokens and 51 output tokens in 17 correct Responses requests,
with no unrelated HTTP inference admitted during those requests. Cold mean
TTFT was 44.206 s. Its sixteen independent cold prompts exceed the 3,367,936
GPU FULL-token capacity and the host-SWA checkpoint capacity.

| Live check | Result |
|---|---|
| 16 unique 256,000-token cold prompts | 16/16 correct, zero cached input as expected |
| Oldest exact 256k replay after all 16 | Correct; 255,744 cached tokens, 1.540 s TTFT |
| Long replay FULL / SWA restoration | 1,022,976 / 1,024 slots across TP4: 255,744 FULL + 256 SWA slots per rank |
| Four header-recall replays, two per API | 4/4 correct; each reused 7,168 tokens from RAM |
| Each header probe's FULL / SWA restore | 28,672 / 1,024 slots across TP4: 7,168 FULL + 256 SWA per rank |
| SWA host occupancy after pressure | 253,440 / 253,440 slots/rank |
| Intermediate SWA slots reclaimed | 139,776 slots/rank |
| Retained boundary slots reclaimed | **0** |
| Minimum available RAM sampled during pressure | 39.232 GiB |
| API failures / fatal scheduler-CUDA-NCCL errors / unexpected restarts | **0 / 0 / 0** |

For comparison, the earlier same-shape 16-prompt experiment's oldest replay
had **zero** cached tokens and **43.391 s** TTFT. The random prompt seeds and
small auto-sized capacities differ between runs, so this is a targeted
behavioral before/after comparison, not a controlled general throughput claim.

Header probes place a random six-digit answer near the beginning, outside the
uncached tail. They check both content and independent FULL/SWA restoration
counters. Responses exposed cached tokens in API usage. The Messages probes
did not expose a cache-read field in their usage events; their 7,168-token RAM
reuse is established by scheduler and per-pool transfer counters, not inferred
from that absent API field. No API usage semantics were changed.

One initial header-probe pass stopped on unsettled HTTP-finalization metrics
after both answers were correct. The harness was changed to wait for settled
counters, as the existing cache acceptance already does. Fresh probes and all
four later exact replays passed uncontended attribution. This was a probe
measurement failure, not a serving failure.

There were **86 CUDA allocation-failure debug log lines** during qualification;
the allocator recovered, requests completed and no fatal error occurred.
These lines are not 86 failed requests or necessarily 86 distinct retries.
The patch does not remove the pre-existing transient allocation-pressure issue.

Evidence under the rollout directory:

- `pressure/results.json`, `pressure/status.json`, per-request streams, metrics
  and exact prompt/request artifacts;
- `anchors-settled/replay-20260918T053618Z/probe-results.json` and
  `anchors/replay-20260918T053622Z/probe-results.json`;
- `acceptance/api-20260918T051947Z/`;
- `qualification.json` for a compact outcome summary.

The endpoint was healthy and fully idle at final inspection. Capture heartbeat
was fresh, packet drops zero and native logging active. Its six logging-refresh
failures occurred during startup unavailability, not model inference.
Separate operational warning: `/` had only about **320 MB free**; the image,
new artifacts and captures are on `/mnt/hot`. No unrelated files were deleted.

Rollback, only after draining active and queued work, uses the saved
`rollback.compose.json` in the rollout directory with `docker compose ... up
-d --no-build --pull never --force-recreate deepseek`. Also restore the
repository Compose/release identity before future launches; otherwise a later
Compose recreation would select the new image again. The old image and
rollback configuration remain available. Restart necessarily discards the
RAM/GPU KV caches; model weights and captures are preserved.

## Evidence map

| Major claim | Best direct evidence | Independent corroboration | Assessment |
|---|---|---|---|
| Older requests lost usable reuse while FULL host KV remained | Prior 64 × 256k capture/report; exact input hashes and cached-token records | Real-tree legacy-policy regression reproduces the mechanism | High confidence in the observed failure; node-by-node historical causality remains an inference |
| Relevant upstream changes were not all in the pinned deployment | Local source/image hash comparison; linked PR patches and merge records | None outside the project; merge status is authoritative GitHub metadata | High confidence in the specific code/status differences |
| Boundary preference preserves reusable windows under the tested pressure | 16 dedicated tests, including all 64 host-only FULL prefix replays | Live 16 × 256k GPU-pressure run and five correct RAM-backed replays | High confidence for these workloads; not a universal retention guarantee |
| API and hardware stack were preserved | Parent/candidate file hashes and 1,202 regressions | Seven live API acceptance calls and successful replay on both APIs | High confidence for scope and tested contracts; not exhaustive runtime equivalence |

## Disagreements, uncertainties and limitations

- The older trim-on-eviction PR reports gains, but maintainers raised complexity
  concerns and it was not merged. Those benchmarks do not justify treating it
  as a proven drop-in patch for this deployment.
- This favors completed-request reuse, not every arbitrary internal branch
  point. A prompt diverging before a retained window can still miss; tokens
  from different prompts cannot safely share mismatched KV.
- No TTL or durability guarantee is introduced. Capacity pressure, FULL-host
  eviction, explicit cache flush and restart can remove retained prefixes.
- Insertion at coarse existing nodes can retain more SWA than two pages. The
  fallback pass bounds memory use but cannot promise a fixed number of reusable
  conversations for every workload.
- CPU index/payload tests do not exercise native CUDA transfer kernels, model
  logits, long concurrent lifetimes or every SWA+Mamba/Rust combination. Live
  tests now demonstrate correct targeted GPU/RAM restoration, but not numerical
  logit/KL equivalence, a full 64-prompt GPU replay sweep, long outputs or a
  sustained concurrent soak.

## Recommendations

1. **Deploy only the reviewed layer after idle**, preserving the memory budget
   and API/kernel stack. This isolates retention behavior from tuning changes.
   Roll back if startup, stream contract or restore-correctness checks fail.
2. **Require actual FULL and SWA host restores in qualification.** Age independent
   prompts beyond GPU capacity, replay exact requests, and check correctness,
   cached tokens, TTFT and per-pool load counters. A recent SWA-only hit is not
   enough to establish two-tier reuse.
3. **Track boundary evictions separately under the real workload.** If that
   gauge grows while FULL RAM is far below capacity and useful replays miss,
   inspect marker/window granularity and pool proportions before allocating
   more RAM. Retention is a preference; a full boundary-only pool must evict.

## Sources

Primary sources are linked at their claims: SGLang PRs #26907, #27009, #27210,
#27444, #34565, #37584, #38138, #38835, #38482 and #38926; issue #26577; current
SWA component source. These share project lineage and are not counted as
independent confirmations. Local independent evidence consists of the original
captured experiment, source/image inspection and the new executable tests.
