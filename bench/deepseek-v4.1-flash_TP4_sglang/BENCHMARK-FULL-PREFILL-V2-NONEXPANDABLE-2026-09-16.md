# Non-expandable allocator restoration and full-window retest — 2026-09-16 UTC

**Passed.** Disabling expandable segments restored startup tool correctness with
the same 1107712-token KV cap that failed in the preceding expandable experiment.
The exact saved 524177-token request then completed correctly. The deployment
remains healthy, but the test still generated **208 recoverable CUDA allocation-
retry warning lines**. No memory-pressure-free or concurrency claim is made.

## Controlled configuration

Only `PYTORCH_ALLOC_CONF` changed from `expandable_segments:True` to
`expandable_segments:False` relative to the capped failed attempt. A normalized
Compose comparison enforced that restriction before recreation. Unchanged:

- Memory fraction **0.90**, explicit KV cap **1107712**, context **524288**.
- TP4/EP4, prefill chunk 2048, concurrency 8, DSPARK block 5, FP8 KV.
- RAM Engram offload, `DSV41_CACHE_GIB=0`, and all image/model/kernel/API bytes.
- Container `dsv41`, port 8000, model alias `deepseek-v4-flash`, no backend key,
  and `unless-stopped` restart policy. Old V12/embedding containers stay stopped.

The previously failed container was already intentionally stopped, so no live
queue was interrupted. It and production-state metadata were preserved privately
before recreation at **07:03:46 UTC**. Starfield was absent. The effective KV pool
was verified at **1107712**, rather than allowing a no-game restart to enlarge it.
Allocator configuration was verified in Docker and the bootstrap environment;
this is not direct worker allocator-state introspection.

## Correctness and comparison

| Measurement | Previous successful no-game run | Fresh non-expandable/capped run |
|---|---:|---:|
| Allocator | Default, expandable off | Explicit expandable off |
| KV pool capacity | 1107712, sized on earlier restart with game running | 1107712, explicit cap |
| Input / cached tokens | 524177 / 0 | 524177 / 0 |
| Answer / output tokens | `42` / 2 | `42` / 2 |
| Server prefill | 100.2216 s | **100.2432 s** |
| Uncached prefill rate | 5230.18 tokens/s | **5229.05 tokens/s** |
| Server end-to-end | 101.0281 s | 101.0003 s |
| Client first text | 103.2933 s | 103.0787 s |
| Client stream wall time | 103.8062 s | **103.5697 s** |
| Recoverable allocation-retry lines | 210 | **208** |
| Retry lines: GPU 0 / 1 / 2 / 3 | 51 / 57 / 51 / 51 | 50 / 58 / 50 / 50 |
| Fatal/API errors / new restarts | 0 / 0 | **0 / 0** |
| Other non-health inference during prefill | 0 | 0 |

All **49 V4.1 configuration/harness tests and 10 capture tests** passed. The boot
`Ready` marker appeared at **07:07:16 UTC**, after arithmetic, structured JSON,
tool call/result continuation and full-budget vision checks passed. The same
289-token startup tool prompt that degenerated to 32768 tokens in both expandable
attempts now emitted the correct tool call in **49 tokens**; its continuation
answered 42 correctly in 12 tokens.

Seven live API requests passed both before and after the long prefill (**14
total**): strict Responses/Messages tool streams and continuations, omitted-
thinking signature replay, and Responses vision. No parser recovery or startup
check bypass was added to make this experiment pass.

The single long request started at **07:08:42 UTC** and validation finished at
**07:10:27 UTC**, following 20 continuous seconds of idle scheduler state.
The payload is byte-for-byte identical to the previous successful test:
SHA-256 `ff7319cb893282458cf77fdcef75ca5aebe0b8a5145708fe616bdab0cdd467df`.
It uses temperature 0, thinking disabled, no tools, a 64-token output budget,
streaming Responses, and no cache flush. API/native usage agrees; terminal
status is `completed`, stream order and assembled output validate, and there
are zero retractions, observer errors, packet drops or identity changes.

## Assessment and limits

With the KV cap fixed, disabling expandable segments changed startup tool output
from a repeat loop to a correct call. This strongly implicates an interaction
with allocator mode in this pinned runtime, but does not diagnose its underlying
kernel/graph/allocator mechanism. The two failed expandable startups never reached
the long prefill, so they provide **no full-prefill performance comparison**.

Compared with the earlier successful non-expandable run, throughput differs by
only -0.022% and retries by two lines (-0.95%): effectively unchanged, not a
meaningful memory-pressure improvement. Largest failed allocation before recovery
was **4261412864 bytes** (~3.97 GiB), and minimum device-free memory reported at a
retry was **22282240 bytes** (21.25 MiB). Sampled GPU peaks were **97163 / 96895 /
97163 / 97163 MiB**; five-second samples may miss transient peaks. Warning counts
are per-rank log lines, not failed requests or an API failure rate.

This qualifies one isolated repeated-reference prompt with a two-token answer
and small API checks. It does not qualify semantic retrieval, long-output
endurance, eight simultaneous full-window calls, or coexistence with the game.
Server prefill timing includes scheduling and allocation overhead, not isolated
GPU kernel time. The two-token answer cannot measure sustained decode TPS.

Worker counters recorded **335872 read bytes** (0.3203 MiB) and zero major faults;
these include all worker reads, not only Engram. Host memory showed 189.127 GiB
mlocked. The existing Engram page-residency caveat is not resolved by this test.

The post-prefill API suite passed, and at **07:10:56 UTC** the backend was healthy
and idle with restart count **0**. Capture is active. No synthetic workload is
left running, and no image rebuild, kernel edit or additional configuration
change was performed.

## Evidence

- Private rollout root: `/mnt/hot/dsv41_state/nonexpandable-090-kvcap-HZT2NRcA/`.
- `before/` preserves the failed capped-expandable Compose/state and private
  Docker metadata. That configuration is a failure specimen, not a healthy rollback.
- `verification.json` records the fresh startup identity and effective settings.
- `prefill-524k/result.json`, `before.json`, native finish/warnings, observer,
  SSE timeline and exact request preserve the benchmark.
- API suites: `acceptance/api-20260916T070804Z/` and
  `acceptance-post-prefill/api-20260916T071045Z/`.
- Container: `e22196e4d44dadacce4b2f08caf32e49b1311d56e595165078cff7eeae5b6e74`.
- Image: `sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca`.
- Response: `resp_ff3dad9be4e24110be0e6dd126e33d55`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260916T070348Z-3227082`.
