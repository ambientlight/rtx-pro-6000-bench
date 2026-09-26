# Full-window prefill on the restored 0.85 profile — 2026-09-16 UTC

**Passed.** The user requested a new full half-native-context prefill after the
earlier no-load-test relaunch. Exactly one saved 524177-token request was replayed
against the existing 0.85 / automatic-KV deployment. It returned `42` correctly,
with **272 recoverable allocator-retry warning lines** and no fatal/API error or
restart. No serving setting, image, model, kernel or API code changed.

## Results and comparison

| Measurement | Prior 0.90 / capped pool | Current 0.85 / automatic pool |
|---|---:|---:|
| Context limit | 524288 | 524288 |
| Effective KV slots | 1107712 | **2259712** |
| Expandable segments | Disabled | Disabled |
| Input / cached tokens | 524177 / 0 | **524177 / 0** |
| Answer / output tokens | `42` / 2 | **`42` / 2** |
| Server prefill | 100.2432 s | **100.3928 s** |
| Uncached prefill rate | 5229.05 tokens/s | **5221.26 tokens/s** |
| Client first text | 103.0787 s | 103.4129 s |
| Client stream wall time | 103.5697 s | **103.9238 s** |
| Server end-to-end | 101.0003 s | 101.1860 s |
| Recoverable allocation-retry lines | 208 | **272** |
| Retry lines: GPU 0 / 1 / 2 / 3 | 50 / 58 / 50 / 50 | **65 / 77 / 65 / 65** |
| Fatal/API errors / new restarts | 0 / 0 | **0 / 0** |
| Retractions / packet drops | 0 / 0 | **0 / 0** |

Inference started at **09:10:38 UTC**; validation finished at **09:12:24 UTC**.
A 20-second continuously idle scheduler window preceded submission. No other
non-health inference overlapped the test, and Starfield was absent. Temperature
0, thinking disabled, streaming Responses, no tools, `store=false`, and a
64-token output cap match the earlier procedure. No cache flush was used.

The request bytes match the earlier test exactly: SHA-256
`ff7319cb893282458cf77fdcef75ca5aebe0b8a5145708fe616bdab0cdd467df`.
Actual tokenization is **524177**, fitting the 524288-token window with 64 output
tokens and 47 tokens spare. API and native usage agree, the terminal response
is `completed`, and stream ordering/assembled output checks pass.

## Interpretation

The measured prefill rate differs by only **-0.149%** from the preceding 0.90
test: effectively unchanged in this single-run comparison. Allocation warning
lines increased by **64 (+30.8%)**. Both the memory fraction and cache capacity
changed; the automatically sized pool is **2.04 times larger** than the earlier
explicitly capped pool. This does not isolate the effect of memory fraction.
The original v1 0.85 run had 270 warning lines, but a different 519989-token
input and 2272512-slot pool, so that is only a contextual reference.

Largest failed allocation before recovery was **4278190080 bytes** (4080 MiB,
~3.984 GiB); minimum device-free memory at a retry was **28573696 bytes**
(27.25 MiB). Five-second samples recorded GPU peaks of **97167 / 96322 / 97167 /
97167 MiB**, with zero observer errors. Samples can miss transient peaks. Warning
counts are per-rank log lines, not failed requests. This remains memory-pressure-
heavy operation, even though every required allocation ultimately succeeded.

Worker counters increased by **335872 read bytes** (0.3203 MiB) and zero major
faults. Those counters include all worker reads, not only Engram. The existing
RAM-offload residency caveat remains unresolved by this measurement.

This qualifies one isolated repeated-reference prompt with a two-token answer.
It does not establish semantic retrieval accuracy, sustained decode throughput,
long-output endurance, simultaneous full-window requests, or coexistence with
the game. Server prefill timing includes scheduling and allocator overhead.
No additional API acceptance or synthetic-soak workload was started.

## Deployment and evidence

The backend remained healthy and idle with restart count **0**. Image and
container identity, 0.85 fraction, no explicit KV cap, TP4/EP4, DSPARK block 5,
2048 prefill chunk, concurrency 8, RAM Engram offload, port 8000, model alias and
restart policy were unchanged. Capture stayed active with zero packet drops.

- Private root: `/mnt/hot/dsv41_state/prefill-524k-restored-085-ZmKGG3uU/`.
- `run_prefill.py` reuses the prior exact-payload harness and refuses a repeat
  into the same evidence folder. `environment/` and `reference/` preserve the
  profile, runtime receipt and original request.
- `prefill-524k/` contains `result.json`, before/after state, request, SSE timeline,
  native finish/warnings, observer and worker-I/O measurements.
- Container: `67f5f11c1a1b65dc8724745ef400e48529969861a1ac7e634566967780a4341d`.
- Image: `sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca`.
- Response: `resp_260acd6da7b94c8b84f2874713e9adb3`.
- Capture: `/mnt/hot/dsv41_dumps/capture-20260916T072715Z-3288697`.
- Prior comparison: [0.90 capped allocator-off test](BENCHMARK-FULL-PREFILL-V2-NONEXPANDABLE-2026-09-16.md).
