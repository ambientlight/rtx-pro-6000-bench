# Expandable-segments rollout — startup correctness failure

Historical record; see the [deployment README](../../README.md) for the supported configuration.

The requested 0.90 / expandable-segments relaunch was attempted with Starfield
closed, but **startup correctness failed before the 524k prefill test**. The
failed experiment was intentionally stopped. A second capped experiment also
failed, as recorded below. See `STATUS.md` for the subsequent allocator rollback;
this document describes failed historical deployments, not current readiness.

## Change and verification

The sole Compose serving change was
`PYTORCH_ALLOC_CONF=expandable_segments:True`, the canonical variable supported
by the pinned PyTorch headers. Image bytes, model, API/parser code, context
524288, memory fraction 0.90, prefill chunk 2048, concurrency 8, TP4/EP4, DSPARK
block 5 and RAM Engram offload were unchanged. Config tests assert the value,
release-lock agreement and absence of conflicting legacy/incorrect variables.
All **49 V4.1 tests + 10 capture tests** passed.

After a confirmed 20-second idle window, container `dsv41` was recreated at
**06:36:19 UTC September 16**, ID
`c29a6db984c18ccf88b580474a54c352bd2f5aac522d4a40676eb1fb1d1224cf`.
The game was absent. Startup KV capacity rose to **5329152** slots, compared with
1107712 in the previous successful no-game prefill (whose cache had been sized
while the game was still running). No explicit KV cap was set.

| Startup check | Result |
|---|---|
| Arithmetic | Passed: `42`, 2 output tokens |
| Structured JSON | Passed: `{ "answer": 42 }`, 8 output tokens |
| Tool request | Failed: 289 input tokens, 32768 output tokens, no tool call |
| Tool-result continuation and vision smoke | Not reached |
| Full-window prefill | **Not submitted** |

The tool request was the existing deterministic `lookup_fixture(key="alpha")`
smoke. Instead of calling it, the model generated unrelated prose and repeated
“The function is not pure because it has side effects.” **2967 times**, ending
at the 32768-token cap. Native finish reason was `length`; the Chat Completions
response contained no `tool_calls`. The boot wrapper's expected-tool assertion
failed at **06:42:54 UTC**. This is an observed generation/correctness failure,
not a recovered long-prefill run or a measured allocator improvement.

No allocator-retry, fatal CUDA OOM or unsupported-expandable-segments warning
was observed during this failed startup. The runtime reached HTTP health, but
never the boot `Ready` marker. The four-rank environment verification and
additional API acceptance checks, which are gated on boot correctness, were
not completed. Docker/container configuration does confirm the requested flag.

## Containment and uncertainty

Docker automatically restarted once at **06:43:17 UTC**. The failed deployment
was then intentionally stopped at **06:43:59 UTC** to prevent a repeat loop.
Restart policy remains `unless-stopped`, so it will stay stopped until explicitly
started. Nothing was deleted. The capture service remains enabled and waits for
the container; recorded evidence is preserved with zero kernel packet drops in
the failed-start capture.

Both allocator mode and the startup-sized KV pool differ from the last working
run. This observation alone does **not** prove expandable segments are the sole
cause, that the larger pool is the cause, or that the mode is unsupported. No
kernel, CUDA-graph, parser or startup-test bypass was applied. The next choice
was requested from the user: retry with the previous 1107712-slot KV cap to
isolate allocator mode, restore the previous allocator with that cap, or keep
the container stopped for investigation. No option was executed automatically.

## Capped retry: same correctness failure

The user approved an explicit **1107712-token KV cap**, matching the previous
working pool while retaining `expandable_segments:True` and fraction 0.90.
Container `c2536ffe573ed4bae9bed596d95f345ec2cb51544720f47d3bb92ce9a6262618`
started at **06:49:13 UTC**; startup logs and the load endpoint confirmed that
capacity. Arithmetic and structured JSON passed. The tool request again failed:
289 input tokens, 32768 output tokens, native finish reason `length`, no tool
call. This time output began `8262.0.0...` and contained **16383 `.0` repeats**.
It finished at **06:54:55 UTC** before the boot assertion.

No allocator retries, fatal OOM or unsupported-expandable warning was recorded.
Docker restarted once at 06:55:17 UTC; the container was intentionally stopped
at **06:55:39 UTC**. The full 524k prefill and additional API suite were **not
submitted**. Zero kernel packet drops were reported in the failed capture.

Evidence: `/mnt/hot/dsv41_state/expandable-090-kvcap-xOxT9HDJ/`, including
`startup-failure.json`, `request-finished.json` and `failed-tool-output.txt`.
Request ID: `eb83698ab83b485b964baee08e791f71`. Failed capture:
`/mnt/hot/dsv41_dumps/capture-20260916T064916Z-3197217`; automatic-restart capture:
`/mnt/hot/dsv41_dumps/capture-20260916T065524Z-3225440`.

Docker and bootstrap environment confirmed the allocator setting. Scheduler
`/proc/.../environ` did not expose it after process-title rewriting, which is
not evidence that the setting was disabled. A late, bounded memory-profile
request was reset during shutdown; no snapshot was produced. Direct worker
allocator-state verification was therefore not obtained.

Matching the old KV capacity did not resolve the failure. An allocator/runtime
interaction is suspected, but neither these tests nor absence of OOM establish
the exact mechanism. At user direction, the next comparison disables expandable
segments **without changing the 1107712-token cap**. No parser/kernel/graph change
or startup-check bypass is part of that comparison.

## Allocator-off control: passed

The next deployment preserved the 1107712-token cap, image and all other serving
configuration, changing only the allocator setting to
`PYTORCH_ALLOC_CONF=expandable_segments:False`. Startup passed at **07:07:16 UTC**.
The same 289-input-token tool request returned `lookup_fixture({"key":"alpha"})`
in **49 output tokens**, followed by a correct tool-result continuation. All
other startup checks also passed.

The subsequent **524177-token uncached prefill passed**, as did seven API
acceptance requests before it and seven afterward. No restart or fatal/API error
occurred. There were **208 recoverable allocator-retry warning lines** during
the full prefill; disabling expandable segments restores correctness here, not
memory-pressure-free operation. See the
[retest report](../../../../bench/deepseek-v4.1-flash_TP4_sglang/BENCHMARK-FULL-PREFILL-V2-NONEXPANDABLE-2026-09-16.md).

This controlled configuration comparison strongly implicates an allocator-mode
interaction in the pinned runtime. It does not establish its internal mechanism
or mean expandable segments are generally broken in PyTorch/SGLang. The failed
expandable attempts never reached full prefill, so no performance comparison
between allocator modes is available.

## First attempt: evidence and rollback

- Private rollout: `/mnt/hot/dsv41_state/expandable-090-YXkuqtGz/`.
- `startup-failure.json`, `request-received.json`, `request-finished.json` and
  `failed-tool-output.txt` preserve the failure.
- `rollback-compose.yaml` preserves the prior allocator configuration. It has
  no explicit KV cap; launching it with the game absent can size a different
  cache than the previous successful run.
- Original state/configuration: `before/`, including private credential-bearing
  Docker/state snapshots. Do not publish these files.
- Failed request ID: `3b286a9c6c1441d19a770282cb870b2e`.
- Failed startup capture: `/mnt/hot/dsv41_dumps/capture-20260916T063624Z-3176497`.
- Auto-restart capture: `/mnt/hot/dsv41_dumps/capture-20260916T064321Z-3193735`.
- The saved `run_prefill.py` was prepared but **never executed**.
