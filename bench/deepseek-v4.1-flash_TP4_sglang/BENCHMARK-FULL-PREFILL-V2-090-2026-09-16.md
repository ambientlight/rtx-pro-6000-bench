# Full-window prefill, v2 / 0.90 — 2026-09-16 UTC

**Failed with a fatal CUDA OOM on GPU 1.** The server could start and pass small
API checks, but the current shared-GPU configuration did not complete this
near-524288-token prefill. No tuning change or second probe was made.

A later user-requested [exact replay after closing the game](BENCHMARK-FULL-PREFILL-V2-NO-GAME-2026-09-16.md)
passed, with 210 recoverable retry warnings. The failure below remains the
record of the original game-running test.

## Measurement

| Measurement | Result |
|---|---:|
| Configured context / memory fraction | 524288 / 0.90 |
| Tokenized input | 524177 (99.98% of the window) |
| Output budget / emitted output | 64 / 0 tokens |
| Inference start | 06:04:34 UTC |
| Fatal OOM | 06:06:08 UTC, about 94 seconds after submission |
| Last scheduled token progress | 387072; 137105 tokens still pending |
| CUDA allocation-retry warning lines | 173 |
| Warning lines, GPU 0 / 1 / 2 / 3 | 27 / 92 / 27 / 27 |
| Fatal allocation request | 1024 MiB on GPU 1 |
| Device free memory reported at fatal OOM | 808.94 MiB |
| Sampled peak GPU 1 memory | 97066 MiB |
| Other non-health inference requests observed | 0 |
| Capture packet drops added | 0 |
| Successful prefill TPS | **Unavailable: prefill did not complete** |

The final scheduler line describes scheduled/in-flight work, not proof that
its last chunk completed. Warning counts are per-rank allocator log lines, not
173 failed requests. GPU memory was sampled every five seconds; seven observer
timeouts after the scheduler failure leave gaps in the measurements.

## What failed

The traceback runs through the V4.1 dense FP4 attention indexer into
`select_candidate_blocks()` in
`python/sglang/srt/layers/attention/dsv4/indexer.py:1142`:

```python
scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
```

This attempted a **1 GiB temporary score-tensor allocation**. At failure, the
exception reported 86.33 GiB used by the inference process, 82.51 GiB allocated
by PyTorch and 2.79 GiB reserved but unallocated; only 808.94 MiB was device-free.
Earlier failed allocations reached 3204448256 bytes (~2.98 GiB). The exact
contribution of fragmentation was not isolated.

This was not a context-length rejection, tool-parser failure or exhaustion of
the nominal 1059840 KV slots: the last scheduler line reported full-token usage
0.37. Long-context attention requires temporary buffers in addition to weights
and the preallocated cache. Increasing the static memory fraction to make
startup KV allocation viable did not establish enough usable runtime headroom.

Starfield was running on GPU 1 throughout the setup (6345 MiB immediately before
submission, with usage varying during the session). The game was not stopped
or modified. Its memory use reduces available headroom, but this test does not
isolate it as the only contributor or prove that stopping it alone fixes 524k.
Earlier successful v1 prefill is not a controlled fraction comparison: external
GPU occupancy and startup-sized KV capacity differed.

## Client outcome and recovery

The Responses endpoint returned HTTP 200 and `response.created` /
`response.in_progress`, but **no generated text and no terminal SSE event**.
The client connection ended after 159.52 seconds; the harness correctly rejected
the missing completion. There is no final API/native token usage record from
which to report successful prefill or decode throughput.

Docker automatically restarted the same container at **06:07:36 UTC**, with
unchanged configuration. It was ready at **06:11:56 UTC**, with all built-in
startup checks passed, and verified healthy/idle at **06:12:44 UTC**. Capture
followed the restart, with native logging active and zero drops. KV capacity
resized from 1059840 to **1107712** slots against current free GPU memory;
fraction 0.90, context 524288 and effective maximum input 524282 are unchanged.
This recovery does not requalify the failed full-window test. Do not interpret
a cached Docker health label during the crash interval as proof the scheduler
was responsive. `recovery.json` retains the verified post-restart state.

## Procedure and evidence

Exactly one bounded synthetic inference request was submitted after a stable
20-second idle window. Tokenization calibrated the repeated-reference prompt
before submission. The prompt had a unique prefix, requested the answer `42`,
disabled thinking, used temperature 0 and streamed `/v1/responses`. No cache
flush, ignored EOS, serving reconfiguration, manual restart or repeat was used.
Automatic health checks and restart-time built-in smokes are separate from the
single synthetic probe. This was a memory-capacity test, not semantic retrieval,
long-output or concurrent-agent qualification.

- Evidence: `/mnt/hot/dsv41_state/prefill-524k-v2-090-VbCEm8U8/`.
- `run_probe.py` sends the request and refuses to rerun after it starts.
- `analyze_failure.py` reanalyzes saved evidence without sending inference.
- `result.json`, `fatal-trace.log`, `allocation-retries.json`,
  `prefill-progress.log`, `timeline.jsonl`, `observer.jsonl` and `response.sse`
  retain the failure evidence.
- Response ID: `resp_58236df672724adcad701cad8ce9dc12`.
- Failed-run capture: `/mnt/hot/dsv41_dumps/capture-20260916T054637Z-3042052`.
- Automatic-restart capture: `/mnt/hot/dsv41_dumps/capture-20260916T060736Z-3122440`.
