# DeepSeek V4.1 Flash: bounded long-call TPS baseline

Measured on 2026-09-15, 22:16–22:24 UTC. Three sequential, no-tool,
streaming `/v1/responses` requests on the existing canary, with no deployment
changes. This is a single-request performance baseline, not a concurrent-agent
benchmark or a production reliability estimate.

## Results

| Actual input tokens | Actual output tokens | First output (s) | Decode tokens/s | Total request (s) | Effective prefill tokens/s | Termination |
|---:|---:|---:|---:|---:|---:|---|
| 32,788 | 6,091 | 15.55 | 73.07 | 98.92 | 2,109 | Natural EOS |
| 131,388 | 8,190 | 39.62 | 73.53 | 151.09 | 3,317 | Output budget |
| 200,304 | 8,190 | 46.17 | 83.29 | 144.65 | 4,338 | Output budget |

Totals: **364,480 input tokens, 22,471 output tokens**, and 394.66 seconds of
client request time. Decode-time-weighted throughput was **76.68 tokens/s**.
Decode speed did not collapse at the larger contexts in this small sample.

- All three streams passed event-sequence and assembled-text consistency checks.
  API usage matched native completion counts and generated token-ID lengths.
- No prefix-cache hits, overlapping inference requests, request retractions,
  new allocator OOM/retry warning lines, terminal OOMs, tracebacks, or
  deleted-TokenizerManager-state warnings were observed during these requests.
- No raw tool markers, Han characters, replacement characters, or exact
  duplicate long lines were found in the outputs. Maximum repeated-character
  run was two. These are heuristic checks, not semantic correctness evaluation.
- Maximum gaps between nonempty output bursts were 65, 148, and 218 ms,
  respectively. These are **burst gaps**, not individual-token latencies.
- The two budget stops emitted `response.incomplete` with
  `reason=max_output_tokens`, as expected. Each requested 8,192 output tokens;
  the current Responses adapter reserved two framing tokens, leaving 8,190.
  The first request voluntarily ended earlier; EOS was not suppressed.
- At the 22:25 UTC post-check, health returned HTTP 200, active/queued requests
  were zero, the container was healthy with zero restarts, and packet capture
  remained active with zero cumulative kernel drops. V12 and embedding
  containers remained stopped. Earlier qualification's 37 recovered allocator
  warnings are still a separate deployment caveat; this benchmark does not
  erase them.

## Configuration and interpretation

Image: `ambientlight/dsv41-sm120:sero-v12-api-candidate`, immutable ID
`sha256:49117bf6688f1ef620bf8cd98bb8a3280427fe29c91972c229718450123f047b`.
Source overlay: `da53ffcef5fa7c479da2e673dfbb408a3e660237`.
Model revision: `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

- Four RTX PRO 6000 Blackwell Max-Q GPUs, 250 W power limits; TP4 / EP4.
- MXFP4 MoE, FP8 KV cache, NVMe Engram offload with 64 GiB host cache.
- DSPARK block size 5, bounded decoder SWA replay, chunked prefill 2,048,
  context capacity 409,600, static memory fraction 0.85.
- Concurrency **1**. Thinking **off**, temperature **1.0**, top-p **0.95**.
- Synthetic, varied reference notes followed by a substantial technical
  handbook request. No counting-only task. Unique prefixes prevented KV prefix
  reuse. Hardware, process, compilation, and Engram caches were not flushed.

Decode TPS uses SGLang's exact `(completion_tokens - 1) / decode_latency`
metadata, not SSE event counts. DSPARK can emit several tokens in one event.
Client first-output latency starts before sending the HTTP request and ends
at the first nonempty model-text delta; `response.created` is not a token.
Client total time ends at the terminal SSE event. Effective prefill TPS is
uncached input divided by client first-output latency; it includes HTTP,
tokenization, queueing, prefill, and first-token overhead and is not pure GPU
prefill throughput.

Mean speculative accept lengths were **2.484, 2.483, and 2.782** tokens per
verification step; draft acceptance rates were **29.68%, 29.67%, and 35.63%**.
The third request's higher TPS coincides with higher speculation acceptance;
it does not establish that longer contexts are faster. Different generated
text, cache warming, and run order can affect these one-shot measurements.
The earlier deterministic counting demonstration (~160–172 tokens/s after
prefill, near-six-token accept lengths) is not an apples-to-apples comparison.

This test did not exercise thinking-enabled output, tool calls, concurrency,
32k-token generation, or the full 409,600-token context capacity.

## Reproduce and preserve

From the benchmark repository, with the canary idle:

```bash
python3 scripts/dsv41_long_tps.py
python3 -m unittest discover -s scripts -p 'test_dsv41_long_tps.py' -v
```

After promotion, the script's defaults target `dsv41` on port 8000 with public
model `deepseek-v4-flash` and no key. The results above remain the original
409600-context, port-8010 canary measurements; a rerun on production must be
reported with its new image/configuration. For a separately restored canary,
pass its explicit base URL, model name, key file, container and context limit.

The script sends at most three sequential requests by default and stops adding
traffic if another inference workload is detected. It does not restart the
server, change settings, or flush caches. Eleven harness tests pass, including
multi-token burst accounting, budget termination, and Docker log normalization.

Private evidence directories (requests, raw SSE, timelines, outputs, GPU/load
observations, native records, and JSON metrics; no auth headers in harness files):

- `/mnt/hot/dsv41_state/benchmarks/long-tps-20260915T221657Z/` — 32k probe.
- `/mnt/hot/dsv41_state/benchmarks/long-tps-20260915T221935Z/` — 131k and 200k probes.
- Wire/native capture: `/mnt/hot/dsv41_dumps/capture-20260915T211647Z-967430/`.
  Unlike the harness files, raw PCAP includes authentication headers and must
  remain private.

The first request completed successfully, but initial offline metric extraction
failed because `docker logs --timestamps` inserts transport prefixes inside
large logical JSON lines. The reader was corrected and regression-tested;
the existing captured response was reprocessed **without resending inference**.
The first directory retains its original harness failure in `completion.json`
and its successful recovery receipt in `analysis.json`. Reprocessing is available:

```bash
python3 scripts/dsv41_long_tps.py --analyze-existing /mnt/hot/dsv41_state/benchmarks/long-tps-20260915T221657Z
```
