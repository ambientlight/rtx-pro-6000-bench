# SWE-bench Verified — DeepSeek-V4-Flash (MXFP4 W4A4, SM120) — v2 tool-calling

Agentic SWE-bench Verified of DeepSeek-V4-Flash served on 4× RTX PRO 6000
(SM120, TP=4) via the native MXFP4 W4A4 + HMMA sparse-kernel stack, driven by
mini-swe-agent **2.4.2 native tool-calling** over litellm/`hosted_vllm`. Serving
recipe:
[DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md](../../../docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md).

This is our **best result to date — 76.0%**, up +3.0pt from the
[2026-06-18 run (73.0%)](../deepseek-v4-flash-2026-06-18/). Same model/serving;
the gain comes from moving the agent to mini-swe-agent 2.4's native tool-calling
(the model's `<｜DSML｜…>` tool-call markup parsed server-side via
`--tool-call-parser deepseekv4`) in place of v1's XML action regex.

## Result

| Metric | Value |
|--------|-------|
| **Resolved / total** | **380 / 500 = 76.0%** |
| Submitted | 500 |
| Unresolved | 116 |
| Empty patch | 4 |
| Error | 0 |
| Output tokens | 11,828,887 |
| Wall-clock | 11h 22m |

## Files

- `scored.json` — the swebench harness report (resolved/unresolved/empty/error ids).
- `config.yaml` — mini-swe-agent run config.
- `trajectories/` — 500 per-instance `*.traj.json` agent trajectories + `preds.json`.

## Configuration

**Model / serving** (sglang fork, `sglang-single.yaml` + `launch-single.sh`; full
recipe in the [deploy doc](../../../docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md)):

| Setting | Value |
|---------|-------|
| Model | DeepSeek-V4-Flash (native MXFP4 W4A4 experts) |
| Hardware | 4× RTX PRO 6000 Blackwell (SM120), TP=4 |
| Context length | 1,048,576 (1M) |
| max-running-requests | 16 |
| chunked-prefill-size | 8192 |
| page-size | 256 |
| kv-cache-dtype | fp8_e4m3 |
| mem-fraction-static | 0.90 |
| cuda-graph batch sizes | 1, 2, 4, 8, 16 |
| Sparse attention | HMMA decode + prefill, split-KV indexer |
| Thinking | on (`SGLANG_DEFAULT_THINKING=1`) |
| Reasoning effort | max (`SGLANG_DSV4_REASONING_EFFORT=max`) |
| Reasoning parser | `deepseek-v4` (strips `<think>` from responses) |
| Tool-call parser | `deepseekv4` (parses `<｜DSML｜…>` markup → structured `tool_calls`) |

**Agent / sampling** (mini-swe-agent `v2.4.2_dev`, `config.yaml`):

| Setting | Value |
|---------|-------|
| Harness | mini-swe-agent 2.4.2 (native tool-calling) |
| Benchmark | SWE-bench Verified (500, test split) |
| Workers | 16 |
| temperature | 0.2 |
| top_p | 0.95 |
| frequency_penalty | 0.1 |
| max_tokens (output) | 65536 |
| step_limit | 250 |
| cost_limit | 3.0 |
| max_consecutive_format_errors | 10 |
| parallel_tool_calls | false |
| env command timeout | 60 s |
| drop_params | false |

## Reproduce

Two forks are required (the agent fork carries the streaming-console /
token-tracking patches; the inference fork carries the SM120 MXFP4 + HMMA
sparse kernels). The **tool-call parser is the load-bearing v2 change** — without
`--tool-call-parser deepseekv4` the server returns the DSML tool-call markup as
raw content and the 2.4 agent fails every turn (`RepeatedFormatError`).

| Component | Fork | Branch / ref |
|-----------|------|--------------|
| Agent harness | `github.com/ambientlight/mini-swe-agent` | `v2.4.2_dev` |
| Inference server | `github.com/ambientlight/sglang` | `feat/sm120-mxfp4-w4a4-moe` |

**1. Serve the model** (4× RTX PRO 6000, SM120) — endpoint on `:8000` from
[`launch-single.sh`](../../../bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh)
(thinking + MAX effort + `--reasoning-parser deepseek-v4` + `--tool-call-parser deepseekv4`) +
[`sglang-single.yaml`](../../../bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml)
(1M context, mrr 16). From the sglang fork's venv:

```bash
bash bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh   # wait for /v1/models
```

**2. Run the agent** over SWE-bench Verified (16 workers):

```bash
git clone -b v2.4.2_dev https://github.com/ambientlight/mini-swe-agent
cd mini-swe-agent && uv sync
mini-extra swebench --subset verified --split test -w 16 \
  -c /path/to/config.yaml \
  -o ./results-deepseek-v4-flash-verified-v2
# config.yaml = the config.yaml in this folder; its api_base targets the :8000
# endpoint served by launch-single.sh above.
```

**3. Score** with the official SWE-bench harness (timeout 1200s was too short for
one CPU-heavy sklearn test under load — use 6000 to avoid a false error). The
`swebench` package is independent of the agent; install it into the same venv
(`uv pip install swebench`) and grade:

```bash
python -m swebench.harness.run_evaluation \
  --run_id deepseek-v4-flash-VERIFIED-v2 \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path ./results-deepseek-v4-flash-verified-v2/preds.json \
  --max_workers 64 --timeout 6000
```

## Trajectory characteristics

Clean run — **0 harness errors**, and 497/500 reached `Submitted` (76.5% of those
resolved). The 3 non-submits: 2× `LimitsExceeded` (hit step-250 empty) + 1×
`RepeatedFormatError` — and that one is **infrastructure, not the tool-call path**:
the Docker container died mid-run (`The container is completely gone`) and the
agent exhausted its format-error budget against a dead env, while still emitting
valid `tool_calls` throughout.

- **Turns separate winners from losers**: resolved median 36 turns, unresolved 55,
  empty 194. Only 2 instances hit the step limit (both empty) — the harness isn't
  leaving solvable work on the table.
- **Loops controlled** (`frequency_penalty: 0.1`): only 2/500 trajectories with a
  severe (>100×) repeated line, vs. the 75 runaway loops that sank the
  no-penalty thinking attempt. Loop severity still costs accuracy (84.6% resolved
  at no/normal repetition → 41.7% at moderate), but it's a tail, not a trend.
- **vs. the 73.0% run**: 28 newly solved, 13 regressed, 352 stable (96.4% of the
  prior wins held). A couple of the regressions are explained (the dead-container
  case above; one moderate loop). Net +15, comfortably above run-to-run variance.

## `frequency_penalty`

MAX-effort thinking would occasionally spin into repetition loops — the same line
repeated hundreds of times until it filled the whole output budget, burning the
step limit. A small `frequency_penalty` alleviates that; 0.1 and 0.2 both work,
this run used 0.1.
