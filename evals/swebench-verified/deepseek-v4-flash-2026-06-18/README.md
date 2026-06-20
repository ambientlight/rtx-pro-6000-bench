# SWE-bench Verified — DeepSeek-V4-Flash (MXFP4 W4A4, SM120)

Agentic SWE-bench Verified of DeepSeek-V4-Flash served on 4× RTX PRO 6000
(SM120, TP=4) via the native MXFP4 W4A4 + HMMA sparse-kernel stack, driven by
mini-swe-agent over litellm/`hosted_vllm`. Serving recipe:
[DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md](../../../docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md).

## Result

| Metric | Value |
|--------|-------|
| **Resolved / total** | **365 / 500 = 73.0%** |
| Submitted | 500 |
| Unresolved | 119 |
| Empty patch | 11 |
| Error | 5 |
| Output tokens | 9,262,551 |
| Wall-clock | 9h 39m |

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

**Agent / sampling** (mini-swe-agent `v1.17.3_dev`, `config.yaml`):

| Setting | Value |
|---------|-------|
| Harness | mini-swe-agent (default agent) |
| Benchmark | SWE-bench Verified (500, test split) |
| Workers | 16 |
| temperature | 0.2 |
| top_p | 0.95 |
| frequency_penalty | 0.2 |
| max_tokens (output) | 65536 |
| step_limit | 250 |
| cost_limit | 3.0 |
| max_consecutive_errors | 10 |
| env command timeout | 600 s |
| drop_params | false |

## Reproduce

Two forks are required (the agent fork carries the custom action-parser /
token-tracking / `max_consecutive_errors` patches this config relies on):

| Component | Fork | Branch / ref |
|-----------|------|--------------|
| Agent harness | `github.com/ambientlight/mini-swe-agent` | `v1.17.3_dev` |
| Inference server | `github.com/ambientlight/sglang` | `feat/sm120-mxfp4-w4a4-moe` |

**1. Serve the model** (4× RTX PRO 6000, SM120) — endpoint on `:8000` from
[`launch-single.sh`](../../../bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh)
(thinking + MAX effort + `--reasoning-parser deepseek-v4`) +
[`sglang-single.yaml`](../../../bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml)
(1M context, mrr 16). From the sglang fork's venv:

```bash
bash bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh   # wait for /v1/models
```

**2. Run the agent** over SWE-bench Verified (16 workers):

```bash
git clone -b v1.17.3_dev https://github.com/ambientlight/mini-swe-agent
cd mini-swe-agent && uv sync
mini-extra swebench --subset verified --split test -w 16 \
  -c /path/to/config.yaml \
  -o ./results-deepseek-v4-flash-VERIFIED
# config.yaml = the config.yaml in this folder; its api_base targets the :8000
# endpoint served by launch-single.sh above.
```

**3. Score** with the official SWE-bench harness (timeout 1200s was too short for
one CPU-heavy sklearn test under load — use 6000 to avoid a false error):

```bash
python -m swebench.harness.run_evaluation \
  -id deepseek-v4-flash-VERIFIED \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path ./results-deepseek-v4-flash-VERIFIED/preds.json \
  --max_workers 64 --timeout 6000
```

## `frequency_penalty`

MAX-effort thinking would seldomly spin into repetition loops — the same line
repeated hundreds of times until it filled the whole output budget, which burned through the step limit. A small `frequency_penalty` alleviates that. Both 0.1 and
0.2 worked fine, this run used 0.2.
