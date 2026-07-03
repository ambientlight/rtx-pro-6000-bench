# SWE-bench Verified — MiniMax-M3 (MXFP4 W4A4, SM120) — v2 tool-calling

Agentic SWE-bench Verified of MiniMax-M3 served on 4× RTX PRO 6000 (SM120, TP=4)
via the native MXFP4 W4A4 (routed experts) + MXFP8 weight-only linears + SM120
Triton MSA block-sparse attention stack, driven by mini-swe-agent **2.4.2** with
native tool-calling. Serving recipe:
[DEPLOY-MXFP4-W4A4-MINIMAX-M3-SM120.md](../../../docs/DEPLOY-MXFP4-W4A4-MINIMAX-M3-SM120.md).

A submit-nudge + 600 s command-timeout config fix eliminated the turn-limit failure class (43→6 `LimitsExceeded`)

## Result

| Metric | Value |
|--------|-------|
| **Resolved / total** | **374 / 500 = 74.8%** |
| Submitted | 500 |
| Completed | 491 |
| Unresolved | 117 |
| Empty patch | 8 |
| Error | 1 |

## Files

- `scored.json` — the swebench harness report (resolved/unresolved/empty/error ids).
- `config.yaml` — mini-swe-agent run config (`swebench-litellm-minimax-m3.latest.yaml`).
- `trajectories/` — 498 per-instance `*.traj.json` agent trajectories + `preds.json`.
  One trajectory (`sphinx-doc__sphinx-9320`, a 1.2 GB repetition-loop blowup that still
  resolved) is omitted — see `trajectories/sphinx-doc__sphinx-9320/OMITTED.md`.

## Configuration

**Model / serving** (sglang, `launch-bwrap-highconc.sh`; full recipe in the
[deploy doc](../../../docs/DEPLOY-MXFP4-W4A4-MINIMAX-M3-SM120.md)):

| Setting | Value |
|---------|-------|
| Model | MiniMax-M3 (native MXFP4 W4A4 routed experts + MXFP8 linears) |
| Hardware | 4× RTX PRO 6000 Blackwell (SM120), TP=4 |
| Context length | 1,048,576 (1M) |
| kv-cache-dtype | fp8_e4m3 |
| Quantization | compressed-tensors (mixed-precision) |
| Sparse attention | SM120 Triton MSA (block-sparse, two-step indexer→main) |
| MoE kernel | flashinfer SM120 MXFP4 fused MoE (clamped SwiGLU-OAI), dynamic-M JIT |
| Thinking | on (chat-template default) |
| Reasoning parser | `minimax-m3` (splits `<think>` into `reasoning_content`) |
| Tool-call parser | `minimax-m3` (parses tool calls → structured `tool_calls`) |

**Agent / sampling** (mini-swe-agent `v2.4.2_dev`, `config.yaml`):

| Setting | Value |
|---------|-------|
| Harness | mini-swe-agent 2.4.2 (native tool-calling) |
| Benchmark | SWE-bench Verified (500, test split) |
| Workers | 16 (scoring) |
| temperature | 1.0 (MiniMax-M3 model-card, thinking on) |
| top_p | 0.95 (model-card) |
| top_k | 40 (model-card, via `drop_params: false`) |
| frequency_penalty | 0.1 (stops thinking-mode repetition loops) |
| max_tokens (output) | 65536 |
| step_limit | 250 |
| cost_limit | 3.0 |
| max_consecutive_format_errors | 10 |
| parallel_tool_calls | false |
| env command timeout | 600 s (bumped from 60 s — the 60 s cap was killing slow tests) |
| drop_params | false |

Unlike the DSV4 run (temperature 0.2), M3 has **no reasoning-effort dial** and its
model card recommends **temperature 1.0** with thinking on — so this is the
model-card sampling (matching 0.2 regressed this benchmark significantly)

## Reproduce

Two forks are required (the agent fork carries the streaming-console / token-tracking
patches; the inference fork carries the SM120 MXFP4 MoE + MXFP8 linear + MSA kernels).

| Component | Fork | Branch / ref |
|-----------|------|--------------|
| Agent harness | `github.com/ambientlight/mini-swe-agent` | `v2.4.2_dev` |
| Inference server | `github.com/ambientlight/sglang` | `feat/sm120-minimax-m3-mxfp4` |
| MoE kernels | `github.com/ambientlight/flashinfer` | `ambientlight/mxfp4-fused-moe-minimax-m3` |

**1. Serve the model** (4× RTX PRO 6000, SM120) — endpoint on `:8000` from
[`launch-bwrap-highconc.sh`](../../../bench/minimax-m3-mxfp4_W300_TP4_sglang/launch-bwrap-highconc.sh)
(`--reasoning-parser minimax-m3` + `--tool-call-parser minimax-m3`, fp8 KV, dynamic-M MoE):

```bash
bash bench/minimax-m3-mxfp4_W300_TP4_sglang/launch-bwrap-highconc.sh   # wait for /v1/models
```

**2. Run the agent** over SWE-bench Verified:

```bash
git clone -b v2.4.2_dev https://github.com/ambientlight/mini-swe-agent
cd mini-swe-agent && uv sync
mini-extra swebench --subset verified --split test -w 16 \
  -c /path/to/config.yaml \
  -o ./results-minimax-m3-verified
# config.yaml = the config.yaml in this folder; its api_base targets the :8000
# endpoint served by launch-bwrap-highconc.sh above.
```

**3. Score** with the official SWE-bench harness. Use `--max_workers 16` (64 OOM-dropped
test containers under load, producing false `error` results) and `--timeout 6000`:

```bash
python -m swebench.harness.run_evaluation \
  --run_id minimax-m3-VERIFIED-v2 \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path ./results-minimax-m3-verified/preds.json \
  --max_workers 16 --timeout 6000
```

## Trajectories

500/500 `Submitted`, 491 completed. The turn-limit failure class (`LimitsExceeded`
at step-250) that plagued earlier runs was cut 43→6 by the submit-nudge prompt (steps
150/200 remind the agent to submit if its fix already works) + the 600 s command
timeout (the 60 s cap was killing slow sklearn/sympy tests mid-run, forcing retries
that burned the step budget).

## `frequency_penalty`

M3's thinking mode at temperature 1.0 would occasionally spin into repetition loops —
the same line repeated hundreds of times until it filled the whole output budget (the
`sphinx-doc__sphinx-9320` trajectory is the extreme case, at 1.2 GB). A small
`frequency_penalty` (0.1) alleviates that.
