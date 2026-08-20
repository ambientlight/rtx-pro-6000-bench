# sglang-sm120-mxfp4

Benchmark-qualified SGLang main serving image for **DeepSeek-V4-Flash with
MXFP4 weights** on **4× RTX PRO 6000 Blackwell GPUs (SM120, PCIe, TP=4)**.

Release `2026.08.0-cu130-sm120a` pins SGLang main commit
[`b03ac355`](https://github.com/sgl-project/sglang/commit/b03ac355e795b3a86b26b8732c47c0965fd71bbc)
on CUDA 13.0.3. It uses SGLang's native SM120 FlashMLA attention path and
FlashInfer 0.6.17 for MoE. The checkpoint weights remain MXFP4, while the
current upstream SM120 expert path uses MXFP8 activations: **W4A8**, not the
older custom W4A4 path.

> **Validated model:** DeepSeek-V4-Flash. ~~MiniMax-M3~~ is not validated on
> this image and is not currently exposed by its entrypoint.

Benchmarks, deployment files, and SWE-bench evaluations:
**https://github.com/ambientlight/rtx-pro-6000-bench**

## Requirements

- 4× RTX PRO 6000 Blackwell (SM120, 96 GB)
- NVIDIA Container Toolkit
- NVIDIA driver 595 or newer
- Linux/amd64

This release is intended and performance-qualified for SM120 only. Other GPU
architectures are unsupported.

## Models

| Model | Hugging Face checkpoint | Status |
|---|---|---|
| DeepSeek-V4-Flash | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | **Validated** |
| ~~MiniMax-M3~~ | ~~[olka-fi/MiniMax-M3-MXFP4](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4)~~ | **Not validated on this image** |

Download the validated checkpoint:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /path/to/DeepSeek-V4-Flash
```

## Run DeepSeek-V4-Flash

Mount the model at `/model`. The server exposes an OpenAI-compatible API on
port `8000` with served model name `deepseek-v4-flash`.

Use the pinned tag for reproducible deployment:

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v /path/to/DeepSeek-V4-Flash:/model:ro \
  ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a
```

Test the server:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"What is the capital of Japan?"}],
    "temperature": 0
  }'
```

## Defaults

All values below are environment-overridable.

| Variable | Default | Notes |
|---|---:|---|
| `MODEL_DIR` | `/model` | Mounted checkpoint directory |
| `SERVED_NAME` | `deepseek-v4-flash` | OpenAI API model name |
| `CONTEXT_LENGTH` | `1048576` | Full 1M context |
| `MEM_FRACTION_STATIC` | `0.65` | KV pool fraction used by the qualified image |
| `MAX_RUNNING` | `16` | Maximum concurrent requests |
| `CHUNKED_PREFILL_SIZE` | `8192` | Chunked-prefill token count |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV-cache data type |
| `PORT` | `8000` | API port |
| `EXTRA_SERVER_ARGS` | empty | Additional `sglang.launch_server` arguments |

The entrypoint fixes tensor parallelism at TP=4 and selects
`flashinfer_mxfp4` as the MoE runner backend.

## W4A4 vs W4A8

This release uses W4A8. In our controlled 8K-input/1K-output probe, replayed
W4A4 had slightly better decode latency (9.86 vs 10.76 ms TPOT), but W4A8 had
far faster prefill (1.38 vs 2.98 s TTFT) and higher end-to-end throughput. The
current W4A8 path is preferred for agentic workloads that can dominated by
long-context prefills

## Qualification

The release passed a matched comparison against the previously deployed custom
SGLang/FlashInfer/HMMA stack on 4× SM120 GPUs at 300 W per GPU. Exact SGLang
main won all nine benchmark cells across 8K and 64K inputs, 1K output tokens,
and concurrency 1/2/4/8/16 without regressing the measured primary metrics.

See the full
[benchmark report](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/bench/deepseek-v4-sm120-main-compare-20260820/REPORT.md)
for methodology, results, and caveats.

## Tags

- `:2026.08.0-cu130-sm120a` — benchmark-qualified pinned release
- `:latest` — moving tag; use the pinned release tag for reproducibility
- `:YYYY.MM.PATCH-cuNNN-sm120a` — release naming convention

The `cu130` component identifies the CUDA 13.0 runtime. The `sm120a` component
identifies the target Blackwell architecture.

## Build sources

- [Release definition](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/docker/sglang-main-sm120/Dockerfile.release)
- [Pinned SGLang-main image](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/docker/sglang-main-sm120/Dockerfile)
- [DeepSeek-V4 entrypoint](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/docker/sglang-main-sm120/entrypoint-dsv4.sh)
