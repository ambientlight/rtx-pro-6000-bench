# sglang-sm120-mxfp4

> **Legacy W4A4 unified image:** this document describes the older custom
> SGLang/FlashInfer/HMMA image for DeepSeek-V4-Flash and MiniMax-M3. For the
> current benchmark-qualified DeepSeek-V4-Flash release, use
> `ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a` (also published as
> `:latest`) and see the
> [new image documentation](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/docker/sglang-main-sm120/DOCKERHUB.md).
> The new image uses W4A8 and has not validated MiniMax-M3.

sglang + flashinfer serving image for **MXFP4 W4A4 MoE** on **RTX PRO 6000 Blackwell (SM120, PCIe, TP=4)** for **DeepSeek-V4-Flash** and **MiniMax-M3**. Routed experts run in MXFP4 (E2M1 + E8M0/32 scales) via flashinfer's CuTe-DSL SM120 fused MoE.

Benchmarks, deploy docs, and SWE-bench evals: **https://github.com/ambientlight/rtx-pro-6000-bench**

## Requirements

- 4× RTX PRO 6000 Blackwell (SM120, 96 GB)
- NVIDIA Container Toolkit + driver ≥ 595

## Models

| Model | Hugging Face checkpoint |
|---|---|
| DeepSeek-V4-Flash | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| MiniMax-M3 | [olka-fi/MiniMax-M3-MXFP4](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4) |

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /path/to/DeepSeek-V4-Flash
huggingface-cli download olka-fi/MiniMax-M3-MXFP4 --local-dir /path/to/minimax-m3-mxfp4
```

## Run

Weights are mounted at `/model`; the model is chosen with `MODEL={dsv4|m3}` (default `m3`). Serves an OpenAI-compatible API on `:8000`

**DeepSeek-V4-Flash** (served name `deepseek-v4-flash`):
```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -e MODEL=dsv4 \
  -v /path/to/DeepSeek-V4-Flash:/model:ro \
  ambientlight/sglang-sm120-mxfp4:2026.07.3-cu131-sm120a
```

**MiniMax-M3** (served name `minimax-m3`):
```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -e MODEL=m3 \
  -v /path/to/minimax-m3-mxfp4:/model:ro \
  ambientlight/sglang-sm120-mxfp4:2026.07.3-cu131-sm120a
```

Test:
```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role":"user","content":"What is the capital of Japan?"}],
  "temperature": 0
}'
```

## Defaults (all env-overridable)

| var | dsv4 | m3 | notes |
|---|---|---|---|
| `CONTEXT_LENGTH` | 1048576 | 1048576 | full 1M |
| `MEM_FRACTION_STATIC` | 0.90 | 0.95 | KV pool fraction |
| `MAX_RUNNING` | 16 | 16 | max concurrent requests |
| `KV_CACHE_DTYPE` | fp8_e4m3 | fp8_e4m3 | |
| `PORT` | 8000 | 8000 | |

## Tags

- `:2026.07.3-cu131-sm120a` — pinned legacy W4A4 unified release
- `:latest` — now points to the newer DeepSeek-only W4A8 release documented above
- `:YYYY.MM.PATCH-cu131-sm120a` — legacy naming convention (date + CUDA + `sm120a` arch; **SASS is compiled for SM120 only — will not run on other GPUs**)

Built from forks of sglang, flashinfer, and the DSV4 HMMA kernel — please ref [Dockerfile](https://github.com/ambientlight/rtx-pro-6000-bench/blob/feat/dsv4-0731-stack/docker/sm120-unified/Dockerfile)
