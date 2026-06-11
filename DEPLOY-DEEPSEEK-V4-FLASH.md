# DeepSeek-V4-Flash-FP8 on 4x RTX PRO 6000 Blackwell

Deploying [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (291B params, MoE) via the [sgl-project FP8 repack](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) on 4x NVIDIA RTX PRO 6000 Blackwell (SM120) workstation GPUs using SGLang + the [0xSero SM120 sparse-decode patch](https://github.com/0xSero/deepseek-v4-flash-sm120).

## Hardware

| Component | Spec |
|-----------|------|
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| VRAM | 96 GB per GPU (384 GB total) |
| RAM | 504 GB DDR5 |
| CPU | 128 threads |
| Storage | 7.1 TB RAID (`/mnt/hot`) |

## Why the SM120 Patch?

The stock `lmsysorg/sglang:deepseek-v4-blackwell` image ships FlashMLA sparse-decode kernels for SM90 (Hopper) and SM100 (Blackwell data-center) but **not SM120** (Blackwell workstation/RTX PRO). Without the patch you get:

```
RuntimeError: Unsupported architecture for sparse decode fwd
```

The [0xSero/deepseek-v4-flash-sm120](https://github.com/0xSero/deepseek-v4-flash-sm120) repo builds a small CUDA extension targeting `sm_120a` and injects it at runtime via a read-only bind mount + `PYTHONPATH`. The SGLang image is never modified.

---

## Step 1: Download the FP8 Weights (~294 GB)

```bash
export MODEL_DIR=/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8
mkdir -p "$MODEL_DIR"

# Option A: huggingface-cli (if available)
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

# Option B: Python API (works with huggingface_hub >= 1.x)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'sgl-project/DeepSeek-V4-Flash-FP8',
    local_dir='$MODEL_DIR',
    max_workers=8
)
print('DONE')
"
```

**Verify:** You should have 46 `.safetensors` shards + config files (55 files total, ~294 GB).

```bash
ls "$MODEL_DIR"/*.safetensors | wc -l   # expect 46
du -sh "$MODEL_DIR"                      # expect ~294G
```

---

## Step 2: Clone the SM120 Patch Repo

```bash
cd /mnt/hot/ambientlight
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
git submodule update --init --recursive
```

---

## Step 3: Pull the SGLang Docker Image

```bash
docker pull lmsysorg/sglang:deepseek-v4-blackwell
```

This image is ~82 GB (compressed ~30 GB download). Expect 10-20 minutes depending on bandwidth.

**Verify:**
```bash
docker images | grep sglang
# lmsysorg/sglang   deepseek-v4-blackwell   ...   82.3GB
```

---

## Step 4: Build the SM120 Kernel Extension

```bash
cd /mnt/hot/ambientlight/deepseek-v4-flash-sm120
bash scripts/build_in_sglang_docker.sh
```

This spins up a **throwaway container** from the SGLang image, compiles the CUDA extension for `sm_120a` (SM120), and copies artifacts to `./build-docker/`. The SGLang image is untouched.

**Verify:** The build should produce these files:
```bash
ls build-docker/deepseek_v4_kernel/
# cuda.cpython-312-x86_64-linux-gnu.so
# __init__.py
# ops.py
# _patch.py
# sitecustomize_hook.py

ls build-docker/sitecustomize.py
# build-docker/sitecustomize.py
```

---

## Step 5: Launch the Server

### Quick Launch (use the provided script)

```bash
MODEL_DIR=/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8 \
CUDA_VISIBLE=0,1,2,3 \
PORT=8000 \
/mnt/hot/ambientlight/deepseek-v4-flash-sm120/scripts/launch_dsv4_flash_sm120.sh
```

> **Important:** The script defaults to `CUDA_VISIBLE=0,2,3,4`. Our system has GPUs `0,1,2,3`, so you **must** set `CUDA_VISIBLE=0,1,2,3` explicitly. Verify your GPU IDs with `nvidia-smi -L`.

### What the Launch Script Does

The script runs `docker run` with:

- **Model mount:** `-v $MODEL_DIR:/workspace/model:ro`
- **Patch injection:** `-v ./build-docker:/dsv4:ro` + `-e PYTHONPATH=/dsv4`
- **GPU access:** `--gpus all` + `CUDA_VISIBLE_DEVICES=0,1,2,3`
- **Shared memory:** `--shm-size=64g --ipc=host`

Key SGLang flags:

| Flag | Value | Purpose |
|------|-------|---------|
| `--tensor-parallel-size` | 4 | One shard per GPU |
| `--context-length` | 131072 | 128K context window (step up to 393216 for Think Max) |
| `--mem-fraction-static` | 0.8 | 80% of VRAM for KV cache |
| `--max-running-requests` | 16 | Concurrent request limit |
| `--kv-cache-dtype` | fp8_e4m3 | FP8 KV cache for memory efficiency |
| `--attention-backend` | compressed | DeepSeek-V4 MLA compressed attention |
| `--fp8-gemm-backend` | triton | FP8 GEMM via Triton |
| `--moe-runner-backend` | triton | MoE dispatch via Triton |
| `--chunked-prefill-size` | 32768 | Larger chunks = fewer prefill rounds = faster TTFT |
| `--tool-call-parser` | deepseekv4 | Native tool-call support |
| `--reasoning-parser` | deepseek-v4 | Thinking/reasoning token support |
| `--preferred-sampling-params` | `{"temperature":0.2,"top_p":0.95}` | Server default sampling (clients can override) |

NCCL tuning for PCIe Max-Q (set as env vars in `docker run`):

| Env Var | Value | Purpose |
|---------|-------|---------|
| `NCCL_PROTO` | `LL` | Low Latency protocol — faster allreduce for prefill |
| `NCCL_ALGO` | `Ring` | Ring algorithm — optimal for PCIe without NVLink |
| `NCCL_MIN_NCHANNELS` | `8` | More parallel NCCL channels |
| `NCCL_NTHREADS` | `512` | More NCCL threads |

> **Do not pass** a custom `--chat-template`. SGLang auto-detects the correct template for DeepSeek-V4.

> **Do not enable EAGLE speculative decoding** on SM120. The `--speculative-algorithm EAGLE` flags from the upstream repo cause garbled/gibberish output after ~50-100 tokens on RTX PRO 6000 GPUs. The draft token acceptance logic produces corrupted sequences on this architecture. Remove all `--speculative-*` flags.

---

## Step 6: Enable Thinking/Reasoning Mode

DeepSeek-V4-Flash supports a thinking mode where chain-of-thought reasoning is separated from the final answer. To enable it, clients must pass `chat_template_kwargs: {"thinking": true}` in the request body:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "temperature": 0,
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "What is 127 * 389?"}],
    "chat_template_kwargs": {"thinking": true}
  }'
```

**Without** `thinking: true`: reasoning leaks into `content` with raw `<think>...</think>` tags, and `reasoning_content` is null.

**With** `thinking: true`: reasoning goes to `reasoning_content` and `content` contains only the clean final answer.

### LiteLLM Configuration

To make this automatic for all requests through LiteLLM, add `extra_body` to the model config:

```yaml
- model_name: deepseek-v4-flash
  litellm_params:
    model: hosted_vllm/deepseek-v4-flash
    api_base: http://localhost:8000/v1
    api_key: "noop"
    max_parallel_requests: 8
    extra_body:
      chat_template_kwargs:
        thinking: true
```

---

## Step 6: Verify

### Wait for Startup

Startup takes ~2-3 minutes:
1. Weight loading (~26s) — 69.86 GB per GPU
2. KV cache allocation — ~14.6 GB avail per GPU after weights
3. CUDA graph capture (~83s) — 3.86 GB per GPU

Watch for the ready message:
```
The server is fired up and ready to roll!
```

You should also see the SM120 patch confirm in logs:
```
INFO _patch.py:261: deepseek_v4_kernel.patch_flash_mla installed (device SM 12.0).
```

### Smoke Test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "temperature": 0,
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Say OK only."}]
  }' | python3 -m json.tool
```

### Health Check

```bash
curl -s http://127.0.0.1:8000/health
```

---

## Memory Layout (Per GPU)

From actual startup logs:

| Stage | VRAM |
|-------|------|
| Total available | ~94 GB |
| After weight load | ~24.2 GB free (69.9 GB weights) |
| After KV cache pool | ~14.6 GB free |
| After CUDA graphs | ~10.7 GB free |

---

## Performance

From the SM120 patch repo benchmarks (4x RTX PRO 6000, EAGLE enabled in upstream — note: we disable EAGLE due to output corruption on SM120):

| Context Length | Decode tok/s |
|---------------|-------------|
| 32K | ~1273 |
| 64K | ~1099 |
| 128K | ~918 |
| 200K | ~772 |
| 300K | ~646 |

> The SM120 patch is correctness-first; performance at very long contexts is limited by the current sparse-decode kernel implementation. Improvements (split-KV, multi-CTA) are in progress upstream.

---

## Known Warnings (Safe to Ignore)

1. **"Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!"** — No tuned kernel configs exist for the RTX PRO 6000 yet. Default configs work correctly but leave performance on the table. You can generate tuned configs with [SGLang's benchmark scripts](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton).

2. **"Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0."** — Expected for this checkpoint; works correctly.

3. **"Using default MoE kernel config."** — Same as #1, for the MoE dispatch kernel.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Unsupported architecture for sparse decode fwd` | SM120 patch not loaded | Check `-v ./build-docker:/dsv4:ro` and `PYTHONPATH=/dsv4` in docker run |
| No `deepseek_v4_kernel.patch_flash_mla installed` in logs | `sitecustomize.py` not imported | Verify `build-docker/sitecustomize.py` exists and PYTHONPATH is set |
| OOM during long context | KV cache exhausted | Lower `--mem-fraction-static` or `--max-running-requests` |
| Tool-call formatting issues | Custom chat template interference | Remove any `--chat-template` flag; keep `--tool-call-parser deepseekv4` |
| Docker pull hangs | Large image (82 GB) | Be patient; layers download sequentially. Check `docker system df` for progress |

---

## Stopping the Server

```bash
docker rm -f sglang-dsv4-flash-sm120
```

## Restarting

Just re-run the launch command from Step 5. The build artifacts persist in `build-docker/` — no rebuild needed unless you update the patch repo.

---

## API Usage

The server exposes an **OpenAI-compatible** API at `http://localhost:8000`:

```bash
# Chat completions
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 1024,
    "temperature": 0.7
  }'

# With reasoning/thinking enabled (via env SGLANG_ENABLE_THINKING=1)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Solve: what is 127 * 389?"}],
    "max_tokens": 4096
  }'

# Model info
curl http://localhost:8000/model_info

# Health
curl http://localhost:8000/health
```

Compatible with any OpenAI SDK client:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

---

## File Locations

| What | Path |
|------|------|
| Model weights | `/mnt/hot/ambientlight/models/DeepSeek-V4-Flash-FP8` |
| SM120 patch repo | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120` |
| Build artifacts | `/mnt/hot/ambientlight/deepseek-v4-flash-sm120/build-docker/` |
| Docker image | `lmsysorg/sglang:deepseek-v4-blackwell` (82.3 GB) |
