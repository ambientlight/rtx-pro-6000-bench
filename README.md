# rtx-pro-6000-bench

Benchmark sweep harness for local model inference (SM120). Sweeps concurrency levels, input/output token lengths, and collects GPU telemetry (power, KV cache, utilization) to produce comparison charts.

## Hardware

- **GPUs**: 4x NVIDIA RTX 6000 Blackwell Pro Max-Q Workstation Edition (96GB x4)
- **CPU**: AMD Ryzen Threadripper PRO 7985WX (64-core)
- **RAM**: 512 GB DDR5 ECC (8x 64 GB Kingston KSM56R46BD4PMI-64HAI)
- **Platform**: ASUS Pro WS WRX90E-SAGE SE
- **PSU**: Super Flower Leadex Titanium 1700W ATX 3.1
- **OS**: Ubuntu 24.04 LTS

## Models Benchmarked

| Model | Architecture | Params (total / active) | Context | Engine | Notes |
|-------|-------------|------------------------|---------|--------|-------|
| **Qwen3.5-397B-A17B** | MoE | 397B / 17B | 131,072 | vLLM | |
| **MiniMax-M2.5** | MoE | 230B / 10B | 196,608 | vLLM | |
| **Devstral-2-123B** | Dense | 123B | 262,144 | vLLM | torch.compile mode 3, CUDAGraphs, fuse_act_quant=false (sm_120) |
| **DeepSeek-V4-Flash** | MoE (MLA + sparse) | 284B / 13B | 1,048,576 | sglang | native MXFP4 W4A4 experts + HMMA tensor-core sparse decode **and** prefill + split-KV long-context indexer (sm_120); TP4 |

All vLLM models: `tensor_parallel_size: 4`, `gpu_memory_utilization: 0.90`, `kv_cache_dtype: fp8_e4m3`, `enable_chunked_prefill: true`, `max_num_seqs: 128`, `max_num_batched_tokens: 65536`. DeepSeek-V4-Flash (sglang): `tp 4`, `mem-fraction-static 0.80`, `kv-cache-dtype fp8_e4m3`, `max-running-requests 128`, `chunked-prefill-size 16384`.

## Results Summary

**Test parameters**: 128 random prompts per run, 1024 output tokens, input lengths from 2K to 64K.

### W300 / TP4: Qwen3.5-397B vs DeepSeek-V4-Flash

The two models swept at **300W, TP4** on this box (same `--output-len 1024 --step-size 8` matrix). Different
families and stacks — Qwen3.5-397B-A17B (NVFP4, vLLM) vs DeepSeek-V4-Flash (native MXFP4 W4A4 + MLA/sparse
attention, sglang) — so this is a "what runs at W300/TP4 here", not really direct apples-to-apples comparison.

#### Peak Output Throughput (tok/s)

| Input Length | Qwen3.5-397B-A17B (NVFP4, vLLM) | DeepSeek-V4-Flash (MXFP4, sglang) |
|:------------:|:-------------------------------:|:---------------------------------:|
| 2,048 | **1,124** @c64 | 756 @c64 |
| 4,096 | **908** @c64 | 444 @c32 |
| 8,192 | **649** @c72 | 263 @c40 |
| 16,384 | **387** @c48 | 163 @c104 |
| 32,768 | **212** @c32 | 79 @c40 |
| 65,536 | **102** @c16 | 40 @c16 |

#### Single-User (concurrency=1, 2K input)

| Metric | Qwen3.5-397B-A17B | DeepSeek-V4-Flash |
|--------|:-----------------:|:-----------------:|
| TTFT p50 | **255 ms** | 604 ms |
| TPOT p50 | **11.3 ms** | 16.8 ms |
| Output throughput | **86.4 tok/s** | 57.6 tok/s |
| Mean power @ 2K peak | 1,062 W | **1,009 W** |

Qwen leads on raw throughput/latency; DeepSeek-V4-Flash is the larger-context (1M) MLA+sparse model and the
only one here on native MXFP4 W4A4 + a custom SM120 attention stack. Per-model detail below.

### Peak Output Throughput at 250W (tok/s)

| Input Length | Qwen3.5-397B MoE | MiniMax-M2.5 | Devstral-2-123B |
|:------------:|:-----------------:|:-----------:|:---------------:|
| 2,048 | 1,041 @c72 | **2,213** @c128 | 1,107 @c64 |
| 4,096 | 865 @c96 | **1,437** @c64 | 1,027 @c64 |
| 8,192 | 616 @c80 | **1,244** @c64 | 894 @c88 |
| 16,384 | **366** @c48 | 408 @c72 | 100 @c64 |
| 32,768 | **205** @c32 | 194 @c48 | 54 @c16 |
| 65,536 | **98** @c16 | 83 @c112 | 25 @c16 |

### Single-User Latency (concurrency=1, 2K input)

| Metric | Qwen3.5-397B MoE | MiniMax-M2.5 | Devstral-2-123B |
|--------|:-----------------:|:-----------:|:---------------:|
| TTFT p50 | 260 ms | **212 ms** | 803 ms |
| TTFT p95 | 264 ms | **216 ms** | 807 ms |
| TTFT p99 | 288 ms | **237 ms** | 808 ms |
| TPOT p50 | **11.5 ms** | 11.9 ms | 33.0 ms |
| TPOT p95 | **11.5 ms** | 11.9 ms | 33.1 ms |
| TPOT p99 | **11.5 ms** | 11.9 ms | 33.1 ms |
| Output throughput (mean) | **85.4 tok/s** | 83.0 tok/s | 29.6 tok/s |
| Output throughput (peak) | **89.0 tok/s** | 86.0 tok/s | 32.0 tok/s |

### DeepSeek-V4-Flash (W300, sglang, native MXFP4 W4A4 + HMMA sparse attention)

Reported separately: different engine (sglang), quantization (native MXFP4 W4A4 MoE), and power cap (W300).
Full 2K–64K × concurrency sweep, **8576/8576 requests succeeded, 0 failed** — including the >11673-token
prefill batches that crash stock sgl-kernel on SM120 (the gap the HMMA sparse-prefill kernel closes). Setup +
kernel details: [docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md](docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md).

#### Peak Output Throughput at 300W (tok/s)

| Input Length | Peak tok/s | @ concurrency | Mean system power |
|:------------:|:----------:|:-------------:|:-----------------:|
| 2,048 | **756** | c64 | 1,009 W |
| 4,096 | **444** | c32 | 1,009 W |
| 8,192 | **263** | c40 | 984 W |
| 16,384 | **163** | c104 | 931 W |
| 32,768 | **79** | c40 | 914 W |
| 65,536 | **40** | c16 | 919 W |

#### Single-User Latency (concurrency=1, by prompt length)

| Input Length | TTFT p50 | TTFT p99 | TPOT p50 | Output tok/s | E2E p50 (s) |
|:------------:|:--------:|:--------:|:--------:|:------------:|:-----------:|
| 2,048 | 604 ms | 819 ms | 16.8 ms | 57.6 | 17.8 |
| 4,096 | 1,285 ms | 1,451 ms | 17.0 ms | 55.0 | 18.6 |
| 8,192 | 2,653 ms | 2,725 ms | 17.5 ms | 49.8 | 20.6 |
| 16,384 | 5,381 ms | 5,428 ms | 18.5 ms | 42.2 | 24.3 |
| 32,768 | 11,199 ms | 11,238 ms | 20.4 ms | 32.0 | 32.1 |
| 65,536 | 23,914 ms | 24,697 ms | 24.2 ms | 21.0 | 48.6 |

TTFT scales with prompt length (prefill cost); TPOT stays ~17–24 ms (decode is sparse-attention top-k, near
prompt-independent), so single-stream output throughput tapers from 58 → 21 tok/s as KV grows.

Per-input-length charts in [bench/deepseek-v4-flash_W300_TP4_sglang/plots/](bench/deepseek-v4-flash_W300_TP4_sglang/plots/);
peak throughput is decode-bound (TPOT ~17 ms single-user), while long-context throughput is gated by the
sparse-attention gather + KV-cache capacity (32K/64K saturate KV at 100%, hence early concurrency peaks).

| | |
|---|---|
| ![Throughput vs concurrency](bench/deepseek-v4-flash_W300_TP4_sglang/plots/deepseek-v4-flash_compare_W300/compare_throughput_vs_concurrency.png) | ![Peak power](bench/deepseek-v4-flash_W300_TP4_sglang/deepseek-v4-flash_random_2048in_1024out_c64_W300/telemetry_power.png) |

<sub>Left: output throughput vs concurrency across input lengths. Right: system power at peak throughput (756 tok/s @ c64, 2048in/1024out) — ~1,009 W mean, 94% GPU util.</sub>

#### Long-context scaling to 1M (single stream, c1)

A separate single-concurrency sweep doubling the prompt 2K → **1,047,552** (= 1M − 1024), one prompt per length, output 1024. Run with
[`sglang-single.yaml`](bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml)
(full 1M context, `mem-fraction-static 0.80`, `chunked-prefill-size 8192`, `max-running-requests 1`) +
[`launch-single.sh`](bench/deepseek-v4-flash_W300_TP4_sglang/launch-single.sh), which enables the **split-KV
indexer** (`SGLANG_SM120_INDEXER_SPLIT`). **All 10 lengths completed, including the full 1M prompt.**

| Input | TTFT (prefill) | TPOT (decode) | Output tok/s | vs no-split | Peak VRAM/GPU | KV |
|------:|:--------------:|:-------------:|:------------:|:-----------:|:-------------:|:--:|
| 2,048 | 0.19 s | 16.7 ms | 59.2 | 1.01× | 85% | 1% |
| 8,192 | 0.20 s | 17.2 ms | 57.6 | 1.02× | 85% | 2% |
| 32,768 | 0.31 s | 18.9 ms | 52.1 | 1.08× | 87% | 7% |
| 131,072 | 0.79 s | 25.8 ms | 37.6 | 1.22× | 87% | 7% |
| 262,144 | 1.58 s | 35.0 ms | 27.4 | 1.32× | 88% | 12% |
| 524,288 | 3.26 s | 53.5 ms | 17.7 | 1.43× | 90% | 23% |
| **1,047,552** | **6.46 s** | **90.2 ms** | **10.4** | **1.51×** | **95%** | **47%** |

| |
|---|
| ![1M power](bench/deepseek-v4-flash_W300_TP4_sglang/single_longsequence_indexersplit/deepseek-v4-flash_random_1047552in_1024out_c1_W300/telemetry_power.png) |

<sub>System power during the 1,047,552-token single-stream run (split-KV indexer; TTFT 6.5 s prefill, then 1024-token decode at ~10 tok/s; ~900 W mean).</sub>




## Installation

```bash
uv pip install -e .
```

Requires a running inference server (OpenAI-compatible API on `http://127.0.0.1:8000`) and `vllm`
installed in the harness venv — `bench-sweep` drives `vllm bench serve --backend openai`, which works
against any OpenAI-compatible endpoint (vLLM **or** sglang). It invokes the bench tool as
`python -m vllm.entrypoints.cli.main bench serve` (so a stale `vllm` console script doesn't block
runs); override with `--bench-cmd` if needed.

### sglang (DeepSeek-V4-Flash)

DeepSeek-V4-Flash is served by our sglang fork (MXFP4 W4A4 + HMMA sparse decode on sm_120). Full
setup: [docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md](docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md).
Launch the server, then sweep with `--tokenizer` pointed at the checkpoint:

```bash
# 1. Start the server (config + env in bench/deepseek-v4-flash_W300_TP4_sglang/)
bash bench/deepseek-v4-flash_W300_TP4_sglang/launch.sh   # wait for /v1/models (~2 min)

# 2. Matrix sweep (output 1024, step-size 8 → c1,2,4,8,16,24,…,128; matches the vLLM models)
bench-sweep --matrix --telemetry \
  --model-id deepseek-v4-flash --watt 300 \
  --tokenizer /mnt/hot/ambientlight/models/DeepSeek-V4-Flash \
  --input-lens 2048,4096,8192,16384,32768,65536 --output-len 1024 \
  --step-size 8 --num-prompts 128 --max-error-rate 0.1
```

The sglang server **must** be launched with `enable_metrics: true` (set in `sglang.yaml`) for telemetry
to capture KV-cache % and request counts; GPU power/util work regardless.

## Usage

```bash
# Full matrix sweep with telemetry
bench-sweep --matrix --telemetry \
  --model-id qwen35-397b-a17b-nvfp4 \
  --tokenizer /path/to/tokenizer \
  --watt 250 \
  --input-lens 2048,4096,8192,16384,32768,65536 \
  --output-len 1024 \
  --step-size 8 \
  --num-prompts 128

# Per-input-len max concurrency caps (avoid OOM on large inputs)
bench-sweep --matrix --telemetry \
  --model-id qwen35-397b-a17b-nvfp4 \
  --tokenizer /path/to/tokenizer \
  --watt 250 \
  --input-lens 2048,4096,8192,16384,32768,65536 \
  --max-concurrency 128,96,96,96,48,48 \
  --output-len 1024

# Re-plot existing results without re-running benchmarks
bench-sweep --matrix --plot-only \
  --model-id qwen35-397b-a17b-nvfp4 --watt 250 \
  --input-lens 2048,4096,8192,16384,32768,65536 --output-len 1024

# Single concurrency sweep
bench-sweep --model-id qwen35-397b-a17b-nvfp4 \
  --tokenizer /path/to/tokenizer --watt 250

# Dry run (print commands only)
bench-sweep --dry-run --model-id qwen35-397b-a17b-nvfp4 \
  --tokenizer /path/to/tokenizer --watt 250
```

## Directory Structure

```
bench/
  {model}_W{watt}_TP{tp}_{engine}/
    vllm.yaml | sglang.yaml             # Server configuration
    b.log                                # bench-sweep invocation command
    bench_sweep.log                      # Full execution log (all runs)

    {model}_random_{in}in_{out}out_c{concurrency}_W{watt}/
      openai-infqps-concurrency{N}-{model}-{YYYYMMDD-HHMMSS}.json
      telemetry.csv
      telemetry_summary.json
      telemetry_power.png
      telemetry_kv_cache.png

    plots/
      {model}_{in}in_{out}out_W{watt}/
        overview.png                   # Combined dashboard
        throughput_vs_concurrency.png
        ttft_vs_concurrency.png
        tpot_vs_concurrency.png
        itl_vs_concurrency.png
        e2el_vs_concurrency.png
        duration_vs_concurrency.png
      {model}_compare_W{watt}/
        compare_overview_p50.png
        compare_overview_p95_p99.png
        compare_throughput_vs_concurrency.png
        compare_ttft_p50_vs_concurrency.png
        compare_tpot_p50_vs_concurrency.png
        compare_itl_p50_vs_concurrency.png
        compare_e2el_p50_vs_concurrency.png
        compare_duration_vs_concurrency.png
        compare_efficiency_vs_concurrency.png   # tok/s per watt
        compare_power_vs_concurrency.png
        compare_gpu_util_vs_concurrency.png
        compare_mem_bw_util_vs_concurrency.png
        compare_kv_cache_vs_concurrency.png
```

## Data Schemas

### Benchmark Results JSON (`openai-infqps-*.json`)

One file per (model, input_len, output_len, concurrency) run. Produced by vLLM's [`benchmark_serving.py`](https://github.com/vllm-project/vllm/tree/main/benchmarks). See upstream for the full schema.

### GPU Telemetry CSV (`telemetry.csv`)

Time-series GPU metrics sampled at ~2.4 Hz during each benchmark run. 4-GPU system (gpu0 through gpu3), with per-GPU columns repeated.

**Header** (30+ columns):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | float | Unix epoch (s) | Absolute timestamp |
| `elapsed_s` | float | seconds | Time since benchmark start |
| `gpu{N}_power_w` | float | Watts | power draw |
| `gpu{N}_mem_used_gb` | float | GB | Memory used |
| `gpu{N}_util_pct` | int | % | Compute utilization (0-100) |
| `gpu{N}_mem_bw_util_pct` | int | % | Memory bandwidth utilization |
| `gpu{N}_temp_c` | int | C | Temperature |
| `gpu{N}_pcie_tx_mb_s` | float | MB/s | PCIe transmit throughput |
| `gpu{N}_pcie_rx_mb_s` | float | MB/s | PCIe receive throughput |
| `kv_cache_pct` | float | % | KV cache utilization (from server /metrics) |
| `requests_running` | int | count | Active inference requests |
| `requests_waiting` | int | count | Queued requests |

Where `{N}` is 0, 1, 2, 3. Columns repeat for each GPU.

**Example row** (abbreviated):
```
1774517244.043,0.000,128.0,81.59,0,0,84,0.5,0.4,111.4,81.63,...,0.0,0,0
```

### Telemetry Summary (`telemetry_summary.json`)

Pre-aggregated statistics from `telemetry.csv` per run (power, GPU util, KV cache, PCIe, etc.). Each per-run directory contains one.

Example plots from Qwen3.5-397B-A17B (W300):

| | |
|---|---|
| ![Overview P50](bench/qwen35-397b-a17b-nvfp4_W300_TP4_vllm/plots/qwen35-397b-a17b-nvfp4_compare_W300/compare_overview_p50.png) | ![Overview P95/P99](bench/qwen35-397b-a17b-nvfp4_W300_TP4_vllm/plots/qwen35-397b-a17b-nvfp4_compare_W300/compare_overview_p95_p99.png) |
| ![Peak Power](bench/qwen35-397b-a17b-nvfp4_W300_TP4_vllm/qwen35-397b-a17b-nvfp4_random_2048in_1024out_c64_W300/telemetry_power.png) | ![Peak KV Cache](bench/qwen35-397b-a17b-nvfp4_W300_TP4_vllm/qwen35-397b-a17b-nvfp4_random_2048in_1024out_c64_W300/telemetry_kv_cache.png) |

<sub>Bottom row: power draw and KV cache at peak throughput (1,124 tok/s @ concurrency 64, 2048in/1024out)</sub>

### Engine Configs

**vLLM models:**

- [Qwen3.5-397B-A17B](bench/qwen35-397b-a17b-nvfp4_W250_TP4_vllm/vllm.yaml) — checkpoint: [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
- [MiniMax-M2.5](bench/minimax_m25-nvfp4_W250_TP4_vllm/vllm.yaml) — checkpoint: [lukealonso/MiniMax-M2.5-NVFP4](https://huggingface.co/lukealonso/MiniMax-M2.5-NVFP4)
- [Devstral-2-123B](bench/devstral-2-123b-instruct-2512_W250_TP4_vllm/vllm.yaml) — checkpoint: [mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512), manually quantized to NVFP4 using [LLM Compressor](https://github.com/vllm-project/llm-compressor) with `transformers` v5 (one-shot calibration on [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), 128 samples, 8192 seq len) — [quantization script](src/misc/quantize_devstral2_123b_nvfp4.py)

served via:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True vllm serve /path/to/model --config /path/to/model/vllm.yaml --port 8000 -O3
```

### Log Files

| File | Content |
|------|---------|
| `b.log` | Single-line `bench-sweep` invocation command with all CLI args |
| `bench_sweep.log` | Full concatenated stdout from all benchmark runs (config dumps, progress, result tables) |

## Power: W250 vs W300 (Qwen3.5-397B-A17B)

Additionally see [notebooks/analysis.ipynb](notebooks/analysis.ipynb).

#### Peak Output Throughput (tok/s)

| Input Length | W250 | W300 | Delta |
|:------------:|:----:|:----:|:-----:|
| 2,048 | 1,041 @c72 | **1,124** @c64 | +7.9% |
| 4,096 | 865 @c96 | **908** @c64 | +5.0% |
| 8,192 | 616 @c80 | **649** @c72 | +5.3% |
| 16,384 | 366 @c48 | **387** @c48 | +5.8% |
| 32,768 | 205 @c32 | **212** @c32 | +3.5% |
| 65,536 | 98 @c16 | **102** @c16 | +4.5% |

#### Single-User Latency (concurrency=1, 2K input)

| Metric | W250 | W300 | Delta |
|--------|:----:|:----:|:-----:|
| TTFT p50 | 260 ms | **255 ms** | -1.9% |
| TPOT p50 | 11.5 ms | **11.3 ms** | -1.7% |
| Output throughput (mean) | 85.4 tok/s | **86.4 tok/s** | +1.2% |

#### Power Draw at Peak Throughput (2K input)

| Metric | W250 | W300 | Delta |
|--------|:----:|:----:|:-----:|
| Mean system power | **913 W** | 1,062 W | +16.4% |
| Peak system power | **1,030 W** | 1,207 W | +17.2% |
| Efficiency (tok/s/W) | **1.14** | 1.06 | -7.2% |

| | |
|---|---|
| ![Throughput & Power](notebooks/plots/w250_vs_w300_throughput_power.png) | ![Prefill vs Decode Power](notebooks/plots/prefill_vs_decode_power.png) |

<sub>Generated by [notebooks/analysis.ipynb](notebooks/analysis.ipynb)</sub>

---

# Disclaimer

This repo (code, README) was predominantly AI-generated using [cc](https://claude.com/claude-code) with opus-4.6 (max).
