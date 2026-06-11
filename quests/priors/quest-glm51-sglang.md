# Quest: GLM-5.1-478B-A42B-REAP-NVFP4 on 4× RTX PRO 6000 Blackwell

## Summary

### Human Timeline
~2 hours (mostly waiting for weight loading + CUDA graph capture).

### Complexity
Medium — the model itself loads fine, but engine compatibility required investigation.

### Goal
Serve GLM-5.1-478B-A42B-REAP-NVFP4 for benchmark sweeps on 4× RTX PRO 6000 Blackwell (sm_120, 96 GB each).

### Result
✅ Running via **sglang 0.5.10.post1** on `http://localhost:8000` (OpenAI-compatible API). vllm is not viable for this model+hardware combo.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| VRAM | 96 GB per GPU (384 GB total) |
| Compute | sm_120 |
| Topology | PCIe (workstation, not NVLink) |

## Model

| Field | Value |
|-------|-------|
| HF repo | `0xSero/GLM-5.1-478B-A42B-REAP-NVFP4` |
| Architecture | `GlmMoeDsaForCausalLM` (DeepSeek V3-style MLA + MoE + DSA) |
| Total params | 478.4B |
| Activated/token | ~42.7B |
| Quantization | NVFP4 (ModelOpt, group_size=16) |
| On-disk | ~285 GB (85 safetensor shards) |
| Local path | `/mnt/hot/ambientlight/models/glm-51-478b-a42b-reap-nvfp4/` |

## Why vllm Does Not Work

Attempted vllm 0.19.2rc1 (dev build from source at `/mnt/hot/ambientlight/repos/vllm/`). Two blockers:

### Blocker 1: DeepGEMM for Sparse Attention Indexer
```
RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.
```
- GLM-5.1 has `index_topk: 2048` in config → vllm creates DSA `Indexer` → requires DeepGEMM.
- **Fix**: installed DeepGEMM 2.5.0 from source (`/tmp/DeepGEMM`, `--no-build-isolation`). This resolved the import error.

### Blocker 2: No MLA+Sparse Attention Backend for sm_120
```
ValueError: No valid attention backend found for cuda with
  AttentionSelectorConfig(use_mla=True, use_sparse=True, ...)
```
Every MLA backend rejects the combination:
- `FLASH_ATTN_MLA`: compute capability not supported, sparse not supported
- `FLASHMLA`: compute capability not supported, sparse not supported
- `FLASHINFER_MLA`: compute capability not supported, sparse not supported
- `TRITON_MLA`: sparse not supported
- `FLASHMLA_SPARSE`: kv_cache_dtype not supported, compute capability not supported

**Root cause**: vllm has no attention backend that supports MLA + sparse (DSA) on Blackwell sm_120. This is a fundamental gap — not patchable without writing a new attention backend.

### Alternatives Considered but Rejected
- **Patching config to remove `index_topk`** (disable DSA): unknown quality impact, still wouldn't fix the MLA compute capability issues
- **GLM-5.1-555B-A14B-REAP-GPTQ-W4A16**: vllm-compatible variant (~297 GB, fits on 4×96 GB), but different model — would need separate download
- **GLM-5.1-555B-A14B-REAP BF16**: vllm-compatible but needs ~1125 GB VRAM (8× H200)

## sglang Setup (What Works)

### Venv
```
~/.venvs/sglang-cu130/
```

### Pinned Stack
| Package | Version |
|---------|---------|
| sglang | 0.5.10.post1 |
| torch | 2.9.1 |
| flashinfer-python | 0.6.7.post3 |
| flashinfer-cubin | 0.6.7.post3 |
| transformers | 5.3.0 |
| triton | 3.5.1 |

### Required Patch: Disable NSA for GLM on sm_120

File: `~/.venvs/sglang-cu130/lib/python3.12/site-packages/sglang/srt/configs/model_config.py`

Remove `"GlmMoeDsaForCausalLM"` from `is_deepseek_nsa()` architectures list (line 75). This forces triton attention instead of NSA kernels, which don't have sm_120 support.

Without this patch, sglang routes GLM-5.1 through NSA code paths that are incompatible with Blackwell workstation GPUs.

### Launch Script
```
/mnt/hot/ambientlight/models/glm-51-478b-a42b-reap-nvfp4/launch.sh
```

Key flags:
```bash
--quantization      modelopt_fp4
--kv-cache-dtype    fp8_e4m3
--tensor-parallel-size 4
--context-length    32768        # can push to 202752
--mem-fraction-static 0.94
--triton-attention-num-kv-splits 64
--moe-runner-backend cutlass
--fp4-gemm-backend  flashinfer_cudnn
--json-model-override-args '{"index_topk_pattern": "FFSFSSSFSS..."}'  # IndexCache
```

Essential env vars (Blackwell workstation NCCL tuning):
```bash
SGLANG_ENABLE_JIT_DEEPGEMM=0    # no sm_120 DeepGEMM kernels
SGLANG_ENABLE_DEEP_GEMM=0
SGLANG_DISABLE_DEEP_GEMM=1
NCCL_IB_DISABLE=1               # PCIe workstation, no InfiniBand
NCCL_P2P_LEVEL=PIX              # PCIe peer-to-peer
```

### Runtime Stats

| Metric | Value |
|--------|-------|
| Weight load time | ~98s |
| Weights per rank | 77.15 GB |
| KV cache per rank | 11.26 GB (fp8_e4m3, 269k tokens) |
| Free VRAM per rank | ~5.3 GB |
| CUDA graph capture | ~142s (bs=1 only) |
| Attention backend | triton (forced by NSA patch) |
| Total startup | ~4.5 min |

### Serving

OpenAI-compatible API at `http://localhost:8000`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-478B-A42B-REAP-NVFP4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.5,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_tokens": 512
  }'
```

Recommended sampling (from HF card): `temperature=0.5, top_p=0.95, frequency_penalty=0.3, repetition_penalty=1.05`.

## Notes for Benchmarking

1. **Engine**: sglang, NOT vllm. Benchmark harness needs to point at the sglang OpenAI API.
2. **Concurrency**: `--max-running-requests 1` in launch script. Increase for throughput benchmarks but watch VRAM (~5 GB headroom).
3. **Context**: Currently 32k. Can push to 202k but needs `--page-size 128` and `--kv-cache-dtype fp8_e4m3` (already set).
4. **IndexCache**: Enabled via `index_topk_pattern`. Saves ~70% indexer time at long contexts. Remove `--json-model-override-args` flag to disable for ablation.
5. **MTP/Speculative decode**: Available but reduces max context to ~65k and requires `--kv-cache-dtype auto` (bf16). Roughly doubles short-prompt tok/s.

## Hard Rules

1. Always use sglang venv at `~/.venvs/sglang-cu130/`
2. Always use model weights at `/mnt/hot/ambientlight/models/glm-51-478b-a42b-reap-nvfp4/`
3. Launch via `launch.sh` in the model directory
4. Do not attempt vllm — it fundamentally lacks MLA+sparse attention support on sm_120
