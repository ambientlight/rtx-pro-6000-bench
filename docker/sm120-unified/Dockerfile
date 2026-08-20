# LEGACY W4A4 unified image. The current benchmark-qualified DeepSeek-V4 W4A8
# release is ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a and is
# defined in ../sglang-main-sm120/. Do not publish this legacy build as :latest.
#
# Unified SM120 image — serves BOTH DeepSeek-V4-Flash AND MiniMax-M3 (TP=4)
# Pick the model at `docker run` via MODEL={dsv4|m3}
# Requires NVIDIA Container Toolkit + driver >= 595
#   docker run --rm --gpus all --ipc=host -p 8000:8000 -e MODEL=dsv4 \
#     -v /path/DeepSeek-V4-Flash:/model:ro \
#     ambientlight/sglang-sm120-mxfp4:2026.07.3-cu131-sm120a

ARG CUDA_BASE=nvidia/cuda:13.1.1-devel-ubuntu24.04
FROM ${CUDA_BASE}

LABEL org.opencontainers.image.title="sglang-sm120-mxfp4" \
      org.opencontainers.image.description="sglang + flashinfer serving image for native MXFP4 W4A4 MoE on RTX PRO 6000 Blackwell (SM120, TP=4). Serves DeepSeek-V4-Flash and MiniMax-M3." \
      org.opencontainers.image.source="https://github.com/ambientlight/rtx-pro-6000-bench"

# SHA-pinned unified forks (ambientlight).
ARG FLASHINFER_REF=ambientlight/sm120-mxfp4-unified
ARG FLASHINFER_SHA=eb68b4d865a497afd5d26ce02e60c4ab5075361e
ARG SGLANG_REF=feat/sm120-mxfp4-unified
ARG SGLANG_SHA=3e5151a1309fbe1759436a643071f2dd54d5fe28
ARG HMMA_REF=feat/hmma-tensor-core-sparse-decode
ARG HMMA_SHA=08bfa86aef9e7068c8ec5da3c208f4e6cea6db07

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="12.0a"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip git ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

RUN pip install --no-cache-dir torch==2.11.0+cu130 \
      --index-url https://download.pytorch.org/whl/cu130

# flashinfer (unified): SOURCE checkout on PYTHONPATH, not pip-installed. Its JIT
# resolves C++/CUDA via flashinfer/data/* symlinks that a bare checkout lacks, so we
# run build_backend._create_data_dir() to create them. Install requirements.txt (minus
# torch), then PIN nvidia-cutlass-dsl 4.5.2 — flashinfer pulls 4.6.0, whose CuTe-DSL JIT
# is incompatible with apache-tvm-ffi 0.1.9 (MoE compile dies at graph-capture with
# make_kwargs_wrapper() unexpected kwarg 'map_dataclass_to_tuple'). 4.5.2 is sglang's pin.
RUN pip install --no-cache-dir "flashinfer-cubin==0.6.12" \
 && git clone https://github.com/ambientlight/flashinfer.git /opt/flashinfer \
 && cd /opt/flashinfer \
 && git fetch --depth 1 origin ${FLASHINFER_SHA} && git checkout ${FLASHINFER_SHA} \
 && git submodule update --init --recursive \
 && grep -ivE '^\s*torch\s*$' requirements.txt > /tmp/fi-reqs.txt \
 && pip install --no-cache-dir -r /tmp/fi-reqs.txt \
 && pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2" \
 && python -c "import build_backend; build_backend._create_data_dir(); print('flashinfer data/ symlinks created')" \
 && test -e flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp

# sglang (unified: DSV4 + M3). Strip the optional gRPC-router Rust ext-module from
# pyproject (needs a Rust toolchain; not on the single-node TP=4 inference path), then
# editable-install pure-Python.
RUN pip install --no-cache-dir --no-build-isolation "transformers==5.8.1" \
 && git clone https://github.com/ambientlight/sglang.git /opt/sglang \
 && cd /opt/sglang \
 && git fetch --depth 1 origin ${SGLANG_SHA} && git checkout ${SGLANG_SHA} \
 && python3 -c "import re,pathlib; p=pathlib.Path('python/pyproject.toml'); s=p.read_text(); s2=re.sub(r'\n\[\[tool\.setuptools-rust\.ext-modules\]\]\n(?:[^\[\n].*\n)*','\n',s); assert s2!=s, 'setuptools-rust ext-modules stanza not found'; p.write_text(s2)" \
 && pip install --no-cache-dir --no-deps -e /opt/sglang/python

# sglang runtime deps (the editable install above was --no-deps). Install its declared
# deps EXCEPT what we provide or would break the stack: torch* (keep 2.11.0+cu130),
# flashinfer* (keep /opt/flashinfer), nvidia-cutlass-dsl (pinned 4.5.2 above), and heavy
# compiled kernels not on the MXFP4 path (flash-attn-4, sgl-kernel, quack, deep-gemm, ...).
# tilelang IS kept (0.1.8): load-bearing on DSV4 (deepseek_v4.py -> mhc.py imports it, else
# the model file fails to import and the registry silently drops it). torch constraint = pin.
RUN cd /opt/sglang \
 && python3 -c "import re,pathlib; s=pathlib.Path('python/pyproject.toml').read_text(); m=re.search(r'^dependencies = \[(.*?)^\]', s, re.S|re.M); reqs=re.findall(r'\"([^\"]+)\"', m.group(1)); [print(r.split(';')[0].strip()) for r in reqs if ';' not in r or 'platform_machine !=' in r]" \
      | grep -ivE '^(torch|torchvision|torchaudio|torchcodec|torchao)([=<>~!].*)?$' \
      | grep -ivE '^(flashinfer_python|flashinfer_cubin|flashinfer-python|flashinfer-cubin)(\[.*\])?([=<>~!].*)?$' \
      | grep -ivE '^(nvidia-cutlass-dsl)(\[.*\])?([=<>~!].*)?$' \
      | grep -ivE '^(flash-attn-4|sglang-kernel|sgl-kernel|quack-kernels|sgl-deep-gemm|tokenspeed_mla|torch_memory_saver)([=<>~!].*)?$' \
      > /tmp/sgl-reqs.txt \
 && echo "torch==2.11.0+cu130" > /tmp/torch-constraint.txt \
 && pip install --no-cache-dir -c /tmp/torch-constraint.txt -r /tmp/sgl-reqs.txt \
 && python -c "import torch; assert torch.__version__ == '2.11.0+cu130', torch.__version__; print('torch pin intact:', torch.__version__)"

# sgl-kernel 0.4.3, built from the vendored source (not on PyPI at 0.4.3; imported at
# boot). Builds the sm90+sm100 trees (527 targets); SM120 loads the sm100 tree, compiled
# with compute_120a under CUDA>=13. BOUNDED PARALLELISM: CMAKE_BUILD_PARALLEL_LEVEL x
# SGL_KERNEL_COMPILE_THREADS is the real fork count (upstream 128x32=4096-way OOM-kills
# co-tenants); capped to 16x4=64 under nice/ionice. TUs peak ~4-8 GB each.
ENV TORCH_CUDA_ARCH_LIST="12.0a"
ARG SGL_BUILD_JOBS=16
ARG SGL_NVCC_THREADS=4
RUN cd /opt/sglang/sgl-kernel \
 && pip install --no-cache-dir "scikit-build-core>=0.10" "cmake>=3.26" ninja \
 && CMAKE_BUILD_PARALLEL_LEVEL="${SGL_BUILD_JOBS}" \
    nice -n 19 ionice -c3 \
    pip install --no-cache-dir --no-build-isolation \
      --config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS="${SGL_NVCC_THREADS}" \
      . \
 && python -c "import importlib.metadata as m; assert m.version('sglang-kernel')=='0.4.3', m.version('sglang-kernel'); print('sglang-kernel 0.4.3 installed')" \
 && python -c "import importlib.util, os, glob; s=importlib.util.find_spec('sgl_kernel'); d=os.path.dirname(s.origin); so=glob.glob(d+'/sm*/common_ops*.so'); assert so, 'no compiled common_ops .so under '+d; print('sgl_kernel compiled .so present:', sorted(os.path.relpath(p,d) for p in so))"

# repo-flashinfer first on PYTHONPATH to override any transitively-pulled wheel.
ENV PYTHONPATH=/opt/flashinfer:/opt/sglang/python \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    FLASHINFER_WORKSPACE_BASE=/ficache

# HMMA sparse decode/prefill kernel (DSV4-only), compiled for sm_120a against the 13.1
# toolkit. Slow build (~minutes).
RUN git clone --depth 1 -b ${HMMA_REF} \
      https://github.com/ambientlight/deepseek-v4-flash-sm120.git /opt/hmma \
 && cd /opt/hmma && git fetch --depth 1 origin ${HMMA_SHA} && git checkout ${HMMA_SHA} \
 && pip install --no-cache-dir -e . --no-deps --no-build-isolation

# RTX PRO 6000 tuned W8A8 + MoE configs (else "Using default W8A8 sub-optimal").
RUN cp /opt/hmma/tuned-configs/w8a8/*.json /opt/sglang/python/sglang/srt/layers/quantization/configs/ \
 && cp /opt/hmma/tuned-configs/moe/*.json  /opt/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/

RUN mkdir -p /ficache

# Verify both models' paths resolve — fails the build early on any drift.
RUN python -c "from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import sm120_moe_supported_quant_modes as f; assert 'mxfp4' in f(); print('flashinfer mxfp4 OK')" \
 && python -c "import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as m; assert hasattr(m,'_STATIC_DYNAMIC_M'); print('dynamic-M present')" \
 && python -c "from sglang.srt.layers.quantization.mxfp4_w4a4_moe import Mxfp4W4A4MoEMethod, Mxfp4W4A4MoEScheme; print('sglang MXFP4 MoE (both DSV4 method + M3 scheme) OK')" \
 && python -c "from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsW8A8Mxfp8; print('sglang M3 MXFP8 linear OK')" \
 && python -c "from deepseek_v4_kernel.ops import sparse_decode_fwd, sparse_prefill_fwd; print('DSV4 HMMA kernels OK')" \
 && python -c "import importlib; [importlib.import_module('sglang.srt.models.'+m) for m in ('deepseek_v4','minimax_m3_vl')]; print('model files import cleanly (deps present)')" \
 && python -c "from sglang.srt.models.registry import ModelRegistry as R; a=set(R.get_supported_archs()); need={'DeepseekV4ForCausalLM','MiniMaxM3SparseForConditionalGeneration'}; miss=need-a; assert not miss, 'UNREGISTERED (registry swallowed a model import -> would fall to generic AutoModel loader): '+str(miss); print('both model archs registered:', sorted(need))"

COPY entrypoint.sh entrypoint-dsv4.sh entrypoint-m3.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/entrypoint-dsv4.sh /usr/local/bin/entrypoint-m3.sh

EXPOSE 8000
# Dispatch on MODEL={dsv4|m3}. Default m3. Override MODEL_DIR + any server flag via -e.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
