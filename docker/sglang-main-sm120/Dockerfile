# Reproducible SGLang-main control image for the DSV4 SM120 A/B.
#
# The base is the official amd64 dev image built from SGLang c0b6474b on
# 2026-08-18.  The source and the three dependency pins that changed before the
# comparison cutoff are then advanced to the exact main commit below.
ARG BASE_IMAGE=lmsysorg/sglang@sha256:4acf360a84b87d1f138649b390f7c34f28cc8ba360ef8f2a599c8fa6453f26c7
FROM ${BASE_IMAGE}

ARG SGLANG_SHA=b03ac355e795b3a86b26b8732c47c0965fd71bbc
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git

LABEL org.opencontainers.image.title="sglang-main-dsv4-sm120-control" \
      org.opencontainers.image.description="Pinned SGLang-main native SM120 control for DeepSeek-V4-Flash-0731" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_SHA}" \
      ai.sglang.comparison.role="native-main-control" \
      ai.sglang.base.digest="sha256:4acf360a84b87d1f138649b390f7c34f28cc8ba360ef8f2a599c8fa6453f26c7"

# c0b6474b -> b03ac355 changed these runtime pins.  Updating them avoids a
# misleading source-only rebase while retaining the official prebuilt CUDA,
# torch, FlashInfer, and sglang-kernel stack from the nearby nightly image.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
      sgl-deep-ep==0.1.2 \
      tilelang==0.1.12 \
      nvidia-nccl-cu13==2.30.7

RUN git init /opt/sglang-main \
 && cd /opt/sglang-main \
 && git remote add origin "${SGLANG_REPO}" \
 && git fetch --depth 1 origin "${SGLANG_SHA}" \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${SGLANG_SHA}" \
 && python3 -m pip install --no-cache-dir --no-deps -e /opt/sglang-main/python

ENV PYTHONPATH=/opt/sglang-main/python \
    SGLANG_BUILD_COMMIT=b03ac355e795b3a86b26b8732c47c0965fd71bbc \
    SGLANG_IMAGE_TAG=ambientlight/sglang-main-sm120:b03ac355-cu130-candidate

# Fail the build if the source overlay or the key native-main SM120 paths drift.
RUN python3 -c "import pathlib,sglang; p=pathlib.Path(sglang.__file__).resolve(); assert str(p).startswith('/opt/sglang-main/python/'), p; print('sglang source:', p)" \
 && python3 -c "import importlib.metadata as m; expected={'flashinfer-python':'0.6.17','sglang-kernel':'0.4.6.post1','tilelang':'0.1.12','sgl-deep-ep':'0.1.2'}; actual={k:m.version(k) for k in expected}; assert actual==expected, (actual,expected); print('runtime pins:', actual)" \
 && python3 -c "from sglang.kernels.ops.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120; from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config; print('native SM120 attention + MXFP4 imports OK')" \
 && python3 -c "from sglang.srt.parser.reasoning_parser import DeepSeekV4Detector; from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector as ToolDetector; print('DSV4 reasoning + tool parsers import OK')"

COPY entrypoint-dsv4.sh /usr/local/bin/entrypoint-dsv4-main-sm120.sh
RUN chmod +x /usr/local/bin/entrypoint-dsv4-main-sm120.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint-dsv4-main-sm120.sh"]
