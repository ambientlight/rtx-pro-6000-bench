# DeepSeek-V4-Flash reasoning bridge — thin FastAPI relay in front of sglang.
#
# Sits between LiteLLM and the sglang DSV4 container: forces thinking on,
# strips/separates reasoning, re-emits native DSML tool-calls as OpenAI
# tool_calls, and repairs streaming tool-call artifacts (multi-Write split,
# zero-arg dedup). Plain HTTP on :8010 — TLS is terminated upstream at LiteLLM.
#
# Local image (not pushed). Build from repo root's docker/ dir:
#   docker build -t ambientlight/dsv4-reasoning-bridge:latest docker/reasoning-bridge
#
# Run (talks to the dsv4 sglang container via the host gateway):
#   docker run -d --name dsv4-reasoning-bridge --add-host=host:host-gateway \
#     -e SGLANG_BASE_URL=http://host:8000/v1 -e BRIDGE_DEBUG=1 \
#     -p 8010:8010 ambientlight/dsv4-reasoning-bridge:latest
FROM python:3.12-slim

# curl: used by launch_all.sh's inline "wait for :8000/health" loop before exec.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Pinned to match the working host venv (dsv4-test).
RUN pip install --no-cache-dir \
      fastapi==0.136.3 \
      uvicorn==0.49.0 \
      httpx==0.28.1

# Bake the bridge script (single self-contained module, stdlib + the 3 deps above).
COPY deepseek_reasoning_bridge.py /app/deepseek_reasoning_bridge.py
WORKDIR /app

# Defaults (all env-overridable at `docker run`):
#   BRIDGE_HOST=0.0.0.0  BRIDGE_PORT=8010  SGLANG_BASE_URL  BRIDGE_DEBUG
# SGLANG_BASE_URL points at the sglang DSV4 server; inside a container that is
# the host gateway, not 127.0.0.1 — set it at run time (see header).
ENV BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=8010 \
    SGLANG_BASE_URL=http://host:8000/v1

EXPOSE 8010
CMD ["python", "deepseek_reasoning_bridge.py"]
