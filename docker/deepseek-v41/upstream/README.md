# DeepSeek V4.1 Flash on upstream SGLang and SM120

This is a separately launchable candidate for the four RTX PRO 6000s. Its first parent before local changes is the November 23 official upstream commit `c64f2256d58358ff65c53f0e38dc3e6bcfac8484`. The local branch is `sm120/production-reconstruction`. The bundle and source hash below restore it from GitHub into an empty directory. [Official source](https://github.com/sgl-project/sglang/commit/c64f2256d58358ff65c53f0e38dc3e6bcfac8484) · [Release lock](release.json)

## What the candidate runs

| Deployment behavior | Implementation |
|---|---|
| 1017 checkpoint, 524,288 context, TP4/EP4, 0.87 static memory fraction, 2,048-token prefill chunks, interleave interval 4, eight running requests, DSPARK block 5 / verify 6 | Offline checkpoint receipt and upstream’s resolved loading, scheduling, native DeepSeek V4 attention and FlashInfer MXFP4 MoE options. Standard mode keeps decode CUDA graphs enabled. [Launcher](boot.py) · [Resolved baseline](receipts/profile.json) |
| Engram on host, zero host KV cache budget | Native CPU placement, TP-sharded tables and all-reduce lookup. The old weight-loader monkey patch and host-cache assembler overlays drop out of the active deployment. Both checkpoint tables are covered by a sampled hardware gather test. [Engram receipt](receipts/sm120-oci-gpu0-engram.json) |
| SM120 sparse attention and index selection | Upstream includes the C1/C2 FP4 split-128 selection fix. FlashInfer 0.7.4 supplies native SM120f sparse MLA. A six-case test exercises 16 local heads with 128- and 256-token cache pages, real forward kernels and changed-input graph replay. Keep the CUDA development headers and writable kernel cache for first-use JITs. [Attention receipt](receipts/sm120-oci-gpu0-attention.json) |
| OpenAI Responses and Anthropic compatibility | Ports tool containment, bounded recovery, progress events and previous-turn thinking signatures around the native DeepSeek recipe. The candidate uses the deployed model alias and the same read-only key. Upstream also allows enabled Anthropic thinking without a supplied budget. [API test receipt](receipts/serving-api.json) |
| Logging and crash diagnostics | Includes the content-free journal publisher, request-path guards, diagnostic entry point, per-start NCCL paths and bounded crash handling. Compose preserves the host cgroup slice, memlock, core limit and loopback binding. Raw request dumps are disabled. [Environment](environment.json) · [Compose](compose.yaml) |
| Pearl | Request admission parks warm direct workers through the host broker and resumes work after the last serving rank’s device fence. The optional public-workload adapter runs a calibrated integer projection for target layer 0 `attn.wo_b` during eager prefill, verify and decode. Draft, padded and sequence-sharded inputs use their ordinary projection. [Resolved eager profile](receipts/profile-pearl.json) · [Native proof receipt](receipts/sm120-projection.json) |

The candidate uses the deployed RAM Engram profile. Historical NVMe Engram mode and host L2 KV caching are outside this profile. The external broker and sliced idle workers are separate processes that stay resident across requests. Proof openings disclose quantized rows. Use synthetic or otherwise public inputs for the Pearl canary. The default Compose service runs stock checkpoint math.

## Restore and build

Run these from the bench repository. The source builder requires git, Python 3.11+, a Rust toolchain, CMake, NASM, the CPython 3.12 interpreter for the PyO3 ABI, the pinned Pearl build artifacts described in the operator guide, and user-space `crane` and `umoci`. The registry digest selects the official linux/amd64 SGLang development image with CUDA 13.0, PyTorch 2.14.1 and the SM120f FlashInfer cache.

```bash
u="$PWD/docker/deepseek-v41/upstream"
out="$(mktemp -d)"
"$u/restore-source.sh" "$out/sglang"
src="$out/sglang"
base=sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213
crane pull --format=oci "docker.io/lmsysorg/sglang@$base" "$out/base"
rev=$(git -C "$src" rev-parse --short=12 HEAD)
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev1+g$rev" \
    uv build "$src/python" --python 3.12 --wheel --out-dir "$out/wheels"
PYO3_PYTHON=$(command -v python3.12) cargo rustc \
    --manifest-path "$src/rust/sglang-server/Cargo.toml" --locked --release \
    --lib --features pyo3/extension-module --crate-type cdylib
operator="$PWD/work/pearl-inference-20261123/prototype/operator/gpu_prepare"
"$operator/build.sh" "$out/pearl"
python3 "$u/build_image.py" --source "$src" --base "$out/base" \
    --output "$out/candidate" --libs "$out/pearl" --wheel "$out"/wheels/*.whl \
    --rust-server "$src/rust/target/release/libsglang_server.so"
umoci stat --image "$out/candidate:dsv41-upstream-sm120"
```

The restored checkout reproduced the candidate Git tree. After unpacking all 72 OCI layers, the installed SGLang and PyO3 server imported, all three Pearl shared libraries loaded, and the content-free journal entry point verified the checkpoint with Linux dump prevention active. [Image receipt](receipts/assembled-image.json)

The image builder layers the tracked source changes onto the official development image, installs the candidate's Python distribution metadata, and replaces the changed native server extension. The Python serving frontend is selected. The three other packaged Rust extensions retain their official binaries. [Builder](build_image.py) · [Pearl build inputs](../../../work/pearl-inference-20261123/prototype/operator/gpu_prepare/README.md)

## Candidate launch

The stock candidate listens on `127.0.0.1:18000`. The four GPUs and sufficient host RAM for the approximately 189 GiB Engram allocation must be available before launching the full model. Its checkpoint and signing key are read-only, and it has separate state and kernel-cache volumes. The launcher verifies the pinned weights and runs arithmetic, structured-output, tool-continuation and vision smokes on startup.

```bash
skopeo copy "oci:$out/candidate:dsv41-upstream-sm120" docker-daemon:dsv41-upstream-sm120:local
docker compose -f "$u/compose.yaml" up -d
```

For the eager public-input canary, start the watched, sliced workers and broker as described in the [prototype instructions](../../../work/pearl-inference-20261123/prototype/README.md). Set `PEARL_CONTROL_DIR` to its private directory containing `broker.sock`, and add `-f "$u/compose.pearl.yaml"` to the Compose invocation. Optionally set `PEARL_REPLAY_POOL_PORT` when a loopback Stratum test peer is listening **in the container's network namespace**. Without it the projection runs the fixed public job. Keep the proof-enabled port for public inputs only.

## Full-checkpoint benchmark

The four-GPU 1017 benchmark covers uncached prefill up to 524,177 tokens, sustained generation, concurrent requests, prefix reuse, and handoff to warm direct workers. The proof-enabled runs used public inputs and a private verifier. External pool credit and broad model quality still need independent evaluation. [Performance comparison](full-canary/performance.md) · [Benchmark configuration](full-canary/full-run.json)

## Earlier component measurements

On upstream, 39 independently verified submissions came from prefill, six-row target verification and one-row decode on GPU 3. The test loads the real FP8 checkpoint shard through the native weight loader. Ten alternating local stock/mining pairs under shared-GPU load measured 3.085/4.170 ms at 2,048 rows, 2.425/2.455 ms at six rows and 2.653/2.474 ms at one. All 117,760 sampled BF16 outputs matched the calibrated integer oracle. Relative L2 error against native FP8 was 3.36–3.69% on these activations. [Projection fixture](check_projection.py) · [Measurements](receipts/sm120-projection.json)

One public HTTP request to a two-layer DeepSeek TP1 canary on GPU 3 produced two accepted v3 witnesses. The sixteen-token prefill used the real 5,120 × 2,048 checkpoint projection shard, and the singleton decode used the same shard. The independent receiver checked all 163,840 disclosed weight bytes. The other parameters used native dummy initialization. The private job receiver ran locally. [HTTP proof fixture](check_http_projection.py) · [Request and proof receipt](receipts/sm120-http-projection.json)

A separate two-layer experiment put a real sliced direct worker and the serving process on GPU 2 together. All 48 warmed requests paused the worker, kept its cursor stationary through generation, then resumed it. HTTP send to observed drain took 4.07 ms median, 7.22 ms p95 and 17.90 ms maximum. Release to observed GPU execution took 2.11 ms median, 2.23 ms p95 and 2.25 ms maximum. Across 32 randomized pairs the within-pair increase in time to first streamed token was 0.34 ms median and 10.96 ms p95. Both experiments exercise real HTTP paths and use offline jobs. [Same-card measurements](receipts/sm120-http-same-gpu.json) · [Separate-device control](receipts/sm120-http-deepseek-handoff.json)

A small deterministic model served known output over both the OpenAI and native HTTP endpoints on GPU 2. The separate historical four-worker handoff test measured warm p99 4.98 ms to drain and 2.83 ms to resume. Those measurements apply to their tested workloads. [HTTP receipt](receipts/sm120-http-llama.json) · [Four-worker timings](../../../work/pearl-inference-20261123/prototype/miner/measurements.json)
