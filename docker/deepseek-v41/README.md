# DeepSeek-V4.1-Flash on 4× RTX PRO 6000

Production deployment files for the 0xSero-based SM120 runtime, with our
Responses/Messages API fixes, SWA-boundary retention, cache telemetry and crash
diagnostics. The public model alias remains `deepseek-v4-flash`.

This page describes the checked-in profile, not a live health report.
[Operations](docs/OPERATIONS.md) covers launch, verification and recovery;
[benchmark results](../../bench/deepseek-v4.1-flash_TP4_sglang/README.md) live under `bench/`.

## Configuration

[compose.api.yaml](compose.api.yaml) defines the next launch. Running containers
may use frozen Compose/source snapshots; a Git edit does not update them.

| Setting | Configured value |
|---|---|
| Container / port / model alias | `dsv41` / `8000` / `deepseek-v4-flash` |
| Image | `ambientlight/dsv41-sm120:cache-attribution-v1` plus explicit source mounts |
| Deployment label | `dsv41-v2-responses-progress-v1` |
| Parallelism / speculation | TP4 / EP4 / DSPARK block 5 |
| Context / prefill chunk / decode interval | 524,288 / 2,048 / 4 |
| Static memory fraction / max running requests | 0.87 / 8 |
| GPU KV capacity / expandable segments | Automatic / disabled |
| Engram | RAM offload; `DSV41_CACHE_GIB=0` |
| RAM HiCache | 256 GB host-total; 67% SWA / 33% FULL tensor payload |
| Sampling defaults | Temperature 1.0; top-p 0.95; output budget 32,768 |
| Reasoning default | `high`; explicit caller settings take precedence |
| Restart / host memory policy | `unless-stopped` / `dsv41.slice`, no swap |
| Backend authentication | Disabled for compatibility; trusted-network access only |

The cache split is by **bytes**, not slots, and does not control GPU pools.
Automatic GPU capacity depends on memory available at startup. The context
limit is not a guarantee of eight simultaneous full-window requests.

## Files and guides

Only build/launch inputs and this README live at the top level.

| Path | Purpose |
|---|---|
| [compose.api.yaml](compose.api.yaml) | Production settings and explicit runtime mounts |
| [Dockerfile.api](Dockerfile.api) | Full API/cache/diagnostic overlay build |
| [diagnostic_entrypoint.py](diagnostic_entrypoint.py) | Per-start CUDA/NCCL evidence directories |
| [baseline.lock.json](baseline.lock.json) | Recipe/model/image pins and dated rollout receipts |
| [compose.baseline.yaml](compose.baseline.yaml) | Original authenticated, localhost-only port-8010 baseline |
| [Operations](docs/OPERATIONS.md) | Prerequisites, launch, capture, builds and recovery |
| [Cache](docs/HICACHE.md) | Engram vs HiCache, FULL/SWA retention and budget semantics |
| [Cache monitoring](docs/CACHE-MONITOR.md) | Occupancy, eviction attribution and reuse counters |
| [Diagnostics](docs/DIAGNOSTICS.md) | Crash artifacts, time bounds and safe inspection |
| [No-swap policy](docs/NO-SWAP.md) | Host slice setup and effective-limit checks |
| [History](docs/history/README.md) | Dated investigations, rollout records and preserved release |

## Launch

Run from the repository root, **only for an authorized deployment** after
checking prerequisites and draining traffic:

```bash
docker compose -f docker/deepseek-v41/compose.api.yaml config --quiet
./launch_all.sh dsv41
```

The explicit `dsv41` target recreates the container; it does not wait for idle.
See [safe rollout](docs/OPERATIONS.md#launch-and-restart) before running it.
A plain `docker restart` retains the existing image, environment and mounts.
The old `launch_all.sh dsv4` target is an alias for V4.1, not a V12 rollback.
