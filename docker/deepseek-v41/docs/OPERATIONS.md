# Deployment operations

Commands below run from the benchmark repository root. The
[deployment README](../README.md) summarizes the checked-in configuration;
dated launch identities and measurements belong in [history](history/README.md),
not in this runbook.

## Prerequisites and ownership

- Four 96 GB RTX PRO 6000 GPUs, NVIDIA Container Toolkit, Docker Compose,
  systemd cgroup v2 and Docker's systemd cgroup driver.
- Pinned weights in `~/models/DeepSeek-V4.1-Flash`
  (`/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash` on this host).
- The reviewed SGLang worktree at
  `/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay`. Compose mounts
  the launcher, host-cache sizing code and Responses adapter from it.
- Private state in `/mnt/hot/dsv41_state/production`, captures in
  `/mnt/hot/dsv41_dumps`, and the existing V12 thinking-signature key at
  `/mnt/hot/dsv4_state/anthropic-thinking.key`. Preserve that key for old history;
  never commit it or dump full container environment/inspection into public logs.
- The external volume `dsv41-sero-baseline_kernel-cache` and installed
  [no-swap slice](NO-SWAP.md). Engram RAM mode also needs `IPC_LOCK` and unlimited
  `memlock`, already set in Compose.

RAM startup checks require the Engram backing-file sizes, the full host-cache
budget and another 24 GiB available: approximately **451.5 GiB** for this profile.
This is a startup check, not a reservation against other applications. No-swap
does not prevent a host OOM.

Only V4.1 is enabled for GPU inference. Embeddings, V12 and canary containers
must remain stopped while it owns the GPUs. Backend port 8000 has no API key;
keep it on a trusted network. LiteLLM frontend authentication is separate.

## Launch and restart

1. Compare the intended profile with the running deployment's frozen Compose,
   image ID and source mounts. `baseline.lock.json` contains dated receipts;
   it is not a live inventory. In particular, the September 25 preserved
   release predates the Responses-progress source override.
2. For a scheduled change, require **60 continuous seconds** of empty scheduler
   queues and no HTTP inference activity. Use fresh `/v1/loads` and `/metrics`;
   zero running requests alone is insufficient during chunked prefill.
   The old change-specific rollout jobs have been retired. Coordinate the idle
   window explicitly; the launcher itself does not enforce it.
3. Immediately recheck idle, then use the authorized launch command:

   ```bash
   docker compose -f docker/deepseek-v41/compose.api.yaml config --quiet
   ./launch_all.sh dsv41
   ```

4. Let the built-in arithmetic, JSON, tool-round-trip and vision checks finish.
   Check health, effective flags, swap policy and that capture/monitoring followed
   the new launch.

The explicit launcher target **force-recreates** the service and does not
perform the idle wait. Polling is not an admission fence: new traffic can arrive
between the last check and shutdown. A plain `docker restart dsv41` cannot pick
up Compose changes. With no target, `launch_all.sh` handles LiteLLM and V4.1,
skipping healthy services unless `--restart` is supplied.

## Read-only status and capture

```bash
docker inspect dsv41 --format '{{.State.Status}} {{.State.Health.Status}} restarts={{.RestartCount}} started={{.State.StartedAt}}'
curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8000/health
systemctl --user status dsv41-wire-dump.service dsv41-cache-monitor.service
readlink -f /mnt/hot/dsv41_dumps/current
jq . /mnt/hot/dsv41_state/cache-monitor/status.json
```

The wire recorder runs as a host user-systemd service and attaches inside the
container network namespace. `interface=any` captures both host-originated
traffic and in-container startup checks. Each new launch gets a private capture
directory containing HTTP PCAP, native logs and recorder identity/status.
Collection waits through container stops and reboots; it does not live in the
model image.

Captures can contain full prompts, responses and credentials. Keep them private.
When analyzing PCAP, use port 8000 and separate the container-IP traffic from
`127.0.0.1` startup checks. Check recorder drop counts before calling a sample
complete. See [cache telemetry](CACHE-MONITOR.md) and [crash diagnostics](DIAGNOSTICS.md)
for their separate locations and interpretation.

## Builds and recovery

[baseline.lock.json](../baseline.lock.json) pins the 0xSero recipe, model revision,
image identities and source provenance. The original base image's SGLang Git
revision was unknown; do not substitute an arbitrary upstream tree or upgrade
kernel dependencies while reproducing that baseline.

`scripts/dsv41_build_api_image.py` builds [Dockerfile.api](../Dockerfile.api)
from a clean, committed overlay worktree, checks its explicit source allow-list
and verifies packaged files and unchanged hardware-runtime fingerprints.
Historical incremental image builders and their Dockerfiles have been retired;
the [history index](history/README.md) identifies the Git revision retaining them.

For an exact restore, follow the private
[September 25 release archive](history/RELEASE-2026-09-25.md), noting that it
predates Responses progress. Weights, warm KV, captures and host drivers are not
part of that archive. `scripts/dsv41_snapshot_release.py` captures settings,
source mounts and private state references; saving the image and Git bundles
are separate steps.

V12 rollback requires its retained image/container **and re-downloaded 0731
weights**; those weights were intentionally removed from this workstation.
`launch_all.sh dsv4` launches V4.1 and must not be used for V12 restoration.

## Maintained scripts

`scripts/` retains capture/monitoring, the load utility, the full image builder,
release snapshots, API acceptance, the encoder/grammar checks and the CPU-only
API-regression runner for the separate SGLang fork. The shared monitoring
utilities live in `dsv41_load.py`; no retired rollout module is imported.

Local unit tests, one-shot migrations and the standalone TPS/cache-pressure
probes have been removed from this checkout and remain recoverable from Git
revision `dc74842`. Benchmark reports and deployment evidence are preserved.
The SGLang fork's own API tests are unchanged.

`dsv41_live_acceptance.py` sends inference requests, including its optional long
suite. Run it only as part of an explicitly requested qualification.
