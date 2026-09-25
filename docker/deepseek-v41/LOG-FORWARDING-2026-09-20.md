# L16: binary-safe, supervised launcher logs

Status at preparation: **armed, waiting for 60 continuous idle seconds**. The
currently serving L15 container is not hot-patched. The authoritative ongoing
status is:

`/mnt/hot/dsv41_state/log-forwarding-rollout-XHq1vglm/status.json`

User service:
`dsv41-log-forwarding-relaunch@log-forwarding-rollout-XHq1vglm.service`

## Scope

Only `/opt/dsv41/boot.py` and deployment label
`dsv41-v2-log-forwarding-v1` change. The existing
`ambientlight/dsv41-sm120:cache-attribution-v1` image is pinned to immutable ID
`sha256:fcd6ae8574240c6f0df3c6277b477234328a028007ef7e97acdcb9ed1db1349c`.
There is no rebuild or change to the SGLang API, kernels, weights or cache code.

Unchanged profile: TP4/EP4, context 524288, static fraction 0.87, concurrency 8,
chunked prefill 2048, prefill/decode interval 4, DSPARK 5, automatic device KV,
RAM Engram offload, 256 GB host HiCache, **67% SWA / 33% FULL**, no NVMe KV tier,
port 8000, model alias `deepseek-v4-flash`, no backend API key, and
`restart: unless-stopped`. Expected host capacity remains 12,570,368 FULL tokens
and 1,656,064 SWA slots per rank (subject to the unchanged planner's guards).

The launcher forwards bounded binary chunks instead of decoding shared child
stdout as strict UTF-8. Key redaction spans chunk boundaries. A failed reader
cannot leave a silently unread pipe: failure stops the child process group,
with bounded TERM/KILL handling and a failing launcher exit. The implementation
guide is `sglang-dsv41-production-overlay/deployment/LOG_FORWARDING.md`.

## Tests and evidence

- Full API/launcher qualification: **1,203 tests and 622 subtests passed** in the
  current image with the source overlay and no GPU access.
- Frozen boot qualification: **31 tests passed** with no network or GPU access.
  These include 17 dedicated log-forwarding regressions and 14 policy tests.
- Deployment/load/monitor qualification: **84 tests passed**.
- Compose validation and whitespace/diff checks passed.
- The host-only first test run lacked Pillow for the existing vision fixture;
  the deployment-image runs include Pillow and pass the entire suite.

Full API evidence: `/mnt/hot/dsv41_state/api-tests-DDgrRTby/`.
Frozen qualification, manifests, source hashes, candidate and private rollback:
`/mnt/hot/dsv41_state/log-forwarding-rollout-XHq1vglm/`.

The isolated reproduction of the old reader yielded the real exception,
65,536 unread pipe bytes and a child blocked in `anon_pipe_write`. The updated
reader completes the same malformed-byte / million-byte output case. Tests also
cover Unicode splitting, concurrent writers, redaction, large lines, short and
failed writes, unexpected EOF, ordinary/nonzero exits and ignored SIGTERM.

## Rollout safety and activation

The watcher reuses the tested idle detector (`/v1/loads` plus HTTP active and
arrival counters). Any new inference, active stream, pending prefill or queued
work resets the minute. Missing/stale telemetry never permits recreation.
Container identity, source fingerprints, image identity and RAM headroom are
checked before the final idle observation. Only one recreation is allowed.
There is no automatic retry or rollback that could interrupt newly arrived work.

After recreation it retains the normal arithmetic, JSON, tool round-trip and
vision startup checks, then verifies health, exact runtime hashes, environment,
four-rank host allocation, diagnostics, active capture and cache monitor. No
additional throughput or cache-pressure request is sent. A successful rollout
creates `complete.json` and an L16 `launch-spec.json`; until those exist, do not
report the candidate as deployed.

Rollback uses the preserved `rollback.compose.json`, **only after another
user-authorized drain**. It restores the old launcher, including its known bug.
The frozen L15 source files and captures were not overwritten.
