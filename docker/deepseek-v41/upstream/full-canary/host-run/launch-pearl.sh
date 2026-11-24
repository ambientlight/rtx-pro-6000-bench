#!/usr/bin/env bash
set -eu
d=/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary
install -d -m 700 "$d/pearl-shapes" "$d/pearl"
systemd-run --user --unit=dsv41-sm120-full-candidate --collect --property=KillMode=control-group --property=MemoryMax=365G --property=MemorySwapMax=0 --property=TimeoutStopSec=70 --property=StandardOutput=append:"$d/logs/pearl.log" --property=StandardError=append:"$d/logs/pearl.log" --setenv=CANARY_STATE="$d/pearl" --setenv=CANARY_PORT=18000 --setenv=CANARY_REQUIRE_API_KEY=1 --setenv=PEARL_BROKER_SOCKET="/run/user/1000/dsv41-sm120-full-miners/broker.sock" --setenv=PEARL_SHAPE_DIR="$d/pearl-shapes" --setenv=PEARL_REPLAY_POOL_PORT=31988 --setenv=PEARL_PUBLIC_REPLAY=1 --setenv=PEARL_GPU_PREPARE=1 --setenv=PEARL_WOB_MODULE=checkpoint_projection "$d/env-run" run
