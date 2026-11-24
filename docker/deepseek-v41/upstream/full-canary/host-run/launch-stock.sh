#!/usr/bin/env bash
set -eu
d=/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary
systemd-run --user --unit=dsv41-sm120-full-candidate --collect --property=KillMode=control-group --property=MemoryMax=365G --property=MemorySwapMax=0 --property=TimeoutStopSec=70 --property=StandardOutput=append:"$d/logs/stock.log" --property=StandardError=append:"$d/logs/stock.log" --setenv=CANARY_STATE="$d/stock" --setenv=CANARY_PORT=18000 "$d/env-run" run
