#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?supply a fresh path for the observer JSONL}
[[ ! -e "$out" ]]
systemd-run --user --collect --unit=dsv41-pearl-graph275-observer \
    -p KillMode=control-group -p MemorySwapMax=0 -p UMask=0077 -p TimeoutStopSec=6 \
    /usr/bin/python3 "$here/observe.py" "$out"
for _ in {1..30}; do
    if [[ -s "$out" ]]; then exit 0; fi
    sleep .05
done
exit 1
