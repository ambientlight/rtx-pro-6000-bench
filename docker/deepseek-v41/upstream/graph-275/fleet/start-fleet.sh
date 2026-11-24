#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
systemd-run --user --collect --unit=dsv41-pearl-graph275-miners --service-type=notify \
  -p NotifyAccess=main -p KillMode=control-group -p MemorySwapMax=0 -p RuntimeDirectory=dsv41-pearl-graph275 \
  -p RuntimeDirectoryMode=0700 -p UMask=0077 -p TimeoutStartSec=150 -p TimeoutStopSec=15 \
  -p Restart=no /usr/bin/python3 "$here/miners.py" serve "${1:-http://127.0.0.1:8000}"
