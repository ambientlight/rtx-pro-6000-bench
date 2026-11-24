#!/usr/bin/env bash
set -euo pipefail

src=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
if [[ $(hostname -s) != htower || $(id -un) != ambientlight ]]; then
    echo 'Run this script as ambientlight on htower.' >&2
    exit 1
fi
socket=dsv41-memory-clock.socket
service=dsv41-memory-clock@.service
cards=$(sed -n 's/^ExecStartPre=.* --id=\([^ ]*\) .*/\1/p' "$src/$service")
expected=$(tr ',' '\n' <<<"$cards" | sort)
actual=$(/usr/bin/nvidia-smi --query-gpu=uuid --format=csv,noheader | sort)
[[ -n $cards && $expected == "$actual" ]] || { echo 'GPU inventory differs from the clock unit.' >&2; exit 1; }
/usr/bin/systemd-analyze verify "$src/$socket" "$src/$service"

# All elevated programs and their destinations are fixed. No repo script runs as root.
/usr/bin/sudo -- /usr/bin/install -o root -g root -m 0644 -t /etc/systemd/system/ -- "$src/$socket" "$src/$service"
/usr/bin/sudo -- /usr/bin/systemctl daemon-reload
/usr/bin/sudo -- /usr/bin/systemctl enable --now "$socket"
/usr/bin/systemctl is-active --quiet "$socket"
[[ -S /run/dsv41-memory-clock.sock ]]
echo 'The clock socket is ready. The serving account can now run the adaptive-clock trial.'
