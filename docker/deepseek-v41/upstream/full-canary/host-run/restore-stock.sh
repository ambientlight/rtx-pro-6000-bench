#!/usr/bin/env bash
# The only deployed object this canary owns temporarily is the stopped incumbent.
set -eu
for unit in dsv41-sm120-full-candidate.service dsv41-sm120-full-miners.service dsv41-sm120-full-peer.service; do
    if systemctl --user is-active --quiet "$unit"; then systemctl --user stop "$unit"; fi
done
docker --context shared-operator start dsv41
