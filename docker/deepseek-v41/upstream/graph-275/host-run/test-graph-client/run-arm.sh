#!/usr/bin/env bash
set -eu
arm="${1:?switch or joined}"
case "$arm" in switch|joined) ;; *) exit 2 ;; esac
root=/mnt/hot/ambientlight/sglang-sm120-runtime
run="$root/graph-production-275"
client="$root/deployment-source/docker/deepseek-v41/upstream/full-canary/graph_compare.py"
export CANARY_API_KEY_FILE="$run/arms/$arm/api-key"
base=http://127.0.0.1:18001
opts=()
if [[ $arm == joined ]]; then
    opts=(--shape-dir "$run/arms/joined/shapes" --peer-ledger "$run/receipts/inference-peer.json" --accepted-log "$run/receipts/inference-acks.jsonl")
fi
for mode in fixed-gap fixed-period; do
    python3 -u "$client" trace --base "$base" --arm "$arm" --mode "$mode" --pairs 4 --gap 1 --period 20 --inputs 2048 --outputs 256 --receipt "$run/receipts/$arm-$mode.json" "${opts[@]}"
done
python3 -u "$client" trace --base "$base" --arm "$arm" --mode fixed-gap --pairs 24 --gap .25 --inputs 256 --outputs 16 --receipt "$run/receipts/$arm-handoff-burst.json" "${opts[@]}"
hist_opts=()
if [[ $arm == joined ]]; then hist_opts=(--shape-dir "$run/arms/joined/shapes"); fi
python3 -u "$client" handbook "${hist_opts[@]}" --base "$base" --arm "$arm" --receipt "$run/receipts/$arm-handbook.json"
