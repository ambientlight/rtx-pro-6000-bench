#!/usr/bin/env bash
# Build an isolated experimental worker from the exact deployed source and overlay.
set -euo pipefail
if [[ $# -ne 2 ]]; then echo 'usage: build.sh SOURCE_REPO NEW_CHECKOUT' >&2; exit 2; fi
source_repo=$(realpath "$1")
checkout=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 13.4 toolkit}"
git -C "$source_repo" worktree add --detach "$checkout" 157ca8bb
git -C "$checkout" submodule update --init --recursive
git -C "$checkout" apply "$here/sliced-worker.patch"
cd "$checkout"
export RUSTC_BOOTSTRAP=1 RUSTFLAGS='-C panic=abort' BPM_VARIANTS_FILE="$here/v1.txt"
cargo +1.97.0 build --locked --release --bin bpm
