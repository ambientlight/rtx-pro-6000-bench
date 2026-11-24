#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
dest=${1:?Usage: restore-source.sh NEW_DIRECTORY [LOCAL_BASE_CHECKOUT]}
base=481e8e66504cb680ccb1316eaf01ba09b03da5cd
ref=refs/heads/sm120/pearl-graph-verify
target=a384e33a52d7bac3e7fac1b0ad24ce9bd4a9f4da
test ! -e "$dest" || { echo 'Choose a new checkout directory' >&2; exit 1; }
(cd "$here" && sha256sum --check SHA256SUMS)
if [[ $# -eq 2 ]]; then
    local_base=$(cd -- "$2" && pwd)
    git init -q "$dest"
    git -C "$dest" fetch --quiet --no-tags --depth=1 "$local_base" "$base"
else
    "$here/../restore-source.sh" "$dest"
fi
git -C "$dest" bundle verify "$here/sglang-graph.bundle"
git -C "$dest" fetch --quiet "$here/sglang-graph.bundle" "$ref:$ref"
git -C "$dest" checkout -q sm120/pearl-graph-verify
test "$(git -C "$dest" rev-parse HEAD)" = "$target"
git -C "$dest" log -1 --format='%H %s'
