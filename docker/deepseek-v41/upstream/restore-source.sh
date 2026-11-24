#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
dest=${1:?Usage: restore-source.sh NEW_DIRECTORY}
base=c64f2256d58358ff65c53f0e38dc3e6bcfac8484
image_source=e96e96757018ce842d570781ec8312f8c470493b
ref=refs/heads/sm120/production-reconstruction
test ! -e "$dest" || { echo 'Choose a new checkout directory' >&2; exit 1; }
(cd "$here" && sha256sum --check source.sha256)
git init -q "$dest"
git -C "$dest" remote add origin https://github.com/sgl-project/sglang.git
git -C "$dest" fetch --quiet --no-tags --depth=1 origin "$base" "$image_source"
git -C "$dest" bundle verify "$here/sglang-sm120.bundle"
git -C "$dest" fetch --quiet "$here/sglang-sm120.bundle" "$ref:$ref"
git -C "$dest" checkout -q sm120/production-reconstruction
git -C "$dest" log -1 --format='%H %s'
