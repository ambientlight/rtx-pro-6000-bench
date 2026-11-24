#!/usr/bin/env bash
# Assemble a private Python/launcher overlay from the base candidate and pinned graph source.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
src=${1:?Usage: stage-runtime.sh RESTORED_GRAPH_SOURCE BASE_OCI_ROOTFS NEW_DIRECTORY}
root=${2:?Missing candidate OCI rootfs}
dest=${3:?Missing new output directory}
base=481e8e66504cb680ccb1316eaf01ba09b03da5cd
target=a384e33a52d7bac3e7fac1b0ad24ce9bd4a9f4da
test ! -e "$dest" || { echo 'Choose a new overlay directory' >&2; exit 1; }
test "$(git -C "$src" rev-parse HEAD)" = "$target"
git -C "$src" diff --quiet HEAD -- python/sglang/srt
(cd "$here" && sha256sum --check SHA256SUMS)
installed="$root/sgl-workspace/sglang/python/sglang"
files=(srt/pearl_runtime.py srt/model_executor/runner/decode_cuda_graph_runner.py)
for path in "${files[@]}"; do
    cmp "$installed/$path" <(git -C "$src" show "$base:python/sglang/$path")
done
mkdir -p "$dest/pkg" "$dest/boot"
cp --reflink=auto -a "$installed" "$dest/pkg/sglang"
for path in "${files[@]}"; do
    cp "$src/python/sglang/$path" "$dest/pkg/sglang/$path"
done
cp "$here/../boot.py" "$here/../checkpoint.json" "$dest/boot/"
patch --quiet --directory="$dest/boot" -p1 --input="$here/boot-graph.patch"
cp "$here/../full-canary/host-run/protected_launch.py" "$dest/boot/"
(cd "$dest" && sha256sum --check "$here/staged-launcher.sha256")
