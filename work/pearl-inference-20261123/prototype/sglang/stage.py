"""Build a sparse SGLang source overlay in a fresh directory, preserving the input tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def stage(source, destination):
    hashes = json.loads((HERE / "source-sha256.json").read_text())
    if destination.exists():
        raise ValueError("choose a new output directory")
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        root = Path(scratch) / "candidate"
        for relative, digest in hashes.items():
            data = (source / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"unexpected SGLang revision for {relative}")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        subprocess.run(["patch", "--silent", "--batch", "--fuzz=0", "-p1", "-i", str(HERE / "hooks.patch")], cwd=root, check=True)
        shutil.copyfile(HERE / "pearl_runtime.py", root / "python/sglang/srt/pearl_runtime.py")
        for file in root.rglob("*.py"):
            compile(file.read_bytes(), str(file), "exec")
        root.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.source.resolve(), args.output.resolve())
