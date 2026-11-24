"""Build a local OCI image from a verified official base and a clean source tree."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[2] / "work/pearl-inference-20261123/prototype"
IMAGE_SOURCE = "e96e96757018ce842d570781ec8312f8c470493b"
BASE = "sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213"
UPSTREAM = "c64f2256d58358ff65c53f0e38dc3e6bcfac8484"
TAG = "dsv41-upstream-sm120"
EPOCH = "2026-11-24T00:00:00Z"


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def read_blob(layout, descriptor):
    digest = descriptor["digest"].split(":", 1)[1]
    data = (layout / "blobs/sha256" / digest).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest or len(data) != descriptor["size"]:
        raise ValueError(f"damaged OCI descriptor {digest}")
    return json.loads(data)


def make_layer(source, libs, wheel, rust_server, path):
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit local source changes before building")
    git(source, "merge-base", "--is-ancestor", UPSTREAM, "HEAD")
    # The immutable upstream dev image installs this tree editable. Change tracked
    # files relative to its labeled commit, retaining its generated native modules.
    original = tomllib.loads(git(source, "show", f"{IMAGE_SOURCE}:python/pyproject.toml").decode())
    current = tomllib.loads((source / "python/pyproject.toml").read_text())
    expected = {"cryptography>=3.4", "jsonschema>=4.18"}  # Both installed in the pinned OCI.
    base_deps = set(original["project"]["dependencies"])
    if set(current["project"]["dependencies"]) != base_deps | expected:
        raise ValueError("review the dependency changes against the base image")
    rev = git(source, "rev-parse", "--short=12", "HEAD").decode().strip()
    version = f"0.0.0.dev1+g{rev}"
    site = "opt/sglang/lib/python3.12/site-packages"
    tree = "sgl-workspace/sglang"
    old = "0.0.0.dev1+ge96e96757"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as layer:
        def put(name, data, mode=0o644, link=None):
            entry = tarfile.TarInfo(name)
            entry.mode = mode
            if link is not None:
                entry.type, entry.linkname = tarfile.SYMTYPE, link
                layer.addfile(entry)
            else:
                entry.size = len(data)
                layer.addfile(entry, io.BytesIO(data))

        raw = git(source, "diff", "--name-status", "-z", "--no-renames", IMAGE_SOURCE, "HEAD")
        fields = raw.decode().rstrip("\0").split("\0")
        for status, name in zip(fields[::2], fields[1::2], strict=True):
            dest = Path("sgl-workspace/sglang") / name
            if status == "D":
                put(str(dest.with_name(".wh." + dest.name)), b"")
                continue
            f = source / name
            if f.is_symlink():
                put(str(dest), b"", link=os.readlink(f))
            else:
                put(str(dest), f.read_bytes(), 0o755 if os.access(f, os.X_OK) else 0o644)
        # The source tree is the editable install. Replace the base Python
        # metadata using the wheel built from this revision, and retire the
        # original VCS/package metadata so version queries see one identity.
        put(f"{tree}/.wh..git", b"")
        put(f"{tree}/python/.wh.sglang.egg-info", b"")
        put(f"{site}/.wh.sglang-{old}.dist-info", b"")
        put(f"{site}/.wh.__editable__.sglang-{old}.pth", b"")
        put(f"{site}/.wh.__editable___sglang_0_0_0_dev1_ge96e96757_finder.py", b"")
        dist = f"sglang-{version}.dist-info"
        editable = "sglang-source.pth"
        with zipfile.ZipFile(wheel) as package:
            for item in ("METADATA", "entry_points.txt", "WHEEL", "top_level.txt"):
                put(f"{site}/{dist}/{item}", package.read(f"{dist}/{item}"))
            generated = package.read("sglang/_version.py")
            if version.encode() not in generated:
                raise ValueError("the wheel was not built from this source revision")
            put(f"{tree}/python/sglang/_version.py", generated)
        put(f"{site}/{editable}", b"/sgl-workspace/sglang/python\n")
        records = [f"{dist}/{item}" for item in ("METADATA", "entry_points.txt", "WHEEL", "top_level.txt", "RECORD")]
        put(f"{site}/{dist}/RECORD", "".join(f"{f},,\n" for f in [*records, editable]).encode())
        put(f"{tree}/python/sglang/srt/rust_extensions/_server.cpython-312-x86_64-linux-gnu.so",
            rust_server.read_bytes(), 0o755)
        for f in sorted((HERE.parent / "privacy").glob("*.py")):
            if f.name != "patch_runtime.py":  # Installed source already contains the ported policy.
                put("opt/content-free/" + f.name, f.read_bytes())
        put("opt/sglang/lib/python3.12/site-packages/content_free.pth",
            (HERE.parent / "privacy/content_free.pth").read_bytes())
        for name in ("boot.py", "checkpoint.json"):
            put("opt/dsv41/" + name, (HERE / name).read_bytes())
        put("opt/dsv41/diagnostic_entrypoint.py", (HERE.parent / "diagnostic_entrypoint.py").read_bytes())
        for name in ("checkpoint_projection.py", "pool_layouts.py", "replay_pool.py", "operator/pearl_operator.py",
                     "operator/gpu_prepare/activation.py"):
            put("opt/pearl/" + name, (PROTO / name).read_bytes())
        for name in ("libpearl_reference.so", "libpearl_wo_b.so", "libpearl_prepare.so"):
            put("opt/pearl/lib/" + name, (libs / name).read_bytes(), 0o755)


def build(source, base, output, libs, wheel, rust_server, umoci):
    source, base, output = source.resolve(), base.resolve(), output.absolute()
    if output.exists():
        raise ValueError("choose a new output directory")
    index = json.loads((base / "index.json").read_text())
    descriptor = next(x for x in index["manifests"] if x["digest"] == BASE)
    manifest = read_blob(base, descriptor)
    config = read_blob(base, manifest["config"])
    if (config["architecture"] != "amd64" or
            config["config"]["Labels"]["org.opencontainers.image.revision"] != IMAGE_SOURCE):
        raise ValueError("unexpected dependency image architecture or source revision")
    rev = git(source, "rev-parse", "HEAD").decode().strip()
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp)
        layer = root / "source.tar"
        make_layer(source, libs, wheel, rust_server, layer)
        layout = root / "image"
        layout.mkdir()
        shutil.copyfile(base / "oci-layout", layout / "oci-layout")
        (layout / "blobs/sha256").mkdir(parents=True)
        for obj in (descriptor, manifest["config"], *manifest["layers"]):
            name = obj["digest"].split(":", 1)[1]
            origin = base / "blobs/sha256" / name
            target = layout / "blobs/sha256" / name
            if origin.stat().st_size != obj["size"]:
                raise ValueError(f"incomplete base blob {name}")
            if target.exists():  # The manifest can reuse an empty layer.
                continue
            try:
                os.link(origin, target)
            except OSError:
                shutil.copyfile(origin, target)
        descriptor["annotations"] = {"org.opencontainers.image.ref.name": TAG}
        index["manifests"] = [descriptor]
        (layout / "index.json").write_text(json.dumps(index))
        ref = f"{layout}:{TAG}"
        subprocess.run([umoci, "raw", "add-layer", "--history.created", EPOCH,
                        "--image", ref, str(layer)], check=True)
        args = [umoci, "config", "--image", ref, "--created", EPOCH, "--history.created", EPOCH,
                "--clear=config.entrypoint", "--clear=config.cmd",
                "--config.label", f"org.opencontainers.image.revision={rev}",
                "--config.label", f"dev.idle.sglang.upstream={UPSTREAM}",
                "--config.label", f"dev.idle.base.digest={BASE}",
                "--config.label", f"ai.sglang.build.commit={rev}",
                "--config.label", f"ai.sglang.image.tag={TAG}",
                "--config.label", "ai.sglang.build.url=",
                "--config.label", "org.opencontainers.image.title=DeepSeek V4.1 Flash on SM120",
                "--config.env", f"SGLANG_BUILD_COMMIT={rev}",
                "--config.env", f"SGLANG_GIT_COMMIT={rev}",
                "--config.env", f"SGLANG_IMAGE_TAG={TAG}",
                "--config.env", "SGLANG_BUILD_URL="]
        env = json.loads((HERE / "environment.json").read_text())
        for key, value in env.items():
            args.extend(["--config.env", key + "=" + value])
        for segment in ("python3", "-S", "-u", "/opt/content-free/journal_entrypoint.py", "--",
                        "python3", "-u", "/opt/dsv41/diagnostic_entrypoint.py"):
            args.extend(["--config.entrypoint", segment])
        subprocess.run([*args, "--config.cmd", "run"], check=True)
        subprocess.run([umoci, "gc", "--layout", str(layout)], check=True)
        layout.rename(output)
    return json.loads((output / "index.json").read_text())["manifests"][0]["digest"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "base", "output", "libs", "wheel", "rust-server"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--umoci", default="umoci")
    print(build(**vars(parser.parse_args())))
