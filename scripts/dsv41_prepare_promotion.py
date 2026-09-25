#!/usr/bin/env python3
"""Preserve the canary and prepare isolated state for its port-8000 replacement.

Does not stop/start containers, modify the old state, or copy model weights.
Inspect snapshots may contain credentials and are always private.
"""

import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from dsv41_load import LOAD_PATH, is_idle, parse_loads


ROOT = Path(__file__).resolve().parents[1]
STATE = Path("/mnt/hot/dsv41_state")
CANARY_IMAGE = "sha256:49117bf6688f1ef620bf8cd98bb8a3280427fe29c91972c229718450123f047b"


def run(*args):
    return subprocess.check_output(args, text=True)


def save(path, value):
    with path.open("x") as out:
        out.write(value if isinstance(value, str) else json.dumps(value, indent=2))


def main():
    os.umask(0o077)
    target = STATE / "production"
    if target.exists():
        raise RuntimeError("Production state already exists; inspect it instead of overwriting")
    if run("docker", "ps", "-aq", "--filter", "name=^/dsv41$").strip():
        raise RuntimeError("A dsv41 container already exists; inspect before replacing it")
    containers = json.loads(run("docker", "inspect", "dsv41-api", "dsv4", "qwen3-embed"))
    by_name = {c["Name"].lstrip("/"): c for c in containers}
    if by_name["dsv41-api"]["Image"] != CANARY_IMAGE:
        raise RuntimeError("The canary image changed; reevaluate before promotion")
    if any(by_name[n]["State"]["Running"] for n in ("dsv4", "qwen3-embed")):
        raise RuntimeError("A preserved V12/embedding container is unexpectedly running")
    old_state = STATE / "api-candidate"
    req = urllib.request.Request(
        "http://127.0.0.1:8010" + LOAD_PATH,
        headers={"Authorization": "Bearer " + (old_state / "api-key").read_text().strip()},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        load = parse_loads(json.load(response))
    if not is_idle(load):
        raise RuntimeError("Canary is busy; wait for in-flight work before promotion")
    now = dt.datetime.now(dt.timezone.utc)
    backup = STATE / ("pre-promotion-" + now.strftime("%Y%m%dT%H%M%SZ"))
    backup.mkdir(mode=0o700)
    save(backup / "containers.private.json", containers)
    save(backup / "capture-unit.txt", run("systemctl", "--user", "cat", "dsv41-wire-dump.service"))
    save(
        backup / "canary-source-boot.py",
        run(
            "git",
            "-C",
            "/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay",
            "show",
            "da53ffcef5fa7c479da2e673dfbb408a3e660237:deployment/boot.py",
        ),
    )
    old_capture = Path("/mnt/hot/dsv41_dumps/current").resolve(strict=True)
    save(
        backup / "capture-position.json",
        {
            "path": str(old_capture),
            "native_bytes": (old_capture / "dsv4-native.log").stat().st_size,
            "status": json.loads((old_capture / "wire/status.json").read_text()),
        },
    )
    for name in (
        "launch.json",
        "verification.json",
        "smoke.json",
        "smoke-structured.json",
        "smoke-tools.json",
        "smoke-vision.json",
    ):
        if (old_state / name).exists():
            shutil.copy2(old_state / name, backup / name)
    save(backup / "status-before.md", (ROOT / "docker/deepseek-v41/STATUS.md").read_text())
    target.mkdir(mode=0o700)
    for name in ("api-key", "verification.json"):
        shutil.copy2(old_state / name, target / name)
        (target / name).chmod(0o600)
    receipt = {
        "prepared_utc": now.isoformat(),
        "backup": str(backup),
        "state": str(target),
        "canary_image_id": CANARY_IMAGE,
        "native_model": "DeepSeek-V4.1-Flash",
        "served_model": "deepseek-v4-flash",
        "port": 8000,
        "context_length": 524288,
        "container": "dsv41",
        "require_api_key": False,
        "restart": "unless-stopped",
        "embedding_enabled": False,
        "containers_changed": False,
    }
    save(target / "promotion-preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
