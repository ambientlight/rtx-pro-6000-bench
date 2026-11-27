"""Stage the real native miner and the SGLang overlay in a private host run directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

import fullmodel as host
from gateway import HOST, PORT, PREFLIGHT, VERIFIER_SHA, secret
from native_report import TILES_PER_NONCE

FILES = ("srt/pearl_runtime.py", "srt/pearl_native_idle.py", "srt/managers/scheduler.py")
BASE_SGLANG = "a384e33a52d7bac3e7fac1b0ad24ce9bd4a9f4da"
MODEL = Path("/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017")
DEFAULT_SLICE_TILES = 12_032
VERIFIER = Path("/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-sglang-prototype/target/cargo/release/examples/verify_receipt")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, library, *, slice_tiles=DEFAULT_SLICE_TILES):
    if not 1 <= slice_tiles <= TILES_PER_NONCE:
        raise ValueError(f"K780 slice size must be between 1 and {TILES_PER_NONCE} tiles")
    host.check_artifacts()
    os.umask(0o077)
    source = source.resolve()
    run = host.RUN
    run.mkdir(mode=0o700, exist_ok=True)
    cfg = Path(host.NATIVE_CONFIG)
    if run.stat().st_uid != os.getuid() or run.stat().st_mode & 0o077 or cfg.exists():
        raise ValueError("Use a fresh config path in an owner-only run directory")
    source_pkg = source/"python/sglang"
    for rel in FILES:
        original = host.OLD/"pkg/sglang"/rel
        if original.exists():
            base = subprocess.check_output(["git","-C",source,"show",f"{BASE_SGLANG}:python/sglang/{rel}"])
            if original.read_bytes() != base:
                raise ValueError(f"Staged graph base differs at {rel}")
    subprocess.run(["cp","-a","--reflink=auto",host.OLD/"pkg",run/"pkg"],check=True)
    for rel in FILES:
        shutil.copyfile(source_pkg/rel, run/"pkg/sglang"/rel)
    shutil.copyfile(library,run/"libpearl_idle.so")
    for name in ("logs","receipts"):
        (run/name).mkdir(mode=0o700)
    (run/"gateway-token").write_text(secrets.token_hex(32))
    with cfg.open("x") as out:
        json.dump(dict(library=str(run/"libpearl_idle.so"),permit=str(host.SOCKET.with_name("idle.permit")),
                       endpoint=f"{HOST}:{PORT}",account=json.loads(secret(PREFLIGHT))["f2pool_mining_account"],
                       worker_prefix="sglnative",log_dir=str(run/"logs"),tiles=slice_tiles,power_w=275),out)
    (run/"staged.json").write_text(json.dumps(dict(library_sha256=digest(run/"libpearl_idle.so"),
        source={rel:digest(run/"pkg/sglang"/rel) for rel in FILES}),indent=2)+"\n")


def gateway():
    if digest(VERIFIER) != VERIFIER_SHA:
        raise ValueError("Unexpected inference proof verifier")
    host.unit(host.GATEWAY,host.OLD/"env-run",host.HERE.with_name("gateway.py"),"--model",MODEL,
              "--verifier",VERIFIER,"--token-file",host.RUN/"gateway-token","--receipt",host.RUN/"receipts/gateway.json",
              "--port","31990","--arm-file",host.RUN/"armed","--worker","sglnativeverify",
              output=host.RUN/"logs/gateway.log")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="command", required=True)
    prep = cmd.add_parser("prepare")
    prep.add_argument("--source", required=True, type=Path)
    prep.add_argument("--library", required=True, type=Path)
    prep.add_argument("--slice-tiles", type=int, default=DEFAULT_SLICE_TILES,
                      help="Maximum K780 tiles per idle CUDA graph (default %(default)s)")
    cmd.add_parser("gateway")
    args=cli.parse_args()
    if not host.NATIVE_CONFIG or not Path(host.NATIVE_CONFIG).is_absolute():
        cli.error("Set PEARL_NATIVE_IDLE_CONFIG to the absolute private config path")
    if args.command == "prepare":
        prepare(args.source, args.library, slice_tiles=args.slice_tiles)
    else:
        gateway()
