#!/usr/bin/env python3
"""Run native Pearl and the TP4 model on the production loopback endpoint."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import fullmodel as host
from native_host import MODEL, VERIFIER, VERIFIER_SHA, digest

HERE = Path(__file__).resolve()
RELEASE = host.HOST / "native-production"
UNIT = "dsv41-pearl"
JOURNAL = host.HOST / "candidate-unpacked/rootfs/opt/content-free/journal_entrypoint.py"
BASE = "http://127.0.0.1:8000"
MODES = ("joined", "non-joined")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def notify(message):
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode(), address.replace("@", "\0", 1) if address.startswith("@") else address)


def selected_mode():
    try:
        mode = (RELEASE / "serving-mode").read_text().strip()
    except FileNotFoundError:
        return "joined"  # Existing deployments predate the mode selector.
    if mode not in MODES:
        raise ValueError("Unknown Pearl serving mode")
    return mode


def set_mode(mode):
    if mode not in MODES:
        raise ValueError("Unknown Pearl serving mode")
    with tempfile.NamedTemporaryFile("w", dir=RELEASE, prefix=".mode-", delete=False) as tmp:
        tmp.write(mode + "\n")
    os.replace(tmp.name, RELEASE / "serving-mode")


def fresh_session(control, mode):
    config = read(RELEASE / "config.json")
    stage = read(RELEASE / "staged.json")
    if digest(Path(config["library"])) != stage["library_sha256"]:
        raise RuntimeError("Staged K780 library changed")
    if mode == "joined" and digest(VERIFIER) != VERIFIER_SHA:
        raise RuntimeError("Joined proof verifier changed")
    for rel, checksum in stage["source"].items():
        if digest(RELEASE / "pkg/sglang" / rel) != checksum:
            raise RuntimeError("Staged SGLang overlay changed")
    runs = RELEASE / "runs"
    runs.mkdir(exist_ok=True)
    run = runs / (time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
    run.mkdir()
    for path in ("logs", "serving"):
        (run / path).mkdir()
    (run / "pkg").symlink_to(RELEASE / "pkg", target_is_directory=True)
    if mode == "joined":
        (run / "serving/shapes").mkdir()
        (run / "receipts").mkdir()
        (run / "gateway-token").write_text(secrets.token_hex(32))
    # Older staging manifests set a cutoff for the in-process miner.
    config.pop("cutoff_c", None)
    config.update(permit=str(control / "idle.permit"), log_dir=str(run / "logs"))
    save(run / "config.json", config)
    tmp = RELEASE / "next"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run, target_is_directory=True)
    tmp.replace(RELEASE / "current")
    return run


def model_environment(run, sock, mode):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PEARL_")}
    env.update(CANARY_PORT="8000", CANARY_REQUIRE_API_KEY="0", CANARY_CONTENT_FREE="1",
               CANARY_STATE=str(run / "serving"), RAYON_NUM_THREADS="2",
               PEARL_NATIVE_IDLE_CONFIG=str(run / "config.json"), PEARL_BROKER_SOCKET=str(sock))
    if mode == "joined":
        env.update(PEARL_SHAPE_DIR=str(run / "serving/shapes"), PEARL_PUBLIC_REPLAY="1",
                   PEARL_WOB_MODULE="checkpoint_graph", PEARL_GRAPH_VERIFY="1", PEARL_GPU_PREPARE="1",
                   PEARL_REPLAY_POOL_HOST="127.0.0.1", PEARL_REPLAY_POOL_PORT="31990",
                   PEARL_REPLAY_GATEWAY_TOKEN_FILE=str(run / "gateway-token"))
    return env


def stock_status():
    deadline = time.monotonic() + 90
    while True:
        try:
            rows = host.run(*host.DOCKER, "ps", "--all", "--format", "{{.Names}} {{.State}}")
            return dict(row.split(maxsplit=1) for row in rows.splitlines()).get("dsv41", "absent")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # The user manager can start before Docker and its restricted operator socket.
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def stock_available():
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as response:
            return [row["id"] for row in json.load(response)["data"]] == ["deepseek-v4-flash"]
    except (OSError, ValueError):
        return False


def exclusive_devices():
    legacy = host.run("systemctl", "show", "pearl-f2pool.service", "-p", "ActiveState", "--value")
    if legacy not in ("failed", "inactive"):
        host.run("sudo", "-n", "/usr/bin/systemctl", "stop", "pearl-f2pool.service", timeout=120)
    if stock_status() == "running":
        # A stock instance can still be loading when a private cutover guard starts it.
        # Check active traffic if the stock API is already accepting requests.
        if stock_available():
            host.ensure_idle()
        host.run(*host.DOCKER, "stop", "dsv41", timeout=120)
    owners = host.run("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader")
    if owners:
        raise RuntimeError("An existing compute process still owns a GPU")


def launch(children, run, name, argv, env=None):
    with (run / "logs" / (name + ".log")).open("ab") as log:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    children[name] = proc


def stop_children(children):
    # The model owns all four CUDA contexts. Keep its control channels up until it exits.
    for proc in reversed(list(children.values())):
        if proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def serve():
    os.umask(0o077)
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    control = Path(os.environ["RUNTIME_DIRECTORY"])
    mode = selected_mode()
    joined = mode == "joined"
    run = fresh_session(control, mode)
    children = {}
    try:
        host.check_artifacts()
        host.cards()  # Verify the four boards and their power limits before cutover.
        exclusive_devices()
        sock = control / "broker.sock"
        token = run / "gateway-token"
        receipt = run / "receipts/gateway.json"
        arm = run / "armed"
        launch(children, run, "broker", [host.PYTHON, host.PROTO / "broker.py", "--socket", sock,
               "--inference-only-ranks", "4", "--native-permit", control / "idle.permit",
               "--timeline", run / "broker.jsonl"])
        if joined:
            launch(children, run, "gateway", [host.OLD / "env-run", HERE.with_name("gateway.py"),
                   "--model", MODEL, "--verifier", VERIFIER, "--token-file", token,
                   "--receipt", receipt, "--port", "31990", "--arm-file", arm, "--worker", "sglnativeverify"])
        env = model_environment(run, sock, mode)
        ready = False
        deadline = time.monotonic() + 15 * 60
        while not stopping.is_set():
            if any(child.poll() is not None for child in children.values()):
                raise RuntimeError("A serving child exited: " + ",".join(k for k,p in children.items() if p.poll() is not None))
            temps = host.cards()
            save(run / "temperature.json", dict(time_ns=time.time_ns(), temperatures_c=temps))
            if "model" not in children and sock.exists() and (not joined or receipt.exists()):
                if joined:
                    arm.touch(mode=0o600, exist_ok=False)
                launch(children, run, "model", [host.PYTHON, "-S", JOURNAL, "--", host.OLD / "env-run",
                       host.HERE, "launch-oci"], env)
                notify("STATUS=Loading DeepSeek on four SM120 GPUs")
            if not ready and all((run / "serving" / name).is_file() for name in host.SMOKES):
                if any(read(run / "serving" / name) != {"status": "passed"} for name in host.SMOKES):
                    raise RuntimeError("A DeepSeek startup smoke check failed")
                with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as response:
                    if [x["id"] for x in json.load(response)["data"]] != ["deepseek-v4-flash"]:
                        raise RuntimeError("Unexpected production model")
                save(run / "ready.json", dict(time_ns=time.time_ns(), smokes=list(host.SMOKES),
                     port=8000, mode=mode, inference_proofs_enabled=arm.is_file()))
                notify(f"READY=1\nSTATUS=DeepSeek TP4 with native Pearl ({mode})")
                ready = True
            if not ready and time.monotonic() >= deadline:
                raise TimeoutError("Native production startup exceeded 15 minutes")
            stopping.wait(1)
    except Exception as exc:
        save(run / "failure.json", dict(time_ns=time.time_ns(), reason=str(exc)))
        raise
    finally:
        (run / "armed").unlink(missing_ok=True)
        stop_children(children)


def install():
    directory = Path.home() / ".config/systemd/user"
    clock_assets = HERE.parent / "native/clock"
    clock_unit = clock_assets / "dsv41-adaptive-clock.service"
    controller = Path.home() / ".local/libexec/dsv41-watch-clock.py"
    unit = f"""[Unit]
Description=DeepSeek TP4 with native Pearl mining

[Service]
Type=notify
ExecStart={host.PYTHON} {HERE} serve
Restart=on-failure
RestartSec=10s
RuntimeDirectory={UNIT}
RuntimeDirectoryMode=0700
UMask=0077
MemoryHigh=infinity
MemoryMax=infinity
MemorySwapMax=infinity
LimitMEMLOCK=infinity
LimitCORE=0
KillMode=mixed
TimeoutStartSec=18min
TimeoutStopSec=160

[Install]
WantedBy=default.target
"""
    files = [(controller, (clock_assets / "watch_clock.py").read_text(), 0o755),
             (directory / clock_unit.name, clock_unit.read_text(), 0o644),
             (directory / (UNIT + ".service"), unit, 0o644)]
    for dst, body, mode in files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
            tmp = Path(staging) / dst.name
            tmp.write_text(body)
            tmp.chmod(mode)
            tmp.replace(dst)
    legacy = directory / (UNIT + "-fallback.service")
    if legacy.exists():
        host.run("systemctl", "--user", "stop", legacy.name)
        legacy.unlink()
    host.run("systemctl", "--user", "daemon-reload")
    host.run("systemctl", "--user", "enable", UNIT + ".service", clock_unit.name)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("serve", "install"):
        commands.add_parser(name)
    commands.add_parser("set-mode", help="Select the mode for the next service start").add_argument("mode", choices=MODES)
    args = cli.parse_args()
    if args.command == "set-mode":
        set_mode(args.mode)
    else:
        {"serve": serve, "install": install}[args.command]()
