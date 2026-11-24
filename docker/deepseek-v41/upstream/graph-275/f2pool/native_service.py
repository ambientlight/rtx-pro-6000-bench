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


def fresh_session(control):
    config = read(RELEASE / "config.json")
    stage = read(RELEASE / "staged.json")
    if digest(Path(config["library"])) != stage["library_sha256"] or digest(VERIFIER) != VERIFIER_SHA:
        raise RuntimeError("Staged K780 library or proof verifier changed")
    for rel, checksum in stage["source"].items():
        if digest(RELEASE / "pkg/sglang" / rel) != checksum:
            raise RuntimeError("Staged SGLang overlay changed")
    runs = RELEASE / "runs"
    runs.mkdir(exist_ok=True)
    run = runs / (time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
    run.mkdir()
    for path in ("logs", "receipts", "serving", "serving/shapes"):
        (run / path).mkdir()
    (run / "pkg").symlink_to(RELEASE / "pkg", target_is_directory=True)
    (run / "gateway-token").write_text(secrets.token_hex(32))
    config.update(permit=str(control / "idle.permit"), log_dir=str(run / "logs"))
    save(run / "config.json", config)
    tmp = RELEASE / "next"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run, target_is_directory=True)
    tmp.replace(RELEASE / "current")
    return run


def stock_status():
    return host.run(*host.DOCKER, "inspect", "-f", "{{.State.Status}}", "dsv41")


def exclusive_devices():
    legacy = host.run("systemctl", "show", "pearl-f2pool.service", "-p", "ActiveState", "--value")
    if legacy not in ("failed", "inactive"):
        host.run("sudo", "-n", "/usr/bin/systemctl", "stop", "pearl-f2pool.service", timeout=120)
    if stock_status() == "running":
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
    run = fresh_session(control)
    children = {}
    try:
        host.check_artifacts()
        if max(host.cards()) >= host.GPU_TEMP_CUTOFF_C:
            raise RuntimeError("Temperature reached the 92 C cutoff before startup")
        exclusive_devices()
        sock = control / "broker.sock"
        token = run / "gateway-token"
        receipt = run / "receipts/gateway.json"
        arm = run / "armed"
        launch(children, run, "broker", [host.PYTHON, host.PROTO / "broker.py", "--socket", sock,
               "--inference-only-ranks", "4", "--native-permit", control / "idle.permit",
               "--timeline", run / "broker.jsonl"])
        launch(children, run, "gateway", [host.OLD / "env-run", HERE.with_name("gateway.py"),
               "--model", MODEL, "--verifier", VERIFIER, "--token-file", token,
               "--receipt", receipt, "--port", "31990", "--arm-file", arm, "--worker", "sglnativeverify"])
        env = dict(os.environ, CANARY_PORT="8000", CANARY_REQUIRE_API_KEY="0", CANARY_CONTENT_FREE="1",
                   CANARY_STATE=str(run / "serving"), PEARL_SHAPE_DIR=str(run / "serving/shapes"),
                   PEARL_NATIVE_IDLE_CONFIG=str(run / "config.json"), PEARL_BROKER_SOCKET=str(sock),
                   PEARL_PUBLIC_REPLAY="1", PEARL_WOB_MODULE="checkpoint_graph", PEARL_GRAPH_VERIFY="1",
                   PEARL_GPU_PREPARE="1", PEARL_REPLAY_POOL_HOST="127.0.0.1", PEARL_REPLAY_POOL_PORT="31990",
                   PEARL_REPLAY_GATEWAY_TOKEN_FILE=str(token), RAYON_NUM_THREADS="2")
        ready = False
        deadline = time.monotonic() + 15 * 60
        while not stopping.is_set():
            if any(child.poll() is not None for child in children.values()):
                raise RuntimeError("A serving child exited: " + ",".join(k for k,p in children.items() if p.poll() is not None))
            temps = host.cards()
            save(run / "temperature.json", dict(time_ns=time.time_ns(), temperatures_c=temps))
            if max(temps) >= host.GPU_TEMP_CUTOFF_C:
                raise RuntimeError("GPU reached the 92 C cutoff")
            if "model" not in children and sock.exists() and receipt.exists():
                # Each run has its own arm file. Winning inference proofs are authorized for production.
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
                     port=8000, inference_proofs_enabled=arm.is_file()))
                notify("READY=1\nSTATUS=DeepSeek TP4 with native Pearl and the F2Pool inference gateway")
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


def fallback():
    # systemd orders this after the native unit and drains its cgroup first.
    if (RELEASE / "current").exists():
        save(RELEASE / "current/fallback.json", dict(time_ns=time.time_ns()))
    host.run(*host.DOCKER, "start", "dsv41", timeout=120)
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as reply:
                if [x["id"] for x in json.load(reply)["data"]] == ["deepseek-v4-flash"] and stock_status() == "running":
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("Stock model failed to return on port 8000")


def install():
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    units = {UNIT: f"""[Unit]
Description=DeepSeek TP4 with native Pearl mining
OnFailure={UNIT}-fallback.service

[Service]
Type=notify
ExecStart={host.PYTHON} {HERE} serve
RuntimeDirectory={UNIT}
RuntimeDirectoryMode=0700
UMask=0077
MemoryMax=365G
MemorySwapMax=0
LimitCORE=0
KillMode=mixed
TimeoutStartSec=18min
TimeoutStopSec=160

[Install]
WantedBy=default.target
""", UNIT + "-fallback": f"""[Unit]
Description=Restore stock DeepSeek after native Pearl stops
Conflicts={UNIT}.service
After={UNIT}.service

[Service]
Type=oneshot
ExecStart={host.PYTHON} {HERE} fallback
UMask=0077
TimeoutStartSec=18min
"""}
    for name, body in units.items():
        dst = directory / (name + ".service")
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(body)
        tmp.replace(dst)
    host.run("systemctl", "--user", "daemon-reload")
    host.run("systemctl", "--user", "enable", UNIT + ".service")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("serve", "fallback", "install"))
    {"serve": serve, "fallback": fallback, "install": install}[cli.parse_args().command]()
