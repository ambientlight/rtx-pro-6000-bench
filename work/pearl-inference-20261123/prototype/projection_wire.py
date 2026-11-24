"""Exercise mining.notify -> SM120 data path -> gzip v2 mining.submit on loopback.

The receiver validates the actual wire payload with the separate deployed-source
Rust receipt verifier before acknowledging. The only client identity is 'fixture'.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np

from checkpoint_projection import K, Projection
from replay_pool import compact

SHARE_BITS = 0x1E100000
SHARE_TARGET = (SHARE_BITS & 0x7fffff) << (8 * ((SHARE_BITS >> 24) - 3))


def send(sock, message):
    sock.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")


def receive(stream):
    line = stream.readline(1 << 20)
    if not line.endswith(b"\n"):
        raise ValueError("missing or oversized Stratum frame")
    return json.loads(line)


def verify_submission(frame, job, verifier):
    params = frame["params"]
    if frame["method"] != "mining.submit" or params["job_id"] != job["job_id"]:
        raise ValueError("missing or stale submission")
    compressed = base64.b64decode(params["plain_proof"], validate=True)
    raw = gzip.decompress(compressed)
    if len(raw) > 1 << 20:
        raise ValueError("fixture proof exceeds expected maximum")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw)
    # The deployed receipt format hashes the Base64 field. Our result identifies the raw witness.
    receipt = dict(header=job["header"], cert_version=3, share_nbits=compact(int(job["target"], 16)),
                   pool_base_target=job["target"], plain_proof=encoded.decode(),
                   proof_sha256=hashlib.sha256(encoded).hexdigest())
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(receipt, f)
        f.flush()
        subprocess.run([str(verifier), f.name], capture_output=True, check=True, timeout=10)
    return dict(job_id=job["job_id"], compressed_bytes=len(compressed),
                verified_bytes=len(raw), proof_sha256=digest)


def run_pool(listener, verifier):
    with listener.accept()[0] as conn, conn.makefile("rb") as stream:
        conn.settimeout(40)
        auth = receive(stream)
        if auth.get("method") != "mining.authorize" or auth.get("params", {}).get("type") != "v2":
            raise ValueError("expected fixture authorization")
        send(conn, dict(id=auth["id"], result=True, error=None, type="v2"))
        receipts = []
        for generation in range(2):
            header = struct.pack("<I32s32sII", 1, bytes([53 + generation]) * 32,
                                 bytes([91 - generation]) * 32, 0x66666666 + generation, 0x1c00ffff)
            job = dict(job_id=f"fixture-{generation}", cert_version=3, header=header.hex(),
                       target=f"{SHARE_TARGET:064x}")
            send(conn, dict(method="mining.notify", params=job))
            frame = receive(stream)
            receipt = verify_submission(frame, job, verifier)
            send(conn, dict(id=frame["id"], result=True, error=None))
            receipts.append(receipt)
        return receipts


def run_projection(address, args):
    with socket.create_connection(address, timeout=40) as sock, sock.makefile("rb") as stream:
        send(sock, dict(id=1, method="mining.authorize",
                        params={"wallet": "fixture", "worker": "public-projection", "type": "v2",
                                "pass": "x", "password": "x", "agent": "pearl-sglang-prototype"}))
        auth = receive(stream)
        if not auth.get("result") or auth.get("type") != "v2":
            raise ValueError("fixture did not negotiate gzip")
        rows = []
        for i, (phase, m) in enumerate((("prefill", 2048), ("decode", 6)), 2):
            notice = receive(stream)
            job = notice["params"]
            if notice["method"] != "mining.notify" or job["cert_version"] != 3:
                raise ValueError("unsupported fixture job")

            def submit(origin, raw):
                if origin != phase:
                    raise ValueError("incorrect proof provenance")
                wire = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
                send(sock, dict(id=i, method="mining.submit", params=dict(job_id=job["job_id"], plain_proof=wire)))
                result = receive(stream)
                if result["id"] != i or result["result"] is not True:
                    raise ValueError("fixture rejected the share")

            with Projection(args.model, args.build, args.rank, args.device, header=bytes.fromhex(job["header"]),
                            share_bits=compact(int(job["target"],16)), on_witness=submit) as worker:
                x = np.random.default_rng(200 + i).standard_normal((m, K), dtype=np.float32)
                y, stats = worker.forward(x, phase, check=True)
                assert stats[0]["verified_witness_bytes"] > 0 and np.isfinite(y).all()
                rows.append(dict(phase=phase, rows=m, job_id=job["job_id"], tickets=stats[0]["tickets"],
                                 checked_integer_outputs=stats[0]["exact_integer_outputs_checked"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(40)
        with ThreadPoolExecutor(1) as t:
            pool = t.submit(run_pool, listener, args.verifier.resolve())
            phases = run_projection(listener.getsockname(), args)
            shares = pool.result()
    result = dict(unix_time=time.time(), pool="local fixture", device=args.device, rank=args.rank,
                  cuda_visible=os.getenv("CUDA_VISIBLE_DEVICES"), phases=phases, accepted=shares)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    for phase, share in zip(phases, shares, strict=True):
        print(f"{phase['phase']} {share['job_id']}: accepted {share['verified_bytes']}-byte proof")


if __name__ == "__main__":
    main()
