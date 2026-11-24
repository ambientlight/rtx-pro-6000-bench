"""Local TP4 qualification peer. Record independent v3 verification and checkpoint shard attribution."""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_http_projection import matrix_opening
from checkpoint_projection import checkpoint_weight, quantize_rows
from projection_wire import SHARE_TARGET, receive, send, verify_submission

MAX_RAW = 1 << 20
MAX_SHARES = 512
RETAINED_SAMPLES = 16
KEEPALIVE_SECONDS = 30


def parse_target(raw):
    try:
        value = int(raw, 16)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive hex share target") from None
    if not 0 < value <= SHARE_TARGET:
        raise argparse.ArgumentTypeError("target must be positive and at most the qualification target")
    return value


def inspect_opening(frame, shards):
    encoded = base64.b64decode(frame["params"]["plain_proof"], validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(encoded)) as data:
        raw = data.read(MAX_RAW + 1)
    if len(raw) > MAX_RAW:
        raise ValueError("proof exceeds fixture limit")
    rows, n, k, noise_rank = struct.unpack_from("<4Q", raw)
    if (n, k, noise_rank) != (5120, 2048, 128):
        raise ValueError("unexpected checkpoint projection shape")
    activations, _, pos = matrix_opening(raw, 32)
    chunks, indices, end = matrix_opening(raw, pos)
    if end + 1 != len(raw) or raw[end] != 0 or not chunks:
        raise ValueError("invalid dense proof opening")
    matches = [rank for rank, shard in enumerate(shards)
               if all(shard[index * 1024 : (index + 1) * 1024] == chunk
                      for index, chunk in zip(indices, chunks, strict=True))]
    if len(matches) != 1:
        raise ValueError("opened weight chunks do not uniquely identify one checkpoint shard")
    return dict(tp_rank=matches[0], proof_rows=rows, opened_weight_bytes=len(chunks) * 1024,
                distinct_opened_activation_codes=len(set(b"".join(activations))))


class Peer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 4

    def __init__(self, args):
        self.limit = threading.BoundedSemaphore(4)
        self.lock = threading.Lock()
        self.sustained = args.sustained
        self.receipt = args.receipt
        self.verifier = args.verifier.resolve()
        self.shards = []
        for rank in range(4):
            bt, _ = quantize_rows(checkpoint_weight(args.model, rank))
            self.shards.append(memoryview(bt).cast("B"))
        header = struct.pack("<I32s32sII", 1, bytes([79]) * 32, bytes([83]) * 32, 0x66666667, 0x1C00FFFF)
        target = SHARE_TARGET
        if self.sustained:
            target = args.share_target if args.share_target is not None else SHARE_TARGET >> 16
        self.job = dict(job_id="full-checkpoint-sustained" if self.sustained else "full-checkpoint-public",
                        cert_version=3, header=header.hex(), target=f"{target:064x}")
        self.results = dict(scope="isolated local Stratum peer at public test difficulty",
                            started_at=time.time(), share_limit=MAX_SHARES, verifier_sha256=hashlib.sha256(
                                self.verifier.read_bytes()).hexdigest(), job=self.job,
                            negotiations=0, rejected_frames=0, accepted=[])
        if self.sustained:
            self.results.update(share_limit=None, accepted_sample_limit=RETAINED_SAMPLES, accepted_total=0,
                                verified_by_rank_rows={str(i): {} for i in range(4)}, keepalive_frames=0)
            if getattr(args, "resume_receipt", False):
                prior = json.loads(self.receipt.read_text())
                if (prior["job"] != self.job or prior["verifier_sha256"] != self.results["verifier_sha256"]
                        or prior["share_limit"] is not None or prior["accepted_sample_limit"] != RETAINED_SAMPLES
                        or not 0 <= len(prior["accepted"]) <= RETAINED_SAMPLES
                        or prior["accepted_total"] != sum(sum(c.values()) for c in prior["verified_by_rank_rows"].values())):
                    raise ValueError("receipt cannot be resumed with these peer settings")
                self.results = prior
                self.results.setdefault("keepalive_frames", 0)
                self.results.setdefault("peer_restarts", []).append(dict(
                    restarted_at=time.time(), negotiations=prior["negotiations"], accepted=prior["accepted_total"],
                    rejected=prior["rejected_frames"], keepalive_seconds=KEEPALIVE_SECONDS))
        super().__init__(("127.0.0.1", args.port), Handler)
        self.save()

    def save(self):
        # Same-filesystem atomic replace keeps a valid receipt on service termination.
        self.receipt.parent.mkdir(exist_ok=True, parents=True)
        with tempfile.NamedTemporaryFile("w", dir=self.receipt.parent, delete=False) as f:
            json.dump(self.results, f, indent=2)
            f.write("\n")
        os.replace(f.name, self.receipt)

    def at_limit(self):
        return not self.sustained and len(self.results["accepted"]) >= MAX_SHARES

    def record(self, row):
        # Held under peer.lock. The bounded sustained receipt keeps restart-safe
        # lifetime counters without retaining a payload per accepted share.
        if self.sustained:
            counts = self.results["verified_by_rank_rows"][str(row["tp_rank"])]
            key = str(row["proof_rows"])
            counts[key] = counts.get(key, 0) + 1
            self.results["accepted_total"] += 1
            if len(self.results["accepted"]) < RETAINED_SAMPLES:
                self.results["accepted"].append(row)
        else:
            self.results["accepted"].append(row)
        self.save()

    def verify_request(self, request, client_address):
        return self.limit.acquire(blocking=False)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.limit.release()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        peer = self.server
        self.connection.settimeout(90)
        stop = threading.Event()
        send_lock = threading.Lock()
        def send_frame(frame):
            with send_lock:
                send(self.connection, frame)

        def keepalive():
            # Informational Stratum notifications leave the replay client's job
            # and pending share acknowledgments intact.
            while not stop.wait(KEEPALIVE_SECONDS):
                try:
                    send_frame(dict(id=None, method="client.show_message", params=["fixture alive"]))
                    with peer.lock:
                        peer.results["keepalive_frames"] += 1
                        peer.save()
                except OSError:
                    return

        heartbeat = None
        try:
            auth = receive(self.rfile)
            if auth.get("method") != "mining.authorize" or auth.get("params", {}).get("type") != "v2":
                return
            send_frame(dict(id=auth["id"], result=True, error=None, type="v2"))
            send_frame(dict(method="mining.notify", params=peer.job))
            with peer.lock:
                peer.results["negotiations"] += 1
                peer.save()
            if peer.sustained:
                self.connection.settimeout(None)
                heartbeat = threading.Thread(target=keepalive, daemon=True)
                heartbeat.start()
            while True:
                frame = receive(self.rfile)
                received_at = time.time()
                try:
                    with peer.lock:
                        if peer.at_limit():
                            return
                    details = inspect_opening(frame, peer.shards)
                    row = verify_submission(frame, peer.job, peer.verifier)
                except (ValueError, KeyError, AssertionError, IndexError, OSError, EOFError, OverflowError,
                        struct.error, TypeError, RuntimeError, subprocess.SubprocessError):
                    send_frame(dict(id=frame.get("id"), result=False, error="invalid test proof"))
                    with peer.lock:
                        peer.results["rejected_frames"] += 1
                        peer.save()
                    return
                with peer.lock:
                    if peer.at_limit():
                        return
                    send_frame(dict(id=frame["id"], result=True, error=None))
                    peer.record(dict(received_at=received_at, acknowledged_at=time.time(), **details, **row))
        except (ValueError, KeyError, OSError, TypeError):
            # An idle client can disconnect or replace its generation at any time.
            return
        finally:
            stop.set()
            if heartbeat:
                heartbeat.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=31988)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sustained", action="store_true", help="accept unlimited verified synthetic shares, retaining only 16 sample records")
    parser.add_argument("--share-target", type=parse_target, help="hex base target for sustained mode; defaults to the qualification target / 65536")
    parser.add_argument("--resume-receipt", action="store_true", help="resume the same sustained job and its verified counters")
    args = parser.parse_args()
    if args.share_target is not None and not args.sustained:
        parser.error("--share-target requires --sustained")
    if args.resume_receipt and not args.sustained:
        parser.error("--resume-receipt requires --sustained")
    with Peer(args) as server:
        server.serve_forever()
