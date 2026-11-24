"""Verify a single loopback DeepSeek HTTP request earns both kinds of public replay proof.

Use the model created by make_smoke_model.py --deepseek-config ... --pearl-projection.
Launch SGLang at TP1 with --load-format dummy, eager prefill/decode, one running request,
PEARL_PUBLIC_REPLAY=1, PEARL_GPU_PREPARE=1, PEARL_WOB_MODULE=checkpoint_projection,
PEARL_SHAPE_DIR and PEARL_REPLAY_POOL_PORT matching this fixture. PEARL_CHECKPOINT_DIR
must name the real V4.1-Flash-1017 weights. The operator selects its TP4 rank-zero slice.
"""

# ruff: noqa: E402 -- import the repository's existing proof verifier client.
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
from threading import Thread
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "work/pearl-inference-20261123/prototype"))
from checkpoint_projection import checkpoint_weight, quantize_rows
from projection_wire import SHARE_TARGET, receive, send, verify_submission


def matrix_opening(raw, pos):
    # Bincode serializes a Vec of length-prefixed chunks followed by leaf
    # indices, leaf count, root, siblings and matrix row indices.
    (count,) = struct.unpack_from("<Q", raw, pos)
    chunks = []
    pos += 8
    for _ in range(count):
        (length,) = struct.unpack_from("<Q", raw, pos)
        pos += 8
        assert length == 1024
        chunks.append(raw[pos : pos + length])
        pos += length
    (indices_count,) = struct.unpack_from("<Q", raw, pos)
    assert indices_count == count
    indices = struct.unpack_from(f"<{count}Q", raw, pos + 8)
    pos += 8 + count * 8 + 8 + 32
    (siblings,) = struct.unpack_from("<Q", raw, pos)
    pos += 8 + siblings * 32
    (rows,) = struct.unpack_from("<Q", raw, pos)
    return chunks, indices, pos + 8 + rows * 8


def opened_matrices(frame, checkpoint_bytes):
    raw = gzip.decompress(base64.b64decode(frame["params"]["plain_proof"], validate=True))
    m, n, k, noise_rank = struct.unpack_from("<4Q", raw)
    assert (n, k, noise_rank) == (5120, 2048, 128)
    activations, _, pos = matrix_opening(raw, 32)
    weights, indices, end = matrix_opening(raw, pos)
    assert end + 1 == len(raw) and raw[end] == 0  # Dense proof with no optional MoE metadata.
    seen = set(b"".join(activations))
    assert len(seen) > 1, "the hybrid model produced a constant activation opening"
    for chunk, index in zip(weights, indices, strict=True):
        assert checkpoint_bytes[index * 1024 : (index + 1) * 1024] == chunk
    return dict(
        proof_rows=m,
        output_width=n,
        input_width=k,
        opened_activation_codes=len(seen),
        checkpoint_weight_bytes_matched=len(weights) * 1024,
    )


def peer(listener, verifier, proofs, checkpoint_bytes):
    with listener.accept()[0] as conn, conn.makefile("rb") as stream:
        auth = receive(stream)
        assert auth["method"] == "mining.authorize" and auth["params"]["type"] == "v2"
        send(conn, dict(id=auth["id"], result=True, error=None, type="v2"))
        head = struct.pack(
            "<I32s32sII", 1, bytes([71]) * 32, bytes([37]) * 32, 0x66666665, 0x1C00FFFF
        )
        job = dict(
            job_id="public-http-0", cert_version=3, header=head.hex(), target=f"{SHARE_TARGET:064x}"
        )
        send(conn, dict(method="mining.notify", params=job))
        while True:
            frame = receive(stream)
            row = verify_submission(frame, job, verifier)
            row.update(opened_matrices(frame, checkpoint_bytes))
            send(conn, dict(id=frame["id"], error=None, result=True))
            proofs.append(row)
            if row["proof_rows"] == 2:
                return


def run(args, sock):
    proofs, failures = [], []
    bt, _ = quantize_rows(checkpoint_weight(args.real_model, 0))
    checkpoint_bytes = memoryview(bt).cast("B")

    def receiver():
        try:
            peer(sock, args.verifier, proofs, checkpoint_bytes)
        except Exception as exc:
            failures.append(exc)

    thread = Thread(target=receiver, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{args.http_port}"

    def post(ids, outputs):
        body = dict(
            input_ids=ids,
            sampling_params=dict(temperature=0, max_new_tokens=outputs, ignore_eos=True),
        )
        request = urllib.request.Request(
            base + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=480) as http:
            out = json.load(http)
            assert (
                http.status == 200
                and out["meta_info"]["completion_tokens"] == len(out["output_ids"]) == outputs
            )
            return dict(
                http_status=http.status,
                prompt_tokens=out["meta_info"]["prompt_tokens"],
                completion_tokens=out["meta_info"]["completion_tokens"],
            )

    def shapes():
        files = list(args.shape_dir.glob("pearl-shapes-*-tp0.json"))
        assert len(files) == 1, "use an isolated fresh shape directory"
        return {
            (r["mode"], r["real_rows"], r["executed_rows"]): r["calls"]
            for r in json.loads(files[0].read_text())
        }

    # All warmups generate exactly one output, hence execute no decode forwards.
    # They also use disjoint prefill shapes. A measured 16-row or two-plane
    # singleton proof cannot have originated in a warmup request.
    for width in (32, 4):
        for trial in range(8):
            if failures:
                raise failures[0]
            ids = [6 + trial + (8 if width == 4 else 0)] + [4] * (width - 1)
            post(ids, 1)
            time.sleep(0.35)
            if any(row["proof_rows"] == width for row in proofs):
                break
        else:
            raise ValueError(f"no independently verified warmup proof for width {width}")
    before = shapes()
    first = time.monotonic_ns()
    response = post([28] + [4, 5, 6] * 5, 2)
    elapsed = (time.monotonic_ns() - first) / 1e6
    thread.join(15)
    if failures:
        raise failures[0]
    assert not thread.is_alive()
    measured = [r for r in proofs if r["proof_rows"] in (16, 2)]
    assert [r["proof_rows"] for r in measured] == [16, 2]
    expected = {("EXTEND", 16, 16): 1, ("DECODE", 1, 1): 1}
    for _ in range(150):
        after = shapes()
        delta = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
        if delta == expected:
            break
        time.sleep(0.02)
    assert delta == expected
    with urllib.request.urlopen(base + "/server_info", timeout=10) as http:
        info = json.load(http)
    from sglang.srt import pearl_runtime

    source = Path(pearl_runtime.__file__).resolve().parents[3]
    return dict(
        scope="One two-layer hybrid DeepSeek V4.1 HTTP request, TP1, synthetic public token IDs; a local independently verifying Stratum peer",
        candidate=subprocess.check_output(
            ["git", "-C", source, "rev-parse", "HEAD"], text=True
        ).strip(),
        source_base="c64f2256d58358ff65c53f0e38dc3e6bcfac8484",
        checkpoint_projection="DeepSeek-V4.1-Flash-1017 1bae0e1088993b9b82de08f59de5be5a34e42170, layers.0.attn.wo_b, TP4 rank-zero slice loaded and INT7-calibrated in the operator",
        model_other_parameters="Native dummy FP8/FP4 weight initialization with reduced layer, head, expert and vocab counts",
        hidden_size=5120,
        activation_width=2048,
        layers=2,
        tp=1,
        target_layer=0,
        serve_model=info["served_model_name"],
        load_format=info["load_format"],
        public_request=dict(**response, wall_ms=elapsed),
        observed_target_layer=[
            dict(mode=k[0], real_rows=k[1], executed_rows=k[2], calls=v) for k, v in delta.items()
        ],
        independently_verified=[
            dict(actual_phase=phase, **row) for phase, row in zip(("prefill", "decode"), measured)
        ],
        warmups="Only 32- or 4-token prefills generating one output. Both layout types were independently verified before the measured request.",
        target=f"{SHARE_TARGET:064x}",
        certificate_version=3,
        verifier_sha256=hashlib.sha256(args.verifier.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-model",
        type=Path,
        required=True,
        help="The full V4.1-Flash-1017 checkpoint for independently checking weight openings",
    )
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--pool-port", type=int, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.pool_port))
        listener.listen(1)
        result = run(args, listener)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
