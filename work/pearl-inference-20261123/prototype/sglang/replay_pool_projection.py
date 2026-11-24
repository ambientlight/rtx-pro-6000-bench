"""Run a native eager SGLang layer and verify each submitted share on loopback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
from threading import Event
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projection_wire import SHARE_TARGET, receive, send, verify_submission


def wait_for(predicate, server=None):
    deadline = time.monotonic() + 15
    while not predicate():
        if server is not None and server.done():
            server.result()
        if time.monotonic() >= deadline:
            raise TimeoutError("native pool replay timed out")
        time.sleep(.005)


def serve(listener, verifier, advance, disconnect, modes_by_job=(("EXTEND",), ("TARGET_VERIFY", "DECODE"))):
    receipts = []
    with listener.accept()[0] as conn, conn.makefile("rb") as stream:
        conn.settimeout(20)
        auth = receive(stream)
        if auth["method"] != "mining.authorize" or auth["params"]["type"] != "v2":
            raise ValueError("expected Stratum v2 authorization")
        send(conn, dict(id=auth["id"], result=True, type="v2", error=None))
        for i, modes in enumerate(modes_by_job):
            if not advance[i].wait(20):
                raise TimeoutError("native projection did not advance to the next job")
            header = struct.pack("<I32s32sII", 1, bytes([83+i])*32, bytes([42+i])*32, 0x66666666+i, 0x1c00ffff)
            job = dict(job_id=f"native-{i}", cert_version=3, header=header.hex(), target=f"{SHARE_TARGET:064x}")
            send(conn, dict(method="mining.notify", params=job))
            for mode in modes:
                frame = receive(stream)
                row = verify_submission(frame, job, verifier)
                receipts.append(dict(mode=mode, **row))
                send(conn, dict(id=frame["id"], error=None, result=True))
        if not disconnect.wait(10):
            raise TimeoutError("native projection did not observe the last acknowledgment")
    return receipts


def run(listener, args):
    from replay_projection import ForwardMode, ReferenceConfig, RowParallelLinear, dist, parallel_state, pearl, projection, torch

    if not (pearl.OPERATOR is projection and os.environ.get("PEARL_PUBLIC_REPLAY") == "1"):
        raise ValueError("set PEARL_PUBLIC_REPLAY=1 and PEARL_WOB_MODULE=checkpoint_projection")
    if pearl.SOCKET:
        raise ValueError("use the isolated native fixture")
    os.environ["PEARL_REPLAY_POOL_HOST"] = "127.0.0.1"
    os.environ["PEARL_REPLAY_POOL_PORT"] = str(listener.getsockname()[1])
    torch.cuda.set_device(0)
    pearl._capturing = torch.cuda.is_current_stream_capturing
    advance = [Event(), Event()]
    disconnect = Event()
    modes = (("EXTEND", 16), ("TARGET_VERIFY", 6), ("DECODE", 1))
    with tempfile.TemporaryDirectory() as scratch:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=(Path(scratch)/"store").as_uri())
        group = parallel_state.GroupCoordinator(
            [[0]], 0, "nccl", use_pynccl=False, use_pymscclpp=False, use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
            use_xpu_communicator=False, use_npu_communicator=False, group_name="pearl_native_pool")
        parallel_state._TP = group
        adapter = projection._candidate(os.environ["PEARL_CHECKPOINT_DIR"], os.environ["PEARL_OPERATOR_BUILD"],
                                        args.rank, 0)
        pool = adapter.pool_layouts.pool
        layer = RowParallelLinear(projection.TP * projection.K, projection.N, bias=False,
                                  prefix="layers.0.attn.wo_b", tp_rank=args.rank, tp_size=projection.TP,
                                  quant_config=ReferenceConfig(adapter.reference_weight), params_dtype=torch.bfloat16)
        pearl.scheduler_rank(args.rank).receive([SimpleNamespace(pearl_dispatch_id=None)])
        generator = torch.Generator().manual_seed(711)
        inputs = [(torch.randn((m, projection.K), generator=generator).to(device=0, dtype=torch.bfloat16),
                   SimpleNamespace(forward_mode=ForwardMode[mode], batch_size=1, extend_num_tokens=m,
                                   spec_info=SimpleNamespace(draft_token_num=6))) for mode, m in modes]

        def forward(i):
            x, batch = inputs[i]
            return layer(x, pearl_batch=batch, skip_all_reduce=True)[0].cpu()

        try:
            with ThreadPoolExecutor(1) as thread:
                server = thread.submit(serve, listener, args.verifier, advance, disconnect)
                try:
                    assert pool.snapshot() is None
                    baseline = [forward(i) for i in range(3)]
                    counts = {"prefill": 0, "decode": 0}
                    for i, (mode, m) in enumerate(modes):
                        compact, phase = mode != "EXTEND", "prefill" if mode == "EXTEND" else "decode"
                        if i < 2:
                            advance[i].set()
                            wait_for(lambda: pool.snapshot() is not None and pool.snapshot().id == f"native-{i}")
                            # The first forward on an incoming job schedules host preparation.
                            torch.testing.assert_close(forward(i), baseline[i], rtol=0, atol=0)
                        wait_for(lambda: adapter.pool_layouts.select(compact, torch.cuda.current_stream()) is not None)
                        if i == 0:
                            # Fill the proof queue while a valid job exists and check that
                            # the real SGLang layer uses the calibrated plain multiply.
                            with adapter._proof_slots, adapter._proof_slots:
                                torch.testing.assert_close(forward(i), baseline[i], rtol=0, atol=0)
                            assert adapter.busy_skips == 1
                        torch.testing.assert_close(forward(i), baseline[i], rtol=0, atol=0)
                        counts[phase] += 1
                        wait_for(lambda: pool.counters()[phase]["ack"] == counts[phase], server)
                    counters = pool.counters()
                finally:
                    for gate in advance:
                        gate.set()
                    disconnect.set()
                receipts = server.result()
            wait_for(lambda: pool.snapshot() is None)
            torch.testing.assert_close(forward(2), baseline[2], rtol=0, atol=0)
            return dict(scope="native eager target layer, one checkpoint shard, public synthetic BF16 inputs",
                        rank=args.rank, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), counters=counters,
                        exact_vs_unmined_by_mode={mode: m * projection.N for mode, m in modes},
                        saturated_queue_exact=True, skipped_for_proof_backpressure=adapter.busy_skips,
                        jobless_decode_after_disconnect_exact=True, accepted=receipts)
        finally:
            adapter.close()
            group.destroy()
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=3)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        sock.settimeout(20)
        result = run(sock, args)
    print(json.dumps(result), flush=True)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
