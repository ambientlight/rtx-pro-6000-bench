"""Single-GPU capture/replay smoke test on a real TP4 checkpoint shard and two public fixture jobs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from checkpoint_graph import GraphVerify
from checkpoint_projection import HEADER, K, Projection, quantize_rows
from replay_pool import parse_job


class LocalFixture:
    def __init__(self):
        self.job = None
        self.witnesses = []

    def rotate(self, serial):
        header = bytearray(HEADER)
        header[4] = serial
        target = (1 << 256) // (2 * 64 * 2048) - 1
        self.job = parse_job(dict(job_id=str(serial), cert_version=3, header=header.hex(),
                                  target=f"{target:064x}"), serial)

    def snapshot(self):
        return self.job

    def submit(self, job, phase, witness):
        assert phase == "decode" and self.job is job
        self.witnesses.append((job.id, hashlib.sha256(witness).hexdigest()))
        return True

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, choices=range(6, 49, 6), default=6)
    args = parser.parse_args()
    os.environ["PEARL_GPU_PREPARE"] = "1"
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    pool = LocalFixture()
    rng = np.random.default_rng(227)
    with Projection(args.model, args.build, args.rank, args.device, replay_pool=pool) as adapter:
        graph = GraphVerify(adapter, args.build, device)
        x = torch.zeros((args.rows, K), device=device, dtype=torch.bfloat16)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            graph.capture(x)
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph, stream=stream):
                output = graph.capture(x)
            assert not pool.witnesses

            def replay(expected, *, eligible=True):
                activation = torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32)).to(torch.bfloat16)
                x.copy_(activation)
                path, proof = graph.begin(args.rows, args.rows, eligible)
                assert path == expected, (path, expected)
                cuda_graph.replay()
                graph.end(proof, True)
                q, scale = quantize_rows(activation.float().numpy())
                product = q.astype(np.int32) @ adapter.bt.astype(np.int32).T
                reference = torch.from_numpy(product.astype(np.float32) * scale * adapter.scale_b.T).to(torch.bfloat16)
                assert torch.equal(output.cpu(), reference), "graph result differs from corrected checkpoint projection"

            replay("target_graph_no_job")
            for serial in (1, 2):
                pool.rotate(serial)
                deadline = time.monotonic() + 90
                while adapter.pool_layouts.select(True, stream) is None and time.monotonic() < deadline:
                    time.sleep(.01)
                replay("target_graph_mined")
                # Proof completion includes independent Rust verification of the GPU commitments and ticket.
                while len(pool.witnesses) < serial and time.monotonic() < deadline:
                    time.sleep(.01)
                assert len(pool.witnesses) == serial, graph.stats
                replay("target_graph_ineligible", eligible=False)
            # Reserve both slots explicitly so this replay must use the captured plain branch.
            assert adapter._proof_slots.acquire(timeout=5) and adapter._proof_slots.acquire(timeout=5)
            try:
                replay("target_graph_backpressure")
            finally:
                adapter._proof_slots.release()
                adapter._proof_slots.release()
        assert [job for job, _ in pool.witnesses] == ["1", "2"]
        assert len({digest for _, digest in pool.witnesses}) == 2
        assert graph.stats["tickets_scanned"] == 2 * (args.rows // 2) * (5120 // 16), graph.stats
        print(json.dumps(dict(rows=args.rows, rank=args.rank, independent_witnesses=len(pool.witnesses),
                              equal_outputs=6, graph_stats=dict(graph.stats))))


if __name__ == "__main__":
    main()
