"""Run four supervised offline miners, a private broker, and a native TP4 layer replay."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "miner"))
from fleet_replay import exchange, until_resumed


class Joint:
    def __init__(self, folder):
        self.broker = folder / "broker.sock"
        self.workers = [folder / f"gpu{i}.sock" for i in range(4)]
        self.steps = {}
        self.slices = None

    async def arrive(self, runtime, wire):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            rows = await asyncio.gather(*(exchange(path) for path in self.workers))
            # Idle workers may already be in their next GPU slice.
            if all(r["completed_slices"] and r["require_watch"] and r["watched"]
                   and not r["holds"] and not r["faulted"] for r in rows):
                break
            await asyncio.sleep(.002)
        else:
            raise TimeoutError("miners did not complete an initial supervised slice")
        arrival = time.monotonic_ns()
        lease = runtime.Admission(self.broker)
        try:
            await lease.start()
            await lease.wait()
            self.steps["arrival_to_drained_ms"] = (time.monotonic_ns() - arrival) / 1e6
            await lease.dispatch(lambda payload: None, wire, [wire])
        finally:
            await lease.close()
        await self.held("arrival")
        return wire.pearl_dispatch_id

    async def held(self, phase):
        rows = await asyncio.gather(*(exchange(path) for path in self.workers))
        assert all(r["holds"] and r["watched"] and r["require_watch"]
                   and not r["faulted"] and not r["gpu_busy"] for r in rows)
        counters = [r["completed_slices"] for r in rows]
        if self.slices is None:
            self.slices = counters
        assert counters == self.slices, f"direct GPU work advanced during {phase}"
        self.steps.setdefault("held_slices", {})[phase] = counters

    async def resumed(self, start):
        rows = await until_resumed(self.workers, start)
        assert all(r["watched"] and r["require_watch"] and not r["holds"] for r in rows)
        assert all(r["completed_slices"] > s for r, s in zip(rows, self.slices))
        self.steps["resumed_slices"] = [r["completed_slices"] for r in rows]
        self.steps["rank0_before_fence_to_all_workers_completed_first_slice_ms"] = (
            max(r["first_slice_end_ns"] for r in rows) - start) / 1e6
        print(json.dumps({"joint_replay": self.steps}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker", type=Path)
    p.add_argument("--gpu", nargs=4, required=True)
    p.add_argument("--slice-tiles", type=int, default=188)
    args = p.parse_args()
    if len(set(args.gpu)) != 4:
        p.error("select four distinct GPUs")
    with tempfile.TemporaryDirectory(prefix="pearl-joint-") as tmp, ExitStack() as stack:
        folder = Path(tmp)
        fixture = Joint(folder)
        processes = []
        def start(argv, env, name):
            logfile = stack.enter_context((folder / f"{name}.log").open("w"))
            proc = subprocess.Popen(argv, env=env, stdout=logfile, stderr=subprocess.STDOUT)
            processes.append(proc)
            return proc
        def wait_paths(paths):
            deadline = time.monotonic() + 30
            while not all(path.exists() for path in paths):
                if time.monotonic() > deadline or any(proc.poll() is not None for proc in processes):
                    raise RuntimeError("offline fixture failed to start")
                time.sleep(.02)
        try:
            for i, gpu in enumerate(args.gpu):
                start([str(args.worker.resolve()), "--bench-rounds", "200", "--bench-job", str(71+i),
                       "--bench-nonce", "1001", "--kernel-variant", "v1", "--k", "65536",
                       "--control-socket", str(fixture.workers[i]), "--slice-tiles", str(args.slice_tiles),
                       "--require-watch", "--device", "0", "--power-limit", "280", "--max-temp", "92"],
                      {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "RAYON_NUM_THREADS": "2"}, f"miner{i}")
            wait_paths(fixture.workers)
            argv = [sys.executable, str(HERE.parent / "broker.py"), "--socket", str(fixture.broker), "--watch-workers"]
            for worker in fixture.workers:
                argv.extend(["--worker", str(worker)])
            start(argv, os.environ, "broker")
            wait_paths([fixture.broker])
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
                            str(HERE / "replay_projection.py"), "--control-dir", tmp],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(args.gpu),
                                "PEARL_BROKER_SOCKET": str(fixture.broker)}, check=True, timeout=180)
        finally:
            for proc in reversed(processes):
                proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
