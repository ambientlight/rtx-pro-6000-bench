"""Compare plain CUDA projections with exact un-noised CPU products on public inputs."""
import argparse
import json
import statistics

import numpy as np
import torch

from pearl_operator import Operator

K = 2048


def check(op, rng, m, n, repeats):
    a = rng.integers(-64, 65, (m, K), dtype=np.int8)
    bt = rng.integers(-64, 65, (n, K), dtype=np.int8)
    a[0] = 64
    bt[0] = -64
    if m > 1:
        a[1] = 0
    # Signed integer sums stay within exact FP64 range. This BLAS routine
    # computes *every* output independently of CUDA, noise, and packing.
    expected = (a.astype("d") @ bt.astype("d").T).astype("i")
    stream = torch.cuda.Stream()
    sentinel = -123456789
    with torch.cuda.stream(stream):
        da = torch.from_numpy(a).cuda()
        db = torch.from_numpy(bt).cuda()
        guarded = torch.full((m * n + 32,), sentinel, dtype=torch.int32, device="cuda")
        out = guarded[16:-16].view(m, n)
        def launch():
            op.launch_plain(da.data_ptr(), db.data_ptr(), out.data_ptr(), m, n, stream.cuda_stream)
        launch()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        if not torch.all(guarded[:16] == sentinel) or not torch.all(guarded[-16:] == sentinel):
            raise AssertionError("plain kernel wrote outside the real output extent")
        timings = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            launch()
            end.record(stream)
            end.synchronize()
            timings.append(start.elapsed_time(end))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        # Replay sees a changed real activation on the caller's non-default stream.
        da[0].zero_()
        graph.replay()
        expected[0].fill(0)
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
    return dict(m=m, n=n, k=K, exact_integer_outputs=m*n, changed_input_graph_replay=True,
                output_extent_guard=True, warm_cuda_event_median_ms=statistics.median(timings),
                warm_cuda_events_ms=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("use a positive repeat count")
    op = Operator(args.build)
    rng = np.random.default_rng(341607)
    results = [check(op, rng, m, n, args.repeats)
               for m,n in ((1,5120), (3,5120), (6,5120), (17,257), (2048,5120))]
    try:
        op.launch_plain(0, 0, 0, 1, 5120)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing input validation")
    print(json.dumps(dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                          torch=torch.__version__, scope="plain INT7 operands, GPU timing at existing host load",
                          rejected_null_buffers=True, results=results), indent=2))


if __name__ == "__main__":
    main()
