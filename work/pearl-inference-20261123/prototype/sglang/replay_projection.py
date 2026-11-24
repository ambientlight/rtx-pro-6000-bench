"""Run the patched SGLang row-parallel forward body on public synthetic activations."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import checkpoint_projection as projection
from sglang.srt import pearl_runtime as pearl
from sglang.srt.distributed import parallel_state
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from joint_replay import Joint


class ReferenceConfig:
    """SGLang quant-method fixture for the unpacked FP8 checkpoint reference."""
    def __init__(self, weight):
        self.weight = weight

    def get_quant_method(self, layer, prefix):
        return self

    def create_weights(self, layer, params_dtype, **kwargs):
        layer.register_parameter("weight", torch.nn.Parameter(
            torch.from_numpy(self.weight).to(dtype=params_dtype), requires_grad=False))

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-rank", type=int, choices=range(4), default=3)
    p.add_argument("--control-dir", type=Path, help="private directory from joint_replay.py")
    args = p.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world not in (1, 4):
        p.error("use one local checkpoint shard or four matching TP4 ranks")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    rank = int(os.environ.get("RANK", 0))
    checkpoint_rank = rank if world == 4 else args.checkpoint_rank
    scratch = tempfile.TemporaryDirectory()
    method = "env://" if world == 4 else Path(scratch.name, "rendezvous").as_uri()
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=method)
    group = parallel_state.GroupCoordinator(
        [list(range(world))], local, "nccl",
        use_pynccl=False, use_pymscclpp=False, use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
        use_xpu_communicator=False, use_npu_communicator=False,
        group_name="pearl_replay_tp",
    )
    parallel_state._TP = group
    if pearl.OPERATOR is not projection or os.environ.get("PEARL_PUBLIC_REPLAY") != "1":
        raise ValueError("set PEARL_WOB_MODULE=checkpoint_projection and PEARL_PUBLIC_REPLAY=1")
    # No model runner is booted in this process. Use the real CUDA stream
    # capture predicate in place of its optional graph-runner global flag.
    pearl._capturing = torch.cuda.is_current_stream_capturing
    lifecycle = pearl.scheduler_rank(checkpoint_rank)
    joint = Joint(args.control_dir) if args.control_dir else None
    wire = SimpleNamespace(pearl_dispatch_id=None)
    adapter = projection._candidate(os.environ["PEARL_CHECKPOINT_DIR"], os.environ["PEARL_OPERATOR_BUILD"],
                                    checkpoint_rank, local)
    layer = RowParallelLinear(projection.TP * projection.K, projection.N, bias=False,
                              prefix="layers.0.attn.wo_b", tp_rank=checkpoint_rank,
                              quant_config=ReferenceConfig(adapter.reference_weight),
                              tp_size=projection.TP, params_dtype=torch.bfloat16)
    if joint:
        if world != 4 or Path(pearl.SOCKET) != joint.broker:
            raise ValueError("joint replay requires four ranks and its private broker")
        lifecycle.idle(torch.cuda.synchronize)
        dist.barrier()
        child = [asyncio.run(joint.arrive(pearl, wire)) if rank == 0 else None]
        dist.broadcast_object_list(child, src=0)
        wire.pearl_dispatch_id = child[0]
    else:
        layer = layer.to(device=local)
    lifecycle.receive([wire])
    proofs = []
    adapter.on_witness = lambda phase, data: proofs.append((phase, len(data)))
    for mode, m in (("EXTEND", 16), ("TARGET_VERIFY", 6), ("DECODE", 1)):
        rng = np.random.default_rng(1017 + 23 * checkpoint_rank + m)
        x = torch.from_numpy(rng.standard_normal((m, projection.K), dtype=np.float32)).to(device=local, dtype=torch.bfloat16)
        batch = SimpleNamespace(forward_mode=ForwardMode[mode], batch_size=1, extend_num_tokens=m,
                                spec_info=SimpleNamespace(draft_token_num=6))
        # Independently compute expected un-noised quantized integer products.
        q, scales = projection.quantize_rows(x.float().cpu().numpy())
        exact = np.rint(q.astype("d") @ adapter.bt.astype("d").T).astype("f")
        expected = torch.from_numpy((exact * scales) * adapter.scale_b.T).to(device=local, dtype=x.dtype)
        actual, bias = layer(x, pearl_batch=batch, skip_all_reduce=world == 1)
        if world == 4:
            dist.all_reduce(expected, group=group.device_group)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        assert bias is None
        # Compare to the unmined BF16 projection of the dequantized checkpoint.
        # This excludes production FP8 activation rounding/kernel selection.
        delta = None
        if joint:
            if rank == 0:
                asyncio.run(joint.held(mode))
        else:
            reference, _ = layer(x, skip_all_reduce=world == 1)
            delta = float((actual.float() - reference.float()).norm() / reference.float().norm())
        print(json.dumps(dict(row_parallel_forward=mode, checkpoint_rank=checkpoint_rank, world=world,
                              relative_l2_dequantized_bf16=delta,
                              input_rows=m, exact_bf16_outputs=actual.numel(),
                              reduction="NCCL all-reduce" if world == 4 else "local shard")), flush=True)
    if joint:
        dist.barrier()
        before_idle = time.monotonic_ns()
        lifecycle.idle(torch.cuda.synchronize)
        dist.barrier()
        if rank == 0:
            asyncio.run(joint.resumed(before_idle))
        # Keep persistent rank streams alive while rank 0 checks resumption.
        dist.barrier()
    adapter.close()
    assert sorted(phase for phase, size in proofs if size > 0) == ["decode", "decode", "prefill"]
    group.destroy()
    dist.destroy_process_group()
    scratch.cleanup()


if __name__ == "__main__":
    main()
