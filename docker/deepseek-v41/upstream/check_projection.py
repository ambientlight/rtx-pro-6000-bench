"""Replay a real DeepSeek wo_b shard and validate each public proof on loopback.

Run from the candidate source environment on one GPU with PEARL_PUBLIC_REPLAY=1,
PEARL_GPU_PREPARE=1 and PEARL_WOB_MODULE=checkpoint_projection. MODEL_PATH and
PEARL_CHECKPOINT_DIR point to the same DeepSeek V4.1 checkpoint.
PEARL_OPERATOR_BUILD contains the three compiled prototype libraries.
"""
# ruff: noqa: E402 -- the standalone fixture adds the benchmark's prototype to sys.path.

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[2] / "work/pearl-inference-20261123/prototype"
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(PROTO / "sglang"))

import checkpoint_projection as projection
from boot import MODEL, REVISION, server_arguments
from replay_pool_projection import serve, wait_for
from sglang.srt import pearl_runtime as pearl
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import (
    get_fp8_gemm_runner_backend,
    initialize_fp8_gemm_config,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.srt.models.deepseek_v4_mhc import AttnOutput
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
from sglang.srt.server_args import prepare_server_args


MODES = (("EXTEND", 16), ("TARGET_VERIFY", 6), ("DECODE", 1))


def native_layer(rank):
    config = Fp8Config.from_config(
        json.loads((MODEL / "config.json").read_text())["quantization_config"]
    )
    prefix = "layers.0.attn.wo_b"
    with torch.device("cuda"):
        layer = RowParallelLinear(
            projection.TP * projection.K,
            projection.N,
            bias=False,
            prefix=prefix,
            parallel_group="attn_tp",
            reduce_results=True,
            params_dtype=torch.bfloat16,
            quant_config=config,
        )
    assert get_group_rank_size(layer.tp_group) == (rank, projection.TP)
    files = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for suffix, param in (("weight", layer.weight), ("scale", layer.weight_scale_inv)):
        key = f"{prefix}.{suffix}"
        with safe_open(MODEL / files[key], framework="pt", device="cpu") as file:
            data = file.get_tensor(key)
            param.weight_loader(param, data)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def inputs(modes=MODES, seed=711):
    rng = torch.Generator().manual_seed(seed)
    for mode, m in modes:
        x = torch.randn((m, projection.K), generator=rng).to(device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(m, device="cuda", dtype=torch.int64)
        lengths = torch.tensor([m], device="cuda", dtype=torch.int32)
        yield (
            x,
            ForwardBatch(
                forward_mode=ForwardMode[mode],
                batch_size=1,
                input_ids=indices,
                req_pool_indices=indices[:1],
                seq_lens=lengths,
                out_cache_loc=indices,
                seq_lens_sum=m,
                extend_num_tokens=m,
                global_num_token_non_padded_cpu=m,
                spec_info=DFlashVerifyInput(
                    draft_token=indices, positions=indices, draft_token_num=m
                )
                if mode == "TARGET_VERIFY"
                else None,
            ),
        )


def metrics(actual, stock):
    actual, stock = actual.float().flatten(), stock.float().flatten()
    return dict(
        relative_l2=((actual - stock).norm() / stock.norm()).item(),
        cosine=torch.nn.functional.cosine_similarity(actual, stock, dim=0).item(),
    )


def independent_int7(x, adapter):
    q, scale = projection.quantize_rows(x.float().cpu().numpy())
    sums = (q.astype(np.float64) @ adapter.bt.astype(np.float64).T).astype(np.float32)
    return torch.from_numpy((sums * scale) * adapter.scale_b.T).bfloat16()


def bench_pair(mode, rows, pairs, model, project_shard, await_ack):
    samples = {"stock_fp8": [], "mined_int7": []}
    for j in range(pairs + 2):
        x, batch = next(inputs(((mode, rows),), seed=103 + 13 * j + rows))
        torch.cuda.current_stream().synchronize()
        order = ("stock_fp8", "mined_int7") if j % 2 == 0 else ("mined_int7", "stock_fp8")
        for arm in order:
            model.pearl_target_layer = arm == "mined_int7"
            start = time.perf_counter_ns()
            out = project_shard(x, batch)
            torch.cuda.current_stream().synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if j >= 2:
                samples[arm].append(elapsed)
            del out
        model.pearl_target_layer = True
        await_ack()
    return dict(
        mode=mode,
        rows=rows,
        samples_ms=samples,
        median_ms={k: float(np.median(v)) for k, v in samples.items()},
        paired_median_delta_ms=float(
            np.median(np.subtract(samples["mined_int7"], samples["stock_fp8"]))
        ),
    )


def check(args, sock):
    assert MODEL.samefile(os.environ["PEARL_CHECKPOINT_DIR"])
    assert pearl.OPERATOR is projection and os.environ["PEARL_PUBLIC_REPLAY"] == "1"
    assert os.environ["PEARL_GPU_PREPARE"] == "1" and not pearl.SOCKET
    torch.cuda.set_device(0)
    # Identify colocated work without capturing arguments, request content, or pool identity.
    gpu_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-compute-apps=process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()
    colocated = dict(
        direct_miner_seen=any(Path(name.strip()).name == "bpm" for name in apps),
        serving_scheduler_seen=any("sglang::scheduler" in name for name in apps),
    )
    os.environ["PEARL_REPLAY_POOL_HOST"] = "127.0.0.1"
    os.environ["PEARL_REPLAY_POOL_PORT"] = str(sock.getsockname()[1])
    source = Path(pearl.__file__).resolve().parents[3]
    candidate = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    config = prepare_server_args(server_arguments())
    publish(config, role="scheduler", ranks=SpawnRanks(world_rank=args.rank, gpu_id=0))
    pearl.validate_config()
    initialize_fp8_gemm_config()
    # c64 constructs a logical partition owner from published ranks when a
    # device group has not been created. Only the local projection runs here.
    with get_parallel().override(attn_tp_group=None, tp_group=None), torch.inference_mode():
        layer = native_layer(args.rank)
        model = SimpleNamespace(wo_b=layer, attn_tp_size=projection.TP, pearl_target_layer=True)
        cases = tuple(inputs())

        def project_shard(x, batch):
            out = MQALayer._project_wo_b(model, x, defer_all_reduce=True, forward_batch=batch)
            assert isinstance(out, AttnOutput)
            return out.partial

        def forward(i, *, batch_override=None):
            x, batch = cases[i]
            return project_shard(x, batch if batch_override is None else batch_override).cpu()

        stock = [forward(i) for i in range(len(MODES))]
        lifecycle = pearl.scheduler_rank(args.rank)
        lifecycle.receive([SimpleNamespace(pearl_dispatch_id=None)])
        adapter = projection._candidate(
            os.environ["PEARL_CHECKPOINT_DIR"], os.environ["PEARL_OPERATOR_BUILD"], args.rank, 0
        )
        gates = [Event(), Event()]
        disconnect = Event()
        pool = adapter.pool_layouts.pool
        try:
            assert adapter.gpu_prepare is not None and pool.snapshot() is None
            plain = [independent_int7(x, adapter) for x, _ in cases]
            for i, (_, batch) in enumerate(cases):
                torch.testing.assert_close(forward(i), plain[i], rtol=0, atol=0)
                # A padded batch must stay on native stock FP8.
                padded = replace(batch, _original_forward_mode=batch.forward_mode)
                torch.testing.assert_close(
                    forward(i, batch_override=padded), stock[i], rtol=0, atol=0
                )
            short_prefill = replace(cases[0][1], global_num_token_non_padded_cpu=12)
            torch.testing.assert_close(
                forward(0, batch_override=short_prefill), stock[0], rtol=0, atol=0
            )
            for i, (x, batch) in enumerate(cases):
                sharded = replace(batch, attn_tp_sequence_sharded=True)
                torch.testing.assert_close(
                    forward(i, batch_override=sharded), stock[i], rtol=0, atol=0
                )
                codes_scales = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
                torch.testing.assert_close(
                    project_shard(codes_scales, batch).cpu(), stock[i], rtol=0, atol=0
                )
            ragged = RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=[4], device=cases[1][0].device, grid=[6]
            )
            batch = replace(cases[1][1], ragged_verify_layout=ragged)
            torch.testing.assert_close(forward(1, batch_override=batch), stock[1], rtol=0, atol=0)
            model.pearl_target_layer = False
            for i in range(len(cases)):
                torch.testing.assert_close(forward(i), stock[i], rtol=0, atol=0)
            model.pearl_target_layer = True
            # Use the same jobs and Stratum receiver for the timed forwards. Each
            # sampled pair changes input. Wait for its verified ACK outside timing.
            timed_rows = (2048, 6, 1)
            schedules = [[], []]
            for i, (mode, _) in enumerate(MODES):
                schedule = schedules[min(i, 1)]
                schedule.append(mode)
                schedule.extend(f"{mode}/warm-or-timed-{j}" for j in range(args.pairs + 2))
            timings = []
            with ThreadPoolExecutor(1) as threads:
                server = threads.submit(serve, sock, args.verifier, gates, disconnect, schedules)
                try:
                    expected_acks = {"prefill": 0, "decode": 0}
                    for i, (mode, _) in enumerate(MODES):
                        compact = mode != "EXTEND"
                        if i < 2:
                            gates[i].set()
                            wait_for(
                                lambda: (
                                    pool.snapshot() is not None
                                    and pool.snapshot().id == f"native-{i}"
                                ),
                                server,
                            )
                            torch.testing.assert_close(forward(i), plain[i], rtol=0, atol=0)
                        wait_for(
                            lambda: (
                                adapter.pool_layouts.select(compact, torch.cuda.current_stream())
                                is not None
                            ),
                            server,
                        )
                        if i == 0:
                            with adapter._proof_slots, adapter._proof_slots:
                                torch.testing.assert_close(forward(i), plain[i], rtol=0, atol=0)
                            assert adapter.busy_skips == 1
                        torch.testing.assert_close(forward(i), plain[i], rtol=0, atol=0)
                        phase = "decode" if compact else "prefill"

                        def await_ack():
                            expected_acks[phase] += 1
                            wait_for(
                                lambda: pool.counters()[phase]["ack"] == expected_acks[phase],
                                server,
                            )

                        await_ack()
                        timings.append(
                            bench_pair(
                                mode, timed_rows[i], args.pairs, model, project_shard, await_ack
                            )
                        )
                    counters = pool.counters()
                finally:
                    for gate in gates:
                        gate.set()
                    disconnect.set()
                all_accepted = server.result()
                accepted = [row for row in all_accepted if "/" not in row["mode"]]
            wait_for(lambda: pool.snapshot() is None)
            torch.testing.assert_close(forward(2), plain[2], rtol=0, atol=0)
            lifecycle.idle(torch.cuda.synchronize)
            torch.testing.assert_close(forward(2), stock[2], rtol=0, atol=0)
            old = json.loads((PROTO / "operator/gpu_prepare/native-pool.json").read_text())
            digest_key = "proof_sha256"
            hashes_match = [r[digest_key] for r in old["accepted"]] == [
                r[digest_key] for r in accepted
            ]
            if args.rank == old["rank"]:
                assert hashes_match
            return dict(
                scope="One logical TP4 target shard on one SM120 GPU, without a TP collective or HTTP",
                source=str(source),
                candidate=candidate,
                base="c64f2256d58358ff65c53f0e38dc3e6bcfac8484",
                model=MODEL.name,
                revision=REVISION,
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                device=torch.cuda.get_device_name(),
                rank=args.rank,
                packages={
                    p: importlib.metadata.version(p)
                    for p in ("torch", "sglang-kernel", "flashinfer-python")
                },
                entry="MQALayer._project_wo_b -> RowParallelLinear.forward -> pearl_runtime.project",
                input="seeded synthetic BF16 activations",
                preparation="GPU keyed BLAKE3 and v3 noise",
                quant_method=type(layer.quant_method).__name__,
                fp8_backend=get_fp8_gemm_runner_backend().value,
                fp8_weight_shape=list(layer.weight.shape),
                stock_block_size=layer.quant_method.weight_block_size,
                stock_fp8_kernel=layer.quant_method.w8a8_block_fp8_linear.func.__name__,
                stock_block_as_mxfp8=getattr(layer, "block_fp8_mxfp8_ready", False),
                per_mode={
                    name: dict(
                        rows=m,
                        corrected_elements_exact=plain[i].numel(),
                        stock_fp8_comparison=metrics(plain[i], stock[i]),
                    )
                    for i, (name, m) in enumerate(MODES)
                },
                jobless_and_backpressured_int7_exact=True,
                padded_and_nontarget_fp8_exact=True,
                idle_fp8_exact=True,
                proof_queue_skips=adapter.busy_skips,
                counters=counters,
                accepted=accepted,
                sharded_and_short_prefill_fp8_exact=True,
                ragged_verify_padding_fp8_exact=True,
                stock_prequantized_tuple_fp8_exact=True,
                deferred_reduction_type="AttnOutput.partial",
                warm_local_timing=dict(
                    pairs=args.pairs,
                    warmup_pairs=2,
                    colocated_at_start=colocated,
                    boundary="Python model projection entry to serving-stream completion, no TP all-reduce",
                    workload="Paired seeded inputs, alternating order, empty proof queue; each mined call had its wire ACK independently verified outside timing",
                    extra_wire_acks=len(all_accepted) - len(accepted),
                    cases=timings,
                ),
                matches_pinned_reference_proofs=hashes_match,
                verifier_sha256=hashlib.sha256(args.verifier.read_bytes()).hexdigest(),
            )
        finally:
            adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=range(4), default=3)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("--pairs requires at least two measured pairs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(20)
        receipt = check(args, listener)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
