"""Resolve the real launch profile without reserving the weights or starting a server."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from boot import MODEL, REVISION, server_arguments
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule, publish
from sglang.srt.server_args import prepare_server_args
from sglang.srt import pearl_runtime


def profile():
    args = prepare_server_args(server_arguments())
    args.resolve_once()
    publish(args, role="tokenizer")
    pearl_runtime.validate_config()
    view = args.resolved_dict()
    e = view.get("engram_config")
    graph = get_exec().graph.cuda_graph_config
    result = dict(checkpoint=REVISION, model=MODEL.name, architecture=json.loads((MODEL / "config.json").read_text())["architectures"][0],
                  tp=get_parallel().tp_size, ep=get_parallel().ep_size,
                  context=view["context_length"], engram=e,
                  attention=view["attention_backend"], moe=view["moe_runner_backend"],
                  speculative_algorithm=view["speculative_algorithm"],
                  speculative_draft_tokens=view["speculative_num_draft_tokens"],
                  block_size=view["speculative_dspark_block_size"],
                  prefill_chunk=get_schedule().chunked_prefill_size,
                  prefill_decode_interval=get_schedule().prefill_decode_interval,
                  concurrency=get_schedule().max_running_requests, memory_fraction=view["mem_fraction_static"],
                  decode_graph=graph.decode.backend, prefill_graph=graph.prefill.backend,
                  cache_l2=view["enable_hierarchical_cache"], parsers=[view["reasoning_parser"], view["tool_call_parser"]],
                  packages={k:importlib.metadata.version(k) for k in
                            ("torch","flashinfer-python","transformers","sgl-deep-gemm","sglang-kernel","deepseek-recipe")})
    expected_graph = "disabled" if os.environ.get("PEARL_WOB_MODULE") else "full"
    if any((result["architecture"] != "DeepseekV41ForCausalLM",
            result["decode_graph"] != expected_graph, result["prefill_graph"] != "disabled",
            result["speculative_algorithm"] != "DSPARK", result["speculative_draft_tokens"] != 6,
            result["memory_fraction"] != 0.87, result["parsers"] != ["deepseek-v41", "deepseekv41"],
            [e[k] for k in ("placement", "replication", "lookup")] != ["cpu", "none", "all_reduce"],
            result["tp"] != 4, result["ep"] != 4, result["context"] != 524288,
            result["block_size"] != 5, result["prefill_chunk"] != 2048,
            result["prefill_decode_interval"] != 4, result["concurrency"] != 8,
            result["cache_l2"], result["attention"] != "dsv4", result["moe"] != "flashinfer_mxfp4")):
        raise ValueError(f"unexpected effective launch profile: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    options = parser.parse_args()
    data = json.dumps(profile(), indent=2) + "\n"
    if options.receipt:
        options.receipt.write_text(data)
    print(data, end="")
