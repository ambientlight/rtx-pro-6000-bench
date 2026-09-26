#!/usr/bin/env python3
"""Compare the integration encoder with all five pinned publisher prompt goldens."""

import argparse
import copy
import importlib.util
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash"))
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
    )
    args = parser.parse_args()
    reference = load("reference_encoder", args.model / "encoding/encoding.py")
    candidate = load("candidate_encoder", args.source / "python/sglang/srt/entrypoints/openai/encoding_dsv41.py")
    fixtures = sorted((args.model / "encoding/tests").glob("test_input_*.json"))
    assert len(fixtures) == 5, "Expected the five locked reference fixtures"
    for fixture in fixtures:
        case = reference.load_cases(str(fixture))[0]
        expected = fixture.with_name(fixture.stem.replace("input", "output") + ".txt").read_text()
        published, _ = reference.encode_case(case, thinking_mode="chat")
        assert published == expected, fixture.name + ": publisher golden differs"
        # Both encoders accept OpenAI-wrapped function tool declarations.
        messages = copy.deepcopy(case["messages"])
        actual = candidate.encode_messages(
            messages,
            thinking_mode=case.get("thinking_mode") or "chat",
            context=case.get("context"),
            reasoning_effort=case.get("reasoning_effort"),
        )
        assert actual == expected, fixture.name + ": candidate differs from publisher golden"
        print(f"PASS {fixture.name}: {len(expected)} characters match exactly")


if __name__ == "__main__":
    main()
