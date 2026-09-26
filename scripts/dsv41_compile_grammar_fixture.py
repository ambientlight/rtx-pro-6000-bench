#!/usr/bin/env python3
"""CPU check inside the built API image, using the real pinned tokenizer."""

import json
from pathlib import Path

from transformers import AutoTokenizer
import xgrammar as xgr

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice, ToolChoiceFuncName
from sglang.srt.function_call.function_call_parser import FunctionCallParser


def main():
    model = Path("/models/DeepSeek-V4.1-Flash")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    config = json.loads((model / "config.json").read_text())
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=config["text_config"]["vocab_size"], stop_token_ids=[config["eos_token_id"]]
    )
    compiler = xgr.GrammarCompiler(info)
    tools = [
        Tool(
            type="function",
            function=Function(
                name="lookup_fixture",
                strict=True,
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string", "enum": ["alpha"]}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            ),
        )
    ]
    parser = FunctionCallParser(tools, "deepseekv41")
    correct = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup_fixture">{"key":"alpha"}</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    for choice in ("required", "auto", ToolChoice(type="function", function=ToolChoiceFuncName(name="lookup_fixture"))):
        kind, tag = parser.get_structure_constraint(choice, parallel_tool_calls=False)
        assert kind == "structural_tag"
        compiled = compiler.compile_structural_tag(tag)
        for text, expected in ((correct, True), (correct.replace('"alpha"', "42"), False)):
            matcher = xgr.GrammarMatcher(compiled)
            accepted = all(matcher.accept_token(token) for token in tokenizer.encode(text, add_special_tokens=False))
            if accepted:
                accepted = matcher.accept_token(tokenizer.eos_token_id) and matcher.is_terminated()
            assert accepted == expected, f"Unexpected grammar result: {choice=} {expected=}"
        print(
            json.dumps(
                {"tool_choice": choice if isinstance(choice, str) else "named", "real_tokenizer": True, "passed": True}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
