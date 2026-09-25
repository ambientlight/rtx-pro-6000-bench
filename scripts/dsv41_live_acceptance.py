#!/usr/bin/env python3
"""Bounded V4.1 API checks with private request/response artifacts; not a soak.

Run `--suite api` after the unmodified Sero baseline passes. `--suite long`
sends one to three long-context requests with small output budgets. No tool
named by a model is executed: the only tool is a fixed in-memory lookup fixture.
"""

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib

from dsv41_load import LOAD_PATH, is_idle, parse_loads


MODEL = "deepseek-v4-flash"
QUESTION = "Use lookup_fixture to retrieve key alpha. Do not guess its value."
SCHEMA = {
    "type": "object",
    "properties": {"key": {"type": "string", "enum": ["alpha"]}},
    "required": ["key"],
    "additionalProperties": False,
}
FUNCTION = {
    "type": "function",
    "name": "lookup_fixture",
    "description": "Look up a fixture value.",
    "parameters": SCHEMA,
    "strict": True,
}
ANTHROPIC_TOOL = {
    "name": "lookup_fixture",
    "description": "Look up a fixture value.",
    "input_schema": SCHEMA,
    "strict": True,
}


def decode_sse(raw):
    events = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def solid_red_png():
    """Deterministic RGB fixture at the original recipe's full-budget size."""

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    width, height = 3024, 588
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * width) * height))
        + chunk(b"IEND", b"")
    )


def completed_responses(events):
    assert events and events[-1]["type"] == "response.completed", "Responses did not complete"
    assert not any(e["type"] in ("error", "response.failed", "response.incomplete") for e in events)
    assert [e["sequence_number"] for e in events] == list(range(len(events))), "Non-contiguous SSE sequence"
    response = events[-1]["response"]
    assert response["status"] == "completed"
    for item in response["output"]:
        if item["type"] == "function_call":
            added = [e for e in events if e["type"] == "response.output_item.added" and e["item"]["id"] == item["id"]]
            assert len(added) == 1, "Missing/duplicated function item start"
            fragments = [
                e["delta"]
                for e in events
                if e["type"] == "response.function_call_arguments.delta" and e["item_id"] == item["id"]
            ]
            assert "".join(fragments) == item["arguments"], "Tool argument stream diverged"
    assert "｜DSML｜" not in json.dumps(response["output"], ensure_ascii=False), "Raw protocol leaked"
    return response


def completed_messages(events):
    assert events and events[-1]["type"] == "message_stop", "Messages did not terminate"
    assert not any(e["type"] == "error" for e in events), "Messages emitted an error event"
    blocks, closed, arguments = {}, set(), {}
    final = None
    for event in events:
        kind, index = event["type"], event.get("index")
        if kind == "content_block_start":
            assert index not in blocks and index == len(blocks), "Invalid content block order"
            assert len(closed) == len(blocks), "Previous block is still open"
            blocks[index] = dict(event["content_block"])
            arguments[index] = ""
        elif kind == "content_block_delta":
            assert index in blocks and index not in closed, "Delta targets a closed/unknown block"
            delta, block = event["delta"], blocks[index]
            if delta["type"] == "input_json_delta":
                assert block["type"] == "tool_use"
                arguments[index] += delta["partial_json"]
            elif delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = block.get("signature", "") + delta["signature"]
        elif kind == "content_block_stop":
            assert index in blocks and index not in closed
            closed.add(index)
            if blocks[index]["type"] == "tool_use":
                blocks[index]["input"] = json.loads(arguments[index] or "{}")
        elif kind == "message_delta":
            final = event["delta"]["stop_reason"]
    assert len(closed) == len(blocks) and final is not None, "Incomplete Messages blocks"
    content = list(blocks.values())
    assert "｜DSML｜" not in json.dumps(content, ensure_ascii=False), "Raw protocol leaked"
    return content, final


class Client:
    def __init__(self, args):
        self.base = args.base_url.rstrip("/")
        self.model = args.model
        self.long_contexts = args.long_contexts
        self.context_limit = args.context_limit
        self.headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if args.key_file is not None:
            secret = args.key_file.read_text().strip()
            assert secret and "\n" not in secret and "\r" not in secret
            self.headers["Authorization"] = "Bearer " + secret
        self.timeout = args.timeout
        self.output = args.output_root / (
            args.suite + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        self.output.mkdir(mode=0o700, parents=True)
        self.records = []
        print("Evidence:", self.output, flush=True)

    def request(self, name, path, payload=None):
        if payload is not None:
            (self.output / (name + ".request.json")).write_text(json.dumps(payload, ensure_ascii=False))
        started = time.monotonic()
        request = urllib.request.Request(
            self.base + path, headers=self.headers, data=None if payload is None else json.dumps(payload).encode()
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode()
                status, media_type = response.status, response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            raw = error.read().decode(errors="replace")
            (self.output / (name + ".http-error.txt")).write_text(raw)
            raise AssertionError(f"{name}: HTTP {error.code}; see private artifact") from None
        elapsed = time.monotonic() - started
        (self.output / (name + ".response.txt")).write_text(raw)
        self.records.append({"name": name, "path": path, "status": status, "seconds": elapsed})
        (self.output / "requests.json").write_text(json.dumps(self.records, indent=2))
        return decode_sse(raw) if "text/event-stream" in media_type else json.loads(raw) if raw else {}

    def passed(self, name, details=None):
        print(json.dumps({"check": name, "passed": True, **(details or {})}), flush=True)


def api_checks(client):
    response_input = [{"role": "user", "content": QUESTION}]
    common = {
        "model": client.model,
        "temperature": 0,
        "max_output_tokens": 1024,
        "chat_template_kwargs": {"thinking": False},
        "store": False,
    }
    response = completed_responses(
        client.request(
            "responses-tool",
            "/v1/responses",
            {**common, "input": response_input, "tools": [FUNCTION], "tool_choice": "required", "stream": True},
        )
    )
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "lookup_fixture"
    assert json.loads(calls[0]["arguments"]) == {"key": "alpha"}
    call = {k: calls[0][k] for k in ("type", "name", "call_id", "arguments")}
    continued = client.request(
        "responses-tool-result",
        "/v1/responses",
        {
            **common,
            "input": response_input
            + [call, {"type": "function_call_output", "call_id": call["call_id"], "output": '{"value":42}'}],
            "tools": [FUNCTION],
            "tool_choice": "none",
        },
    )
    assert continued["status"] == "completed"
    texts = [
        p["text"]
        for item in continued["output"]
        if item["type"] == "message"
        for p in item["content"]
        if p["type"] == "output_text"
    ]
    assert "42" in "".join(texts) and not any(i["type"] == "function_call" for i in continued["output"])
    client.passed("responses-strict-stream-and-stateless-tool-roundtrip", {"usage": response.get("usage")})

    messages = [{"role": "user", "content": QUESTION}]
    anthropic = {
        "model": client.model,
        "temperature": 0,
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "tools": [ANTHROPIC_TOOL],
    }
    blocks, stop = completed_messages(
        client.request(
            "messages-tool",
            "/v1/messages",
            {
                **anthropic,
                "messages": messages,
                "tool_choice": {"type": "tool", "name": "lookup_fixture"},
                "stream": True,
            },
        )
    )
    tools = [b for b in blocks if b["type"] == "tool_use"]
    assert stop == "tool_use" and len(tools) == 1
    assert tools[0]["name"] == "lookup_fixture" and tools[0]["input"] == {"key": "alpha"}
    continued = client.request(
        "messages-tool-result",
        "/v1/messages",
        {
            **anthropic,
            "messages": messages
            + [
                {"role": "assistant", "content": blocks},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tools[0]["id"], "content": '{"value":42}'}],
                },
            ],
            "tool_choice": {"type": "none"},
        },
    )
    assert continued["stop_reason"] == "end_turn"
    assert "42" in "".join(b.get("text", "") for b in continued["content"])
    client.passed("messages-strict-stream-and-tool-roundtrip")

    think_request = {
        "model": client.model,
        "max_tokens": 2048,
        "temperature": 0,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
        "messages": [{"role": "user", "content": "What is 17 times 19? Briefly answer the number."}],
        "stream": True,
    }
    blocks, stop = completed_messages(client.request("messages-omitted-thinking", "/v1/messages", think_request))
    hidden = [b for b in blocks if b["type"] == "thinking"]
    assert hidden and all(
        not b.get("thinking") and b.get("signature", "").startswith("sglang_thinking_v1.") for b in hidden
    )
    assert stop == "end_turn" and "323" in "".join(b.get("text", "") for b in blocks)
    replay = {
        **think_request,
        "messages": think_request["messages"]
        + [
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": "Add 1 to that result. Answer only the number."},
        ],
    }
    replay_blocks, stop = completed_messages(client.request("messages-thinking-replay", "/v1/messages", replay))
    assert stop == "end_turn" and "324" in "".join(b.get("text", "") for b in replay_blocks)
    client.passed("messages-omitted-thinking-signature-roundtrip")

    vision = completed_responses(
        client.request(
            "responses-vision",
            "/v1/responses",
            {
                **common,
                "max_output_tokens": 64,
                "stream": True,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Name the single solid color filling this image. Reply with one lowercase color word.",
                            },
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64," + base64.b64encode(solid_red_png()).decode(),
                            },
                        ],
                    }
                ],
            },
        )
    )
    text = "".join(
        p["text"]
        for item in vision["output"]
        if item["type"] == "message"
        for p in item["content"]
        if p["type"] == "output_text"
    )
    assert text.strip().lower() == "red"
    assert vision["usage"]["input_tokens"] >= 1024, "Image budget was not represented in model input"
    client.passed("responses-native-image-input", {"usage": vision["usage"]})


def long_checks(client):
    # Tokenize before inference so the bounds are measured, not estimated.
    seed = "Reference record: the cache stores recent entries; preserve ordering and validate invariants.\n"
    for target in client.long_contexts:
        repeats = max(1, target // 20)
        for attempt in range(3):
            text = (
                f"Qualification case {target}. Read reference notes; do not summarize them.\n"
                + seed * repeats
                + "\nThe requested verification number is 42. Reply with exactly 42 and nothing else."
            )
            messages = [{"role": "user", "content": text}]
            counted = client.request(
                f"long-{target}-tokenize-{attempt}",
                "/v1/tokenize",
                {"model": client.model, "messages": messages, "chat_template_kwargs": {"thinking": False}},
            )["count"]
            if abs(counted - target) < 100:
                break
            repeats = max(1, round(repeats * target / counted))
        assert 0.99 * target <= counted <= 1.01 * target and counted + 64 < client.context_limit
        response = completed_responses(
            client.request(
                f"long-{target}-responses",
                "/v1/responses",
                {
                    "model": client.model,
                    "input": messages,
                    "temperature": 0,
                    "max_output_tokens": 64,
                    "chat_template_kwargs": {"thinking": False},
                    "stream": True,
                    "store": False,
                },
            )
        )
        text = "".join(
            p["text"]
            for item in response["output"]
            if item["type"] == "message"
            for p in item["content"]
            if p["type"] == "output_text"
        )
        assert text.strip() == "42", "Long-context verification answer differed"
        client.passed("long-context", {"input_tokens": counted, "usage": response.get("usage")})


def cancellation_checks(client):
    """Close each API stream while it is decoding and verify scheduler drain."""
    for endpoint in ("responses", "messages"):
        before = parse_loads(client.request(endpoint + "-load-before", LOAD_PATH))
        assert is_idle(before), "Cancellation probe needs an idle canary"
        prompt = "Print every integer from 1 to 2000, one per line. Do not summarize or use ellipses."
        if endpoint == "responses":
            payload = {
                "model": client.model,
                "input": prompt,
                "stream": True,
                "store": False,
                "temperature": 0,
                "max_output_tokens": 4096,
                "chat_template_kwargs": {"thinking": False},
            }
            delta_type = "response.output_text.delta"
        else:
            payload = {
                "model": client.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0,
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            }
            delta_type = "content_block_delta"
        name = endpoint + "-disconnect"
        (client.output / (name + ".request.json")).write_text(json.dumps(payload))
        request = urllib.request.Request(
            client.base + "/v1/" + endpoint, headers=client.headers, data=json.dumps(payload).encode()
        )
        received, deltas = bytearray(), 0
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            for line in response:
                received.extend(line)
                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                    event = json.loads(line[5:])
                    assert event.get("type") not in ("error", "response.failed"), "Stream failed before cancellation"
                    deltas += event.get("type") == delta_type
                    if deltas >= 3:
                        active = parse_loads(client.request(endpoint + "-load-active", LOAD_PATH))
                        assert any(rank["num_running_reqs"] > 0 for rank in active), (
                            "Generation already ended before disconnect"
                        )
                        break
            assert deltas >= 3, "Stream ended before the cancellation point"
        # Exiting the response context closes the actual HTTP connection.
        (client.output / (name + ".partial-sse.txt")).write_bytes(received)
        started = time.monotonic()
        for attempt in range(31):
            state = parse_loads(client.request(f"{endpoint}-load-after-{attempt}", LOAD_PATH))
            if is_idle(state):
                break
            time.sleep(1)
        else:
            raise AssertionError("Scheduler did not drain within 30 seconds after client disconnect")
        client.request(endpoint + "-health-after", "/health")
        client.passed(
            endpoint + "-disconnect-drains-scheduler",
            {"drain_seconds": time.monotonic() - started, "deltas_seen": deltas},
        )


def mixed_checks(client):
    """Exactly two overlapping requests: Responses decode plus Messages prefill."""
    long_text = (
        "Mixed scheduling qualification. Use lookup_fixture for key alpha after these notes.\n"
        + "The cache stores recently accessed entries and preserves replacement ordering.\n" * 6000
        + "\nNow call lookup_fixture with key alpha, with no other output."
    )
    tokenized = client.request(
        "mixed-tokenize",
        "/v1/tokenize",
        {
            "model": client.model,
            "messages": [{"role": "user", "content": long_text}],
            "chat_template_kwargs": {"thinking": False},
        },
    )
    assert 60000 < tokenized["count"] < 150000
    first_delta = threading.Event()
    timing = []

    def decode():
        payload = {
            "model": client.model,
            "input": "Print each integer from 1 to 2000, one per line. Do not abbreviate.",
            "stream": True,
            "store": False,
            "temperature": 0,
            "max_output_tokens": 1024,
            "chat_template_kwargs": {"thinking": False},
        }
        (client.output / "mixed-decode.request.json").write_text(json.dumps(payload))
        request = urllib.request.Request(
            client.base + "/v1/responses", headers=client.headers, data=json.dumps(payload).encode()
        )
        data = bytearray()
        try:
            with urllib.request.urlopen(request, timeout=client.timeout) as response:
                for line in response:
                    data.extend(line)
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        event = json.loads(line[5:])
                        if event.get("type") == "response.output_text.delta":
                            timing.append(time.monotonic())
                            first_delta.set()
        finally:
            (client.output / "mixed-decode.response.txt").write_bytes(data)
        events = decode_sse(data.decode())
        assert not any(e["type"] in ("error", "response.failed") for e in events)
        assert events[-1]["type"] in ("response.completed", "response.incomplete")
        if events[-1]["type"] == "response.incomplete":
            assert events[-1]["response"]["incomplete_details"]["reason"] == "max_output_tokens"
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        return events[-1]["response"]["status"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(decode)
        assert first_delta.wait(timeout=60), "Short decode did not start in 60 seconds"
        prefill_start = time.monotonic()
        blocks, stop = completed_messages(
            client.request(
                "mixed-prefill",
                "/v1/messages",
                {
                    "model": client.model,
                    "messages": [{"role": "user", "content": long_text}],
                    "max_tokens": 256,
                    "temperature": 0,
                    "thinking": {"type": "disabled"},
                    "tools": [ANTHROPIC_TOOL],
                    "tool_choice": {"type": "tool", "name": "lookup_fixture"},
                    "stream": True,
                },
            )
        )
        decode_status = future.result()
    calls = [b for b in blocks if b["type"] == "tool_use"]
    assert stop == "tool_use" and len(calls) == 1 and calls[0]["input"] == {"key": "alpha"}
    assert len(timing) > 1 and timing[-1] > prefill_start, "Requests did not overlap"
    largest_gap = max(b - a for a, b in zip(timing, timing[1:]))
    measurement = {
        "prefill_input_tokens": tokenized["count"],
        "decode_status": decode_status,
        "decode_delta_count": len(timing),
        "largest_decode_gap_s": largest_gap,
    }
    (client.output / "mixed-timing.json").write_text(json.dumps(measurement, indent=2))
    client.passed("mixed-prefill-decode", measurement)


def metric_total(raw, name, required_labels=None):
    """Sum selected finite nonnegative Prometheus samples, excluding _created."""
    total = 0
    for line in raw.splitlines():
        match = re.fullmatch(re.escape(name) + r"(?:\{([^}]*)\})?\s+(\S+)(?:\s+\d+)?", line)
        if not match:
            continue
        labels = {
            key: json.loads('"' + value + '"')
            for key, value in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[1] or "")
        }
        if any(labels.get(key) != value for key, value in (required_labels or {}).items()):
            continue
        value = float(match[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid cache metric")
        total += value
    return total


def cache_checks(client):
    """Four bounded calls; verify warm-prefix reuse and RAM write-through.

    This deliberately never flushes/evicts production caches. A device hit is
    not evidence of restoring an evicted prefix from RAM.
    """
    from dsv41_deferred_relaunch import activity_metrics

    def snapshot(name):
        request = urllib.request.Request(client.base + "/metrics", headers=client.headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode()
        (client.output / (name + ".metrics.txt")).write_text(raw)
        active, requests = activity_metrics(raw)
        counters = {
            mode: metric_total(raw, "sglang:prefill_effective_tokens_total", {"mode": mode, "tp_rank": "0"})
            for mode in ("input", "device_hit", "host_hit", "storage_hit")
        }
        counters["backup_bytes"] = metric_total(raw, "sglang:hicache_backup_bytes_total")
        return active, dict(requests), counters

    nonce = os.urandom(16).hex()
    results = []
    for endpoint in ("responses", "messages"):
        assert is_idle(parse_loads(client.request(endpoint + "-cache-load", LOAD_PATH))), (
            "Cache probe needs idle service"
        )
        text = (
            f"Synthetic cache check {nonce}-{endpoint}. These are inert reference records.\n"
            + "\n".join(
                f"Record {i}: state=ready, code={hashlib.sha256(f'{nonce}-{endpoint}-{i}'.encode()).hexdigest()[:12]}."
                for i in range(256)
            )
            + "\nIgnore the reference records when answering. "
            + QUESTION
        )
        messages = [{"role": "user", "content": text}]
        counted = client.request(
            endpoint + "-cache-tokenize",
            "/v1/tokenize",
            {"model": client.model, "messages": messages, "chat_template_kwargs": {"thinking": False}},
        )["count"]
        assert 1024 <= counted <= 16384 and counted + 256 < client.context_limit
        common = {"model": client.model, "temperature": 0, "stream": True}
        if endpoint == "responses":
            payload = {
                **common,
                "input": messages,
                "tools": [FUNCTION],
                "tool_choice": "required",
                "max_output_tokens": 256,
                "store": False,
                "chat_template_kwargs": {"thinking": False},
            }
        else:
            payload = {
                **common,
                "messages": messages,
                "tools": [ANTHROPIC_TOOL],
                "tool_choice": {"type": "tool", "name": "lookup_fixture"},
                "max_tokens": 256,
                "thinking": {"type": "disabled"},
            }
        for phase in ("cold", "warm"):
            name = endpoint + "-cache-" + phase
            active, before_http, before = snapshot(name + "-before")
            assert active == 0, "Other inference is active; cache attribution would be ambiguous"
            events = client.request(name, "/v1/" + endpoint, payload)
            if endpoint == "responses":
                response = completed_responses(events)
                calls = [item for item in response["output"] if item["type"] == "function_call"]
                assert len(calls) == 1 and calls[0]["name"] == "lookup_fixture"
                assert json.loads(calls[0]["arguments"]) == {"key": "alpha"}
                usage = response.get("usage")
            else:
                blocks, stop = completed_messages(events)
                calls = [block for block in blocks if block["type"] == "tool_use"]
                assert stop == "tool_use" and len(calls) == 1
                assert calls[0]["name"] == "lookup_fixture" and calls[0]["input"] == {"key": "alpha"}
                usage = [
                    event.get("usage", event.get("message", {}).get("usage"))
                    for event in events
                    if event["type"] in ("message_start", "message_delta")
                ]
            for attempt in range(16):
                active, after_http, after = snapshot(name + f"-after-{attempt}")
                deltas = {key: after[key] - value for key, value in before.items()}
                observed = deltas["device_hit"] + deltas["host_hit"]
                if active == 0 and (
                    observed >= counted * 0.75
                    if phase == "warm"
                    else deltas["input"] > 0 and deltas["backup_bytes"] > 0
                ):
                    break
                time.sleep(1)
            http_deltas = {
                key: after_http.get(key, 0) - before_http.get(key, 0)
                for key in before_http.keys() | after_http.keys()
                if after_http.get(key, 0) != before_http.get(key, 0)
            }
            assert active == 0 and http_deltas == {("/v1/" + endpoint, "POST"): 1}, (
                "Concurrent inference invalidates metric attribution"
            )
            assert all(value >= 0 for value in deltas.values()), "Counters reset during cache probe"
            if phase == "warm":
                assert observed >= counted * 0.75, "Repeated prompt did not produce substantial prefix hits"
            else:
                assert deltas["backup_bytes"] > 0, "No RAM write-through observed"
            result = {
                "endpoint": endpoint,
                "phase": phase,
                "tokenized_input": counted,
                "metrics_delta": deltas,
                "usage": usage,
            }
            results.append(result)
            (client.output / "cache-results.json").write_text(json.dumps(results, indent=2))
            client.passed(name, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--key-file", type=Path, help="Optional; production preserves V12's no-key access")
    parser.add_argument("--long-contexts", type=int, nargs="+", default=[32768, 131072, 200000])
    parser.add_argument("--context-limit", type=int, default=524288)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/hot/dsv41_state/acceptance"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--suite", choices=("api", "long", "cancellation", "mixed", "cache"), default="api")
    args = parser.parse_args(argv)
    # Permit near-limit targets: long_checks verifies the actual tokenized
    # input plus its 64-token output budget before sending any inference.
    if not 1 <= len(args.long_contexts) <= 3 or any(
        n < 1024 or n + 64 >= args.context_limit for n in args.long_contexts
    ):
        parser.error("Choose one to three bounded prompt lengths below the configured context limit")
    return args


def main():
    args = parse_args()
    os.umask(0o077)
    client = Client(args)
    try:
        client.request("health", "/health")
        {
            "api": api_checks,
            "long": long_checks,
            "cancellation": cancellation_checks,
            "mixed": mixed_checks,
            "cache": cache_checks,
        }[args.suite](client)
    except Exception as error:
        (client.output / "result.json").write_text(json.dumps({"passed": False, "error": str(error)}, indent=2))
        raise
    (client.output / "result.json").write_text(json.dumps({"passed": True, "suite": args.suite}, indent=2))


if __name__ == "__main__":
    main()
