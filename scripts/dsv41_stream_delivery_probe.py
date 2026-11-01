#!/usr/bin/env python3
"""Isolated HTTP stream-delivery lab; never calls a model or executes a tool.

Run inside the deployed LiteLLM image with Docker --network none. Mount this
file and the deployed azure_tool_names.py read-only under /probe, and mount a
new private output directory at /out. No production configuration is loaded.
The hook is copied into /out beside a synthetic configuration for its loader.
The real proxy app/router/provider/stream wrappers serve a synthetic Anthropic
backend, or replay recorded backend SSE payloads with --replay-dir. Production
lifespan/background services and the remote client are not reproduced. Captured
replay preserves payloads and relative body-event timing, not HTTP chunk bytes,
the original frontend request, or the delay before backend response headers.
--scale 0.05 is a smoke test; --scale 1 tests a 30-second idle timer.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def sse(kind: str, **fields) -> bytes:
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


class FrontHeartbeat:
    """TEST ONLY: comments after translation, outside the real proxy app."""

    def __init__(self, app, interval):
        self.app, self.interval = app, interval

    async def __call__(self, scope, receive, send):
        enabled = (b"x-stream-lab-heartbeat", b"comment") in scope.get("headers", [])
        if not enabled:
            return await self.app(scope, receive, send)
        started, finished = asyncio.Event(), asyncio.Event()
        lock = asyncio.Lock()
        is_stream = False

        async def wrapped_send(message):
            nonlocal is_stream
            async with lock:
                if message["type"] == "http.response.start":
                    is_stream = any(k == b"content-type" and b"text/event-stream" in v
                                    for k, v in message.get("headers", []))
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    finished.set()
                await send(message)
                if message["type"] == "http.response.start":
                    started.set()

        async def ticker():
            await started.wait()
            while is_stream and not finished.is_set():
                await asyncio.sleep(self.interval)
                async with lock:
                    if not finished.is_set():
                        await send({"type": "http.response.body", "body": b": stream-lab-heartbeat\n\n", "more_body": True})

        heartbeat = asyncio.create_task(ticker())
        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat


class ObserveDisconnect:
    """Observe existing ASGI receive calls without adding a second consumer."""

    def __init__(self, app, hop, log):
        self.app, self.hop, self.log = app, hop, log

    async def __call__(self, scope, receive, send):
        case = dict(scope.get("headers", [])).get(b"x-stream-lab-case", b"").decode()
        if case:
            scope["stream_lab_case"] = case

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                self.log("http_disconnect", hop=self.hop, case=scope.get("stream_lab_case"))
            return message

        await self.app(scope, observed_receive, send)


async def run(args):
    import httpx
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    scale = args.scale
    idle, delay, hb = 30 * scale, 45 * scale, 10 * scale
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    origin = time.monotonic()
    records, backend, attempts = [], [], {}
    content = ('Unicode: café / 日本語 / 🌲; quote: "hello"; slash: \\\n' * 320)
    write_args = json.dumps({"path": "synthetic-only.md", "content": content}, ensure_ascii=False)
    read_args = json.dumps({"path": "synthetic-only.md"})
    replay = {}
    if args.replay_dir:
        replay_dir = Path(args.replay_dir)
        transport = json.loads((replay_dir / "transport.json").read_text())
        source_rows = {row["rid"]: row for row in transport["rows"] if row.get("target")}
        for path in sorted(replay_dir.glob("msg_*.events.private.json")):
            source_bytes = path.read_bytes()
            events = json.loads(source_bytes)
            rid = events[0]["data"]["message"]["id"]
            source = source_rows[rid]
            base = dt.datetime.fromisoformat(source["response_headers_utc"]).timestamp()
            expected, blocks = {}, {}
            for event in events:
                data = event["data"]
                if data["type"] == "content_block_start" and data["content_block"]["type"] == "tool_use":
                    block = data["content_block"]
                    blocks[data["index"]] = {"name": block["name"], "id": block["id"], "arguments": ""}
                if data["type"] == "content_block_delta" and data["delta"]["type"] == "input_json_delta":
                    blocks[data["index"]]["arguments"] += data["delta"]["partial_json"]
            for block in blocks.values():
                json.loads(block["arguments"])
                expected[block["id"]] = block
            if not expected or events[-1]["data"]["type"] != "message_stop":
                raise ValueError(f"Incomplete recorded tool stream: {rid}")
            replay[rid] = {"events": events, "base": base, "expected": expected, "source": source,
                           "events_file_sha256": digest(source_bytes), "source_file": str(path),
                           "duration": events[-1]["time"] - base}
        if not replay or args.replay_id not in replay:
            raise ValueError("--replay-id must name a complete recorded stream in --replay-dir")

    def log(kind, **fields):
        row = {"t": round(time.monotonic() - origin, 6), "kind": kind, **fields}
        records.append(row)
        if kind in {"client_end", "backend_start", "backend_end"}:
            print(json.dumps(row), flush=True)
        return row["t"]

    async def messages(request: Request):
        body = await request.json()

        def find_spec(value):
            if isinstance(value, str) and value.startswith("STREAM_LAB:"):
                return json.loads(value[len("STREAM_LAB:"):])
            if isinstance(value, dict):
                value = value.values()
            if isinstance(value, (list, tuple)) or type(value).__name__ == "dict_values":
                for child in value:
                    found = find_spec(child)
                    if found is not None:
                        return found
            return None

        spec = find_spec(body)
        if spec is None:
            raise ValueError("Only synthetic STREAM_LAB requests are allowed")
        case = spec["case"]
        request.scope["stream_lab_case"] = case
        attempt = attempts[case] = attempts.get(case, 0) + 1
        retry = spec.get("retry", False) and attempt > 1
        duration = (5 if retry else spec["delay"]) * scale
        tool_name, arguments = ("read_file", read_args) if retry else ("write_document", write_args)
        captured = replay.get(spec.get("replay_id"))
        row = {"case": case, "attempt": attempt, "request_sha256": digest(json.dumps(body, sort_keys=True)),
               "tool_name": tool_name, "args_sha256": digest(arguments), "args_bytes": len(arguments.encode()),
               "start": log("backend_start", case=case, attempt=attempt), "heartbeats": [], "completed": False}
        backend.append(row)

        if captured:
            row.pop("tool_name")
            row.pop("args_sha256")
            row.pop("args_bytes")
            row.update(replay_id=spec["replay_id"], replay_speed=spec.get("replay_speed", 1),
                       recorded_tools=[{"name": tool["name"], "id": tool["id"],
                                        "args_sha256": digest(tool["arguments"]), "args_bytes": len(tool["arguments"].encode())}
                                       for tool in captured["expected"].values()], max_schedule_lateness_seconds=0)

            async def recorded_events():
                reason = "generator_closed"
                started = time.monotonic()
                next_ping = started + hb
                mode = spec.get("backend_heartbeat", "none")
                try:
                    for recorded in captured["events"]:
                        target = started + (recorded["time"] - captured["base"]) * scale * spec.get("replay_speed", 1)
                        if mode != "none":
                            while next_ping < target:
                                await asyncio.sleep(max(0, next_ping - time.monotonic()))
                                row["heartbeats"].append(log("backend_heartbeat", case=case, attempt=attempt, mode=mode))
                                yield b": backend heartbeat\n\n" if mode == "comment" else sse("ping")
                                next_ping += hb
                        await asyncio.sleep(max(0, target - time.monotonic()))
                        row["max_schedule_lateness_seconds"] = max(row["max_schedule_lateness_seconds"], time.monotonic() - target)
                        data = recorded["data"]
                        if data["type"] == "content_block_start" and data["content_block"]["type"] == "tool_use":
                            row["tool_emitted"] = log("backend_tool", case=case, attempt=attempt)
                        yield sse(data["type"], **{k: v for k, v in data.items() if k != "type"})
                    row["completed"] = True
                    reason = "completed"
                except asyncio.CancelledError:
                    reason = "cancelled"
                    raise
                finally:
                    row["end"] = log("backend_end", case=case, attempt=attempt, reason=reason)
                    row["end_reason"] = reason

            return StreamingResponse(recorded_events(), media_type="text/event-stream")

        async def events():
            reason = "generator_closed"
            try:
                yield sse("message_start", message={"id": f"msg_{case}_{attempt}", "type": "message", "role": "assistant",
                          "content": [], "model": "deepseek-v4-flash", "stop_reason": None, "stop_sequence": None,
                          "usage": {"input_tokens": 20, "output_tokens": 0}})
                yield sse("content_block_start", index=0, content_block={"type": "text", "text": ""})
                yield sse("content_block_delta", index=0, delta={"type": "text_delta", "text": "Preparing the synthetic tool call."})
                yield sse("content_block_stop", index=0)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    await asyncio.sleep(min(hb, max(0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        break
                    mode = spec.get("backend_heartbeat", "none")
                    if mode != "none":
                        row["heartbeats"].append(log("backend_heartbeat", case=case, attempt=attempt, mode=mode))
                        yield b": backend heartbeat\n\n" if mode == "comment" else sse("ping")
                row["tool_emitted"] = log("backend_tool", case=case, attempt=attempt)
                yield sse("content_block_start", index=1, content_block={"type": "tool_use", "id": f"tool_{case}_{attempt}", "name": tool_name, "input": {}})
                # Deliberately split in the middle of JSON escapes/Unicode. Each
                # partial_json itself is JSON-escaped by the enclosing SSE event.
                for pos in range(0, len(arguments), 137):
                    yield sse("content_block_delta", index=1, delta={"type": "input_json_delta", "partial_json": arguments[pos:pos + 137]})
                yield sse("content_block_stop", index=1)
                yield sse("message_delta", delta={"stop_reason": "tool_use", "stop_sequence": None}, usage={"output_tokens": 6000})
                yield sse("message_stop")
                row["completed"] = True
                reason = "completed"
            except asyncio.CancelledError:
                reason = "cancelled"
                raise
            finally:
                row["end"] = log("backend_end", case=case, attempt=attempt, reason=reason)
                row["end_reason"] = reason

        return StreamingResponse(events(), media_type="text/event-stream")

    # The deployed hook loads configuration relative to its own source file.
    # Keep that lookup inside the lab, using an exact copy of the mounted hook.
    hook_dir = out / "callback"
    hook_dir.mkdir(exist_ok=True)
    (hook_dir / "azure_tool_names.py").write_bytes(Path("/probe/azure_tool_names.py").read_bytes())
    config = {
        "model_list": [{"model_name": "deepseek-v4-flash", "litellm_params": {
            "model": "anthropic/deepseek-v4-flash", "api_base": "http://127.0.0.1:18081",
            "api_key": "synthetic-not-a-secret", "max_tokens": 65536, "max_parallel_requests": 8 if replay else 16}}],
        "router_settings": {"num_retries": 0},
        "litellm_settings": {"request_timeout": 600, "drop_params": True, "telemetry": False,
                             "callbacks": [f"{hook_dir}/azure_tool_names.azure_tool_name_hook"]},
        "general_settings": {"max_parallel_requests": 64},
    }
    config_file = out / "synthetic-config.json"
    config_text = json.dumps(config, indent=2) + "\n"
    config_file.write_text(config_text)
    (hook_dir / "litellm_config.yaml").write_text(config_text)
    from litellm.proxy import proxy_server
    await proxy_server.initialize(config=str(config_file), telemetry=False)
    apps = [ObserveDisconnect(Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])]), "backend", log),
            ObserveDisconnect(FrontHeartbeat(proxy_server.app, hb), "proxy", log)]
    servers = [uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off", access_log=False))
               for app, port in zip(apps, (18081, 18080))]
    server_tasks = [asyncio.create_task(server.serve()) for server in servers]
    for _ in range(200):
        if all(server.started for server in servers):
            break
        if any(task.done() for task in server_tasks):
            await asyncio.gather(*server_tasks)
            raise RuntimeError("Server exited before startup")
        await asyncio.sleep(0.05)
    else:
        raise TimeoutError("Local servers did not start")

    async def client(spec, *, timer="event", front_hb=False, abandon=False, cancel=False):
        case = spec["case"]
        captured = replay.get(spec.get("replay_id"))
        body = {"model": "deepseek-v4-flash", "stream": True, "max_output_tokens": 32768,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "STREAM_LAB:" + json.dumps(spec, sort_keys=True)}]}],
                "tools": [{"type": "function", "name": name, "description": "Synthetic test only; never executed.",
                           "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}
                          for name, properties in [("write_document", {"path": {"type": "string"}, "content": {"type": "string"}}),
                                                   ("read_file", {"path": {"type": "string"}})]]}
        if captured:
            body["tools"] = [{"type": "function", "name": name, "description": "Recorded response replay; never execute.",
                              "parameters": {"type": "object", "properties": {}, "additionalProperties": True}}
                             for name in sorted({tool["name"] for tool in captured["expected"].values()})]
        row = {"case": case, "timer": timer, "front_heartbeat": front_hb, "abandon_without_close": abandon,
               "request_sha256": digest(json.dumps(body, sort_keys=True)), "start": log("client_start", case=case),
               "byte_times": [], "event_times": [], "event_types": {}, "comment_count": 0, "accepted": [], "valid_calls": []}
        if captured:
            row["replay_id"] = spec["replay_id"]
        queue = asyncio.Queue()
        logical_expired = asyncio.Event()
        first_event = asyncio.Event()

        async def reader():
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, read=None), trust_env=False) as http:
                headers = {"x-stream-lab-case": case}
                if front_hb:
                    headers["x-stream-lab-heartbeat"] = "comment"
                async with http.stream("POST", "http://127.0.0.1:18080/v1/responses", json=body, headers=headers) as response:
                    row["http_status"] = response.status_code
                    if response.status_code != 200:
                        raise RuntimeError((await response.aread()).decode()[:1500])
                    buf = b""
                    fragments = {}
                    async for chunk in response.aiter_raw():
                        row["byte_times"].append(log("client_bytes", case=case, length=len(chunk)))
                        await queue.put("byte")
                        buf += chunk
                        while b"\n\n" in buf:
                            frame, buf = buf.split(b"\n\n", 1)
                            if frame.startswith(b":"):
                                row["comment_count"] += 1
                            data = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
                            if not data or data == b"[DONE]":
                                continue
                            event = json.loads(data)
                            kind = event.get("type", "unknown")
                            row["event_times"].append(log("client_event", case=case, event=kind))
                            row["event_types"][kind] = row["event_types"].get(kind, 0) + 1
                            first_event.set()
                            await queue.put("event")
                            if kind == "response.function_call_arguments.delta":
                                key = event.get("item_id", event.get("output_index"))
                                fragments[key] = fragments.get(key, "") + event["delta"]
                            if kind == "response.output_item.done" and event.get("item", {}).get("type") == "function_call":
                                item = event["item"]
                                actual = item["arguments"]
                                expected = write_args if item["name"] == "write_document" else read_args
                                identity_exact = True
                                if captured:
                                    original = captured["expected"][item["call_id"]]
                                    expected = original["arguments"]
                                    identity_exact = item["name"] == original["name"] and item["call_id"] == original["id"]
                                parsed = json.loads(actual)
                                check = {"name": item["name"], "sha256": digest(actual), "bytes": len(actual.encode()),
                                         "exact": actual == expected, "json_equal": parsed == json.loads(expected),
                                         "call_identity_exact": identity_exact,
                                         "delta_reconstruction_exact": fragments.get(item["id"]) == actual,
                                         "after_deadline": logical_expired.is_set()}
                                row["valid_calls"].append(check)
                                if not logical_expired.is_set():
                                    row["accepted"].append(check)
                    if buf.strip():
                        row["trailing_bytes"] = len(buf)
            await queue.put("eof")

        task = asyncio.create_task(reader())
        # Start the idle timer at response streaming, not Python/provider import.
        try:
            ready = asyncio.create_task(first_event.wait())
            done, _ = await asyncio.wait({ready, task}, timeout=90, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                ready.cancel()
                raise TimeoutError("No initial stream event")
            if task in done:
                ready.cancel()
                await task
            if cancel:
                await asyncio.sleep(2 * scale)
                row["outcome"] = "explicit_cancel"
                row["deadline"] = log("client_cancel", case=case)
                task.cancel()
            else:
                last = time.monotonic()
                while True:
                    remaining = idle - (time.monotonic() - last)
                    try:
                        notification = await asyncio.wait_for(queue.get(), max(0, remaining))
                    except TimeoutError:
                        row["outcome"] = "idle_timeout"
                        row["deadline"] = log("client_deadline", case=case)
                        logical_expired.set()
                        if not abandon:
                            task.cancel()
                        break
                    if notification == "eof":
                        row["outcome"] = "completed"
                        break
                    if timer == "byte" or notification == "event":
                        last = time.monotonic()
                if abandon:
                    # Deliberately keep draining the old HTTP connection. A retry
                    # may start while this reader discards the late valid output.
                    row["background_reader"] = task
            if not abandon or row["outcome"] == "completed":
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        except Exception as exc:
            row["outcome"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        row["end"] = log("client_end", case=case, outcome=row["outcome"], accepted=len(row["accepted"]))
        return row

    async def direct_cancel_control():
        case = "direct_backend_cancel_control"
        spec = {"case": case, "delay": 45}
        row = {"case": case, "start": log("client_start", case=case), "accepted": [], "valid_calls": [], "direct_backend": True}
        async with httpx.AsyncClient(timeout=90, trust_env=False) as http:
            body = {"model": "synthetic", "stream": True, "messages": [{"role": "user", "content": "STREAM_LAB:" + json.dumps(spec)}]}
            async with http.stream("POST", "http://127.0.0.1:18081/v1/messages", json=body) as response:
                row["http_status"] = response.status_code
                iterator = response.aiter_raw()
                await anext(iterator)
                await asyncio.sleep(2 * scale)
                row["deadline"] = log("client_cancel", case=case)
                await response.aclose()
        row["outcome"] = "explicit_cancel"
        row["end"] = log("client_end", case=case, outcome=row["outcome"], accepted=0)
        return row

    results = []
    try:
        # Warm up the real provider before short smoke-test timing begins.
        results.append(await client({"case": "fast_control", "delay": 0.1}))
        if results[0]["outcome"] != "completed" or not results[0]["accepted"]:
            raise RuntimeError("Fast control failed: " + json.dumps(results[0]))

        async def retries(case, abandon):
            spec = {"case": case, "delay": 45, "retry": True}
            first = await client(spec, abandon=abandon)
            second = await client(spec)
            drain = first.pop("background_reader", None)
            if drain:
                await drain
            return [first, second]

        if replay:
            # Integrity through the complete HTTP proxy, with capture time
            # compressed 100x. Timing assertions use the separate real-time run.
            for rid in replay:
                results.append(await client({"case": f"capture_integrity_{rid}", "delay": 0,
                                             "replay_id": rid, "replay_speed": 0.01}))
            selected = {"delay": 0, "replay_id": args.replay_id}
            jobs = [
                direct_cancel_control(),
                client({**selected, "case": "capture_no_heartbeat"}),
                client({**selected, "case": "capture_backend_ping", "backend_heartbeat": "ping"}),
                client({**selected, "case": "capture_front_comment_event_timer"}, front_hb=True),
                client({**selected, "case": "capture_front_comment_byte_timer"}, front_hb=True, timer="byte"),
                client({**selected, "case": "capture_explicit_cancel"}, cancel=True),
            ]
            delay = replay[args.replay_id]["duration"] * scale
        else:
            jobs = [
                direct_cancel_control(),
                client({"case": "no_heartbeat", "delay": 45}),
                client({"case": "backend_ping", "delay": 45, "backend_heartbeat": "ping"}),
                client({"case": "backend_comment", "delay": 45, "backend_heartbeat": "comment"}),
                client({"case": "front_comment_event_timer", "delay": 45}, front_hb=True),
                client({"case": "front_comment_byte_timer", "delay": 45}, front_hb=True, timer="byte"),
                client({"case": "explicit_cancel", "delay": 45}, cancel=True),
                retries("retry_after_close", False),
                retries("retry_while_draining", True),
            ]
        for result in await asyncio.gather(*jobs):
            results.extend(result if isinstance(result, list) else [result])
        # Give every original synthetic generation time to finish if cancellation
        # did not reach it. Record this before shutting down either local server.
        pending_deadline = time.monotonic() + delay + 2
        while any("end" not in row for row in backend) and time.monotonic() < pending_deadline:
            await asyncio.sleep(0.1)
    finally:
        for row in results:
            row.pop("background_reader", None)
        files = ["proxy/proxy_server.py", "proxy/common_request_processing.py", "proxy/response_api_endpoints/endpoints.py",
                 "responses/litellm_completion_transformation/streaming_iterator.py", "llms/anthropic/chat/handler.py",
                 "litellm_core_utils/streaming_handler.py"]
        import litellm
        package_root = Path(litellm.__file__).parent
        source_hashes = {name: digest((package_root / name).read_bytes()) for name in files}
        source_hashes["azure_tool_names.py"] = digest(Path("/probe/azure_tool_names.py").read_bytes())
        report = {"schema": 1, "scale": scale, "idle_seconds": idle, "delay_seconds": delay, "heartbeat_seconds": hb,
                  "require_cancellation": args.require_cancellation,
                  "boundaries": {"real_litellm_http_proxy": True, "synthetic_anthropic_backend": True, "controlled_client": True,
                                 "production_lifespan": False, "model_execution": False, "tool_execution": False,
                                 "captured_backend_event_replay": bool(replay), "original_frontend_request": False,
                                 "replays_pre_header_delay": False, "exact_http_chunk_bytes": False,
                                 "front_heartbeat_is_test_only": True},
                  "versions": {name: importlib.metadata.version(name) for name in ["litellm", "httpx", "starlette", "uvicorn"]},
                  "source_package_root": str(package_root), "source_hashes": source_hashes, "clients": results, "backend": backend,
                  "replay_sources": {rid: {"source_file": value["source_file"], "events_file_sha256": value["events_file_sha256"],
                                           "event_count": len(value["events"]), "duration_from_headers_seconds": value["duration"],
                                           "original_request_utc": value["source"]["request_utc"],
                                           "original_max_event_gap_seconds": value["source"]["max_event_gap_seconds"]}
                                     for rid, value in replay.items()},
                  "disconnects": [row for row in records if row["kind"] == "http_disconnect"]}
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        print("RESULTS " + str(out / "results.json"), flush=True)
        for server in servers:
            server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.gather(*server_tasks), timeout=5)
        except TimeoutError:
            for task in server_tasks:
                task.cancel()
    failures = [row for row in results if row["outcome"] == "error"]
    if failures:
        raise RuntimeError(f"{len(failures)} lab clients failed unexpectedly")
    for row in results:
        for call in row["valid_calls"]:
            assert call["exact"] and call["json_equal"] and call["delta_reconstruction_exact"] and call["call_identity_exact"], (row["case"], call)
    direct = next(row for row in backend if row["case"] == "direct_backend_cancel_control")
    assert direct["end_reason"] == "cancelled", direct
    direct_client = next(row for row in results if row["case"] == direct["case"])
    assert 0 <= direct["end"] - direct_client["deadline"] < 1, (direct, direct_client)
    for row in results:
        if row["case"].startswith("capture_integrity_"):
            assert row["outcome"] == "completed" and len(row["accepted"]) == len(replay[row["replay_id"]]["expected"]), row["case"]
    long_write_case = "capture_front_comment_byte_timer" if replay else "front_comment_byte_timer"
    long_write = next(row for row in results if row["case"] == long_write_case)
    assert long_write["outcome"] == "completed" and len(long_write["accepted"]) == 1, long_write
    if args.require_cancellation:
        for row in results:
            if row["outcome"] not in {"explicit_cancel", "idle_timeout"} or row.get("abandon_without_close"):
                continue
            original = next(item for item in backend if item["case"] == row["case"] and item["attempt"] == 1)
            assert original["end_reason"] == "cancelled" and not original["completed"], (row["case"], original)
            assert 0 <= original["end"] - row["deadline"] < 1, (row["case"], original, row["deadline"])
            hops = {item["hop"] for item in report["disconnects"] if item["case"] == row["case"]}
            assert "backend" in hops and (row.get("direct_backend") or "proxy" in hops), (row["case"], hops)
        if not replay:
            closed = [row for row in backend if row["case"] == "retry_after_close"]
            assert len(closed) == 2 and closed[0]["end"] <= closed[1]["start"], closed
            draining = [row for row in backend if row["case"] == "retry_while_draining"]
            assert len(draining) == 2 and draining[0]["end"] > draining[1]["start"], draining


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/out")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--replay-dir", help="Read-only directory containing captured events and transport.json")
    parser.add_argument("--replay-id", help="Recorded message ID for the real-timing cases; all recorded IDs get integrity tests")
    parser.add_argument("--require-cancellation", action="store_true", help="Require downstream close to cancel the backend within one second")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")
    os.umask(0o077)
    os.environ.update({"LITELLM_DONT_SHOW_FEEDBACK_BOX": "true", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DO_NOT_TRACK": "1"})
    sys.path.insert(0, "/probe")
    asyncio.run(run(args))
