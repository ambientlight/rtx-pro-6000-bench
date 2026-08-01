#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8010"))
# Set BRIDGE_DEBUG=1 to log, per turn: incoming tool-call turns + cache hit/miss on
# re-injection, and per response the finish_reason / content-len / tool_calls with a
# JSON-validity check on each tool call's arguments. Diagnoses malformed-tool-call
# ("Invalid tool parameters") and reasoning-cache-miss issues without dumping bodies.
BRIDGE_DEBUG = os.environ.get("BRIDGE_DEBUG", "0") == "1"

logging.basicConfig(
    level=logging.DEBUG if BRIDGE_DEBUG else logging.INFO,
    format="%(asctime)s bridge %(levelname)s %(message)s",
)
log = logging.getLogger("bridge")


def _tool_call_arg_report(tool_calls: list[dict[str, Any]] | None) -> str:
    """Summarize tool calls + whether each one's arguments parse as JSON."""
    if not tool_calls:
        return "none"
    out = []
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                json.loads(args)
                valid = "ok"
            except Exception:
                valid = f"BAD_JSON(len={len(args)})"
        elif isinstance(args, dict):
            valid = "ok(dict)"
        else:
            valid = f"MISSING({type(args).__name__})"
        out.append(f"{name}:{valid}")
    return ",".join(out)


app = FastAPI()

# For single-process local use this is fine.
# For multiple bridge workers, replace this with Redis or SQLite.
REASONING_CACHE: dict[str, str] = {}

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Stray/unpaired think tags. reasoning_content sometimes carries a trailing (or
# leading) tag with no matching partner — clean those when surfacing it as content.
STRAY_THINK_RE = re.compile(r"</?think>\s*")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def surface_reasoning_as_content(reasoning: str) -> str:
    """Turn reasoning_content into clean visible content for the empty-turn fallback:
    remove paired think blocks, then any stray/unpaired think tags, then trim."""
    text = clean_inline_think(reasoning) or reasoning
    return STRAY_THINK_RE.sub("", text).strip()


def clean_inline_think(text: str | None) -> str | None:
    if not isinstance(text, str):
        return text
    return THINK_RE.sub("", text)


def strip_anthropic_thinking_blocks(content: Any, role: str | None = None) -> Any:
    """Remove Anthropic thinking/redacted_thinking blocks from content arrays."""
    if isinstance(content, str):
        return clean_inline_think(content) if role == "assistant" else content

    if not isinstance(content, list):
        return content

    cleaned = []
    for block in content:
        if not isinstance(block, dict):
            cleaned.append(block)
            continue

        block_type = block.get("type")
        if block_type in ("thinking", "redacted_thinking"):
            continue

        new_block = deepcopy(block)
        if role == "assistant" and block_type == "text" and isinstance(new_block.get("text"), str):
            new_block["text"] = clean_inline_think(new_block["text"])

        cleaned.append(new_block)

    return cleaned


def sanitize_message_for_upstream(msg: dict[str, Any]) -> dict[str, Any]:
    msg = deepcopy(msg)
    role = msg.get("role")

    # SGLang/OpenAI route does not understand Anthropic thinking content blocks.
    msg.pop("thinking_blocks", None)
    msg.pop("thinking", None)

    # We will inject reasoning_content from our cache, not trust client-visible blocks.
    msg.pop("reasoning_content", None)

    if "content" in msg:
        msg["content"] = strip_anthropic_thinking_blocks(msg["content"], role=role)

    return msg


def normalize_message_for_key(msg: dict[str, Any]) -> dict[str, Any]:
    msg = sanitize_message_for_upstream(msg)

    # Normalize assistant tool-call content shape. Different clients use None, "",
    # or omitted content for tool-call assistant messages.
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        msg["content"] = msg.get("content") or ""

    return msg


def prefix_cache_key(model: str, messages: list[dict[str, Any]], assistant_index: int) -> str:
    prefix = [normalize_message_for_key(m) for m in messages[: assistant_index + 1]]
    raw = canonical_json({"model": model, "prefix": prefix})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def has_tool_call_shape(msg: dict[str, Any]) -> bool:
    if msg.get("tool_calls"):
        return True

    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)

    return False


def prepare_request_body(body: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    body = deepcopy(body)
    model = body.get("model", "deepseek-v4-flash")

    if BRIDGE_DEBUG:
        log.debug(
            "REQ-SAMPLING temperature=%s top_p=%s top_k=%s frequency_penalty=%s "
            "presence_penalty=%s max_tokens=%s parallel_tool_calls=%s stream=%s",
            body.get("temperature"), body.get("top_p"), body.get("top_k"),
            body.get("frequency_penalty"), body.get("presence_penalty"),
            body.get("max_tokens"), body.get("parallel_tool_calls"), body.get("stream"),
        )

    # Claude Code/LiteLLM may forward Anthropic-ish knobs. SGLang wants
    # chat_template_kwargs for DeepSeek V4 thinking.
    body.pop("thinking", None)
    body.pop("output_config", None)

    body.setdefault("chat_template_kwargs", {})
    body["chat_template_kwargs"]["thinking"] = True

    # Keep reasoning separated so it does not leak as raw <think>...</think>.
    body.setdefault("separate_reasoning", True)

    messages = body.get("messages") or []
    sanitized_messages = [sanitize_message_for_upstream(m) for m in messages]

    # Re-inject cached reasoning_content for assistant tool-call messages.
    tc_turns = 0
    hits = 0
    for i, msg in enumerate(sanitized_messages):
        if msg.get("role") == "assistant" and has_tool_call_shape(msg):
            tc_turns += 1
            key = prefix_cache_key(model, sanitized_messages, i)
            cached_reasoning = REASONING_CACHE.get(key)
            if cached_reasoning:
                msg["reasoning_content"] = cached_reasoning
                hits += 1

    if BRIDGE_DEBUG:
        log.debug(
            "REQ msgs=%d tool_call_turns=%d reinject_hits=%d/%d cache_size=%d",
            len(sanitized_messages), tc_turns, hits, tc_turns, len(REASONING_CACHE),
        )
        # Prefix-stability probe: per-message (role, content-len, per-message sha8) for the
        # first messages. Diffing consecutive turns shows the first index whose hash changes
        # -> pinpoints where the client mutates history and breaks prefix (radix) caching.
        fp = []
        for mm in sanitized_messages[:6]:
            c = mm.get("content")
            cl = len(c) if isinstance(c, str) else len(canonical_json(c)) if c is not None else 0
            h = hashlib.sha256(canonical_json(mm).encode()).hexdigest()[:8]
            fp.append(f"{mm.get('role','?')[:4]}:{cl}:{h}")
        log.debug("REQ-PREFIX-FP %s", " | ".join(fp))

    body["messages"] = sanitized_messages
    return body, sanitized_messages


def store_reasoning_for_assistant(
    model: str,
    request_messages: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    reasoning_content: str | None,
) -> None:
    if not reasoning_content:
        return

    assistant_msg = normalize_message_for_key(assistant_msg)

    # Only tool-call assistant turns strictly need re-injection for DeepSeek V4.
    # Storing all is harmless, but injecting only tool-call turns avoids bloating normal turns.
    if not has_tool_call_shape(assistant_msg):
        return

    full_messages = deepcopy(request_messages) + [assistant_msg]
    key = prefix_cache_key(model, full_messages, len(full_messages) - 1)
    REASONING_CACHE[key] = reasoning_content


def strip_response_reasoning_and_store(
    response_json: dict[str, Any],
    model: str,
    request_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    for choice in response_json.get("choices", []):
        msg = choice.get("message") or {}

        reasoning_content = msg.pop("reasoning_content", None)
        msg.pop("thinking_blocks", None)

        if isinstance(msg.get("content"), str):
            msg["content"] = clean_inline_think(msg["content"])

        # Empty-turn fallback: DeepSeek-V4-Flash sometimes concludes *inside* the
        # think block and stops with no post-</think> content and no tool call. After
        # we strip reasoning that leaves a completely blank assistant turn, which
        # makes agent loops (Claude Code) halt — there is nothing to act on. When
        # that happens, surface the reasoning as visible content so the turn is
        # non-empty and the client can continue. (We only do this for genuinely
        # blank turns; turns with content or tool_calls are untouched.)
        has_content = bool((msg.get("content") or "").strip())
        has_tool_calls = bool(msg.get("tool_calls"))
        fallback_fired = False
        recovered_calls = False
        if not has_content and not has_tool_calls and reasoning_content:
            preamble, dsml_calls = recover_dsml_from_text(reasoning_content)
            if dsml_calls:
                # The turn's tool-call block was mis-channeled into reasoning; recover
                # it as real tool_calls instead of leaking DSML as text.
                msg["tool_calls"] = dsml_calls
                msg["content"] = preamble
                has_tool_calls = True
                recovered_calls = True
                fallback_fired = True
            else:
                fallback = preamble
                if fallback:
                    msg["content"] = fallback
                    fallback_fired = True

        if BRIDGE_DEBUG:
            log.debug(
                "RESP finish=%s content_len=%d tool_calls=[%s] reasoning_len=%d fallback=%s dsml_recovered=%s",
                choice.get("finish_reason"),
                len(msg.get("content") or ""),
                _tool_call_arg_report(msg.get("tool_calls")),
                len(reasoning_content or ""),
                fallback_fired,
                recovered_calls,
            )

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content") or "",
        }

        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]

        store_reasoning_for_assistant(
            model=model,
            request_messages=request_messages,
            assistant_msg=assistant_msg,
            reasoning_content=reasoning_content,
        )

    return response_json


def merge_tool_call_delta(acc: dict[int, dict[str, Any]], tool_call: dict[str, Any]) -> None:
    index = tool_call.get("index", 0)
    slot = acc.setdefault(index, {"function": {"name": "", "arguments": ""}})

    if tool_call.get("id"):
        slot["id"] = tool_call["id"]
    if tool_call.get("type"):
        slot["type"] = tool_call["type"]

    fn = tool_call.get("function") or {}
    slot.setdefault("function", {"name": "", "arguments": ""})

    if fn.get("name"):
        # Names are usually emitted once. Assignment is safer than append.
        slot["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


def split_concatenated_json_args(s: Any) -> list[str] | None:
    """Split an arguments string that may be several concatenated JSON objects
    (e.g. '{...}{...}') into the individual object strings. Returns None if it is a
    single object or cannot be cleanly split. raw_decode respects JSON string
    boundaries, so a literal '}{' inside a value won't cause a false split.

    A single zero-arg tool call (e.g. EnterPlanMode) can arrive as a duplicated empty
    object '{}{}' from the streaming parser; that is one call, not two. So empty '{}'
    objects are dropped and identical objects are de-duplicated before deciding whether
    a genuine multi-call split occurred — only distinct, non-empty objects count."""
    if not isinstance(s, str):
        return None
    stripped = s.strip()
    if not stripped:
        return None
    dec = json.JSONDecoder()
    parts: list[tuple[str, Any]] = []  # (raw_substring, decoded_value)
    idx, n = 0, len(stripped)
    while idx < n:
        while idx < n and stripped[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(stripped, idx)
        except json.JSONDecodeError:
            return None  # not cleanly splittable — leave as-is
        parts.append((stripped[idx:end], obj))
        idx = end
    if not parts:
        return None
    # Tool-call arguments are a params object. A single zero-arg call can arrive as
    # '{}""' or '{}{}' (an empty object plus a spurious empty-string/duplicate token),
    # which must stay ONE call — not be split. Keep only parts that decode to a
    # NON-EMPTY dict (real distinct calls); drop empty dicts, bare strings/numbers, and
    # duplicates. If nothing meaningful remains it was a zero-arg call -> '{}'.
    seen: set[str] = set()
    meaningful: list[str] = []
    for raw, obj in parts:
        if not (isinstance(obj, dict) and obj):
            continue  # skip {}, "", numbers, lists — artifacts, not real calls
        key = raw.strip()
        if key in seen:
            continue
        seen.add(key)
        meaningful.append(raw)
    if not meaningful:
        return ["{}"]
    return meaningful


def repair_tool_calls(acc: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Normalize accumulated streaming tool calls, repairing the sglang streaming bug
    that can collapse multiple same-name calls into one index with their JSON arguments
    concatenated. Splits such a call back into separate calls and reindexes/reids the
    whole list. Returns (repaired_calls, split_happened)."""
    ordered = [acc[i] for i in sorted(acc.keys())]
    repaired: list[dict[str, Any]] = []
    split_happened = False
    for call in ordered:
        fn = call.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        base_id = call.get("id")
        ctype = call.get("type") or "function"
        parts = split_concatenated_json_args(args) if isinstance(args, str) else None
        if parts and len(parts) > 1:
            split_happened = True
            for k, part in enumerate(parts):
                repaired.append({
                    "id": f"{base_id}_{k}" if base_id else None,
                    "type": ctype,
                    "function": {"name": name, "arguments": part},
                })
        else:
            # Single logical call. If the splitter normalized the args (e.g. collapsed a
            # duplicated empty-object '{}{}' from a zero-arg call down to one '{}'), use
            # that cleaned value so we never forward invalid concatenated JSON.
            if parts and len(parts) == 1:
                clean_args = parts[0]
            elif isinstance(args, str):
                clean_args = args
            else:
                clean_args = ""
            # A zero-arg call may arrive with empty/blank args; emit valid '{}' so the
            # client never sees invalid JSON for the arguments field.
            if not clean_args.strip():
                clean_args = "{}"
            repaired.append({
                "id": base_id,
                "type": ctype,
                "function": {"name": name, "arguments": clean_args},
            })
    for i, tc in enumerate(repaired):
        tc["index"] = i
        if not tc.get("id"):
            tc["id"] = f"call_{i}"
    return repaired, split_happened


# DeepSeek-V4 native tool-call ("DSML") tokens. Mirrors sglang's DeepSeekV4Detector.
# We need to parse these in the bridge because DSV4 sometimes emits the whole tool-call
# block INSIDE the think channel (no closing </think> first). With separate_reasoning
# the block then lands in reasoning_content, so sglang's function-call detector — which
# only runs on the content channel — never parses it, and it would otherwise leak as raw
# assistant text. We recover it here and re-emit as real tool_calls.
DSML_TOOLCALLS_RE = re.compile(
    r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>", re.DOTALL
)
DSML_INVOKE_RE = re.compile(
    r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
    r"(?:(?P<self_close>/>)|>(?P<body>.*?)</｜DSML｜invoke>)",
    re.DOTALL,
)
DSML_PARAM_RE = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>',
    re.DOTALL,
)


def _parse_dsml_params(body: str) -> dict[str, Any]:
    """Parse one invoke body into a params dict. Supports the XML parameter-tag form
    and the direct-JSON form, matching sglang's _parse_parameters_from_xml."""
    body_stripped = (body or "").strip()
    if body_stripped.startswith("{"):
        try:
            obj = json.loads(body_stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass  # fall through to XML parsing
    params: dict[str, Any] = {}
    for m in DSML_PARAM_RE.finditer(body or ""):
        name, ptype, value = m.group(1), m.group(2), m.group(3)
        if ptype == "true":  # string-typed
            params[name] = value.strip()
        else:
            try:
                params[name] = json.loads(value.strip())
            except Exception:
                params[name] = value.strip()
    return params


def extract_dsml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract DSML invoke blocks from text into OpenAI-shaped tool_call dicts.
    Returns [] if no complete invoke block is present. Handles the wrapped
    <｜DSML｜tool_calls>...</｜DSML｜tool_calls> form and a bare run of invokes
    (e.g. when the closing wrapper was truncated)."""
    if not isinstance(text, str) or "<｜DSML｜invoke" not in text:
        return []
    block_match = DSML_TOOLCALLS_RE.search(text)
    if block_match:
        block = block_match.group(1)
    else:
        # Unwrapped/truncated: parse from the first invoke onward.
        block = text[text.find("<｜DSML｜invoke"):]
    calls: list[dict[str, Any]] = []
    for i, m in enumerate(DSML_INVOKE_RE.finditer(block)):
        name = m.group("name").strip()
        body = "" if m.group("self_close") else (m.group("body") or "")
        params = _parse_dsml_params(body)
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "index": i,
            "function": {"name": name, "arguments": json.dumps(params, ensure_ascii=False)},
        })
    return calls


def recover_dsml_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """For the empty-turn fallback: a mis-channeled tool-call block may sit in the
    reasoning. Return (visible_preamble, tool_calls). tool_calls is the recovered DSML
    invoke blocks (or [] if none complete); preamble is the text before the first DSML
    marker, cleaned of think tags. When there are no calls, preamble is the full cleaned
    text (preserving the prior surface-reasoning-as-content behavior)."""
    calls = extract_dsml_tool_calls(text)
    if not calls:
        return surface_reasoning_as_content(text), []
    markers = [text.find("<｜DSML｜tool_calls>"), text.find("<｜DSML｜invoke")]
    markers = [m for m in markers if m != -1]
    cut = min(markers) if markers else len(text)
    return surface_reasoning_as_content(text[:cut]), calls


async def stream_filtered_response(
    upstream_response: httpx.Response,
    model: str,
    request_messages: list[dict[str, Any]],
) -> AsyncIterator[bytes]:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}

    # Whether we ever emitted visible content or a tool call to the client. Used for
    # the empty-turn fallback: DeepSeek-V4-Flash sometimes concludes inside the think
    # block and stops with no post-</think> content and no tool call, which — after we
    # strip reasoning — is a blank turn that halts agent loops (Claude Code). Finish /
    # role chunks alone do NOT count as visible.
    emitted_visible = False
    # Hold finish_reason-bearing chunks so we can inject a synthetic content chunk
    # BEFORE them when the turn would otherwise be blank (clients may finalize on
    # finish_reason, so content must precede it).
    pending_finish: list[dict[str, Any]] = []
    chunk_template: dict[str, Any] | None = None

    async for line in upstream_response.aiter_lines():
        if not line:
            continue

        if not line.startswith("data:"):
            yield (line + "\n").encode("utf-8")
            continue

        data = line[len("data:") :].strip()

        if data == "[DONE]":
            # Empty-turn fallback: nothing visible was emitted but the model produced
            # reasoning. Two sub-cases: (a) the tool-call block was mis-channeled into
            # reasoning (DSV4 emitted DSML before closing </think>) -> recover it as real
            # tool_calls; (b) plain reasoning-only turn -> surface it as content so the
            # turn isn't blank.
            recovered_calls: list[dict[str, Any]] = []
            if not emitted_visible and reasoning_parts:
                preamble, recovered_calls = recover_dsml_from_text("".join(reasoning_parts))
                fallback_text = preamble if recovered_calls else surface_reasoning_as_content("".join(reasoning_parts))
                if fallback_text:
                    content_parts.append(fallback_text)
                    base = deepcopy(chunk_template) if chunk_template else {
                        "object": "chat.completion.chunk",
                        "model": model,
                    }
                    base["choices"] = [
                        {"index": 0, "delta": {"content": fallback_text}, "finish_reason": None}
                    ]
                    payload = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
                    yield f"data: {payload}\n\n".encode("utf-8")

            # Emit the accumulated tool calls now, repaired and correctly indexed, as
            # clean per-call delta chunks. Must precede any finish_reason chunk because
            # clients finalize the turn on finish_reason. Recovered DSML calls (from the
            # mis-channeled-reasoning case) are already well-formed, so use them directly.
            repaired_calls: list[dict[str, Any]] = []
            split_happened = False
            if tool_calls_acc:
                repaired_calls, split_happened = repair_tool_calls(tool_calls_acc)
            elif recovered_calls:
                repaired_calls = recovered_calls
            for tc in repaired_calls:
                base = deepcopy(chunk_template) if chunk_template else {
                    "object": "chat.completion.chunk",
                    "model": model,
                }
                base["choices"] = [
                    {"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}
                ]
                payload = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
                yield f"data: {payload}\n\n".encode("utf-8")

            # Release any buffered finish_reason chunk(s).
            for fin in pending_finish:
                payload = json.dumps(fin, ensure_ascii=False, separators=(",", ":"))
                yield f"data: {payload}\n\n".encode("utf-8")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts),
            }

            if repaired_calls:
                assistant_msg["tool_calls"] = repaired_calls

            store_reasoning_for_assistant(
                model=model,
                request_messages=request_messages,
                assistant_msg=assistant_msg,
                reasoning_content="".join(reasoning_parts),
            )

            if BRIDGE_DEBUG:
                log.debug(
                    "RESP(stream) content_len=%d tool_calls=[%s] reasoning_len=%d "
                    "fallback=%s emitted_visible=%s split_repair=%s dsml_recovered=%s",
                    len("".join(content_parts)),
                    _tool_call_arg_report(assistant_msg.get("tool_calls")),
                    len("".join(reasoning_parts)),
                    (not emitted_visible and bool(reasoning_parts)),
                    emitted_visible,
                    split_happened,
                    bool(recovered_calls),
                )

            yield b"data: [DONE]\n\n"
            continue

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            yield (line + "\n\n").encode("utf-8")
            continue

        if chunk_template is None:
            # Capture id/created/model/etc. to shape a well-formed synthetic chunk.
            chunk_template = {k: v for k, v in chunk.items() if k != "choices"}

        visible = False
        has_finish = False

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}

            reasoning_delta = delta.pop("reasoning_content", None)
            delta.pop("thinking_blocks", None)

            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)

            if isinstance(delta.get("content"), str):
                delta["content"] = clean_inline_think(delta["content"])
                if delta["content"]:
                    content_parts.append(delta["content"])
                    visible = True
                    emitted_visible = True

            if delta.get("tool_calls"):
                # Buffer tool-call deltas rather than forwarding them live: the sglang
                # streaming parser can collapse several same-name calls (e.g. multiple
                # Write) onto one index with concatenated JSON args. We accumulate here
                # and emit repaired, correctly-indexed calls at [DONE]. A turn with tool
                # calls is not blank, so it still counts as emitted for the fallback.
                for tc in delta["tool_calls"]:
                    merge_tool_call_delta(tool_calls_acc, tc)
                delta.pop("tool_calls", None)
                emitted_visible = True

            if choice.get("finish_reason") is not None:
                has_finish = True
            elif delta.get("role"):
                visible = True

        # Buffer finish-only chunks (no visible content/tool_calls in them) so the
        # empty-turn fallback can inject content before the client finalizes. Chunks
        # that also carry content/tool_calls are emitted normally below.
        if has_finish and not visible:
            pending_finish.append(chunk)
            continue

        if visible:
            payload = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
            yield f"data: {payload}\n\n".encode("utf-8")


@app.get("/v1/models")
async def models(request: Request):
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(f"{SGLANG_BASE_URL}/models")
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    incoming = await request.json()
    body, sanitized_messages = prepare_request_body(incoming)
    model = body.get("model", "deepseek-v4-flash")

    headers = {"Content-Type": "application/json"}

    if body.get("stream") is True:
        async def stream_gen():
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    f"{SGLANG_BASE_URL}/chat/completions",
                    json=body,
                    headers=headers,
                ) as upstream:
                    if upstream.status_code >= 400:
                        error_body = await upstream.aread()
                        yield error_body
                        return

                    async for b in stream_filtered_response(upstream, model, sanitized_messages):
                        yield b

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{SGLANG_BASE_URL}/chat/completions",
            json=body,
            headers=headers,
        )

    if r.status_code >= 400:
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    response_json = r.json()
    response_json = strip_response_reasoning_and_store(
        response_json=response_json,
        model=model,
        request_messages=sanitized_messages,
    )
    return JSONResponse(response_json)


@app.get("/health")
async def health():
    return {"ok": True, "cached_reasoning_items": len(REASONING_CACHE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT)
