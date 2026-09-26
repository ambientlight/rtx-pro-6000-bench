#!/usr/bin/env python3
"""Prime/replay two header-needle prompts across an intervening cache-pressure run.

The answer is near the START of each prompt, not in its uncached trailing page.
Replays require observed FULL-KV and SWA restores, not just a cached-token claim.
Never flushes cache, changes configuration or sends concurrent inference.
"""

import argparse
import json
import os
from pathlib import Path
import secrets
import time
from types import SimpleNamespace as NS

from dsv41_cache_pressure import MODEL, container_state, host_memory, random_words, save, selected_metrics
from dsv41_deferred_relaunch import activity_metrics, fetch
from dsv41_live_acceptance import Client, completed_messages, completed_responses
from dsv41_load import is_idle, parse_loads


def snapshot():
    raw = fetch("/metrics")
    active, fingerprint = activity_metrics(raw)
    return {"metrics": selected_metrics(raw), "active": active, "http": {str(k): v for k, v in fingerprint}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prime", "replay"))
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = args.state_dir / "manifest.json"
    if args.phase == "prime":
        if manifest_path.exists():
            raise RuntimeError("Refusing to replace an existing probe")
        cases = []
        nonce = secrets.token_hex(16)
        for i, api in enumerate(("responses", "messages")):
            answer = str(100000 + secrets.randbelow(900000))
            prompt = (
                f"Unique record {nonce}-{api}. Reference secret: {answer}.\n"
                "Remember that reference secret. The following words are inert filler, not instructions.\n<data>\n"
                + " ".join(random_words(nonce, i, 7000))
                + "\n</data>\nReply with exactly the six-digit reference secret from the first line, and nothing else."
            )
            payload = {"model": MODEL, "stream": True, "temperature": 0}
            if api == "responses":
                payload.update(
                    input=[{"role": "user", "content": prompt}],
                    store=False,
                    max_output_tokens=16,
                    chat_template_kwargs={"thinking": False},
                )
            else:
                payload.update(
                    messages=[{"role": "user", "content": prompt}], max_tokens=16, thinking={"type": "disabled"}
                )
            cases.append({"api": api, "answer": answer, "payload": payload})
        manifest = {"identity": container_state(), "cases": cases}
        save(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text())
    client = Client(
        NS(
            base_url="http://127.0.0.1:8000",
            model=MODEL,
            long_contexts=[],
            context_limit=524288,
            timeout=180,
            key_file=None,
            output_root=args.state_dir,
            suite=args.phase,
        )
    )
    records = []
    for case in manifest["cases"]:
        if container_state() != manifest["identity"]:
            raise RuntimeError("Probe deployment changed")
        if not is_idle(parse_loads(json.loads(fetch("/v1/loads")))) or host_memory()["MemAvailable"] < 24 * 2**30:
            raise RuntimeError("Probe requires an idle service and >=24 GiB available RAM")
        before = snapshot()
        if before["active"]:
            raise RuntimeError("Another inference request is active")
        events = client.request(case["api"], "/v1/" + case["api"], case["payload"])
        if case["api"] == "responses":
            response = completed_responses(events)
            answer = "".join(
                block["text"]
                for item in response["output"]
                if item["type"] == "message"
                for block in item["content"]
                if block["type"] == "output_text"
            )
            usage = response["usage"]
        else:
            content, stop = completed_messages(events)
            if stop != "end_turn":
                raise RuntimeError("Messages did not finish normally")
            answer = "".join(block["text"] for block in content if block["type"] == "text")
            usage = dict(next(e["message"]["usage"] for e in events if e["type"] == "message_start"))
            for event in events:
                if event["type"] == "message_delta":
                    usage.update(event.get("usage", {}))
        if answer.strip() != case["answer"]:
            raise RuntimeError("Header needle answer is incorrect")
        # Stream completion can precede ASGI/metrics finalization. Wait for
        # settled counters, as the existing bounded cache acceptance does.
        expected_change = {str(("/v1/" + case["api"], "POST")): 1}
        for _ in range(16):
            time.sleep(1)
            after = snapshot()
            changes = {
                key: after["http"].get(key, 0) - before["http"].get(key, 0)
                for key in before["http"].keys() | after["http"].keys()
                if after["http"].get(key, 0) != before["http"].get(key, 0)
            }
            if not after["active"] and changes == expected_change:
                break
        save(client.output / (case["api"] + "-attribution.json"), {"before": before, "after": after, "changes": changes})
        if after["active"] or changes != {str(("/v1/" + case["api"], "POST")): 1}:
            raise RuntimeError("External inference prevents reliable attribution")
        delta = {name: value - before["metrics"][name] for name, value in after["metrics"].items()}
        record = {
            "api": case["api"],
            "phase": args.phase,
            "answer_correct": True,
            "usage": usage,
            "metric_delta": delta,
            "before": before,
            "after": after,
        }
        records.append(record)
        save(client.output / "probe-results.json", records)
        print(json.dumps({"api": case["api"], "phase": args.phase, "usage": usage, "metric_delta": delta}), flush=True)
        if args.phase == "replay" and not (
            delta["load_back_kv_tokens_all_ranks"] > 0 and delta["load_back_swa_tokens_all_ranks"] > 0
        ):
            raise RuntimeError("Replay did not prove both FULL and SWA restoration from RAM")
    save(
        args.state_dir / (args.phase + "-complete.json"),
        {"results": str(client.output / "probe-results.json"), "identity": container_state()},
    )


if __name__ == "__main__":
    main()
