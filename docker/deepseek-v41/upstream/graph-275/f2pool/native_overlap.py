"""Verify a concurrent Responses stream, an early disconnect and resumption of all native ranks."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time

import fullmodel as host
from native_check import ARM, BASE, GATEWAY, logs, wait
from qualify import request
from public_handbook import make_prompt
from throughput import sse, write


def stream(prompt, budget):
    return request(BASE,"/v1/responses",dict(model="deepseek-v4-flash",store=False,stream=True,
        input=[dict(role="user",content=prompt)],temperature=0,max_output_tokens=budget,
        chat_template_kwargs={"thinking":False}),timeout=45)


def main():
    os.environ["CANARY_API_KEY_FILE"]=str(host.STATE/"api-key")
    ARM.touch(mode=0o600,exist_ok=False)
    try:
        wait(lambda:json.loads(GATEWAY.read_text())["armed"],10,"public proof arm")
        first,second=threading.Event(),threading.Event()
        def cancel_one():
            with stream(make_prompt("public-native-cancel",32),4096) as incoming:
                for event in sse(incoming):
                    if event["type"]=="response.output_text.delta" and event.get("delta"):
                        first.set()
                        if not second.wait(20):
                            raise TimeoutError("concurrent request did not produce its first token")
                        return time.monotonic_ns()
            raise RuntimeError("cancellation stream produced no output")
        def survive():
            if not first.wait(20):
                raise TimeoutError("first stream did not start")
            with stream(make_prompt("public-native-survivor",15),160) as incoming:
                for event in sse(incoming):
                    if event["type"]=="response.output_text.delta":
                        second.set()
                    if event["type"] in ("response.completed","response.incomplete"):
                        return dict(http=incoming.status,output_tokens=event["response"]["usage"]["output_tokens"])
            raise RuntimeError("concurrent response failed")
        with ThreadPoolExecutor(2) as pool:
            a,b=pool.submit(cancel_one),pool.submit(survive)
            closed,survivor=a.result(),b.result()
        answer=[]
        with stream("What is 19 + 23? Reply only with the number.",96) as incoming:
            for event in sse(incoming):
                if event["type"]=="response.output_text.delta":
                    answer.append(event["delta"])
                if event["type"]=="response.completed":
                    break
        if "".join(answer).strip().rstrip(".")!="42":
            raise ValueError("arithmetic Responses smoke failed")
        def resumed():
            status=host.broker_status()
            if status["claims"] or status["serving"] or not all(status["idle"]):
                return False
            epoch=status["native_epoch"]
            ranks=logs()
            return len(ranks)==4 and all(any(e["event"]=="native_reentry" and e["epoch"]==epoch and e["gpu_seen_ns"]>closed
                                             for e in events) for events in ranks.values())
        wait(resumed,10,"all four native graphs after a disconnected stream")
        write(host.RUN/"overlap.json",dict(canceled_after_delta=True,concurrent_peer=survivor,
             arithmetic_42=True,all_four_resumed=True))
    finally:
        ARM.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
