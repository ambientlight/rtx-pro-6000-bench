"""Check four real rank transports on a disarmed loopback gateway without using a GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from gateway import digest, secret
from replay_pool import ReplayPool


def check(receipt, token, port, timeout):
    def snapshot():
        return json.loads(receipt.read_text())

    initial = snapshot()
    if (not initial["connected"] or initial["armed"] or initial["authorized_clients"]
            or initial["submitted"] or initial["accepted"]):
        raise ValueError("Transport check requires a fresh, connected and disarmed gateway with zero clients")
    clients = []
    try:
        for _ in range(4):
            clients.append(ReplayPool("127.0.0.1", port, token=token))
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            state = snapshot()
            if state["run_id"] != initial["run_id"] or state["armed"]:
                raise RuntimeError("Gateway changed during the transport check")
            offered = state["job"]
            snapshots = [client.snapshot() for client in clients]
            if (state["connected"] and state["authorized_clients"] == 4 and offered
                    and all(job and job.id == offered["id"]
                            and digest(job.header) == offered["header_sha256"]
                            and job.target == int(offered["target"], 16)
                            and job.share_bits == offered["share_bits"] for job in snapshots)):
                return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), connected=True,
                            upstream_endpoint=state["endpoint"], authenticated_transport_clients=4,
                            client="ReplayPool", phase="disarmed preflight without GPUs or model requests",
                            header_sha256=offered["header_sha256"], target=offered["target"],
                            individual_job_and_target_matches=4, upstream_submissions=state["submitted"],
                            accepted=state["accepted"])
            time.sleep(.05)
        raise TimeoutError("Four rank transports did not acquire the gateway's live job")
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--gateway-receipt", required=True, type=Path)
    parser.add_argument("--port", default=31990, type=int)
    args = parser.parse_args()
    print(json.dumps(check(args.gateway_receipt, secret(args.token_file), args.port, 15), indent=2))
