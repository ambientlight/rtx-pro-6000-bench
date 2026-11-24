"""Exercise both model proof paths with an independent easy-target verifier on loopback."""
import json
import os
from pathlib import Path
import time

import fullmodel as host
import native_host
from native_bench import measure
from native_check import ARM, BASE, GATEWAY, wait
from native_paired import park
from throughput import write


def main():
    os.environ["CANARY_API_KEY_FILE"]=str(host.STATE/"api-key")
    if ARM.exists() or json.loads(GATEWAY.read_text())["armed"]:
        raise ValueError("Disarm the public-input gateway before the isolated proof test")
    local=host.RUN/"receipts/local.json"
    verified=host.RUN/"receipts/local-proofs.jsonl"
    if local.exists() or verified.exists():
        raise ValueError("Use a fresh private proof receipt")
    with park("local-proof"):
        if host.run("systemctl","--user","show",host.GATEWAY,"-p","ActiveState","--value") in ("active","deactivating"):
            host.run("systemctl","--user","stop",host.GATEWAY,timeout=125)
        try:
            # This peer listens only on loopback. It has no upstream pool connection.
            host.unit(host.SAMPLER,host.OLD/"env-run",host.GRAPH.parent/"full-canary/public_pool.py","--port","31990",
                      "--model",native_host.MODEL,"--verifier",native_host.VERIFIER,"--sustained",
                      "--share-target",format(1<<224,"064x"),"--receipt",local,"--accepted-log",verified,
                      output=host.RUN/"logs/local-proof.log")
            wait(lambda:local.exists() and json.loads(local.read_text())["negotiations"]>=4,30,"four local verifier clients")
            for i in range(12):
                measure(BASE,600+i,70,192)
                time.sleep(.25)
                data=json.loads(local.read_text())
                by_rank=data["verified_by_rank_rows"]
                if all(v.get("6",0)>0 and v.get("2048",0)>0 for v in by_rank.values()):
                    break
            else:
                raise RuntimeError("Missing independent prefill or decode proof on at least one rank")
            proofs=[json.loads(x) for x in verified.read_text().splitlines()]
            if data["rejected_frames"] or len({x["proof_sha256"] for x in proofs})!=len(proofs):
                raise RuntimeError("Duplicate or invalid local proof")
            write(host.RUN/"proof-check.json",dict(scope="loopback qualification target",requests=i+1,
                  target=data["job"]["target"],verifier=data["verifier_sha256"],accepted=data["accepted_total"],
                  unique=len({x["proof_sha256"] for x in proofs}),rejected=data["rejected_frames"],
                  verified_by_rank_rows=by_rank))
        finally:
            if host.active(host.SAMPLER):
                host.run("systemctl","--user","stop",host.SAMPLER)
            if host.active(host.CANDIDATE) and not (host.STATE/"guard.json").exists():
                GATEWAY.rename(GATEWAY.with_name("gateway-first-session.json"))
                native_host.gateway()
                wait(lambda:GATEWAY.exists() and json.loads(GATEWAY.read_text())["authorized_clients"]==4
                     and json.loads(GATEWAY.read_text())["connected"],40,"real F2Pool inference gateway reconnect")


if __name__=="__main__":
    main()
