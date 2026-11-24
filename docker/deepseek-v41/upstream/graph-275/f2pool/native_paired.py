"""Interleave identical uncached Responses prompts with the native K780 engine parked or enabled."""
from contextlib import contextmanager, nullcontext
import json
import os
import socket
import time

import fullmodel as host
from native_bench import measure
from native_check import ARM, BASE, GATEWAY, wait
from throughput import write


@contextmanager
def park(i, *, broker=None):
    token=f"native-control-{i}"
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(8)
        connection.connect(str(broker or host.SOCKET))
        connection.sendall(f"LEASE {token}\n".encode())
        if connection.makefile("rb").readline()!=b"DRAINED\n":
            raise RuntimeError("Control arm failed to drain idle work")
        yield


def main():
    os.environ["CANARY_API_KEY_FILE"]=str(host.STATE/"api-key")
    rows=[]
    ARM.touch(mode=0o600,exist_ok=False)
    try:
        wait(lambda:json.loads(GATEWAY.read_text())["armed"],10,"public proof arm")
        for i in range(8):
            for mode in (("held","enabled") if i%2==0 else ("enabled","held")):
                with park(i) if mode=="held" else nullcontext():
                    time.sleep(.35)
                    row=measure(BASE,300+i,14 if i<6 else 150,192,cache_salt=f"native-pair-{i}-{mode}")
                if row["cached_tokens"]!=0:
                    raise ValueError("Expected an uncached control request")
                rows.append(dict(mode=mode,**row))
                write(host.RUN/"paired.json",{"rows":rows})
    finally:
        ARM.unlink(missing_ok=True)


if __name__=="__main__":
    main()
