import asyncio,json,subprocess,time
from pathlib import Path
from miners import HERE,snapshot
async def main():
    ledger=HERE.parent/'pool-sustained.json'
    gpu=HERE/'gpu-telemetry.csv'; temps=HERE/'thermal-telemetry.csv'
    while True:
        status=subprocess.run(['systemctl','--user','is-active','dsv41-sm120-full-peer.service'],capture_output=True,text=True).stdout.strip()
        try: fleet=await snapshot()
        except Exception as e: fleet={'snapshot_error':type(e).__name__}
        record=dict(time=time.time(),peer_service=status,fleet=fleet,hard_peer=json.loads(ledger.read_text()),latest_gpu_rows=gpu.read_text().splitlines()[-4:],latest_temp_rows=temps.read_text().splitlines()[-4:])
        p=HERE/('peer-at-stop.json' if status!='active' else 'peer-last-running.json')
        tmp=p.with_suffix('.part');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
        if status!='active': return
        await asyncio.sleep(.75)
asyncio.run(main())
