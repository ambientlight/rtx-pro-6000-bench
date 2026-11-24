"""Observe the original container's GPU quota at cgroup recreation and model readiness."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
from check_restored_limits import CGROUP, HERE, inspect

def write(name,record):
    p=HERE/name
    staged=p.with_suffix('.part')
    staged.write_text(json.dumps(record,indent=2)+'\n')
    staged.replace(p)

def health():
    # The shared Docker proxy exposes health through ps Status, while inspect
    # only includes Running, Status and ExitCode. No model calls are needed.
    result=subprocess.run(['docker','--context','shared-operator','ps','--filter','name=^/dsv41$','--format','{{.Status}}'],capture_output=True,text=True)
    state=result.stdout.strip()
    return 'healthy' if '(healthy)' in state else state or 'unknown'

def main():
    while not CGROUP.is_dir():
        time.sleep(3)
    with (HERE/'restoration-timeline.jsonl').open('a',buffering=1) as timeline:
        while CGROUP.is_dir():
            try:
                record=inspect()
            except (OSError,subprocess.SubprocessError):
                time.sleep(3)
                continue
            record['health']=health()
            timeline.write(json.dumps(record,separators=(',',':'))+'\n')
            if record['health']=='healthy':
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/server_info',timeout=3) as response:
                        info=json.load(response)
                    record['serving']={name:info.get(name) for name in ['context_length','tp_size','ep_size','mem_fraction_static','max_total_num_tokens']}
                except (OSError,ValueError) as exc:
                    record['serving_lookup']=type(exc).__name__
                write('restored-budget.json',record)
                return
            time.sleep(20)

if __name__=='__main__':
    main()
