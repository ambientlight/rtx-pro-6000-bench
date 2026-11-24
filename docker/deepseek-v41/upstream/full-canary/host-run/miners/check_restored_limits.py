#!/usr/bin/env python3
"""Read only: compare restored dsv41 namespace against the confirmed four-card mining policy."""
import json
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
CONTAINER='087c42c0a93e58dc2ab902b13c1719a03ba6a2a261a3b685129d7743df5e3130'
CGROUP=Path('/sys/fs/cgroup/dsv41.slice') / f'docker-{CONTAINER}.scope'
SOURCE=Path('/mnt/hot/ambientlight/repos/mining/pool-trial/five-gpu')
def run(*args):
    return subprocess.check_output(args,text=True).strip()

def inspect():
    expected=json.loads((SOURCE/'preflight.json').read_text())['linux']['dsv41']['hard_limit_each_MiB']
    snapshot=json.loads((SOURCE/'continuous-vram-20261122T0557Z.json').read_text())
    active=run('docker','--context','shared-operator','inspect','dsv41','--format','{{.Id}} {{.State.Status}}').split()
    if active[0] != CONTAINER:raise ValueError(f'dsv41 container changed: {active[0]}')
    cards=run('nvidia-smi','--query-gpu=index,uuid,memory.total,memory.reserved','--format=csv,noheader,nounits')
    rows=[]
    for card in cards.splitlines():
        i,uuid,total,reserved=(x.strip() for x in card.split(','))
        old=snapshot['cards'][i] if isinstance(snapshot['cards'],dict) else snapshot['cards'][int(i)]
        target=int(total)-int(reserved)-1024
        assert target==expected==old['deepseek_limit_MiB']
        rows.append({'index':int(i),'uuid':uuid,'total_mib':int(total),'driver_reserved_mib':int(reserved),'expected_hard_mib':target})
    applied=False
    if CGROUP.is_dir():
        text=run('nvidia-smi','memory-limits','-n',str(CGROUP),'--get')
        for match in re.finditer(r'^GPU (\d+):[^\n]+\n\s*namespace:[^\n]+\n\s*soft limit:[^\n]+\n\s*hard limit:\s*(max|\d+)(?: MiB)?\n\s*current used:\s*(\d+) MiB',text,re.M):
            idx,actual,used=match.groups(); row=rows[int(idx)]
            row.update(actual_hard_mib='max' if actual=='max' else int(actual),used_mib=int(used))
        applied=all(row.get('actual_hard_mib')==row['expected_hard_mib'] for row in rows)
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),original_container_state=active[1],cgroup=str(CGROUP),namespace_present=CGROUP.is_dir(),prior_policy_met=applied,rows=rows,verified_sources=['five-gpu/preflight.json','five-gpu/continuous-vram-20261122T0557Z.json'],
        recovery='F2Pool live() checks existing namespace limits without applying them. The released fleet deepseek_limits(apply=True) only sets a per-GPU target after current usage is <= that target. A healthy running incumbent is mandatory before supervisor start.')
if __name__=='__main__':
    print(json.dumps(inspect(),indent=2))
