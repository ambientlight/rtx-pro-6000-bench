import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parent
state = Path('/mnt/hot/dsv41_state/production')
bench = root.parent/'deployment-source/docker/deepseek-v41/upstream/full-canary'
sys.path.insert(0, str(bench))
from qualify import request

base = 'http://127.0.0.1:8000'
started = int((root/'receipts/original-restart-ns').read_text())
os.environ['CANARY_API_KEY_FILE'] = str(state/'api-key')
for _ in range(450):
    try:
        with request(base, '/health', timeout=4):
            pass
        names = ('smoke.json', 'smoke-structured.json', 'smoke-tools.json', 'smoke-vision.json')
        files = {name:(state/name).lstat() for name in names}
        if all(s.st_uid == 0 and stat.S_ISREG(s.st_mode) and s.st_mtime_ns >= started for s in files.values()):
            break
    except OSError:
        pass
    time.sleep(2)
else:
    raise RuntimeError('original startup smoke checks did not complete')
with request(base, '/server_info') as response:
    server = json.load(response)
with request(base, '/v1/chat/completions', {'model':server['served_model_name'], 'messages':[{'role':'user','content':'What is 21 + 21? Give just the number.'}], 'temperature':0, 'max_tokens':96, 'chat_template_kwargs':{'thinking':False}}) as response:
    payload=json.load(response)
message = payload['choices'][0]['message']
answer = str(message.get('content','')).strip()
if answer.rstrip('.') != '42':
    raise RuntimeError('original public arithmetic check did not return 42')
result={'original_restart_epoch_ns':started,'verified_epoch_ns':time.time_ns(),'smoke_mtimes_ns':{k:s.st_mtime_ns for k,s in files.items()}, 'arithmetic_returned_42':True,'capacity':server['max_total_num_tokens'],'decode_graph':server['cuda_graph_config']['decode']['backend']}
result['power_limits_w']=[float(s.strip()) for s in subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).splitlines()]
result['container']=subprocess.check_output(['docker','--context','shared-operator','inspect','-f','{{.Id}} {{.State.Status}}','dsv41'],text=True).strip()
assert result['container']==(root/'receipts/preexisting-container.txt').read_text().strip() and result['power_limits_w']==[275]*4
(root/'receipts/restored.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
