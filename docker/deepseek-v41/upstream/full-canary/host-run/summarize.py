import json
from pathlib import Path
root=Path(__file__).parent
rows=[json.loads(l) for l in (root/'footprint.jsonl').read_text().splitlines() if l]
active=[r for r in rows if 'cgroup_gib' in r]
if active:
 print('candidate memory, GiB', max(r['cgroup_gib'] for r in active),'; min available',min(r['available_gib'] for r in active))
print(rows[-1])
