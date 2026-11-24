import hashlib,json,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent; shape=p/'arms/joined/shapes'; previous=None
with (p/'receipts/joined-handbook-transitions.jsonl').open('x') as log:
 while True:
  data={f.name:json.loads(f.read_text()) for f in sorted(shape.glob('*.json'))}; signature=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
  if signature!=previous:
   log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'data':data},separators=(',',':'))+'\n');log.flush();previous=signature
  if json.loads((p/'receipts/joined-handbook.json').read_text()).get('complete'):break
  time.sleep(.15)
