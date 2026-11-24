import json, os, signal, subprocess, sys, pathlib, time
target=pathlib.Path(sys.argv[1]); worker=int(sys.argv[2]); rows=[]
try:
  while pathlib.Path(f'/proc/{worker}/cmdline').read_bytes():
    free=int(subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free','--format=csv,noheader,nounits']))
    rows.append(free)
    if free < 96:
      os.kill(worker, signal.SIGTERM)
      break
    time.sleep(.25)
except (FileNotFoundError,ProcessLookupError):
  pass
finally:
  target.write_text(json.dumps(dict(samples=len(rows),min_free_mib=min(rows,default=None),max_free_mib=max(rows,default=None))))
