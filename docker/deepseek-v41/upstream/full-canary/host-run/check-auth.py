import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import add_api_key_middleware
from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here/"protected-boot"))
import boot
argv=[]
class Captured(Exception):
    pass
def capture(command, **kw):
    argv.extend(command)
    raise Captured()
with patch.object(boot.subprocess,"Popen",side_effect=capture):
    try:boot.serve()
    except Captured:pass
secret=boot.key()
assert secret not in argv and "--api-key" not in argv
idx=argv.index("--config")
config=Path(argv[idx+1])
assert config.stat().st_mode & 0o777 == 0o600 and (boot.STATE/"api-key").stat().st_mode & 0o777 == 0o600
argp=argparse.ArgumentParser();ServerArgs.add_cli_args(argp)
parsed=argp.parse_args(ConfigArgumentMerger(argp).merge_config_with_args(argv[3:]))
assert parsed.api_key == secret
app=FastAPI()
@app.post("/generate")
def generate():return {"ok":True}
add_api_key_middleware(app,get_keys=lambda:(parsed.api_key,None))
with TestClient(app) as client:
    codes=[client.post('/generate',headers=h).status_code for h in ({},{"Authorization":"Bearer reject-me"},{"Authorization":"Bearer "+secret})]
assert codes==[401,401,200], codes
print(json.dumps({"unauthenticated":codes[0],"wrong_key":codes[1],"correct_key":codes[2],"key_in_child_argv":False,"key_mode":"0600","yaml_mode":"0600"}))
