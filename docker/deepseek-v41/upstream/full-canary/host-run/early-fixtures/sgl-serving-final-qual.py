import json,os,sys,unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
import pytest, huggingface_hub
base=Path(os.environ['ROOT'])/'test/registered/unit/entrypoints'
real=huggingface_hub.hf_hub_download
def local(repo_id,filename,*args,**kwargs):
 if repo_id=='deepseek-ai/DeepSeek-V4.1-Flash-1017' and filename=='tokenizer.json':
  return '/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017/tokenizer.json'
 return real(repo_id,filename,*args,**kwargs)
print({n:version(n) for n in ['torch','transformers','deepseek-recipe','cryptography','jsonschema','tml-renderers']})
with patch.object(huggingface_hub,'hf_hub_download',local):
 sys.exit(pytest.main(['-q','-rs','--disable-warnings','--junitxml=/tmp/sgl-serving-final-qual.xml',str(base/'anthropic/test_serving.py'),str(base/'openai/test_protocol.py'),str(base/'openai/test_responses_protocol.py'),str(base/'openai/test_serving_responses.py'),str(base/'openai/test_serving_responses_stream.py'),str(base/'openai/test_serving_deepseek_recipe.py'),str(base/'openai/test_dsv41_local_contract.py')]))
