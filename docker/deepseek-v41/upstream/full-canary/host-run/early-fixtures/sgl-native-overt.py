import sys,os,unittest
from pathlib import Path
base=Path(os.environ['ROOT'])/'test/registered/unit/entrypoints/openai'
sys.path.insert(0,str(base))
from test_serving_deepseek_recipe import _serving
from transformers import PreTrainedTokenizerFast
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
t=PreTrainedTokenizerFast(tokenizer_file='/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017/tokenizer.json')
os.environ['SGLANG_DSV41_TOOL_CONTAINMENT']='1'
s=_serving(unittest.TestCase(),t)
req=ChatCompletionRequest(messages=[{'role':'user','content':'hi'}],tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object','properties':{'x':{'type':'string'}}},'strict':True}}]); out=s._encoding_output()
valid='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="x" string="true">1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
for x in (valid,valid.replace('name="f"','name="invalid"'),valid[:50],valid[:60],valid.replace('string="true">1','string="false">1')):
 ids=t.encode('Checking'+x,add_special_tokens=False)
 try:
  p=out._new_output_parser(req); out1=[p.feed([i]) for i in ids]; y=p.finish([],{'type':'stop'});print(type(p).__name__, sum(len(i.tool_calls) for i in out1), repr(''.join(i.content for i in out1))[:100], repr(y)[:170])
 except Exception as e: print(type(e).__name__, str(e)[:180])
