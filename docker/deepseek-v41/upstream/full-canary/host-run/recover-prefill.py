import json,re,hashlib
from pathlib import Path
from tokenizers import Tokenizer
root=Path('/mnt/hot/dsv41_state/prefill-524k-restored-085-ZmKGG3uU/prefill-524k')
tokens=json.loads((root/'tokenized.json').read_text())['tokens']
tokenizer=Tokenizer.from_file('/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017/tokenizer.json')
text=tokenizer.decode(tokens,skip_special_tokens=False)
seed='Reference record: the cache stores recent entries; preserve ordering and validate invariants.\n'
match=re.search(r'Unique qualification ([a-f0-9]{32})\. Read reference notes; do not summarize them\.\n',text)
assert match, 'different synthetic fixture'
n=text.count(seed)
s=match.group(0)+seed*n+'\nThe requested verification number is 42. Reply with exactly 42 and nothing else.'
assert s in text
payload={'model':'deepseek-v4-flash','input':[{'role':'user','content':s}],'temperature':0,'max_output_tokens':64,'chat_template_kwargs':{'thinking':False},'stream':True,'store':False}
digest=hashlib.sha256((json.dumps(payload,indent=2,ensure_ascii=False)+'\n').encode()).hexdigest()
assert digest=='ff7319cb893282458cf77fdcef75ca5aebe0b8a5145708fe616bdab0cdd467df',digest
print(match.group(1),n,digest)
