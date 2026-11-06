"""Validate external per-Claim knowledge store; optionally export Lucene JSONL."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--split',default='full500',choices=['full500','targeted40','full169']);p.add_argument('--export-lucene');p.add_argument('--seal',required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];claims=json.loads((R/'data'/a.split/'claims.json').read_text());store=Path(a.store);hashes={};docs=[]
for r in claims:
 file=store/f'{r["claim_id"]}.json';raw=file.read_bytes();hashes[file.name]=hashlib.sha256(raw).hexdigest();count=0
 for line in raw.decode().splitlines():
  item=json.loads(line)
  if not isinstance(item.get('url'),str) or not isinstance(item.get('url2text'),list):raise ValueError('Invalid store schema: '+str(file))
  for s in item['url2text']:
   if not isinstance(s,str):raise ValueError('Invalid sentence')
   if a.export_lucene:docs.append(dict(id=f'{r["claim_id"]}_{count}',contents=s,claim_id=r['claim_id'],url=item['url']))
   count+=1
 if not count:raise ValueError('Empty claim knowledge: '+str(file))
Path(a.seal).parent.mkdir(parents=True,exist_ok=True);Path(a.seal).write_text(json.dumps({'split':a.split,'claim_count':len(claims),'sha256':hashes},indent=2)+'\n')
if a.export_lucene:
 out=Path(a.export_lucene);out.mkdir(parents=True,exist_ok=True)
 with (out/'docs.jsonl').open('x') as f:
  for d in docs:f.write(json.dumps(d,ensure_ascii=False)+'\n')
print('Verified',len(claims),'knowledge files')
