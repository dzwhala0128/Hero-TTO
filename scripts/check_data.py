"""Offline inventory, runtime Gold-key audit, AF mapping hashes and package seal."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.common import read,file_sha
from pipeline.build_inputs import no_gold,load

def check():
 counts={}
 for split,n in [('full500',500),('targeted40',40),('full169',169)]:
  claims=read(ROOT/'data'/split/'claims.json');ids=[r['claim_id'] for r in claims]
  assert len(ids)==len(set(ids))==n,(split,'claim count')
  if split!='full500':assert ids==read(ROOT/'data'/split/'claim_ids.json')
  rows=load(split,task='baseline');assert [r['claim_id'] for r in rows]==ids
  assert sum(len(r['evidence']) for r in rows)==10*n
  counts[split]={'claims':n,'qa':10*n,'frozen_vp':n if split!='targeted40' else 'separate 9B frozen arm inputs'}
 for af,n in [('human63',63),('merged67',67)]:
  rows=load('targeted40',af);mapping=read(ROOT/'data/targeted40'/af/'mapping.json')
  assert len(rows)==len(mapping)==40 and sum(len(r['afs']) for r in rows)==n
  assert [r['claim_id'] for r in rows]==read(ROOT/'data/targeted40/claim_ids.json')
  assert sum(len(r['evidence']) for r in rows)==154
  for r,m in zip(rows,mapping):
   assert r['claim_id']==m['claim_id']
   assert r['afs']==[a['text'] for a in m['afs']]
   for a in m['afs']:assert hashlib.sha256(a['text'].encode()).hexdigest()==a['sha256']
   assert [{k:v for k,v in q.items() if k in ['question','answer','url']} for q in m['selected_qas']]==r['evidence']
   assert r['initial_solution'] in r['evaluator_user']
  counts[af]={'claims':40,'afs':n,'qa':154}
 a=load('targeted40','human63');b=load('targeted40','merged67')
 assert [r['evidence'] for r in a]==[r['evidence'] for r in b]
 assert [r['initial_solution'] for r in a]==[r['initial_solution'] for r in b]
 for p in (ROOT/'data').rglob('*.json'):
  if not p.is_relative_to(ROOT/'data/evaluation_only') and not p.is_relative_to(ROOT/'data/manifests'):
   no_gold(read(p))
 for split in ['full500','full169']:
  for r in load(split,task='baseline'):
   assert set(r['evidence'][0])<={'question','answer','url'}
 seal=ROOT/'data/manifests/package_sha256.json'
 if seal.exists():
  for name,sha in read(seal).items():assert file_sha(ROOT/name)==sha,'Package drift: '+name
 counts['fine146']={'status':'missing frozen data, not runnable, no invented AF'}
 print(json.dumps(counts,ensure_ascii=False,indent=2));return counts
if __name__=='__main__':check()
