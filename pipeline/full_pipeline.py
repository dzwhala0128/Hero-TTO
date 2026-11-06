"""Original stage scripts, explicit external assets; no target Gold reads."""
import os,sys,subprocess,importlib.util
from .common import ROOT,save,read
from .build_inputs import no_gold

def run(args,folder):
 claims=read(ROOT/'data'/args.split/'claims.json')
 if args.claim_id is not None:claims=[r for r in claims if r['claim_id']==args.claim_id]
 if args.limit:claims=claims[:args.limit]
 if not claims:raise ValueError('No selected claims')
 if args.mock:raise ValueError('Full retrieval cannot be simulated as a successful HerO pipeline')
 if not os.getenv('KNOWLEDGE_STORE'):raise ValueError('Missing external KNOWLEDGE_STORE; see docs/ASSETS.md')
 from pathlib import Path
 store=Path(os.environ['KNOWLEDGE_STORE'])
 missing=[str(store/f'{r["claim_id"]}.json') for r in claims if not (store/f'{r["claim_id"]}.json').is_file()]
 if missing:raise ValueError('Missing knowledge assets: '+str(missing[:5]))
 for key in ['HYDE_MODEL_PATH','QG_MODEL_PATH','SFR_MODEL_PATH','CUDA_VISIBLE_DEVICES']:
  if not os.getenv(key):raise ValueError('Missing explicit '+key)
 for mod in ['torch','vllm','transformers','sentence_transformers','rank_bm25','numpy','pandas','nltk','sklearn']:
  if importlib.util.find_spec(mod) is None:raise ValueError('Missing retrieval dependency: '+mod)
 from .common import file_sha
 asset_hashes={f'{r["claim_id"]}.json':file_sha(store/f'{r["claim_id"]}.json') for r in claims}
 if args.preflight:print('Full-pipeline asset paths/dependencies present; GPU execution still unverified');return
 if folder.exists() and not args.resume:raise ValueError('Output exists; use --resume')
 folder.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(folder/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (folder/'freeze.json').exists():raise ValueError('Frozen run cannot be changed')
 if (folder/'knowledge_assets.json').exists() and read(folder/'knowledge_assets.json')!=asset_hashes:raise ValueError('Knowledge assets changed on resume')
 save(folder/'knowledge_assets.json',asset_hashes)
 save(folder/'selected_claims.json',claims)
 # Each complete stage has an input/code/config fingerprint. Partial stage output is never accepted.
 from .common import digest,file_sha
 def stage(name,script,params):
  output=folder/name;marker=folder/(name+'.complete.json')
  signature=digest({'script':file_sha(ROOT/'hero/stages'/script),'shared_utils':file_sha(ROOT/'hero/stages/hyde_doc_utils.py'),'knowledge_assets':asset_hashes,'input_files':{v:file_sha(v) for v in params if Path(v).is_file() and Path(v)!=output},'params':params,'claims':claims,'env':{k:os.getenv(k) for k in ['HYDE_MODEL_PATH','QG_MODEL_PATH','SFR_MODEL_PATH','CUDA_VISIBLE_DEVICES','HERO_TP']}})
  if marker.exists():
   seal=read(marker)
   if seal!={'signature':signature,'sha256':file_sha(output)}:raise ValueError('Stage resume mismatch: '+name)
   return
  if output.exists():raise ValueError('Partial stage exists; preserve it and use a new run-id: '+name)
  subprocess.run([sys.executable,str(ROOT/'hero/stages'/script),*params],check=True,cwd=ROOT)
  save(marker,{'signature':signature,'sha256':file_sha(output)})
 stage('hyde.json','hyde_fc_generation.py',['-i',str(folder/'selected_claims.json'),'-o',str(folder/'hyde.json'),'-m',os.environ['HYDE_MODEL_PATH']])
 stage('retrieval.jsonl','retrieval.py',['--target_data',str(folder/'hyde.json'),'-k',str(store),'-o',str(folder/'retrieval.jsonl')])
 stage('reranking.jsonl','reranking.py',['--target_data',str(folder/'retrieval.jsonl'),'-o',str(folder/'reranking.jsonl')])
 stage('qa.jsonl','question_generation.py',['-i',str(folder/'reranking.jsonl'),'-o',str(folder/'qa.jsonl'),'--model',os.environ['QG_MODEL_PATH'],'--reference_corpus',str(ROOT/'data/qg_reference.json')])
 rows=[__import__('json').loads(s) for s in (folder/'qa.jsonl').read_text().splitlines()];no_gold(rows)
 from .runner import settings,freeze
 from .baseline import Baseline
 from .common import digest
 run_cfg={'task':'full-pipeline','split':args.split,'claim_ids':[r['claim_id'] for r in rows],'mock':False,'input_sha256':digest(rows)}
 if (folder/'run.json').exists() and read(folder/'run.json')!=run_cfg:raise ValueError('Full-pipeline resume changed')
 save(folder/'run.json',run_cfg);vp=Baseline(settings('hero'),max_attempts=args.max_attempts)
 for r in rows:
  dest=folder/'claims'/str(r['claim_id'])
  if not (dest/'result.json').exists():save(dest/'result.json',vp.run(r,dest))
  else:vp.top_k=read(dest/'result.json').get('top_k',vp.top_k)
 if args.freeze:freeze(folder)
