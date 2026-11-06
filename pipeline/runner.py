"""Unified no-gold runtime. Evaluation is a separate process/module."""
import argparse,json,os,sys,fcntl,traceback
from .common import ROOT,read,save,config,digest,file_sha,safe_output
from .build_inputs import load

def settings(role,stage=None,protocol=None):
 models=config('models.yaml');m=models['textgrad_e18'] if role=='textgrad' and protocol and 'historical_evaluator' in protocol else models[role]
 result=dict(endpoint=os.getenv(role.upper()+'_BASE_URL',m['endpoint']),model=os.getenv(role.upper()+'_MODEL',m['model']),timeout=(protocol or {}).get('timeout_seconds',300),max_context=(protocol or {}).get('max_context',4096 if role=='hero' else 8192))
 if protocol and stage:result.update(protocol['stages'][stage])
 return result

def optimize(item,folder,args,cfg):
 from .engine import TraceEngine
 from .evaluator import call_for
 from .native_backward import backward
 from .optimizer import step
 from .validator import validate
 import textgrad as tg
 faithful=args.split=='full169';initial=item['initial_solution'];solution=tg.Variable(initial,requires_grad=True,role_description='CURRENT_SOLUTION')
 baseline=validate(initial)
 # E18 baseline can be an explicit two-field wrapper around an unchanged raw VP response.
 if not baseline['valid']:raise ValueError('Frozen baseline is not structurally parseable')
 result=dict(claim_id=item['claim_id'],claim=item['claim'],evidence=item['evidence'],pred_label=item['pred_label'],llm_output=item['llm_output'],mock=args.mock,status='baseline_fallback',accepted=False,iterations=0)
 try:
  for iteration in range(args.max_iterations):
   current_folder=folder/f'iteration_{iteration+1}'
   replay=item['frozen_evaluation'] if faithful and not args.mock else None
   loss=call_for(item,solution,TraceEngine(current_folder,'evaluator',settings('textgrad','evaluator',cfg),args.mock,replay),faithful)
   save(current_folder/'evaluator_feedback.json',{'text':loss.value,'frozen_replay':replay is not None})
   grad=backward(loss,solution,TraceEngine(current_folder,'backward',settings('textgrad','backward',cfg),args.mock),faithful)
   save(current_folder/'textual_gradient.json',grad)
   if faithful:
    status='MATERIAL_ISSUE' if args.mock else item['frozen_status']
    expected='CHANGE_NEEDED:' if status=='MATERIAL_ISSUE' else 'NO_CHANGE:'
    if not grad['text'].strip().startswith(expected):raise ValueError('Faithful backward status contract failed')
    context='CLAIM:\n'+item['claim']+'\n\nQA_EVIDENCE:\n'+item['qa_evidence_text']+'\n\nCURRENT_SOLUTION:\n'+initial
    for g in solution.gradients:solution.gradients_context[g]['context']=context
    if status=='NO_MATERIAL_ISSUE':
     result.update(status='no_change',accepted=False);break
   candidate=step(solution,TraceEngine(current_folder,'optimizer',settings('textgrad','optimizer',cfg),args.mock),cfg['optimizer_constraints'])
   save(current_folder/'optimizer_solution.json',{'solution':candidate})
   check=validate(candidate,allow_plain=not faithful);save(current_folder/'structural_validator.json',check)
   if not check['valid']:raise ValueError('Optimizer candidate failed structure validation')
   result.update(pred_label=check['parsed']['verdict'],llm_output=candidate,status='optimized',accepted=True,iterations=iteration+1)
   solution.gradients.clear();solution.gradients_context.clear()
 except Exception as exc:
  save(folder/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
  result.update(pred_label=item['pred_label'],llm_output=item['llm_output'],status='baseline_fallback',accepted=False,iterations=0,error=str(exc))
 return result

def freeze(folder):
 run=read(folder/'run.json');rows=[]
 for cid in run['claim_ids']:
  p=folder/'claims'/str(cid)/'result.json'
  if not p.exists():raise ValueError('Cannot freeze incomplete run; missing '+str(cid))
  rows.append(read(p))
 pred=folder/'predictions.json'
 if (folder/'freeze.json').exists():raise ValueError('Already frozen; outputs are immutable')
 save(pred,rows)
 save(folder/'freeze.json',dict(schema_version=1,claim_ids=run['claim_ids'],predictions_sha256=file_sha(pred),run_sha256=file_sha(folder/'run.json'),mock=run['mock'],result_sha256={str(r['claim_id']):file_sha(folder/'claims'/str(r['claim_id'])/'result.json') for r in rows},counts={s:sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}}))
 print('Frozen:',pred)

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['baseline','textgrad','status','freeze'])
 p.add_argument('--split',choices=['targeted40','full169','full500'],default='targeted40');p.add_argument('--af',choices=['human63','merged67','fine146'])
 p.add_argument('--mode',choices=['vp-only','full-pipeline'],default='vp-only');p.add_argument('--claim-id',type=int);p.add_argument('--limit',type=int)
 p.add_argument('--run-id');p.add_argument('--mock',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--resume',action='store_true')
 p.add_argument('--max-iterations',type=int,default=1);p.add_argument('--max-attempts',type=int,default=10)
 return p

def main(argv=None):
 args=parser().parse_args(argv)
 if args.max_iterations<1:raise ValueError('--max-iterations must be positive')
 if args.split=='full169' and args.max_iterations!=1:raise ValueError('Full169 frozen evaluator replay is a one-step historical protocol')
 run_id=args.run_id or f'{args.task}_{args.split}_{args.af or "qa"}'+('_mock' if args.mock else '')
 if '/' in run_id or run_id in ('.','..'):raise ValueError('run-id must be a single directory name')
 folder=safe_output('outputs/'+run_id)
 if args.task=='status':
  run=read(folder/'run.json');done=sum((folder/'claims'/str(i)/'result.json').exists() for i in run['claim_ids']);print(f'{done}/{len(run["claim_ids"])} completed; frozen={(folder/"freeze.json").exists()}');return
 if args.task=='freeze':freeze(folder);return
 if args.mode=='full-pipeline' and args.task!='baseline':raise ValueError('full-pipeline is a baseline evidence-generation mode')
 if args.mode=='full-pipeline':
  from .full_pipeline import run
  return run(args,folder)
 rows=load(args.split,args.af,args.task)
 if args.task=='textgrad' and args.split=='full500':raise ValueError('Full500 has 500 QA/VP records, but no frozen AF/Evaluator protocol. Run baseline; do not claim AF500 is available.')
 if args.claim_id is not None:
  rows=[r for r in rows if r['claim_id']==args.claim_id]
  if not rows:raise ValueError('Claim ID outside fixed split')
 if args.limit is not None:
  if args.limit<1:raise ValueError('limit must be positive')
  rows=rows[:args.limit]
 cfg=config((args.af+'.yaml') if args.af else 'textgrad.yaml') if args.task=='textgrad' else config('hero_baseline.yaml')
 for r in rows:
  if args.task=='textgrad' and not {'initial_solution','evaluator_system','evaluator_user'}<=r.keys():raise ValueError('Missing frozen TextGrad inputs')
 if args.preflight:
  print(json.dumps({'valid':True,'task':args.task,'split':args.split,'af':args.af,'count':len(rows),'claim_ids':[r['claim_id'] for r in rows],'protocol':cfg.get('protocol','E18'),'gpu_verified':False},ensure_ascii=False));return
 run=dict(task=args.task,split=args.split,af=args.af,claim_ids=[r['claim_id'] for r in rows],mock=args.mock,input_sha256=digest(rows),config_sha256=digest(cfg),iterations=args.max_iterations,model_settings={s:settings(s,protocol=cfg) for s in ['hero','textgrad']},code_sha256={str(p.relative_to(ROOT)):file_sha(p) for d in ['pipeline','textgrad_native','configs','atomic_facts/common'] for p in sorted((ROOT/d).rglob('*')) if p.is_file() and p.suffix in ['.py','.yaml']})
 if (folder/'freeze.json').exists():raise ValueError('Run frozen; select a new --run-id')
 folder.mkdir(parents=True,exist_ok=True)
 with (folder/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (folder/'run.json').exists():
   if not args.resume:raise ValueError('Run exists; use --resume or a new --run-id')
   if read(folder/'run.json')!=run:raise ValueError('Resume input/config/source mismatch')
  else:save(folder/'run.json',run)
  from .baseline import Baseline
  baseline=Baseline(settings('hero'),args.mock,args.max_attempts)
  for index,item in enumerate(rows,1):
   dest=folder/'claims'/str(item['claim_id']);result_path=dest/'result.json'
   if result_path.exists():
    baseline.top_k=read(result_path).get('top_k',baseline.top_k);continue
   if args.task=='baseline':result=baseline.run(item,dest)
   else:result=optimize(item,dest,args,cfg)
   save(result_path,result)
   with (folder/'progress.jsonl').open('a') as f:f.write(json.dumps({'completed':index,'total':len(rows),'claim_id':item['claim_id'],'status':result['status']})+'\n')
   print(f'{index}/{len(rows)} claim={item["claim_id"]} status={result["status"]}',flush=True)
  if args.freeze:freeze(folder)
if __name__=='__main__':
 try:main()
 except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
