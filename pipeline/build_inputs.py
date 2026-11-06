"""Strict runtime whitelist. Never opens evaluation_only/."""
from .common import ROOT,read
FORBIDDEN={'gold_label','ground_truth','correct_label','is_correct','gold','label','baseline_correct','target_label'}
ALLOWED={'claim_id','claim','evidence','pred_label','llm_output','afs','initial_solution','evaluator_system','evaluator_user','frozen_evaluation','frozen_evaluation_raw','frozen_status','qa_evidence_text'}
def no_gold(value):
 if isinstance(value,dict):
  if FORBIDDEN & value.keys():raise ValueError('Forbidden runtime annotation keys: '+str(FORBIDDEN & value.keys()))
  for v in value.values():no_gold(v)
 elif isinstance(value,list):
  for v in value:no_gold(v)
def load(split,af=None,task='textgrad'):
 if split=='targeted40':
  if task=='baseline':path=ROOT/'data/targeted40/original_qa/runtime.json'
  elif af in ('human63','merged67','fine146'):path=ROOT/'data/targeted40'/af/'runtime.json'
  else:raise ValueError('Targeted40 TextGrad requires --af human63|merged67|fine146')
 elif af is not None:raise ValueError('No frozen AF mapping for '+split+'; do not reuse Targeted40 AF')
 else:path=ROOT/'data'/split/'runtime.json'
 if not path.exists():raise ValueError('Missing real frozen input: '+str(path.relative_to(ROOT))+'; see docs/EXPERIMENT_MATRIX.md')
 rows=read(path)
 for r in rows:
  if set(r)-ALLOWED:raise ValueError('Unknown runtime fields')
  no_gold(r)
 return rows
