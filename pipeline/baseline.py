"""Original HerO VP prompt/decoding and original precedence parser, HTTP transport."""
import ast,json
from .common import ROOT,save
from .engine import TraceEngine
# Extract original literals without importing GPU dependencies.
tree=ast.parse((ROOT/'hero/stages/veracity_prediction.py').read_text())
PROMPT=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in n.targets))
SAMPLING=dict(temperature=0.9,top_p=0.7,top_k=1,skip_special_tokens=False,max_tokens=512,stop=['<|endoftext|>','</s>','<|im_end|>','[INST]','[/INST]','<|eot_id|>','<|end|>'])
def parse_original(output):
 if 'Not Enough Evidence' in output:return 'Not Enough Evidence'
 if 'Conflicting Evidence/Cherrypicking' in output or 'Cherrypicking' in output or 'Conflicting Evidence' in output:return 'Conflicting Evidence/Cherrypicking'
 if 'Supported' in output or 'supported' in output:return 'Supported'
 if 'Refuted' in output or 'refuted' in output:return 'Refuted'
 return None
class Baseline:
 def __init__(self,settings,mock=False,max_attempts=10):self.settings=settings;self.mock=mock;self.max_attempts=max_attempts;self.top_k=1
 def run(self,item,folder):
  prompt=PROMPT+'\n\nClaim: '+item['claim']+'\n\n'+'\n\n'.join(f"Q{i+1}: {q['question']}\nA{i+1}: {q['answer']}" for i,q in enumerate(item['evidence']))
  for attempt in range(self.max_attempts):
   eng=TraceEngine(folder,f'vp_attempt_{attempt+1}',self.settings,self.mock)
   if self.mock:raw='Mock only. Supported'
   else:
    eng.verify_model()
    # Original apply_chat_template default has no added assistant generation header.
    orig=eng.settings['endpoint'];eng.settings=dict(eng.settings,endpoint=orig.removesuffix('/v1'))
    try:tokens=eng.http('/tokenize',dict(model=eng.model_string,messages=[dict(role='user',content=prompt)],add_generation_prompt=False))
    finally:eng.settings['endpoint']=orig
    if len(tokens['tokens'])+SAMPLING['max_tokens']>self.settings.get('max_context',4096):raise ValueError('Original HerO VP 4096-token context exceeded; no truncation')
    req=dict(model=eng.model_string,prompt=tokens['tokens'],**dict(SAMPLING,top_k=self.top_k))
    from .common import read,digest
    request_key=digest(req)
    if (eng.folder/'response.json').exists():
     cached=read(eng.folder/'response.json')
     if cached['request_sha256']!=request_key:raise ValueError('Baseline resume request changed')
     response=cached['raw']
    else:
     if (eng.folder/'pending.json').exists():raise RuntimeError('Unresolved VP request; no automatic resend')
     save(eng.folder/'request.json',req);save(eng.folder/'pending.json',{'request_sha256':request_key})
     response=eng.http('/completions',req);save(eng.folder/'response.json',{'request_sha256':request_key,'raw':response})
    raw=response['choices'][0]['text'].strip()
   label=parse_original(raw);save(eng.folder/'parsed.json',dict(raw=raw,pred_label=label,top_k=self.top_k,prompt=prompt))
   if label:return dict(claim_id=item['claim_id'],claim=item['claim'],evidence=item['evidence'],pred_label=label,llm_output=raw,top_k=self.top_k,status='baseline',mock=self.mock)
   self.top_k=2
  raise RuntimeError('No original HerO label found; operational attempt bound reached')
