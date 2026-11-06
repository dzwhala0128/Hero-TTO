"""Durable request journal, explicit stage settings, no automatic model retries."""
import json,os,urllib.request,time
from .common import save,read,digest
class TraceEngine:
 def __init__(self,folder,stage,settings,mock=False,response=None):
  self.folder=folder/stage;self.stage=stage;self.settings=settings;self.mock=mock;self.response=response
  self.model_string=settings.get('model','mock');self.calls=[]
 def http(self,path,payload=None):
  url=self.settings['endpoint'].rstrip('/')+path
  req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json','Authorization':'Bearer '+os.getenv('MODEL_API_KEY','local-placeholder')})
  with urllib.request.urlopen(req,timeout=self.settings.get('timeout',300)) as r:return json.load(r)
 def verify_model(self):
  models=self.http('/models')
  if self.model_string not in [m['id'] for m in models['data']]:raise ValueError('Configured alias absent from /v1/models: '+self.model_string)
  save(self.folder/'models.json',models)
 def __call__(self,prompt,system_prompt=None,**kwargs):
  messages=([dict(role='system',content=system_prompt)] if system_prompt else [])+[dict(role='user',content=prompt)]
  req=dict(model=self.model_string,messages=messages,temperature=self.settings.get('temperature',0),max_tokens=self.settings.get('max_tokens',1024),seed=12345,chat_template_kwargs={'enable_thinking':False})
  key=digest(req);cached=self.folder/'response.json'
  if cached.exists():
   record=read(cached)
   if record['request_sha256']!=key:raise ValueError('Cached request changed; use a new run ID')
   self.calls.append(record);return record['text']
  if (self.folder/'pending.json').exists():raise RuntimeError('Unresolved pending request; inspect trace and use a new run ID (no silent resend)')
  save(self.folder/'request.json',req)
  if self.response is not None: text=self.response;raw={'replayed_frozen_evaluation':True}
  elif self.mock:
   text={'evaluator':'STATUS: MATERIAL_ISSUE\nFEEDBACK: Check evidence alignment.', 'backward':'CHANGE_NEEDED: Reconsider the justification against supplied QA.', 'optimizer':'<IMPROVED_VARIABLE>{"justification":"Mock graph test only.","verdict":"Supported"}</IMPROVED_VARIABLE>'}[self.stage];raw={'simulation':True}
  else:
   self.verify_model()
   token_req=dict(model=self.model_string,messages=messages,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False})
   # /tokenize belongs to the server root, not /v1.
   original=self.settings['endpoint'];self.settings=dict(self.settings,endpoint=original.removesuffix('/v1'))
   try:audit=self.http('/tokenize',token_req)
   finally:self.settings['endpoint']=original
   save(self.folder/'token_audit.json',audit)
   if audit['count']+req['max_tokens']+128>self.settings.get('max_context',8192):raise ValueError('Token budget exceeded; no truncation allowed')
   save(self.folder/'pending.json',{'request_sha256':key,'started':time.time()})
   try:raw=self.http('/chat/completions',req)
   except Exception as e:
    save(self.folder/'error.json',{'type':type(e).__name__,'error':str(e)});raise
   save(self.folder/'raw_response.json',raw)
   if raw['choices'][0].get('finish_reason')!='stop':raise RuntimeError('Truncated/non-stop response; baseline fallback')
   text=raw['choices'][0]['message']['content']
  record=dict(request_sha256=key,text=text,raw_response=raw,stage=self.stage,model_call=not self.mock and self.response is None)
  save(cached,record);self.calls.append(record);return text
 generate=__call__
