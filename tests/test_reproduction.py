import unittest,tempfile,json,subprocess,sys,ast,importlib,pkgutil
from pathlib import Path
from unittest.mock import patch
from pipeline.common import ROOT,read
from pipeline.build_inputs import load,no_gold
from pipeline.validator import validate
from pipeline.runner import optimize,parser
from pipeline.engine import TraceEngine

class Reproduction(unittest.TestCase):
 def test_data_counts_mapping_sha(self):
  from scripts.check_data import check
  check()
 def test_original_vp_parser_precedence(self):
  from pipeline.baseline import parse_original,SAMPLING,PROMPT
  self.assertEqual(parse_original('Supported, but Not Enough Evidence'),'Not Enough Evidence')
  self.assertEqual(parse_original('Refuted; supported'),'Supported')
  self.assertIsNone(parse_original('unknown'))
  self.assertEqual((SAMPLING['temperature'],SAMPLING['top_p'],SAMPLING['max_tokens']),(0.9,0.7,512))
  self.assertIn('Disregard irrelevant question-answer pairs',PROMPT)
 def test_historical_explicit_parser(self):
  from pipeline.historical_parser import parser_b
  from pipeline.baseline import parse_original
  text='Not Enough Evidence is discussed.\n\nFinal Answer: Refuted'
  self.assertEqual(parser_b(text)[0],'Refuted')
  self.assertEqual(parse_original(text),'Not Enough Evidence')
  for af in ['human63','merged67']:
   for item in load('targeted40',af):self.assertEqual(parser_b(item['llm_output'])[0],item['pred_label'])
 def test_structural_only(self):
  text=json.dumps({'justification':'Invented semantics and https://unseen.example; no relevance check.','verdict':'Refuted'})
  self.assertTrue(validate(text)['valid'])
  self.assertFalse(validate('{"justification":"x","verdict":"unknown"}')['valid'])
  self.assertFalse(validate('{"justification":"x","verdict":"Supported","confidence":1}')['valid'])
  self.assertTrue(validate('Verdict: Refuted\n\nJustification: Only formatting.')['valid'])
 def test_gold_keys_rejected(self):
  for key in ['gold_label','ground_truth','label','is_correct']:
   with self.assertRaises(ValueError):no_gold({'evidence':[{key:'Refuted'}]})
 def run_graph(self,split,af=None):
  item=load(split,af)[0];cfg=read(ROOT/'configs'/((af+'.yaml') if af else 'textgrad.yaml'))
  args=parser().parse_args(['textgrad','--split',split,'--mock']+(['--af',af] if af else []))
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   folder=Path(temp)
   original_open=Path.open
   def guard(path,*a,**kw):
    self.assertNotIn('evaluation_only',str(path),'runtime attempted Gold read')
    return original_open(path,*a,**kw)
   with patch.object(Path,'open',guard):result=optimize(item,folder,args,cfg)
   self.assertEqual(result['status'],'optimized',result)
   self.assertTrue(result['accepted'])
   for name in ['evaluator_feedback.json','textual_gradient.json','optimizer_solution.json','structural_validator.json']:
    self.assertTrue((folder/'iteration_1'/name).exists(),name)
   self.assertEqual(read(folder/'iteration_1/evaluator/request.json')['messages'],[{'role':'system','content':item['evaluator_system']},{'role':'user','content':item['evaluator_user']}])
   feedback=read(folder/'iteration_1/evaluator_feedback.json')['text']
   grad=read(folder/'iteration_1/textual_gradient.json')['text']
   self.assertNotEqual(feedback,grad)
   for stage in ['evaluator','backward','optimizer']:
    self.assertTrue((folder/'iteration_1'/stage/'request.json').exists())
 def test_native_e18_human63(self):self.run_graph('targeted40','human63')
 def test_native_e18_merged67(self):self.run_graph('targeted40','merged67')
 def test_native_full169_faithful(self):self.run_graph('full169')
 def test_fallback_not_success(self):
  item=load('targeted40','human63')[0];cfg=read(ROOT/'configs/human63.yaml');args=parser().parse_args(['textgrad','--af','human63','--mock'])
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   with patch('pipeline.optimizer.step',return_value='bad candidate'):
    result=optimize(item,Path(temp),args,cfg)
   self.assertEqual(result['status'],'baseline_fallback');self.assertFalse(result['accepted']);self.assertEqual(result['llm_output'],item['llm_output'])
 def test_full169_no_change_requires_backward(self):
  item=load('full169')[0];item=dict(item,frozen_status='NO_MATERIAL_ISSUE',frozen_evaluation='STATUS: NO_MATERIAL_ISSUE\nFEEDBACK: Consistent.')
  cfg=read(ROOT/'configs/textgrad.yaml');args=parser().parse_args(['textgrad','--split','full169'])
  orig=TraceEngine.__call__;calls=[]
  def fake(engine,prompt,system_prompt=None,**kwargs):
   calls.append(engine.stage)
   if engine.stage=='backward':return 'NO_CHANGE: The upstream evaluation identifies no material issue; preserve the current solution.'
   return orig(engine,prompt,system_prompt,**kwargs)
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   with patch.object(TraceEngine,'__call__',fake):result=optimize(item,Path(temp),args,cfg)
   self.assertEqual(result['status'],'no_change',result);self.assertEqual(calls,['evaluator','backward']);self.assertFalse(result['accepted'])
 def test_fine146_missing_is_explicit(self):
  with self.assertRaisesRegex(ValueError,'Missing real frozen input'):load('targeted40','fine146')
 def test_all_core_imports(self):
  from pipeline import common
  for package in ['pipeline','textgrad','hero_extension']:
   module=importlib.import_module(package)
   for info in pkgutil.walk_packages(module.__path__,package+'.'):importlib.import_module(info.name)
 def test_frozen_evaluation_guard(self):
  from pipeline.runner import main
  from pipeline.evaluate import evaluate
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   runid=Path(temp).name
   # Temp directory starts empty; runtime writes journal and freezes one claim.
   main(['baseline','--mock','--claim-id','9','--run-id',runid,'--freeze'])
   with self.assertRaisesRegex(ValueError,'Mock outputs'):evaluate(Path(temp))
   metrics=evaluate(Path(temp),True);self.assertEqual(metrics['count'],1)
   p=Path(temp)/'predictions.json';p.write_text('[]')
   with self.assertRaisesRegex(ValueError,'Frozen output'):evaluate(Path(temp),True)
 def test_every_supported_preflight(self):
  from pipeline.runner import main
  for split in ['targeted40','full169','full500']:main(['baseline','--split',split,'--preflight'])
  for af in ['human63','merged67']:main(['textgrad','--af',af,'--preflight'])
  main(['textgrad','--split','full169','--preflight'])
  with self.assertRaisesRegex(ValueError,'no frozen AF'):main(['textgrad','--split','full500','--preflight'])

if __name__=='__main__':unittest.main()
