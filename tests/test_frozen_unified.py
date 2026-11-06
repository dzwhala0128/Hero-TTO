"""Advisor acceptance: zero new VP, frozen baselines, external prompts and native graph."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline.common import ROOT,read
from pipeline.unified import selected,run_one,fewshot_text,evaluator_input,stage_settings,configured_hashes

class FrozenBaselineAdvisor(unittest.TestCase):
    def test_frozen_inputs_all_scales(self):
        for n in (1,10,40,169,500):
            rows,split,af,paths=selected(n)
            self.assertEqual(len(rows),n)
            self.assertEqual(len({r['claim_id'] for r in rows}),n)
            self.assertTrue(paths['frozen_vp'].startswith('data/'))
            self.assertTrue(all(r['frozen']['verdict'] for r in rows))
            self.assertTrue(all('"verdict"' in r['frozen']['solution'] for r in rows))
        for af,total in [('human63',63),('merged67',67)]:
            rows,_,_,_=selected(40,af)
            self.assertEqual(sum(len(r['afs']) for r in rows),total)
            self.assertEqual(sum(len(r['evidence']) for r in rows),154)
        rows,_,_,_=selected(40,'original')
        self.assertEqual(sum(len(r['evidence']) for r in rows),400)
        self.assertTrue(all(not r['afs'] for r in rows))
        self.assertEqual([r['frozen'] for r in rows],[r['frozen'] for r in selected(40,'human63')[0]])
        with self.assertRaises(ValueError):selected(169,'human63')
    def test_prompts_shots_and_no_initial_vp_stage(self):
        row=selected(1)[0][0];shots=fewshot_text();self.assertEqual(shots.count('### DEMONSTRATION '),2)
        req=evaluator_input(row,row['frozen']['solution'],shots)
        self.assertIn('AF1 [batch_AF1]',req)
        self.assertIn('QA1 [batch_QA1]',req)
        c=read(ROOT/'configs/qwen9b.json')
        self.assertNotIn('initial_vp',c['stages'])
        with self.assertRaisesRegex(ValueError,'forbidden'):
            stage_settings(c,'initial_vp')
        self.assertIn('fewshot/fixed2.jsonl',configured_hashes())
        self.assertIn('prompts/evaluator_system.txt',configured_hashes())
    def test_native_mock_zero_vp_calls(self):
        row=selected(1)[0][0]
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            folder=Path(tmp)
            original_open=Path.open
            def guarded_open(path,*a,**kwargs):
                self.assertNotIn('/evaluation_only/',str(path))
                return original_open(path,*a,**kwargs)
            with patch.object(Path,'open',guarded_open):
                result=run_one(row,folder,read(ROOT/'configs/qwen9b.json'),fewshot_text(),mock=True)
            self.assertEqual(result['status'],'optimized',result)
            self.assertEqual(result['initial_verdict'],row['frozen']['verdict'])
            self.assertTrue((folder/'frozen_initial_vp.json').exists())
            self.assertEqual(read(folder/'frozen_initial_vp.json')['model_calls_for_initial_vp'],0)
            for stage in ['evaluator','backward','optimizer']:
                self.assertTrue((folder/stage/'request.json').exists(),stage)
            self.assertFalse((folder/'initial_vp').exists())
    def test_original_hero_sources_untouched(self):
        import hashlib
        original=Path('/mnt/data/HerO_TextGrad_Advisor_Clean/hero')
        target=ROOT/'hero'
        if original.exists():
            for p in original.rglob('*.py'):
                q=target/p.relative_to(original)
                self.assertEqual(hashlib.sha256(p.read_bytes()).digest(),hashlib.sha256(q.read_bytes()).digest())

if __name__=='__main__':unittest.main()
