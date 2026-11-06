"""Offline only. Gold is opened only after verifying frozen predictions."""
import argparse,json
from .common import ROOT,read,file_sha,save,safe_output
from hero_extension.hero_solution import VERDICTS

def evaluate(folder,allow_mock=False,official=False):
 freeze=read(folder/'freeze.json');pred=folder/'predictions.json'
 if file_sha(pred)!=freeze['predictions_sha256'] or file_sha(folder/'run.json')!=freeze['run_sha256']:raise ValueError('Frozen output/config changed')
 for cid,sha in freeze['result_sha256'].items():
  if file_sha(folder/'claims'/cid/'result.json')!=sha:raise ValueError('Frozen trajectory changed')
 if freeze['mock'] and not allow_mock:raise ValueError('Mock outputs cannot be reported as research metrics; use --allow-mock only for testing')
 rows=read(pred)
 if [r['claim_id'] for r in rows]!=freeze['claim_ids'] or len(set(freeze['claim_ids']))!=len(rows):raise ValueError('Frozen ID coverage mismatch')
 gold=read(ROOT/'data/evaluation_only/dev.json')
 for r in rows:
  if r['claim']!=gold[r['claim_id']]['claim'] or r['pred_label'] not in VERDICTS:raise ValueError('Claim/label mismatch')
 matrix={a:{b:0 for b in VERDICTS} for a in VERDICTS}
 for r in rows:matrix[gold[r['claim_id']]['label']][r['pred_label']]+=1
 f1={}
 for lab in VERDICTS:
  tp=matrix[lab][lab];fp=sum(matrix[a][lab] for a in VERDICTS if a!=lab);fn=sum(matrix[lab][b] for b in VERDICTS if b!=lab)
  f1[lab]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
 out=dict(count=len(rows),accuracy=sum(matrix[a][a] for a in VERDICTS)/len(rows),macro_f1=sum(f1.values())/4,f1=f1,confusion_matrix=matrix,mock=freeze['mock'],predictions_sha256=freeze['predictions_sha256'])
 if official:
  from hero.stages.averitec_evaluate import AVeriTeCEvaluator
  ev=AVeriTeCEvaluator();out['official_averitec_score']=ev.evaluate_averitec_score(rows,[gold[r['claim_id']] for r in rows]).tolist()
 save(folder/'offline_metrics.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--allow-mock',action='store_true');p.add_argument('--official',action='store_true');a=p.parse_args();evaluate(safe_output('outputs/'+a.run_id),a.allow_mock,a.official)
if __name__=='__main__':main()
