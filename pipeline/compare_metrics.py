"""Offline comparison of the frozen initial VP vs TextGrad VP. Gold is opened only after freeze checks."""
import argparse
import json
from .common import ROOT,read,save,file_sha,safe_output
from hero_extension.hero_solution import VERDICTS


def score(rows,gold,key):
    matrix={a:{b:0 for b in VERDICTS} for a in VERDICTS}
    for row in rows:
        if row[key] not in VERDICTS:raise ValueError('Invalid four-class prediction')
        matrix[gold[row['claim_id']]['label']][row[key]]+=1
    f1={}
    for lab in VERDICTS:
        tp=matrix[lab][lab]
        fp=sum(matrix[a][lab] for a in VERDICTS if a!=lab)
        fn=sum(matrix[lab][b] for b in VERDICTS if b!=lab)
        f1[lab]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return {'accuracy':sum(matrix[x][x] for x in VERDICTS)/len(rows),
            'macro_f1':sum(f1.values())/4,'per_class_f1':f1,'confusion_matrix':matrix}


def compare(folder,allow_mock=False):
    frozen=read(folder/'freeze.json');rows=read(folder/'predictions.json')
    if file_sha(folder/'predictions.json')!=frozen['predictions_sha256'] or file_sha(folder/'run.json')!=frozen['run_sha256']:
        raise ValueError('Frozen predictions/run changed')
    for cid,sha in frozen['result_sha256'].items():
        if file_sha(folder/'claims'/cid/'result.json')!=sha:raise ValueError('Frozen result changed: '+cid)
    if [r['claim_id'] for r in rows]!=frozen['claim_ids'] or len(rows)!=len(set(frozen['claim_ids'])):
        raise ValueError('Incomplete/duplicated claim list')
    if frozen['mock'] and not allow_mock:raise ValueError('Mock outputs cannot be used as experimental results')
    gold=read(ROOT/'data/evaluation_only/dev.json')
    for r in rows:
        if r['claim']!=gold[r['claim_id']]['claim']:raise ValueError('Claim mismatch')
    a=score(rows,gold,'initial_verdict')
    b=score(rows,gold,'pred_label')
    out={'count':len(rows),'initial_vp_model_calls':0,'baseline':a,'after_textgrad':b,
         'delta_accuracy':b['accuracy']-a['accuracy'],'delta_macro_f1':b['macro_f1']-a['macro_f1'],
         'changed_labels':sum(r['initial_verdict']!=r['pred_label'] for r in rows),
         'optimized_count':sum(r['status']=='optimized' for r in rows),
         'fallback_count':sum(r['status']=='baseline_fallback' for r in rows),
         'mock':frozen['mock']}
    save(folder/'baseline_vs_textgrad_metrics.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2));return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--allow-mock',action='store_true')
    a=p.parse_args();compare(safe_output('outputs/'+a.run_id),a.allow_mock)

if __name__=='__main__':main()
