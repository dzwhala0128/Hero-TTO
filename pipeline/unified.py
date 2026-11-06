"""One advisor protocol: frozen VP -> one compact evaluator + fixed two-shot -> native TextGrad.

This module NEVER calls HerO VP and NEVER generates a new initial prediction.
"""
import argparse
import fcntl
import json
import os
import re
import sys
import traceback
from pathlib import Path

from .common import ROOT, read, save, digest, file_sha, safe_output
from .build_inputs import no_gold
from .qwen_engine import TraceEngine
from .validator import validate

PROMPTS = ROOT / 'prompts'
FEWSHOT = ROOT / 'fewshot/fixed2.jsonl'
COUNTS = (1, 10, 40, 169, 500)


def prompt(name):
    return (PROMPTS/name).read_text(encoding='utf-8').rstrip('\n')


def frozen_solution(record):
    """Parse an existing, frozen baseline prediction; preserve source verbatim separately."""
    from hero_extension.hero_solution import parse_baseline_output, serialize_solution, VERDICTS
    label = record.get('pred_label')
    if label not in VERDICTS:
        raise ValueError('Missing/invalid frozen baseline verdict for claim '+str(record.get('claim_id')))
    raw = record.get('initial_solution') or record.get('llm_output')
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Missing frozen baseline justification for claim '+str(record.get('claim_id')))
    parsed = parse_baseline_output(raw, fallback_verdict=label)
    if parsed['verdict'] != label or not parsed['justification']:
        raise ValueError('Frozen baseline text/verdict inconsistent for claim '+str(record.get('claim_id')))
    # Serialization changes only the wrapper (two fields); no model inference or label change.
    return {'raw':raw,'source_llm_output':record.get('llm_output',raw),
            'solution':serialize_solution(parsed['justification'],label),'verdict':label}


def selected(count,af='auto',claim_id=None):
    if count not in COUNTS:raise ValueError('Count must be 1, 10, 40, 169, 500')
    split='targeted40' if count<=40 else 'full'+str(count)
    if af=='auto':af='human63' if count<=40 else 'original'
    if af not in ('human63','merged67','original'):raise ValueError('AF mode must be human63, merged67, original')
    if count>40 and af!='original':
        raise ValueError('人工63/合并67 AF仅覆盖40条；169/500没有这些AF冻结输入。')

    if count<=40:
        # Even with original 10-QA, reuse the SAME frozen VP from human63/merged67.
        # The original-QA runtime deliberately has no VP: it must never trigger regeneration.
        baseline_path=ROOT/'data/targeted40/human63/runtime.json'
        qa_path=(ROOT/'data/targeted40/original_qa/runtime.json') if af=='original' else ROOT/f'data/targeted40/{af}/runtime.json'
    else:
        baseline_path=ROOT/f'data/{split}/runtime.json'
        qa_path=baseline_path
    baselines=read(baseline_path)
    evidence_rows=read(qa_path)
    no_gold(baselines);no_gold(evidence_rows)
    by_id={r['claim_id']:r for r in baselines}
    if len(by_id)!=len(baselines):raise ValueError('Duplicate baseline claim IDs')
    rows=[]
    for e in evidence_rows:
        b=by_id.get(e['claim_id'])
        if b is None or b['claim']!=e['claim']:
            raise ValueError('Baseline / QA claim alignment failure: '+str(e['claim_id']))
        frozen=frozen_solution(b)
        rows.append({'claim_id':e['claim_id'],'claim':e['claim'],
                     'evidence':e['evidence'],'afs':e.get('afs',[]) if af!='original' else [],
                     'frozen':frozen})
    if len(rows)!=len(baselines):raise ValueError('Baseline and QA set sizes differ')
    if claim_id is not None:
        if count!=1:raise ValueError('--claim-id can only be used with --count 1')
        rows=[r for r in rows if r['claim_id']==claim_id]
    else:rows=rows[:count]
    if len(rows)!=count:raise ValueError('Not enough aligned frozen samples')
    return rows,split,af,{'frozen_vp':str(baseline_path.relative_to(ROOT)),
                          'qa_af':str(qa_path.relative_to(ROOT))}


def render_qa(row):
    return '\n\n'.join(f'QA{i} [batch_QA{i}]\nQ: {q["question"]}\nA: {q["answer"]}'
                      for i,q in enumerate(row['evidence'],1))


def render_af(row):
    return '\n'.join(f'AF{i} [batch_AF{i}]: {a}' for i,a in enumerate(row['afs'],1)) or '(No AF decomposition available; use QA evidence.)'


def fewshot_text():
    from .fewshot import render_demonstration
    demos=[json.loads(s) for s in FEWSHOT.read_text(encoding='utf-8').splitlines() if s.strip()]
    if [d['train_index'] for d in demos]!=[165,6]:raise ValueError('Expected fixed2 train indices [165,6]')
    if [d['evaluation']['status'] for d in demos]!=['MATERIAL_ISSUE','NO_MATERIAL_ISSUE']:
        raise ValueError('Invalid fixed2 statuses')
    for d in demos:no_gold(d)
    return '\n\n'.join(render_demonstration(s,i) for i,s in enumerate(demos,1))


def evaluator_input(row,solution,shots):
    template=prompt('evaluator_user.txt')
    substitutions={'<<FEW_SHOT_EXAMPLES>>':shots,'<<CLAIM>>':row['claim'],
                   '<<AF>>':render_af(row),'<<QA>>':render_qa(row),'<<CURRENT_SOLUTION>>':solution}
    for key,value in substitutions.items():
        if template.count(key)!=1:raise ValueError('Evaluator template placeholder must occur once: '+key)
        template=template.replace(key,value)
    if template.count(solution)!=1:raise ValueError('Expected one trainable CURRENT_SOLUTION occurrence')
    return template


def parse_feedback(raw):
    m=re.fullmatch(r'\s*STATUS:\s*(MATERIAL_ISSUE|NO_MATERIAL_ISSUE)[ \t]*\r?\n[ \t]*FEEDBACK:[ \t]*([^\r\n]+)[ \t]*\s*',raw)
    if m is None:raise ValueError('Evaluator must produce only STATUS and FEEDBACK')
    return {'status':m.group(1),'feedback':m.group(2).strip()}


def stage_settings(config,stage):
    if stage not in ('evaluator','backward','optimizer'):raise ValueError('Initial VP stage is forbidden here')
    value={'endpoint':os.getenv('QWEN_BASE_URL',os.getenv('TEXTGRAD_BASE_URL',config['endpoint'])),
           'model':os.getenv('QWEN_MODEL',os.getenv('TEXTGRAD_MODEL',config['model'])),
           'max_context':config['max_context'],'timeout':240,
           'expected_model_directory':config['expected_model_directory']}
    value.update(config['stages'][stage])
    if value['model']!='qwen3.5-9B':raise ValueError('Qwen optimizer needs qwen3.5-9B; got '+value['model'])
    return value


def run_one(row,folder,config,shots,mock=False):
    import textgrad as tg
    from textgrad.autograd import llm_ops
    from textgrad.autograd.llm_ops import FormattedLLMCall
    from textgrad.optimizer.optimizer import TextualGradientDescent
    from .native_backward import backward

    frozen=row['frozen']
    baseline=frozen['verdict']
    result={'claim_id':row['claim_id'],'claim':row['claim'],'evidence':row['evidence'],
            'afs':row['afs'],'initial_verdict':baseline,'pred_label':baseline,
            'initial_solution':frozen['raw'],'baseline_raw_output':frozen['source_llm_output'],
            'llm_output':frozen['source_llm_output'],
            'status':'baseline_fallback','accepted':False,'iterations':0,'mock':mock,
            'optimization_model':'qwen3.5-9B'}
    # These immutable snapshots are written BEFORE any model call; no initial VP stage exists.
    save(folder/'frozen_initial_vp.json',{'verdict':baseline,'raw':frozen['raw'],
                                          'raw_llm_output':frozen['source_llm_output'],
                                          'normalized_for_textgrad':frozen['solution'],
                                          'model_calls_for_initial_vp':0})
    try:
        solution=tg.Variable(frozen['solution'],requires_grad=True,role_description='CURRENT_SOLUTION')
        rendered=evaluator_input(row,frozen['solution'],shots)
        before,after=rendered.split(frozen['solution'])
        template=before.replace('{','{{').replace('}','}}')+'{CURRENT_SOLUTION}'+after.replace('{','{{').replace('}','}}')
        evaluator=FormattedLLMCall(
            engine=TraceEngine(folder,'evaluator',stage_settings(config,'evaluator'),mock),
            format_string=template,fields={'CURRENT_SOLUTION':None},
            system_prompt=tg.Variable(prompt('evaluator_system.txt'),requires_grad=False,role_description='Evaluator instructions'))
        loss=evaluator({'CURRENT_SOLUTION':solution},response_role_description='evaluation of CURRENT_SOLUTION')
        feedback=parse_feedback(loss.value)
        result.update(evaluator_status=feedback['status'],evaluator_feedback=feedback['feedback'])
        save(folder/'evaluator_feedback.json',feedback)
        # The native graph computes its textual gradient; the system prompt is in prompts/.
        llm_ops.BACKWARD_SYSTEM_PROMPT=prompt('native_backward_system.txt')
        grad=backward(loss,solution,TraceEngine(folder,'backward',stage_settings(config,'backward'),mock),faithful=False)
        save(folder/'textual_gradient.json',grad)
        if feedback['status']=='NO_MATERIAL_ISSUE':
            result['status']='no_change'
            return result
        opt=TextualGradientDescent(
            parameters=[solution],engine=TraceEngine(folder,'optimizer',stage_settings(config,'optimizer'),mock),
            constraints=read(PROMPTS/'optimizer_constraints.json'),
            optimizer_system_prompt=prompt('native_optimizer_system.txt'),
            in_context_examples=[],gradient_memory=0)
        opt.step()
        candidate=solution.value
        save(folder/'optimizer_solution.json',{'solution':candidate})
        valid=validate(candidate,allow_plain=False)
        save(folder/'structural_validator.json',valid)
        if not valid['valid']:raise ValueError('Optimizer solution failed structural validation: '+str(valid['errors']))
        result.update(pred_label=valid['parsed']['verdict'],llm_output=candidate,
                      status='optimized',accepted=True,iterations=1)
    except Exception as exc:
        save(folder/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        result.update(status='baseline_fallback',accepted=False,iterations=0,
                      pred_label=baseline,llm_output=frozen['source_llm_output'],error=str(exc))
    return result


def configured_hashes():
    paths=(list(PROMPTS.rglob('*'))+list((ROOT/'fewshot').rglob('*'))+
           list((ROOT/'pipeline').glob('*.py'))+[ROOT/'configs/qwen9b.json'])
    return {str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths) if p.is_file()}


def parser():
    p=argparse.ArgumentParser(description='在冻结Baseline VP上运行统一9B TextGrad；绝不重新预测VP')
    p.add_argument('--count',type=int,required=True,choices=COUNTS)
    p.add_argument('--af',choices=['auto','original','human63','merged67'],default='auto')
    p.add_argument('--claim-id',type=int)
    p.add_argument('--run-id')
    p.add_argument('--preflight',action='store_true',help='检查冻结VP、QA、AF与文件；零模型调用')
    p.add_argument('--mock',action='store_true',help='仅供离线测试，不能作研究结果')
    p.add_argument('--freeze',action='store_true',help='完成后冻结供离线评估')
    p.add_argument('--resume',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    rows,split,af,paths=selected(args.count,args.af,args.claim_id)
    shots=fewshot_text()
    demos=[json.loads(s) for s in FEWSHOT.read_text(encoding='utf-8').splitlines() if s.strip()]
    norm=lambda value:re.sub(r'\W+','',value.lower())
    if [(d['train_index'],r['claim_id']) for d in demos for r in rows if norm(d['claim'])==norm(r['claim'])]:
        raise ValueError('Few-shot training example matches a target claim')
    cfg=read(ROOT/'configs/qwen9b.json')
    model=stage_settings(cfg,'evaluator')
    if args.preflight:
        print(json.dumps({'valid':True,'count':len(rows),'first_claim_ids':[r['claim_id'] for r in rows[:10]],'last_claim_id':rows[-1]['claim_id'],
                          'frozen_vp_source':paths['frozen_vp'],'qa_af_source':paths['qa_af'],
                          'af':af,'all_frozen_vp_present':True,
                          'qa_per_claim_min':min(len(r['evidence']) for r in rows),
                          'qa_per_claim_max':max(len(r['evidence']) for r in rows),
                          'optimizer_model':model['model'],
                          'evaluator_prompt':'prompts/evaluator_system.txt',
                          'fewshot':'fewshot/fixed2.jsonl','fewshot_count':2,
                          'initial_vp_model_calls':0,'model_calls':0},ensure_ascii=False,indent=2))
        return 0
    run_id=args.run_id or 'textgrad_'+str(args.count)+'_'+af
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ('.','..'):
        raise ValueError('Invalid run-id')
    folder=safe_output('outputs/'+run_id)
    run={'task':'textgrad_frozen_baseline','split':split,'af':af,
         'claim_ids':[r['claim_id'] for r in rows], 'mock':args.mock,
         'optimizer_model':model,'sources':paths,
         'input_sha256':digest(rows),'code_and_prompts_sha256':configured_hashes(),
         'fixed2_train_indices':[165,6], 'initial_vp_model_calls':0}
    if (folder/'freeze.json').exists():raise ValueError('Run already frozen; use another --run-id')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'run.json').exists():
            if not args.resume:raise ValueError('Run exists; use --resume or a new run-id')
            if read(folder/'run.json')!=run:raise ValueError('Resume changed input/model/prompt/source')
        else:save(folder/'run.json',run)
        # Fail fast if the Qwen service is absent/wrong. Do not freeze 500 silent fallback rows.
        if not args.mock:
            TraceEngine(folder,'service_check',stage_settings(cfg,'evaluator')).verify_model()
        for i,row in enumerate(rows,1):
            dest=folder/'claims'/str(row['claim_id'])
            if (dest/'result.json').exists():continue
            result=run_one(row,dest,cfg,shots,args.mock)
            save(dest/'result.json',result)
            with (folder/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'completed':i,'total':len(rows),'claim_id':row['claim_id'],'status':result['status']})+'\n')
            print(f'{i}/{len(rows)} claim={row["claim_id"]} status={result["status"]}',flush=True)
        if args.freeze:
            from .runner import freeze
            freeze(folder)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(type(e).__name__+': '+str(e),file=sys.stderr)
        sys.exit(1)
