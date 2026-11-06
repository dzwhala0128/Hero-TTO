import json

import re

MAX_TEXT = 180_000

def make_evaluator_input(item,config):
    afs='\n'.join(f'AF{i} [batch_AF{i}]: {a}' for i,a in enumerate(item['afs'],1))
    qas='\n\n'.join(f'{q["label"]} [batch_{q["label"]}]\nQ: {q["question"]}\nA: {q["answer"]}' for i,q in enumerate(item['qas'],1))
    vars={'claim':item['claim'],'selected_afs':afs,'selected_qas':qas,'initial_solution':item['baseline_solution'],
          'af':'','af_label':'','extra_inputs':''}
    vars['inputs']='\n\n'.join(x for x in [
        '【Claim】\n'+vars['claim'], '【AF】\n'+afs if afs else '', '【QA】\n'+qas,
        '【初始Solution】\n'+vars['initial_solution']] if x)
    if config['promptMode']=='auto':
        task=('任务：依据上述输入执行证据分析。' if config.get('template')=='custom' else
              ('任务：'+config['userTemplate'].split('任务：')[-1] if '任务：' in config['userTemplate'] else ''))
        out=vars['inputs']+'\n\n'+task
    else:
        templ=config['userTemplate']
        out=re.sub(r'{{\s*([a-z_]+)\s*}}',lambda m:vars.get(m.group(1),m.group(0)),templ)
        if re.search(r'{{[^{}]+}}',out):raise ValueError('Evaluator Prompt有未知变量')
    if out.count(item['baseline_solution'])!=1:
        raise ValueError('Evaluator输入必须包含且仅包含一次完整初始Solution；请检查Prompt模板')
    if len(out)>MAX_TEXT:raise ValueError('Evaluator请求超过允许的文本上限')
    return out

VERDICTS = frozenset(('Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking'))

_HEADER = re.compile(r'^(Verdict|Justification):[ \t]*(.*)$', re.I)

def normalize_plain_fields(raw: str) -> str | None:
    """Convert only exactly two explicit text fields into a JSON serialization.

    A different order is accepted, but any extra heading, missing field,
    unknown class or ambiguous field occurrence fails closed. No rewriting.
    """
    if not isinstance(raw, str) or len(raw) > 50000 or '\x00' in raw:
        return None
    lines = raw.strip().splitlines()
    if not lines:
        return None
    hits = []
    for idx, line in enumerate(lines):
        match = _HEADER.fullmatch(line)
        if match:
            hits.append((idx, match.group(1).lower(), match.group(2)))
    if len(hits) != 2 or {x[1] for x in hits} != {'verdict', 'justification'}:
        return None
    first, second = hits
    if first[0] != 0 or first[1] == second[1]:
        return None
    fields = {}
    for item, end in ((first, second[0]), (second, len(lines))):
        idx, key, inline = item
        content = '\n'.join([inline] + lines[idx+1:end]).strip()
        if not content or re.search(r'(?m)^\s*(?:Verdict|Justification|Confidence|Metadata)\s*:', content, re.I):
            return None
        fields[key] = content
    if fields['verdict'] not in VERDICTS or len(fields['justification']) > 40000:
        return None
    # Format conversion only: no evidence edits, verdict inference or LLM call.
    return json.dumps({'justification': fields['justification'], 'verdict': fields['verdict']}, ensure_ascii=False)
