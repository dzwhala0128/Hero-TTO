"""Historical explicit final-label parser; comparison only, not original HerO VP."""
import re
LABELS=['Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking']

def parser_b(text):
    # Uniformly applied to all40 and the70B comparison; never searches reasoning for labels.
    clean=text.replace('**','')
    labels='|'.join(re.escape(x) for x in LABELS)
    assignments=re.compile(r'(?im)^[ \t]*((?:revised\s+)?(?:final\s+)?(?:label|verdict|answer))\s*[:：]\s*[\'\"]?('+labels+r')[\'\"]?[.!]?[ \t]*$')
    headings=re.compile(r'(?im)^#{1,6}\s*((?:revised\s+)?final\s+(?:label|verdict|answer|conclusion)|label)\s*\n\s*('+labels+r')[.!]?[ \t]*$')
    candidates=[{'offset':m.start(),'label':m.group(2),'marker':m.group(1),'evidence':m.group(0)} for pattern in (assignments,headings) for m in pattern.finditer(clean)]
    candidates.sort(key=lambda x:x['offset'])
    if not candidates:return None,candidates,'no explicit final assignment'
    final=candidates[-1]
    earlier={c['label'] for c in candidates[:-1]}
    # Multiple distinct final assignments require an explicit revised/final declaration.
    if earlier-{final['label']} and not any(s in final['marker'].lower() for s in ('revised','final')):
        return None,candidates,'conflicting final assignments without explicit revision'
    return final['label'],candidates,None
