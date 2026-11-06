"""Only shape/field validation; no evidence, URLs, label-preservation or semantics."""
from .common import ROOT
from hero_extension.hero_solution import parse_solution,VERDICTS
import importlib.util
spec=importlib.util.spec_from_file_location('historical_protocol',ROOT/'atomic_facts/common/historical_protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
def validate(text,allow_plain=True):
 try:parsed=parse_solution(text)
 except (ValueError,SyntaxError,TypeError):
  normalized=protocol.normalize_plain_fields(text) if allow_plain else None
  if normalized is None:return {'valid':False,'errors':['Expected exactly justification and verdict string fields'],'parsed':None}
  try:parsed=parse_solution(normalized)
  except (ValueError,SyntaxError,TypeError):return {'valid':False,'errors':['Invalid two-field normalization'],'parsed':None}
 errors=[]
 if not parsed['justification']:errors.append('Empty justification')
 if parsed['verdict'] not in VERDICTS:errors.append('Invalid four-class verdict')
 return {'valid':not errors,'errors':errors,'parsed':parsed,'semantic_validation_performed':False}
