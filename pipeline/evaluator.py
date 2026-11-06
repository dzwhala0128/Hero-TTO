"""Frozen historical request template, one trainable CURRENT_SOLUTION field."""
from .common import ROOT
from textgrad.autograd.llm_ops import FormattedLLMCall
from hero_extension.hero_faithful_backward_v2_compact import FaithfulFormattedLLMCallV2
import textgrad as tg
def call_for(item,solution,engine,faithful=False):
 original=item['initial_solution'];user=item['evaluator_user']
 if user.count(original)!=1:raise ValueError('Expected exactly one complete frozen solution in Evaluator input')
 prefix,suffix=user.split(original)
 template=prefix.replace('{','{{').replace('}','}}')+'{CURRENT_SOLUTION}'+suffix.replace('{','{{').replace('}','}}')
 cls=FaithfulFormattedLLMCallV2 if faithful else FormattedLLMCall
 call=cls(engine=engine,format_string=template,fields={'CURRENT_SOLUTION':None},system_prompt=tg.Variable(item['evaluator_system'],requires_grad=False,role_description='frozen Evaluator system prompt'))
 return call({'CURRENT_SOLUTION':solution},response_role_description='evaluation of the a prediction' if faithful else 'evaluation of CURRENT_SOLUTION')
