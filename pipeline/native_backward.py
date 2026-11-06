"""Use the native graph's backward; do not replace with a manual model call."""
def backward(loss,solution,engine,faithful=False):
 from textgrad.autograd.llm_ops import LLMCall
 if not faithful and loss.grad_fn.backward_fn.__func__ is not LLMCall.backward:raise RuntimeError('Native LLMCall.backward binding changed')
 before=solution.value
 loss.backward(engine=engine)
 if solution.value!=before or len(solution.gradients)!=1:raise RuntimeError('Invalid native gradient inventory')
 return {'text':solution.get_gradient_text(),'gradients':[{'value':g.value,'context':solution.gradients_context[g]} for g in solution.gradients]}
