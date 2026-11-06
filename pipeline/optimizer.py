from .common import save
import textgrad as tg
def step(solution,engine,constraints):
 optimizer=tg.TextualGradientDescent(parameters=[solution],engine=engine,constraints=constraints,in_context_examples=[],gradient_memory=0)
 optimizer.step()
 return solution.value
