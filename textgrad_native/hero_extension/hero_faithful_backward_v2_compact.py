"""
Hero faithful backward v2.

Difference from faithful_backward_v1:
- Preserve upstream diagnostic semantics.
- Do not propagate or recommend fact-checking verdict labels.
- Do not modify native TextGrad vendor code.
- Do not use gold.
- No claim-specific rules.
- No programmatic gate.
"""

from textgrad.autograd.llm_ops import FormattedLLMCall
from textgrad.autograd.llm_backward_prompts import CONVERSATION_TEMPLATE
from textgrad.loss import MultiFieldEvaluation
from textgrad.variable import Variable

FAITHFUL_BACKWARD_VARIANT = "faithful_backward_v2_compact"


VERDICT_NEUTRAL_BACKWARD_SYSTEM_PROMPT = """
You are the textual-gradient backward stage of an optimization system.

The upstream evaluator has already evaluated the current solution.

Your task is NOT to independently evaluate the claim or evidence again.
Your task is to faithfully convert the upstream diagnostic into actionable
feedback about the reasoning and evidence alignment of CURRENT_SOLUTION.

The upstream evaluation is authoritative for this backward pass.

GENERAL RULES

1. Do not independently re-evaluate the claim, evidence, or task.

2. Do not contradict or override the upstream evaluation.

3. Do not introduce a new factual problem, discrepancy, evidence relation,
   or external fact that is absent from the upstream diagnostic.

4. If the upstream evaluation says NO_MATERIAL_ISSUE, return exactly:

NO_CHANGE: The upstream evaluation identifies no material issue; preserve the current solution.

Do not add anything else.

5. If the upstream evaluation says MATERIAL_ISSUE, begin exactly with:

CHANGE_NEEDED:

Then translate only the evidence-grounded reasoning problem identified by
the upstream evaluator into concise revision guidance.

VERDICT-NEUTRALITY RULES

The textual gradient is diagnostic feedback, not a classification decision.

- Do not recommend a replacement verdict.
- Do not tell the optimizer which fact-checking class to choose.
- Do not say that a particular verdict is correct, appropriate, preferable,
  required, or should replace the current verdict.
- Do not say "change the verdict", "the verdict should be", or equivalent.
- Do not repeat a replacement-verdict recommendation even if the upstream
  evaluator itself contains one.
- If the upstream evaluator mentions a replacement verdict, discard only
  that label recommendation while preserving the underlying diagnostic
  reason.

For example, if the upstream evaluation says that the evidence does not
establish a proposition and then recommends a particular class, preserve
only the diagnostic idea that the current solution overstates what the
evidence establishes.

If the upstream evaluation says that evidence materially contradicts or
supports a proposition, describe that evidence relation without prescribing
a final class.

Focus on:
- what reasoning or evidence-alignment problem exists;
- which supplied evidence establishes that problem;
- what aspect of the reasoning must be reconsidered.

Do not:
- write a revised solution;
- choose a verdict;
- recommend a verdict;
- use external knowledge.

You are a verdict-neutral gradient transformer, not a second evaluator and
not the optimizer.
""".strip()



def _construct_compact_faithful_base_prompt(
    *,
    response_value: str,
    variable_desc: str,
    variable_short: str,
) -> str:
    """
    Compact terminal/base backward transport.

    The upstream evaluator has already consumed CLAIM, QA_EVIDENCE,
    CURRENT_SOLUTION, and any evaluation demonstrations.

    Faithful Backward V2 is not a second evaluator. Therefore the
    backward model only receives:
      - the authoritative upstream evaluation,
      - the variable role,
      - the current variable.

    The original evaluation conversation is intentionally not replayed
    into the backward-model request.
    """

    return f"""
The upstream evaluator has already evaluated CURRENT_SOLUTION.

<UPSTREAM_EVALUATION>
{response_value}
</UPSTREAM_EVALUATION>

Generate a faithful textual gradient for:

<ROLE>
{variable_desc}
</ROLE>

<CURRENT_SOLUTION>
{variable_short}
</CURRENT_SOLUTION>

Use only the diagnostic contained in UPSTREAM_EVALUATION.

Do not independently reassess the claim or QA evidence.
Do not introduce a new factual issue.
Do not choose or recommend a replacement verdict.

If STATUS is NO_MATERIAL_ISSUE, follow the required NO_CHANGE
output exactly.

If STATUS is MATERIAL_ISSUE, translate only the stated diagnostic
into concise revision guidance beginning with CHANGE_NEEDED:.
""".strip()


class FaithfulFormattedLLMCallV2(FormattedLLMCall):

    def backward(
        self,
        response: Variable,
        prompt: str,
        system_prompt: str,
        backward_engine,
    ):
        # Preserve native TextGrad chain backward if this operation ever
        # receives a downstream gradient. The current Hero use is terminal/base.
        if response.get_gradient_text() != "":
            return super().backward(
                response=response,
                prompt=prompt,
                system_prompt=system_prompt,
                backward_engine=backward_engine,
            )

        for variable in response.predecessors:
            if not variable.requires_grad:
                continue

            # Compact faithful transport:
            # do not replay the full evaluation conversation into
            # the backward-model request.
            backward_prompt = _construct_compact_faithful_base_prompt(
                response_value=response.get_value(),
                variable_desc=variable.get_role_description(),
                variable_short=variable.get_short_value(),
            )

            gradient_value = backward_engine(
                backward_prompt,
                system_prompt=VERDICT_NEUTRAL_BACKWARD_SYSTEM_PROMPT,
            )

            var_gradient = Variable(
                value=gradient_value,
                role_description=(
                    f"verdict-neutral faithful feedback to "
                    f"{variable.get_role_description()}"
                ),
            )

            variable.gradients.add(var_gradient)

            native_context = CONVERSATION_TEMPLATE.format(
                response_value=response.get_value(),
                prompt=prompt,
                system_prompt=system_prompt,
            )

            variable.gradients_context[var_gradient] = {
                "context": native_context,
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description(),
            }

            if response._reduce_meta:
                var_gradient._reduce_meta.extend(
                    response._reduce_meta
                )
                variable._reduce_meta.extend(
                    response._reduce_meta
                )


class HeroFaithfulMultiFieldEvaluationV2(MultiFieldEvaluation):

    def __init__(
        self,
        evaluation_instruction,
        role_descriptions,
        engine=None,
        system_prompt=None,
    ):
        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=role_descriptions,
            engine=engine,
            system_prompt=system_prompt,
        )

        self.formatted_llm_call = FaithfulFormattedLLMCallV2(
            engine=self.engine,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt,
        )
