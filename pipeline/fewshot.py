"""Unchanged Fixed-6 renderer inherited from Fixed-5; extracted AST only."""
import json

def render_demonstration(row, number):
    blocks = []
    for i, qa in enumerate(row['qa_pairs'], 1):
        lines = [f'QA {i}', f"Question: {qa['question']}"]
        for j, answer in enumerate(qa['answers'], 1):
            lines += [f"Answer {j}: {answer['answer']}", f"Answer type: {answer['answer_type']}"]
            if 'boolean_explanation' in answer:
                lines.append(f"Boolean explanation: {answer['boolean_explanation']}")
        blocks.append('\n'.join(lines))
    evidence = '\n\n'.join(blocks)
    solution = json.dumps(row['current_solution'], ensure_ascii=False, indent=2)
    return (f"### DEMONSTRATION {number}\n\nCLAIM:\n{row['claim']}\n\n"
            f"QA EVIDENCE:\n{evidence}\n\nCURRENT SOLUTION:\n{solution}\n\n"
            f"EVALUATION:\nSTATUS: {row['evaluation']['status']}\n"
            f"FEEDBACK: {row['evaluation']['feedback']}")
