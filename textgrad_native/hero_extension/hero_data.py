"""Read HerO baseline predictions without consulting evaluation annotations."""

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .hero_solution import parse_baseline_output


def load_baseline(path: str) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("baseline file must contain a JSON list")
    required = {"claim_id", "claim", "evidence", "llm_output", "pred_label"}
    cleaned = []
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("baseline row is missing a required runtime field")
        solution = parse_baseline_output(row["llm_output"], row["pred_label"])
        cleaned.append({
            "claim_id": row["claim_id"],
            "claim": row["claim"],
            "evidence": row["evidence"],
            "baseline_solution": solution,
        })
    return cleaned


def select_samples(
    rows: Iterable[dict], start: int = 0, end: Optional[int] = None,
    claim_ids: Optional[Iterable[int]] = None,
) -> List[dict]:
    selected = list(rows)[start:end]
    if claim_ids is None:
        return selected
    wanted = set(claim_ids)
    return [row for row in selected if row["claim_id"] in wanted]


def evidence_text(evidence: list) -> str:
    blocks = []
    for index, item in enumerate(evidence, 1):
        blocks.append(
            f"QA {index}\nQuestion: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
        )
    return "\n\n".join(blocks)
