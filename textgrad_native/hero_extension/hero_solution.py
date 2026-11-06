"""Serialization and parsing for the single-instance VP solution variable."""

import ast
import json
import re
from typing import Any, Dict

VERDICTS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


def serialize_solution(justification: str, verdict: str) -> str:
    return json.dumps(
        {"justification": str(justification).strip(), "verdict": str(verdict).strip()},
        ensure_ascii=False,
        indent=2,
    )


def _mapping_from_text(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        value = ast.literal_eval(candidate)
    if not isinstance(value, dict):
        raise ValueError("solution must be a JSON object")
    return value


def parse_solution(text: str) -> Dict[str, str]:
    value = _mapping_from_text(text)
    if set(value) != {"justification", "verdict"}:
        raise ValueError("solution must contain exactly justification and verdict")
    if not isinstance(value["justification"], str) or not isinstance(value["verdict"], str):
        raise ValueError("solution fields must be strings")
    return {"justification": value["justification"].strip(), "verdict": value["verdict"].strip()}


def parse_baseline_output(text: str, fallback_verdict: str = "") -> Dict[str, str]:
    try:
        return parse_solution(text)
    except (ValueError, SyntaxError):
        justification_match = re.search(
            r"justification\s*:\s*(.*?)(?=\n\s*verdict\s*:|$)", text, re.DOTALL | re.IGNORECASE
        )
        verdict_match = re.search(r"verdict\s*:\s*['\"]?([^'\"\n]+)", text, re.IGNORECASE)
        justification = justification_match.group(1).strip() if justification_match else text.strip()
        verdict = verdict_match.group(1).strip() if verdict_match else fallback_verdict.strip()
        return {"justification": justification, "verdict": verdict}
