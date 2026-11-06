"""Fail-closed structural and evidence-boundary validation."""

import re
from typing import Dict, List

from .hero_solution import VERDICTS, parse_solution

_LEAK_KEYS = re.compile(
    r"(?:reference[_ -]?answer|correct[_ -]?label|target[_ -]?label|hidden[_ -]?label|ground[_ -]?truth|dev[_ -]?label)",
    re.IGNORECASE,
)


def validate_solution(text: str, claim: str, evidence: list) -> Dict[str, object]:
    errors: List[str] = []
    parsed = None
    try:
        parsed = parse_solution(text)
    except (ValueError, SyntaxError) as exc:
        errors.append(str(exc))
    if parsed is not None:
        if not parsed["justification"]:
            errors.append("justification is empty")
        if parsed["verdict"] not in VERDICTS:
            errors.append("verdict is not one of the four allowed values")
        if _LEAK_KEYS.search(text):
            errors.append("solution contains a prohibited supervision field")
        allowed_urls = {str(item.get("url", "")) for item in evidence}
        output_urls = set(re.findall(r"https?://[^\s\"']+", parsed["justification"]))
        if any(url.rstrip(".,)") not in allowed_urls for url in output_urls):
            errors.append("solution introduces a URL outside the supplied evidence")
    return {"valid": not errors, "errors": errors, "parsed": parsed}
