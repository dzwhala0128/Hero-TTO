import re
from collections import Counter


STOP_WORDS = {
    "the", "a", "an", "in", "on", "of", "to", "for", "and", "or",
    "was", "were", "is", "are", "be", "been", "being", "with", "by",
    "from", "that", "this", "it", "as", "at",
}

QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how", "did", "does", "do",
    "is", "are", "was", "were", "can", "could", "should", "would",
    "has", "have",
}

CHINESE_QUESTION_TERMS = [
    "是否", "有没有", "吗", "么", "谁", "什么", "为何", "为什么", "怎样", "如何",
]

META_PATTERNS = [
    "i cannot",
    "as an ai language model",
    "as a language model",
    "i can't",
    "i am unable",
    "i do not have access",
]

VERDICT_ONLY_PATTERNS = {
    "supported", "refuted", "true", "false", "likely true", "likely false",
    "not enough evidence", "conflicting evidence", "cherrypicking",
}


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", text or ""))


def english_char_ratio(text: str) -> float:
    counted = re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", text or "")
    if not counted:
        return 0.0
    english = [char for char in counted if re.match(r"[A-Za-z]", char)]
    return len(english) / len(counted)


def is_question_like(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if "?" in text or "？" in text:
        return True

    first_word = re.match(r"^\W*([A-Za-z]+)\b", text)
    if first_word and first_word.group(1).lower() in QUESTION_STARTERS:
        return True

    return any(term in text for term in CHINESE_QUESTION_TERMS)


def claim_keywords(claim: str) -> list[str]:
    claim = claim or ""
    keywords = []

    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim):
        keywords.append(match.group(0))

    for word in re.findall(r"\b[A-Za-z]{4,}\b", claim):
        if word.lower() not in STOP_WORDS:
            keywords.append(word)

    deduped = []
    seen = set()
    for keyword in keywords:
        key = keyword.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(keyword)
    return deduped


def keyword_hit_count(text: str, keywords: list[str]) -> int:
    text = text or ""
    count = 0
    for keyword in keywords:
        if not keyword:
            continue
        if " " in keyword:
            if keyword.lower() in text.lower():
                count += 1
        elif re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
            count += 1
    return count


def validate_hyde_doc(doc: str, claim: str, min_words: int = 50, max_words: int = 220) -> tuple[bool, list[str]]:
    raw_doc = doc or ""
    doc = clean_hyde_doc(raw_doc)
    reasons = []

    if not doc:
        return False, ["empty"]

    if contains_cjk(doc):
        reasons.append("contains_cjk")

    if english_char_ratio(doc) < 0.85:
        reasons.append("low_english_char_ratio")

    if is_question_like(doc):
        reasons.append("question_like")

    words = re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)?\b", doc)
    word_count = len(words)
    if word_count < min_words:
        reasons.append("too_short")
    if word_count > max_words:
        reasons.append("too_long")

    keywords = claim_keywords(claim)
    min_hits = max(2, min(4, len(keywords) // 3))
    if keywords and keyword_hit_count(doc, keywords) < min_hits:
        reasons.append("low_claim_keyword_hits")

    sentence_like_parts = [part for part in re.split(r"[.!?]+", doc) if part.strip()]
    if len(sentence_like_parts) <= 1 and word_count < min_words:
        reasons.append("too_short_or_single_query")

    lowered = doc.lower()
    if any(pattern in lowered for pattern in META_PATTERNS):
        reasons.append("llm_refusal_or_meta")

    stripped_lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
    bullet_lines = [line for line in stripped_lines if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line)]
    if stripped_lines and len(bullet_lines) == len(stripped_lines):
        reasons.append("bullet_only_keyword_list")

    if lowered.strip(" .!\n\t") in VERDICT_ONLY_PATTERNS:
        reasons.append("verdict_only")

    comma_terms = [term.strip() for term in re.split(r"[,;|]", doc) if term.strip()]
    if len(comma_terms) >= 5 and word_count < min_words:
        reasons.append("search_query_or_keyword_list")

    hard_reasons = [reason for reason in reasons if reason != "too_long"]
    return len(hard_reasons) == 0, reasons


def clean_hyde_doc(doc: str) -> str:
    doc = doc or ""
    doc = re.sub(r"^\s{0,3}#{1,6}\s*", "", doc)
    doc = re.sub(r"^\s*(?:Passage|Rewritten|Answer|Question)\s*:\s*", "", doc, flags=re.IGNORECASE)
    doc = re.sub(r"\s+", " ", doc)
    return doc.strip()


def dedupe_docs(docs: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for doc in docs or []:
        if not isinstance(doc, str):
            continue
        cleaned = clean_hyde_doc(doc)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def select_hyde_docs(row: dict, all_docs_field: str = "hypo_fc_docs_all") -> list[str]:
    all_docs = row.get(all_docs_field)
    if all_docs:
        return list(all_docs)
    return list(row.get("hypo_fc_docs", []) or [])


def get_hyde_docs_for_retrieval(row: dict, all_docs_field: str = "hypo_fc_docs_all") -> list[str]:
    return select_hyde_docs(row, all_docs_field=all_docs_field)


def combine_hyde_docs(original_docs: list[str], supplemental_docs: list[str]) -> list[str]:
    return dedupe_docs(list(original_docs or []) + list(supplemental_docs or []))


def build_additive_hyde_fields(
    row: dict,
    supplemental_docs_raw,
    supplemental_docs_parsed,
    claim: str,
    max_supplemental_docs: int = 4,
    all_docs_field: str = "hypo_fc_docs_all",
    textgrad_docs_field: str = "hypo_fc_docs_textgrad",
    textgrad_raw_docs_field: str = "hypo_fc_docs_textgrad_raw",
) -> dict:
    updated = dict(row)
    original_docs = list(row.get("hypo_fc_docs", []) or [])
    candidates = list(supplemental_docs_parsed or [])[:max_supplemental_docs]
    filtered_docs, report_items = filter_hyde_docs(candidates, claim, min_keep=0)
    filtered_docs = dedupe_docs(filtered_docs)[:max_supplemental_docs]

    if filtered_docs:
        fallback_reason = ""
    elif supplemental_docs_parsed is None:
        fallback_reason = "supplemental_json_parse_failed"
    elif not candidates:
        fallback_reason = "no_supplemental_docs_generated"
    else:
        fallback_reason = "all_supplemental_docs_filtered"

    updated["hypo_fc_docs"] = original_docs
    updated[textgrad_raw_docs_field] = supplemental_docs_raw if supplemental_docs_raw is not None else ""
    updated[textgrad_docs_field] = filtered_docs
    updated[all_docs_field] = combine_hyde_docs(original_docs, filtered_docs) if filtered_docs else original_docs
    updated["hypo_fc_docs_filter_report"] = {
        "fallback_to_original": not bool(filtered_docs),
        "reason": fallback_reason,
        "items": report_items,
    }
    return updated


def filter_hyde_docs(
    docs: list[str],
    claim: str,
    original_docs: list[str] | None = None,
    min_keep: int = 1,
) -> tuple[list[str], list[dict]]:
    filtered_docs = []
    report = []
    seen = set()

    for index, doc in enumerate(docs or []):
        cleaned = clean_hyde_doc(doc)
        valid, reasons = validate_hyde_doc(doc, claim)
        key = cleaned.lower()
        kept = valid and key not in seen
        if kept:
            filtered_docs.append(cleaned)
            seen.add(key)
        elif valid and key in seen:
            reasons = ["duplicate"]

        report.append({
            "index": index,
            "kept": kept,
            "reasons": reasons,
            "preview": cleaned[:120],
        })

    if len(filtered_docs) < min_keep:
        for index, doc in enumerate(original_docs or []):
            cleaned = clean_hyde_doc(doc)
            valid, reasons = validate_hyde_doc(doc, claim)
            key = cleaned.lower()
            kept = valid and key not in seen
            if kept:
                filtered_docs.append(cleaned)
                seen.add(key)
            report.append({
                "index": f"original_{index}",
                "kept": kept,
                "reasons": reasons if not kept else [],
                "preview": cleaned[:120],
            })
            if len(filtered_docs) >= min_keep:
                break

    return filtered_docs, report


def build_hyde_rewrite_instruction(claim: str, original_doc: str) -> str:
    return f"""Rewrite the hypothetical fact-checking document for retrieval.

Rules:

1. Keep the output in English.
2. Do not translate into another language.
3. Do not turn it into a question.
4. Keep it as a short fact-checking article passage.
5. The passage should be 80-150 words.
6. Preserve important named entities and lexical retrieval keywords from the claim.
7. Remove unsupported verdicts, speculation, and overconfident conclusions.
8. Focus on terms that help BM25 and dense retrieval find real evidence.
9. Do not output bullet points unless necessary.
10. Output only the rewritten passage.

Claim:
{claim}

Original hypothetical fact-checking document:
{original_doc}"""


def count_drop_reasons(report: list[dict]) -> dict:
    counter = Counter()
    for item in report:
        if item.get("kept"):
            continue
        counter.update(item.get("reasons", []))
    return dict(counter)
