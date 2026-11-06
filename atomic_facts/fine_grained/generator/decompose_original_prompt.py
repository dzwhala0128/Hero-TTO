#!/usr/bin/env python3

"""Lightweight copy of scripts/run_atomic_claim_decomposition_v2.py.
One full-claim call; original 7 fixed + 1 BM25 demonstrations.
"""

import argparse
import json
from pathlib import Path

import re


DEFAULT_DEMOS = Path(__file__).with_name("demons.json")

DEFAULT_BASE_URL = "http://127.0.0.1:8006/v1"
DEFAULT_MODEL = "qwen36-atomic"


ADAPTED_INSTRUCTION = """Now decompose the FULL CLAIM into the smallest factual propositions with independent verification value.
Separate independently checkable events, locations, times, identities and attributes where meaningful. Allow overlap between a weaker proposition and a more qualified proposition when each has verification value, as in FActScore; do not mechanically enumerate every entailed weaker proposition or produce exact duplicates.
Resolve references using only the full claim so each fact is self-contained. If a reference cannot be resolved, retain it without guessing.
Keep facts about the same event linked through sufficient identifying context; do not create facts the claim does not assert. Retain the claim's specific information across the facts, and preserve qualifiers wherever removing them changes scope or meaning.
Causal, conditional and negative relations can themselves be atomic facts. Preserve negation, quantifiers, conditions, causality, comparisons and attribution; never turn conditional or quoted content into unconditional assertions. Do not invent information or fact-check.
Use the legacy examples for fine-grained splitting and useful overlap, but not for unresolved references or meaning-changing omissions.
Return ONLY a list, one fact per line prefixed with "- ".

FULL CLAIM:
"""


def load_demos(path):
    from rank_bm25 import BM25Okapi

    with open(path, "r", encoding="utf-8") as f:
        demos = json.load(f)

    demo_sentences = list(demos.keys())

    tokenized_corpus = [
        sentence.split(" ")
        for sentence in demo_sentences
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return demos, demo_sentences, bm25


def get_best_demo(
    claim,
    demo_sentences,
    bm25,
):
    return bm25.get_top_n(
        claim.split(" "),
        demo_sentences,
        1,
    )


def build_prompt(
    claim,
    demos,
    demo_sentences,
    bm25,
):
    prompt = ""

    # Original FActScore strategy:
    # 7 fixed demonstrations
    for i in range(7):
        demo_sentence = demo_sentences[i]

        prompt += (
            "Please breakdown the following sentence "
            "into independent facts: "
            f"{demo_sentence}\n"
        )

        for fact in demos[demo_sentence]:
            prompt += f"- {fact}\n"

        prompt += "\n"

    # + 1 BM25-selected demonstration
    retrieved = get_best_demo(
        claim,
        demo_sentences,
        bm25,
    )

    for demo_sentence in retrieved:
        prompt += (
            "Please breakdown the following sentence "
            "into independent facts: "
            f"{demo_sentence}\n"
        )

        for fact in demos[demo_sentence]:
            prompt += f"- {fact}\n"

        prompt += "\n"

    # HerO / AVeriTeC context-aware adaptation
    prompt += "Please breakdown the following sentence into independent facts: "
    prompt += claim
    prompt += "\n"

    return prompt, retrieved


def parse_atomic_facts(text):
    """Original bullet format, with whole-line parsing and exact deduplication."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")
    facts = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- "):
            raise ValueError("expected one '- ' bullet per line")
        fact = line[2:].strip()
        if not fact or fact.lower() in {"null", "none", "n/a"}:
            raise ValueError("empty or null atomic fact")
        if fact not in facts:
            facts.append(fact)
    if not facts:
        raise ValueError("empty atomic facts")
    return facts


def make_record(claim_id, claim, raw, finish_reason="stop"):
    record = {"claim_id": claim_id, "original_claim": claim,
              "raw_decomposition": raw, "atomic_facts": [], "status": "parsed_unreviewed"}
    try:
        record["atomic_facts"] = parse_atomic_facts(raw)
    except ValueError as exc:
        record.update(status="parse_failed", reason=str(exc))
        return record
    if finish_reason != "stop":
        record.update(status="incomplete", reason=f"finish_reason={finish_reason}")
    elif any(re.search(r"\b(he|she|it|they|him|her|his|their|them|there|this|these|those)\b", f, re.I)
             for f in record["atomic_facts"]):
        # Token presence cannot establish whether a reference is unresolved.
        # Keep parsed_unreviewed and Stage A export; never claim coreference success.
        record["reference_note"] = (
            "Reference words present; they may be bound within each fact. "
            "Non-blocking reminder to check self-contained meaning manually; "
            "no coreference judgment or rewrite was performed."
        )
    return record


def to_stage_a(record):
    """V1.6 prepare_inputs consumes claim_id/claim/atomic_facts; assigns integer IDs."""
    if record["status"] != "parsed_unreviewed":
        raise ValueError("Record needs review or failed; inspect the decomposition record")
    facts = record["atomic_facts"]
    if not facts:
        raise ValueError("empty atomic facts")
    return {"claim_id": record["claim_id"], "claim": record["original_claim"],
            "atomic_facts": list(facts)}


def decompose_claim(
    claim,
    client,
    model,
    demos,
    demo_sentences,
    bm25,
):
    prompt, retrieved_demo = build_prompt(
        claim=claim,
        demos=demos,
        demo_sentences=demo_sentences,
        bm25=bm25,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )

    raw_output = (
        response
        .choices[0]
        .message
        .content
    )

    return raw_output, response.choices[0].finish_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", required=True, help="One manual claim; no dataset is read")
    parser.add_argument("--claim-id", default="manual_1")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.claim.strip() or not args.claim_id.strip():
        parser.error("claim and claim-id must be nonempty")
    # Exclusive directory creation prevents replacing any earlier experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    demos, sentences, bm25 = load_demos(args.demos)
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0, timeout=120.0)
    try:
        raw, finish = decompose_claim(args.claim, client, args.model, demos, sentences, bm25)
        record = make_record(args.claim_id, args.claim, raw, finish)
    except Exception as exc:
        record = {"claim_id": args.claim_id, "original_claim": args.claim,
                  "raw_decomposition": None, "atomic_facts": [],
                  "status": "request_failed", "reason": str(exc)}
    (args.output_dir / "decomposition.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    if record["status"] == "parsed_unreviewed":
        (args.output_dir / "stage_a_input.jsonl").write_text(json.dumps(to_stage_a(record), ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if record["status"] != "parsed_unreviewed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
