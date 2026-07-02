"""Phase-2 diagnostic: inspect what the LLM rerank actually saw and produced.

For the two failing eval cases and two passing controls, reproduce the BGE
top-8 candidate set that ``TripleRetriever`` would have sent to the LLM
re-ranker, then:

  1. Print the EXACT prompt (so we can see whether the gold provision's
     300-char snippet contains the answering clause).
  2. If ``RAG_LLM_URL`` env var is set (typical: http://localhost:8000/v1),
     call the rerank LLM with three prompt variants and print each ranking:
       a. current prompt (300-char snippets)
       b. longer snippets (1500 chars)
       c. "answers the question" framing (instead of "relevance")
  3. Highlight the position of the gold article in each variant.

No fixes here -- evidence only.
"""

from __future__ import annotations

import os

from src.config import settings
from src.retrieval.triple_retriever import (
    _LLM_RERANK_MAX_CANDIDATES,
    TripleRetriever,
)

CASES: dict[str, tuple[str, str]] = {
    "comp_2  FAIL ": (
        "Are AI systems intended for emotion recognition from biometric data always prohibited?",
        "Article 5",
    ),
    "comp_11 FAIL ": (
        "Must AI-generated deepfake image or video content be disclosed as artificially generated?",
        "Article 50",
    ),
    "comp_5  PASS*": (
        "Is social scoring of natural persons by public authorities prohibited under the AI Act?",
        "Article 5",
    ),
    "comp_10 PASS*": (
        "Do providers of AI systems that interact directly with people, such as chatbots, "
        "have to inform users they are interacting with an AI?",
        "Article 50",
    ),
}


def _bge_top8(r: TripleRetriever, q: str) -> list[dict]:
    """Replicate TripleRetriever.retrieve() up through BGE rerank, but return top-8."""
    dense = r._dense_search(q, top_k=30)
    bm25 = r._bm25_search(q)
    xref = r._xref_expand(dense + bm25)
    merged = r._rrf_merge([], dense, bm25, [], xref=xref)
    seen: set[str] = set()
    uniq: list[dict] = []
    for d in merged:
        a = d.get("article_id", "")
        if a and a in seen:
            continue
        seen.add(a)
        uniq.append(d)
    return r.reranker.rerank(q, uniq, top_k=_LLM_RERANK_MAX_CANDIDATES)


def _prompt_current(query: str, docs: list[dict], snip: int) -> str:
    lines = [
        "You are a legal-relevance judge for the EU AI Act.",
        "Given the QUESTION and a numbered list of PROVISIONS, output ONLY the",
        "provision numbers in order from most to least relevant to the question.",
        "Use only the numbers, separated by commas (e.g. '3,1,4,2').",
        "Do NOT explain -- just the comma-separated ranking.",
        "",
        f"QUESTION: {query}",
        "",
        "PROVISIONS:",
    ]
    for i, d in enumerate(docs, start=1):
        aid = d.get("article_id", f"doc-{i}")
        snippet = (d.get("content_raw") or "")[:snip].replace("\n", " ")
        lines.append(f"{i}. [{aid}] {snippet}")
    lines.append("")
    lines.append("Ranking (most relevant first):")
    return "\n".join(lines)


def _prompt_answers(query: str, docs: list[dict], snip: int) -> str:
    lines = [
        "You are a legal-citation selector for the EU AI Act.",
        "Given the QUESTION and a numbered list of PROVISIONS, pick the provisions",
        "that DIRECTLY ANSWER the question. Rank them from the one that most directly",
        "answers it to the one that least directly answers it. A provision merely",
        "ABOUT the same topic is less relevant than one stating the actual rule.",
        "Output ONLY the provision numbers, comma-separated (e.g. '3,1,4,2'). No prose.",
        "",
        f"QUESTION: {query}",
        "",
        "PROVISIONS:",
    ]
    for i, d in enumerate(docs, start=1):
        aid = d.get("article_id", f"doc-{i}")
        snippet = (d.get("content_raw") or "")[:snip].replace("\n", " ")
        lines.append(f"{i}. [{aid}] {snippet}")
    lines.append("")
    lines.append("Direct-answer ranking:")
    return "\n".join(lines)


def _gold_rank(gold: str, docs: list[dict], raw_ranking: str) -> str:
    seen, ordered = set(), []
    for tok in raw_ranking.replace(";", ",").split(","):
        try:
            idx = int(tok.strip().strip(".")) - 1
        except ValueError:
            continue
        if 0 <= idx < len(docs) and idx not in seen:
            seen.add(idx)
            ordered.append(docs[idx])
    for i, doc in enumerate(docs):
        if i not in seen:
            ordered.append(doc)
    for i, d in enumerate(ordered, start=1):
        if d.get("article_id") == gold:
            return f"#{i}"
    return "ABSENT"


def _call_llm(prompt: str, base_url: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="none")
    resp = client.chat.completions.create(
        model=getattr(settings, "utility_model", "default"),
        temperature=0,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> int:
    r = TripleRetriever()
    llm_url = os.environ.get("RAG_LLM_URL", "")

    for label, (q, gold) in CASES.items():
        print("\n" + "#" * 78)
        print(f"## {label}   gold = {gold}")
        print(f"   QUERY: {q}")
        bge = _bge_top8(r, q)
        bge_order = [d.get("article_id") for d in bge]
        print(f"   BGE top-{len(bge)}: {bge_order}")

        # 1) print the actual prompt the LLM saw in the eval
        print("\n--- PROMPT-A (current: 300-char snippets, relevance framing) ---")
        prompt_a = _prompt_current(q, bge, snip=300)
        print(prompt_a)

        if not llm_url:
            print("\n[skipping LLM calls -- set RAG_LLM_URL=http://host:port/v1 to enable]")
            continue

        # 2) call each variant
        for vname, prompt in [
            ("A: relevance / 300", _prompt_current(q, bge, snip=300)),
            ("B: relevance / 1500", _prompt_current(q, bge, snip=1500)),
            ("C: answers / 300", _prompt_answers(q, bge, snip=300)),
            ("D: answers / 1500", _prompt_answers(q, bge, snip=1500)),
        ]:
            raw = _call_llm(prompt, llm_url)
            gold_pos = _gold_rank(gold, bge, raw)
            print(f"\n   variant {vname:<22} -> raw='{raw[:60]}'  gold {gold_pos}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
