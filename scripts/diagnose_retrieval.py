"""Phase-1 retrieval diagnostic: find WHERE the gold article drops out.

For the two failing eval questions (comp_2 → Article 5, comp_11 → Article 50)
plus two PASSING controls that target the same articles (comp_5 → Article 5,
comp_10 → Article 50), report the rank of the gold article_id at each pipeline
boundary:

  raw-dense (NO 0.3 threshold)  →  dense (with threshold)  →  bm25  →
  RRF-merge+dedup  →  BGE rerank top-k

This isolates the failing stage WITHOUT involving SGLang/LightRAG/LLM-rerank.
No fixes here — evidence only.
"""

from __future__ import annotations

from src.config import settings
from src.retrieval.triple_retriever import TripleRetriever

CASES = {
    "comp_27 FAIL ": (
        "What measures does the AI Act provide to support SMEs and start-ups?",
        "Article 62",
    ),
    "comp_2  FAIL ": (
        "Are AI systems intended for emotion recognition from biometric data always prohibited?",
        "Article 5",
    ),
    "comp_25 PASS*": (
        "What additional obligations apply to providers of general-purpose AI models "
        "with systemic risk?",
        "Article 55",
    ),
    "comp_10 PASS*": (
        "Do providers of AI systems that interact directly with people, such as chatbots, "
        "have to inform users they are interacting with an AI?",
        "Article 50",
    ),
}


def rank_of(gold: str, docs: list[dict]) -> str:
    for i, d in enumerate(docs):
        if d.get("article_id") == gold:
            sc = d.get("rerank_score", d.get("score"))
            return f"#{i + 1} (score={sc:.3f})" if isinstance(sc, float) else f"#{i + 1}"
    return "ABSENT"


def main() -> int:
    r = TripleRetriever()

    for label, (q, gold) in CASES.items():
        # raw dense WITHOUT the 0.3 threshold — true cosine ranking
        vec = r.embed_model.encode(q, normalize_embeddings=True).tolist()
        raw = r.qdrant.query_points(
            collection_name=settings.qdrant_collection, query=vec, limit=30
        ).points
        raw_docs = [{**(p.payload or {}), "score": p.score} for p in raw]

        dense = r._dense_search(q, top_k=30)          # with score_threshold=0.3
        bm25 = r._bm25_search(q)
        merged = r._rrf_merge([], dense, bm25, [], xref=[])
        seen: set[str] = set()
        uniq: list[dict] = []
        for d in merged:
            a = d.get("article_id", "")
            if a and a in seen:
                continue
            seen.add(a)
            uniq.append(d)
        bge = r.reranker.rerank(q, uniq, top_k=10)

        print(f"\n{'#' * 70}\n## {label}  query gold = {gold}")
        print(f"  raw-dense (no thresh) : gold {rank_of(gold, raw_docs)}   "
              f"top5={[d.get('article_id') for d in raw_docs[:5]]}")
        print(f"  dense (thresh 0.3)    : gold {rank_of(gold, dense)}   "
              f"top5={[d.get('article_id') for d in dense[:5]]}")
        print(f"  bm25                  : gold {rank_of(gold, bm25)}   "
              f"top5={[d.get('article_id') for d in bm25[:5]]}")
        print(f"  RRF-merged+dedup      : gold {rank_of(gold, uniq)}   "
              f"top5={[d.get('article_id') for d in uniq[:5]]}")
        print(f"  BGE rerank (top10)    : gold {rank_of(gold, bge)}")
        reaches = "YES" if rank_of(gold, bge[:5]) != "ABSENT" else "NO"
        print(f"     -> reaches final top-5? {reaches}   "
              f"final5={[d.get('article_id') for d in bge[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
