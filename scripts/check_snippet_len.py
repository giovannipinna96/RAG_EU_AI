"""Check whether 2000 chars is enough to capture the answering clause.

For each failing case, reproduce the BGE top-8 the LLM rerank receives and
report:
  - the length of each candidate's content_raw
  - the character offset of the answering keyword(s) for the gold provision
  - whether the snippet at the chosen length (300 / 2000 / 4000) includes it

Also reports the longest content_raw across all chunks in Qdrant, so we can
sanity-check our snippet budget against the full corpus.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from src.config import settings
from src.retrieval.triple_retriever import (
    _LLM_RERANK_MAX_CANDIDATES,
    TripleRetriever,
)

CASES: list[tuple[str, str, str, list[str]]] = [
    (
        "comp_2",
        "Are AI systems intended for emotion recognition from biometric data always prohibited?",
        "Article 5",
        ["emotion recognition", "biometric"],
    ),
    (
        "comp_11",
        "Must AI-generated deepfake image or video content be disclosed as artificially generated?",
        "Article 50",
        ["deep fake", "deepfake", "artificially generated"],
    ),
]


def _bge_top8(r: TripleRetriever, q: str) -> list[dict]:
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


def _first_offset(text: str, keywords: list[str]) -> int:
    """Lowercase-insensitive earliest match across keywords; -1 if none."""
    lo = text.lower()
    offs = [lo.find(k.lower()) for k in keywords]
    offs = [o for o in offs if o >= 0]
    return min(offs) if offs else -1


def main() -> int:
    r = TripleRetriever()

    for label, q, gold, keywords in CASES:
        print(f"\n{'#' * 70}\n## {label}   gold = {gold}   keywords = {keywords}")
        bge = _bge_top8(r, q)
        for i, d in enumerate(bge, start=1):
            aid = d.get("article_id", "?")
            text = d.get("content_raw") or ""
            ln = len(text)
            mark = " <- GOLD" if aid == gold else ""
            print(f"  {i}. [{aid:<12}] len={ln:>5}{mark}")
            if aid == gold:
                off = _first_offset(text, keywords)
                if off < 0:
                    print(f"       keyword NOT FOUND in this chunk -- snippet alone can't help")
                else:
                    in300 = "yes" if off < 300 else "no"
                    in2000 = "yes" if off < 2000 else "no"
                    in4000 = "yes" if off < 4000 else "no"
                    end = min(off + 80, ln)
                    print(f"       first keyword hit at offset {off}")
                    print(f"       fits in 300?  {in300}   in 2000? {in2000}   in 4000? {in4000}")
                    print(f"       context: ...{text[max(0,off-30):end]!r}...")

    # corpus-wide stats
    print(f"\n{'#' * 70}\n## Corpus stats")
    client = QdrantClient(path=settings.qdrant_local_path or "data/qdrant")
    coll = settings.qdrant_collection
    lens: list[int] = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=coll, limit=512, offset=offset, with_payload=True
        )
        for p in pts:
            t = (p.payload or {}).get("content_raw") or ""
            lens.append(len(t))
        if offset is None:
            break
    lens.sort()
    n = len(lens)
    p50 = lens[n // 2]
    p90 = lens[int(n * 0.9)]
    p99 = lens[int(n * 0.99)]
    pmax = lens[-1]
    over2k = sum(1 for x in lens if x > 2000)
    over4k = sum(1 for x in lens if x > 4000)
    print(f"  n={n} chunks   p50={p50}  p90={p90}  p99={p99}  max={pmax}")
    print(f"  chunks > 2000 chars: {over2k} ({100*over2k/n:.1f}%)")
    print(f"  chunks > 4000 chars: {over4k} ({100*over4k/n:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
