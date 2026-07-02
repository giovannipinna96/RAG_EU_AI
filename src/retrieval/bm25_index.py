"""BM25S in-memory lexical search over the persisted index.

The persisted bm25s corpus stores one *record per chunk* — a dict carrying the
original text plus ``article_id``/``granularity`` — so a hit can be attributed
to its provision. Older indexes that stored bare strings (no metadata) are still
handled: their hits simply carry an empty ``article_id``.

``bm25s``/``Stemmer`` are imported lazily (inside methods) so this module is
importable in offline tests without the ``ml`` extra.
"""

from __future__ import annotations

from ..config import settings


def _build_hit(doc: object, score: float) -> dict:
    """Turn a bm25s corpus item into a retriever hit dict.

    ``doc`` is either a dict record ({"text", "article_id", "granularity"}) from
    a current index, or a bare string from a legacy index.
    """
    if isinstance(doc, dict):
        text = doc.get("text", "")
        article_id = doc.get("article_id", "")
        granularity = doc.get("granularity", "")
    else:
        text, article_id, granularity = str(doc), "", ""
    return {
        "content_raw": text,
        "article_id": article_id,
        "granularity": granularity,
        "score": float(score),
    }


class BM25Index:
    def __init__(self, index_dir: str | None = None) -> None:
        import bm25s
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25.load(
            index_dir or settings.bm25_index_dir, load_corpus=True
        )

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        import bm25s

        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(tokens, k=top_k, sorted=True)
        return [
            _build_hit(doc, score)
            for doc, score in zip(results[0], scores[0], strict=False)
            if score > 0
        ]
