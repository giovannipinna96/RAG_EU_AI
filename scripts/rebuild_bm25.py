"""Rebuild the BM25 index from Qdrant payloads, attaching article_id metadata.

Applies the BM25 article_id fix WITHOUT a full LLM re-ingest: it reads the
already-indexed chunks straight from Qdrant (content_raw + article_id +
granularity) and rewrites the bm25s index as dict records. The chunk text is
identical to what the original ingest tokenised, so the lexical index is
equivalent — only the per-doc metadata is added.
"""

from __future__ import annotations

from pathlib import Path

import bm25s
import Stemmer
from qdrant_client import QdrantClient

from src.config import settings


def main() -> int:
    path = settings.qdrant_local_path or "data/qdrant"
    client = QdrantClient(path=path)
    coll = settings.qdrant_collection

    texts: list[str] = []
    records: list[dict] = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=coll, limit=256, offset=offset, with_payload=True
        )
        for p in pts:
            pl = p.payload or {}
            txt = pl.get("content_raw") or ""
            if not txt:
                continue
            texts.append(txt)
            records.append(
                {
                    "text": txt,
                    "article_id": pl.get("article_id", ""),
                    "granularity": pl.get("granularity", ""),
                }
            )
        if offset is None:
            break

    print(f"[rebuild_bm25] read {len(records)} chunks from Qdrant ({coll})")
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    save_dir = Path(settings.bm25_index_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(save_dir), corpus=records)
    print(f"[rebuild_bm25] saved index with article_id metadata -> {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
