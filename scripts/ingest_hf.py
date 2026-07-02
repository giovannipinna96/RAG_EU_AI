"""Ingestion pipeline using the Hugging Face dataset as the source corpus.

Identical to scripts/ingest.py downstream of parsing, but loads ArticleNodes from
the structured parquet (scripts/download_hf.py) instead of regex-parsing EUR-Lex
HTML. Heavy: ML extras + a reachable SGLang server. Run via slurm/ingest_hf.slurm.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog

from src.ingestion.chunker import SACChunker
from src.ingestion.hf_loader import PARQUET_FILE, HFDatasetLoader
from src.ingestion.indexer import BM25Indexer, LightRAGIndexer, QdrantIndexer

log = structlog.get_logger(__name__)
PARQUET = Path("data/raw") / PARQUET_FILE


async def main() -> None:
    print("=== LOADING HF DATASET ===")
    nodes = HFDatasetLoader(str(PARQUET), language="en").load()
    articles = [n for n in nodes if n.article_type == "article"]
    annexes = [n for n in nodes if n.article_type == "annex"]
    print(f"Loaded {len(articles)} articles, {len(annexes)} annexes")
    assert len(articles) >= 100 and len(annexes) >= 10, "Unexpected corpus size"

    print("\n=== CHUNKING ===")
    chunks = SACChunker().chunk_all(nodes)

    print("\n=== INDEXING QDRANT ===")
    qdrant_indexer = QdrantIndexer()
    qdrant_indexer.create_collection()
    qdrant_indexer.index_chunks(chunks)

    print("\n=== BUILDING BM25 INDEX ===")
    BM25Indexer().build_index(chunks)

    if os.environ.get("SKIP_LIGHTRAG") == "1":
        print("\n=== SKIPPING KNOWLEDGE GRAPH (SKIP_LIGHTRAG=1) ===")
    else:
        print("\n=== BUILDING KNOWLEDGE GRAPH ===")
        full_text = "\n\n".join(n.full_text for n in nodes)
        await LightRAGIndexer().build_graph(full_text)

    print(f"\n{'=' * 50}")
    print("INGESTION COMPLETE (HF source)")
    print(f"  Nodes:   {len(nodes)}")
    print(f"  Chunks:  {len(chunks)}")
    print(f"  Graph:   {'skipped' if os.environ.get('SKIP_LIGHTRAG') == '1' else 'built'}")


if __name__ == "__main__":
    asyncio.run(main())
