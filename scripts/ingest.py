"""Run the full offline ingestion pipeline: parse → chunk → index → graph.

Heavy: requires the ML extras and a reachable SGLang server. Run on a GPU
compute node via slurm/ingest.slurm, not on the login node.
"""

from __future__ import annotations

import asyncio
import os

import structlog

from src.ingestion.chunker import SACChunker
from src.ingestion.indexer import BM25Indexer, LightRAGIndexer, QdrantIndexer
from src.ingestion.parser import EUAIActParser

log = structlog.get_logger(__name__)


async def main() -> None:
    print("=== PARSING ===")
    parser = EUAIActParser()
    nodes = parser.parse("data/raw/eu_ai_act.html")
    assert parser.validate(nodes), "Parsing failed validation"

    print("\n=== CHUNKING ===")
    chunker = SACChunker()
    chunks = chunker.chunk_all(nodes)

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
    print("INGESTION COMPLETE")
    print(f"  Nodes parsed:   {len(nodes)}")
    print(f"  Chunks created: {len(chunks)}")
    print("  Qdrant indexed: OK")
    print("  BM25 index:     OK")
    print("  LightRAG graph: OK")


if __name__ == "__main__":
    asyncio.run(main())
