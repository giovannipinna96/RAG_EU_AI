"""Index chunks into Qdrant (dense vectors + HyPE questions), BM25S (lexical),
and LightRAG (knowledge graph)."""

from __future__ import annotations

import uuid
from pathlib import Path

import bm25s
import Stemmer
import structlog
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ..clients import get_embedding_model, make_qdrant_client
from ..config import settings
from .chunker import ProcessedChunk
from .hype import HyPEGenerator

log = structlog.get_logger(__name__)


class QdrantIndexer:
    def __init__(self) -> None:
        self.qdrant = make_qdrant_client()
        self.embed_model = get_embedding_model()
        self.hype = HyPEGenerator()

    def create_collection(self) -> None:
        coll = settings.qdrant_collection
        if self.qdrant.collection_exists(coll):
            self.qdrant.delete_collection(coll)

        self.qdrant.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        # Keyword indices for exact-match payload filtering.
        self.qdrant.create_payload_index(coll, "article_id", PayloadSchemaType.KEYWORD)
        self.qdrant.create_payload_index(coll, "granularity", PayloadSchemaType.KEYWORD)
        self.qdrant.create_payload_index(coll, "type", PayloadSchemaType.KEYWORD)
        log.info("collection_created", collection=coll)

    def index_chunks(self, chunks: list[ProcessedChunk]) -> None:
        """Index original chunk embeddings plus HyPE question embeddings."""
        all_points: list[PointStruct] = []

        for i, chunk in enumerate(chunks):
            vec = self.embed_model.encode(chunk.content, normalize_embeddings=True)
            all_points.append(self._make_point(chunk, vec.tolist(), "chunk"))

            # HyPE questions only for LARGE chunks (cost/coverage trade-off).
            if chunk.granularity == "large":
                for q in self.hype.generate_questions(chunk, n=5):
                    q_vec = self.embed_model.encode(q, normalize_embeddings=True)
                    all_points.append(
                        self._make_point(chunk, q_vec.tolist(), "hype", hype_question=q)
                    )

            if (i + 1) % 20 == 0:
                log.info("embedding_progress", done=i + 1, total=len(chunks))

        batch = 64
        for i in range(0, len(all_points), batch):
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=all_points[i : i + batch],
            )
        log.info("qdrant_indexed", points=len(all_points), chunks=len(chunks))

    def _make_point(
        self,
        chunk: ProcessedChunk,
        vector: list[float],
        point_type: str,
        hype_question: str = "",
    ) -> PointStruct:
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "article_id": chunk.article_id,
                "article_type": chunk.article_type,
                "content_raw": chunk.content_raw,
                "title": chunk.title,
                "context": chunk.context,
                "paragraph_refs": chunk.paragraph_refs,
                "granularity": chunk.granularity,
                "type": point_type,
                "hype_question": hype_question,
            },
        )


class BM25Indexer:
    def __init__(self) -> None:
        self.stemmer = Stemmer.Stemmer("english")

    def build_index(self, chunks: list[ProcessedChunk]) -> None:
        """Build a BM25S in-memory index and persist it to disk.

        The corpus is stored as one dict record per chunk (text + article_id +
        granularity) so a search hit can be attributed to its provision. The
        tokenizer still works on the raw text strings.
        """
        texts = [c.content_raw for c in chunks]
        records = [
            {
                "text": c.content_raw,
                "article_id": c.article_id,
                "granularity": c.granularity,
            }
            for c in chunks
        ]
        corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        save_dir = Path(settings.bm25_index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(save_dir), corpus=records)
        log.info("bm25_index_saved", path=str(save_dir), documents=len(records))


class LightRAGIndexer:
    async def build_graph(self, full_text: str) -> None:
        """Index the full Act text into a LightRAG knowledge graph.

        Uses the shared :func:`build_lightrag` factory so the graph is built with
        exactly the embedding/LLM funcs the retriever later queries it with.
        """
        from ..clients import build_lightrag

        rag = build_lightrag(settings.lightrag_working_dir)
        await rag.initialize_storages()

        log.info("lightrag_indexing_started")
        await rag.ainsert(full_text)
        await rag.finalize_storages()
        log.info("lightrag_graph_built")
