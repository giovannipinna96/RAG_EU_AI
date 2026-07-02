"""BGE cross-encoder reranker (BAAI/bge-reranker-v2-m3)."""

from __future__ import annotations

import structlog

from ..config import settings

log = structlog.get_logger(__name__)


class BGEReranker:
    def __init__(self) -> None:
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(settings.reranker_model, use_fp16=True)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        pairs = [[query, doc.get("content_raw", "")] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores, strict=False):
            doc["rerank_score"] = float(score)

        return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
