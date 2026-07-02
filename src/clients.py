"""Shared client factories.

Centralises construction of the Qdrant client and the (process-cached)
SentenceTransformer embedding model so ingestion and retrieval stay in sync and
the heavy embedding weights load at most once per process.

Imports of :mod:`sentence_transformers` are deferred into the functions so this
module — and anything that only needs the Qdrant client — can be imported on a
machine without the ML extras installed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient

from .config import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


def make_qdrant_client() -> QdrantClient:
    """Return a Qdrant client.

    Uses embedded on-disk mode when ``QDRANT_LOCAL_PATH`` is configured
    (no server needed), otherwise connects to the configured server URL.
    """
    if settings.qdrant_local_path:
        return QdrantClient(path=settings.qdrant_local_path)
    return QdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer model by name."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name or settings.embedding_model)


def embed(text: str | list[str], **kwargs: Any):
    """Convenience: encode text with the default model, L2-normalised."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True, **kwargs)


def build_lightrag(working_dir: str | None = None):
    """Construct a LightRAG instance wired to SGLang (LLM) and the local
    embedding model.

    Used for BOTH graph construction (ingestion) and graph query (retrieval) so
    the two stay configured identically — querying needs the same
    ``embedding_func`` that was used to build the graph, otherwise LightRAG
    raises "embedding_func is required for vector storage". LightRAG imports are
    deferred so this module stays importable without the ``ml`` extra.
    """
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=settings.sglang_base_url,
            api_key="none",
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(embedding_dim=settings.embedding_dim, max_token_size=8192)
    async def embed_func(texts: list[str]):
        return get_embedding_model().encode(texts, normalize_embeddings=True)

    return LightRAG(
        working_dir=working_dir or settings.lightrag_working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
    )
