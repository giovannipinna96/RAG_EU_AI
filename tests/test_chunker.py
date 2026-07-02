"""Offline tests for the SAC chunker, using a stub LLM client."""

from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.chunker import DOC_FINGERPRINT, SACChunker
from src.ingestion.parser import ArticleNode


class _StubClient:
    """Minimal stand-in for the OpenAI client returning a fixed context."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content="Context sentence about the provision.")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _node() -> ArticleNode:
    return ArticleNode(
        article_id="Article 5",
        article_type="article",
        number="5",
        title="Prohibited practices",
        full_text="Article 5\nProhibited practices\n1. The following are prohibited.",
        paragraphs=[{"num": "1", "ref": "Article 5.1", "text": "The following are prohibited."}],
    )


def test_creates_large_and_small_chunks():
    chunker = SACChunker(client=_StubClient())
    chunks = chunker.chunk_all([_node()])

    granularities = sorted(c.granularity for c in chunks)
    assert granularities == ["large", "small"]


def test_large_chunk_is_enriched():
    chunker = SACChunker(client=_StubClient())
    large = next(c for c in chunker.chunk_all([_node()]) if c.granularity == "large")

    assert large.content.startswith(DOC_FINGERPRINT)
    assert "[CONTEXT]:" in large.content
    # Raw text is preserved unenriched for BM25 / reranker.
    assert large.content_raw == _node().full_text


def test_small_chunk_carries_paragraph_ref():
    chunker = SACChunker(client=_StubClient())
    small = next(c for c in chunker.chunk_all([_node()]) if c.granularity == "small")
    assert small.paragraph_refs == ["Article 5.1"]
