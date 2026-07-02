"""SAC chunker: Structure-Aware + Summary-Augmented + multi-granularity.

Produces LARGE (full-article) and SMALL (paragraph-level) chunks. Each chunk is
prefixed with a document fingerprint and an LLM-generated context sentence
(Contextual Retrieval) for the embedded text, while the raw text is preserved
separately for BM25 and the reranker.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import OpenAI

from ..config import settings
from .parser import ArticleNode

log = structlog.get_logger(__name__)


DOC_FINGERPRINT = (
    "[DOCUMENT]: EU AI Act — Regulation (EU) 2024/1689 of 13 June 2024. "
    "Contains 113 Articles across 13 Chapters and 13 Annexes (I-XIII). "
    "Covers risk classification of AI systems, obligations for providers "
    "and deployers, prohibited practices, conformity assessment, and penalties."
)


@dataclass
class ProcessedChunk:
    content: str  # SAC-enriched text (for embedding)
    content_raw: str  # original text (for BM25 and reranker)
    article_id: str
    article_type: str
    paragraph_refs: list[str]
    granularity: str  # "large" or "small"
    title: str = ""
    context: str = ""


class SACChunker:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(base_url=settings.sglang_base_url, api_key="none")

    def chunk_all(self, nodes: list[ArticleNode]) -> list[ProcessedChunk]:
        chunks: list[ProcessedChunk] = []
        for i, node in enumerate(nodes):
            log.info("chunking", article=node.article_id, index=i + 1, total=len(nodes))
            context = self._generate_context(node)

            # LARGE chunk: full article.
            large_content = f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n{node.full_text}"
            chunks.append(
                ProcessedChunk(
                    content=large_content,
                    content_raw=node.full_text,
                    article_id=node.article_id,
                    article_type=node.article_type,
                    paragraph_refs=[p["ref"] for p in node.paragraphs],
                    granularity="large",
                    title=node.title,
                    context=context,
                )
            )

            # SMALL chunks: one per numbered paragraph.
            if node.paragraphs:
                for para in node.paragraphs:
                    small_content = (
                        f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n"
                        f"[{para['ref']}]: {para['text']}"
                    )
                    chunks.append(
                        ProcessedChunk(
                            content=small_content,
                            content_raw=para["text"],
                            article_id=node.article_id,
                            article_type=node.article_type,
                            paragraph_refs=[para["ref"]],
                            granularity="small",
                            title=node.title,
                            context=context,
                        )
                    )
            elif node.word_count > 400:
                # Sliding-window split for long articles without numbered paragraphs.
                words = node.full_text.split()
                for j in range(0, len(words), 250):
                    block = " ".join(words[j : j + 300])
                    small_content = f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n{block}"
                    chunks.append(
                        ProcessedChunk(
                            content=small_content,
                            content_raw=block,
                            article_id=node.article_id,
                            article_type=node.article_type,
                            paragraph_refs=[],
                            granularity="small",
                            title=node.title,
                            context=context,
                        )
                    )

        large = sum(1 for c in chunks if c.granularity == "large")
        small = sum(1 for c in chunks if c.granularity == "small")
        log.info("chunking_complete", total=len(chunks), large=large, small=small)
        return chunks

    def _generate_context(self, node: ArticleNode) -> str:
        """Contextual Retrieval: 1-2 sentences describing the article's scope."""
        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                max_tokens=100,
                extra_body=settings.llm_extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"In 1-2 sentences, explain what {node.article_id} "
                            f"('{node.title}') covers in the EU AI Act. "
                            f"Be specific.\n\nText: {node.full_text[:2000]}"
                        ),
                    }
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 — degrade gracefully if LLM is down
            log.warning("context_generation_failed", article=node.article_id, error=str(exc))
            return f"{node.article_id}: {node.title}"
