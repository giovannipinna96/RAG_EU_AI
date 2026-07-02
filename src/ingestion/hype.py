"""HyPE — Hypothetical Prompt Embeddings.

Pre-generates, at indexing time, the questions each chunk would answer. Embedding
those questions alongside the chunk closes the query/document style gap with zero
runtime cost (unlike HyDE, which generates at query time).
"""

from __future__ import annotations

import structlog
from openai import OpenAI

from ..config import settings
from .chunker import ProcessedChunk

log = structlog.get_logger(__name__)


class HyPEGenerator:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(base_url=settings.sglang_base_url, api_key="none")

    def generate_questions(self, chunk: ProcessedChunk, n: int = 5) -> list[str]:
        """Return up to ``n`` hypothetical questions this chunk would answer."""
        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.3,
                max_tokens=400,
                extra_body=settings.llm_extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Generate exactly {n} diverse questions that the following "
                            f"EU AI Act provision would answer. Include yes/no questions, "
                            f"what/how questions, and scenario-based questions.\n"
                            f"Output ONLY the questions, one per line, no numbering.\n\n"
                            f"[{chunk.article_id}]:\n{chunk.content_raw[:1500]}"
                        ),
                    }
                ],
            )
            lines = (resp.choices[0].message.content or "").strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:n]
        except Exception as exc:  # noqa: BLE001 — HyPE is best-effort enrichment
            log.warning("hype_generation_failed", article=chunk.article_id, error=str(exc))
            return []
