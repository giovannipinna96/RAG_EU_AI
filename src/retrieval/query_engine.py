"""Adaptive query engine.

Stages, in order:
1. Multi-turn resolution — rewrite the latest question to be self-contained.
2. Explicit reference extraction — regex over the query and prior user turns.
3. Two-level complexity detection — linguistic patterns first, LLM fallback only
   for the genuinely ambiguous cases.
4. Conditional sub-query decomposition — only for complex questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import OpenAI

from ..config import settings
from .article_matcher import ArticleMatcher

log = structlog.get_logger(__name__)


@dataclass
class ProcessedQuery:
    original_query: str
    resolved_query: str
    explicit_refs: list[str]
    sub_queries: list[str] | None = None
    is_complex: bool = False


COMPLEXITY_SIGNALS = [
    " or ", " versus ", " compared to ", " differ",
    " because ", " resulted in ", " caused by ",
    " relationship between ", " how do ", " how does ",
]


class QueryEngine:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(base_url=settings.sglang_base_url, api_key="none")
        self.matcher = ArticleMatcher()

    def process(self, query: str, history: list[dict]) -> ProcessedQuery:
        resolved = self._resolve_multi_turn(query, history)

        refs = self.matcher.extract_refs(resolved)
        for msg in history:
            if msg.get("role") == "user":
                refs.extend(self.matcher.extract_refs(msg.get("content", "")))
        refs = sorted(set(refs))

        is_complex = self._detect_complexity(resolved)
        sub_queries = self._decompose(resolved) if is_complex else None

        pq = ProcessedQuery(
            original_query=query,
            resolved_query=resolved,
            explicit_refs=refs,
            sub_queries=sub_queries,
            is_complex=is_complex,
        )
        log.info(
            "query_processed",
            refs=refs,
            is_complex=is_complex,
            sub_queries=sub_queries,
        )
        return pq

    def _resolve_multi_turn(self, query: str, history: list[dict]) -> str:
        if len(history) <= 1:
            return query
        try:
            convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-6:])
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=200,
                extra_body=settings.llm_extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the last question to be fully self-contained. "
                            "Resolve all pronouns and references.\n\n"
                            f"Conversation:\n{convo}\n\nRewritten question:"
                        ),
                    }
                ],
            )
            return (resp.choices[0].message.content or query).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("multiturn_resolution_failed", error=str(exc))
            return query

    def _detect_complexity(self, query: str) -> bool:
        # Level 1: linguistic heuristics (0 ms).
        q_lower = query.lower()
        signal_count = sum(1 for s in COMPLEXITY_SIGNALS if s in q_lower)
        ref_count = len(self.matcher.extract_refs(query))
        question_marks = query.count("?")

        score = signal_count + (1 if ref_count >= 2 else 0) + (1 if question_marks >= 2 else 0)
        if score == 0:
            return False
        if score >= 2:
            return True

        # Level 2: LLM classification, only when genuinely uncertain (~50 ms).
        try:
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=10,
                extra_body=settings.llm_extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Classify as SIMPLE or COMPLEX.\n"
                            "SIMPLE: one fact, one article, direct answer.\n"
                            "COMPLEX: multiple articles, comparison, multi-step reasoning.\n\n"
                            f"Question: {query}\nClassification:"
                        ),
                    }
                ],
            )
            return "COMPLEX" in (resp.choices[0].message.content or "").upper()
        except Exception as exc:  # noqa: BLE001
            log.warning("complexity_detection_failed", error=str(exc))
            return False

    def _decompose(self, query: str) -> list[str]:
        try:
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=300,
                extra_body=settings.llm_extra_body,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Break this EU AI Act question into 2-3 simpler sub-questions.\n"
                            "Output only the sub-questions, one per line.\n\n"
                            f"Question: {query}"
                        ),
                    }
                ],
            )
            lines = (resp.choices[0].message.content or "").strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:3]
        except Exception as exc:  # noqa: BLE001
            log.warning("decomposition_failed", error=str(exc))
            return [query]
