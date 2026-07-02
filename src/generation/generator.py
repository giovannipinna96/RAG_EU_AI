"""LLM answer generation via the SGLang OpenAI-compatible endpoint."""

from __future__ import annotations

import json

import structlog
from openai import OpenAI

from ..config import settings
from .normalizer import ReferenceNormalizer
from .prompts import SYSTEM_PROMPT

log = structlog.get_logger(__name__)


class Generator:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(base_url=settings.sglang_base_url, api_key="none")
        self.normalizer = ReferenceNormalizer()

    def generate(
        self,
        history: list[dict],
        chunks: list[dict],
        explicit_refs: list[str] | None = None,
    ) -> dict:
        context = self._build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"RELEVANT PROVISIONS:\n\n{context}"},
            *history,
        ]

        # OpenAI SDK strict overloads reject list[dict] messages; the server
        # accepts the standard dict form at runtime. `extra_body` carries the
        # thinking toggle (see settings.llm_extra_body) — non-standard field.
        resp = self.client.chat.completions.create(  # type: ignore[call-overload]
            model=settings.llm_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=settings.llm_max_tokens,
            extra_body=settings.llm_extra_body,
        )

        msg = resp.choices[0].message
        raw = self._parse_json(msg.content)
        references = self._ground_references(raw.get("references", []), chunks, explicit_refs)

        # Prefer the structured `reasoning` field the model was asked to emit; if
        # it is empty (e.g. thinking was enabled and the JSON omitted it) fall
        # back to the model's native `reasoning_content` so reasoning is never
        # silently dropped.
        reasoning = raw.get("reasoning", "") or self._reasoning_content(msg) or ""

        return {
            "reasoning": reasoning,
            "answer": raw.get("answer", ""),
            "references": references,
        }

    @staticmethod
    def _reasoning_content(msg: object) -> str:
        """Extract the model's native chain-of-thought if the server exposes it.

        Reasoning models served over the OpenAI API (gemma-4 on llama.cpp,
        SGLang `separate_reasoning`) return a non-standard `reasoning_content`
        field. The OpenAI SDK stores unknown fields on the pydantic model's
        `model_extra`; older stubs may set it as a plain attribute.
        """
        direct = getattr(msg, "reasoning_content", None)
        if direct:
            return str(direct)
        extra = getattr(msg, "model_extra", None)
        if isinstance(extra, dict) and extra.get("reasoning_content"):
            return str(extra["reasoning_content"])
        return ""

    def _ground_references(
        self,
        llm_refs: list[str],
        chunks: list[dict],
        explicit_refs: list[str] | None,
    ) -> list[str]:
        """Merge the LLM's chosen citations with authoritative retrieval signals.

        The LLM left to itself drops or substitutes references (e.g. citing the
        parent Article instead of the Annex a question explicitly named). We add
        back two high-precision sources so they can't be silently lost:

        * ``explicit_refs`` — provisions the user named in the query;
        * ``article_id`` of exact-match retrieval hits (same origin as the
          explicit refs, surfaced by payload-filtered lookup).

        We deliberately do NOT dump every retrieved chunk's id — that would wreck
        precision. If nothing valid survives, fall back to the single top-ranked
        retrieved provision so an answer is never returned uncited.
        """
        grounded: list[str] = list(llm_refs)
        grounded.extend(explicit_refs or [])
        grounded.extend(
            c["article_id"]
            for c in chunks
            if c.get("source") == "exact" and c.get("article_id")
        )

        references = self.normalizer.normalize(grounded)
        if not references:
            for c in chunks:
                if c.get("article_id"):
                    references = self.normalizer.normalize([c["article_id"]])
                    break
        return references

    @staticmethod
    def _parse_json(content: str | None) -> dict:
        text = Generator._strip_code_fence(content or "")
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError:
            # gemma-4 / llama.cpp does not honour response_format as a hard JSON
            # grammar, so the object can arrive with stray prose around it. As a
            # last resort, parse the widest brace-delimited span.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            log.warning("llm_json_parse_failed", content=(content or "")[:200])
            return {}

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove a leading ```json / ``` fence and its closing ``` if present."""
        t = text.strip()
        if t.startswith("```"):
            newline = t.find("\n")
            if newline != -1:
                t = t[newline + 1 :]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip()

    def _build_context(self, chunks: list[dict]) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for c in chunks:
            aid = c.get("article_id", "")
            # Only dedup chunks that carry an article_id. BM25-only hits have no
            # article_id (the lexical index stores raw text) — keep them all
            # rather than collapsing every unlabeled hit into one.
            if aid and aid in seen:
                continue
            if aid:
                seen.add(aid)
            text = c.get("content_raw", "")[:1500]
            parts.append(f"--- {aid or 'Provision'} ---\n{text}")
        return "\n\n".join(parts)
