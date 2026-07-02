"""Triple Retriever: Dense + BM25 + LightRAG + Exact-Match, fused with RRF.

All four sources run concurrently via ``asyncio.gather``. Blocking calls are
off-loaded with ``asyncio.to_thread`` so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog
from qdrant_client.models import FieldCondition, Filter, MatchValue

from ..clients import get_embedding_model, make_qdrant_client
from ..config import settings
from .article_matcher import ArticleMatcher
from .bm25_index import BM25Index
from .query_engine import ProcessedQuery
from .reranker import BGEReranker

log = structlog.get_logger(__name__)

# Maximum candidates passed to LLM rerank (keeps prompt small).
_LLM_RERANK_MAX_CANDIDATES = 8
# Snippet length per candidate sent to the LLM rerank. Combined with the
# adaptive-centering in `_adaptive_snippet` this is the size of the window
# centered on the most informative query-keyword match in the chunk, not the
# first N chars. 3000 with adaptive centering covers Art 5(1)(f) (emotion-
# recognition prohibition, offset ~2763) and Art 50(4) (deepfake disclosure,
# offset ~1022) on the actual corpus; with 8 candidates the prompt stays
# below ~24K chars (~6K tokens).
_LLM_RERANK_SNIPPET_CHARS = 3000
# Drop query stopwords shorter than this when picking the centering keyword.
_LLM_RERANK_KEYWORD_MIN_LEN = 4
# Max tokens for the LLM rerank response (a short list of numbers/ids).
_LLM_RERANK_MAX_TOKENS = 120


def _adaptive_snippet(text: str, query: str, length: int) -> str:
    """Return up to *length* chars from *text*, centered on the best query match.

    Picks the earliest in-text occurrence of any query content-word
    (``\\w+`` of length >= ``_LLM_RERANK_KEYWORD_MIN_LEN``) and centers a
    *length*-char window around it. If the text already fits in *length*
    chars, returns the whole text. If no query word matches, falls back to
    ``text[:length]`` (legacy behavior). Window boundaries that hit the
    chunk boundary are shifted to keep the full *length* whenever possible.
    Truncation is marked with ``[...]`` so the LLM sees the context is
    partial. Centering is purely lexical -- no embeddings, no tokenizer.
    """
    if length <= 0 or not text:
        return ""
    if len(text) <= length:
        return text

    words = [
        w
        for w in re.findall(r"\w+", query.lower())
        if len(w) >= _LLM_RERANK_KEYWORD_MIN_LEN
    ]
    text_lower = text.lower()
    hits = [text_lower.find(w) for w in words]
    hits = [h for h in hits if h >= 0]
    if not hits:
        return text[:length] + "[...]"

    center = min(hits)  # earliest match -- usually the start of the relevant paragraph
    half = length // 2
    start = max(0, center - half)
    end = min(len(text), start + length)
    start = max(0, end - length)  # shift back if we hit the right boundary

    snippet = text[start:end]
    prefix = "[...]" if start > 0 else ""
    suffix = "[...]" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


class TripleRetriever:
    def __init__(self) -> None:
        self.qdrant = make_qdrant_client()
        self.embed_model = get_embedding_model()
        self.bm25 = BM25Index()
        self.matcher = ArticleMatcher()
        self.reranker = BGEReranker()
        self._rag = None
        self._rag_lock = asyncio.Lock()
        # LLM rerank client: lazily created on first call; injectable for tests.
        self.llm_rerank_client: Any | None = None

    async def retrieve(self, pq: ProcessedQuery, top_k: int = 5) -> list[dict]:
        names = ("exact", "dense", "bm25", "graph")
        results = await asyncio.gather(
            asyncio.to_thread(self._exact_match, pq.explicit_refs),
            asyncio.to_thread(self._dense_search, pq.resolved_query),
            asyncio.to_thread(self._bm25_search, pq.resolved_query),
            self._graph_search(pq.resolved_query),
            return_exceptions=True,
        )

        cleaned: list[list[dict]] = []
        for r, name in zip(results, names, strict=True):
            if isinstance(r, BaseException):
                log.warning("retrieval_source_failed", source=name, error=str(r))
                cleaned.append([])
            else:
                cleaned.append(list(r))
        exact, dense, bm25, graph = cleaned

        # Expand recall for complex queries by also searching each sub-query.
        if pq.sub_queries:
            sub_tasks = []
            for sq in pq.sub_queries:
                sub_tasks.append(asyncio.to_thread(self._dense_search, sq))
                sub_tasks.append(asyncio.to_thread(self._bm25_search, sq))
            for r in await asyncio.gather(*sub_tasks, return_exceptions=True):
                if not isinstance(r, BaseException):
                    dense.extend(r)

        xref = self._xref_expand(exact + dense + bm25 + graph)
        candidates = self._rrf_merge(exact, dense, bm25, graph, xref=xref)

        return self._final_rerank(pq.resolved_query, candidates, top_k=top_k)

    def _final_rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        """Rerank the RRF-merged candidates down to ``top_k`` provisions.

        Two modes (``settings.enable_chunk_voting``):

        * **classic** (default): dedup by article first, BGE-rerank the survivors,
          then optionally LLM-rerank — one chunk per article throughout.
        * **voting**: BGE-rerank the *un-deduped* pool to ``rerank_candidate_chunks``
          chunks (several may share an article), LLM-rank them, and aggregate the
          ranking into a per-article result by positional voting. This recovers
          articles whose answering chunk would otherwise be dropped by the
          pre-rerank dedup (see ``_aggregate_chunk_votes``).
        """
        if getattr(settings, "enable_chunk_voting", False):
            n = getattr(settings, "rerank_candidate_chunks", 12)
            if getattr(settings, "enable_pool_blend", False):
                # Blend pool selection: BGE-score the whole candidate set, then
                # reserve the top ``pool_rrf_reserve`` RRF candidates before
                # filling the rest in BGE order. A strong fusion signal (high
                # RRF rank) is no longer silently dropped by a low cross-encoder
                # score — the comp_27 case where Art 62 was RRF #1 but BGE #8.
                bge_ranked = self.reranker.rerank(
                    query, candidates, top_k=len(candidates)
                )
                reserve = getattr(settings, "pool_rrf_reserve", 4)
                pool = self._blend_pool(candidates, bge_ranked, n, reserve)
            else:
                pool = self.reranker.rerank(query, candidates, top_k=n)
            return self._llm_rerank_vote(query, pool, top_k=top_k)

        unique = self._dedup_by_article(candidates)
        bge_ranked = self.reranker.rerank(query, unique, top_k=top_k)
        if getattr(settings, "enable_llm_rerank", True):
            bge_ranked = self._llm_rerank(query, bge_ranked, top_k=top_k)
        return bge_ranked

    def _exact_match(self, refs: list[str]) -> list[dict]:
        results: list[dict] = []
        for ref in refs:
            hits, _ = self.qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="article_id", match=MatchValue(value=ref)),
                        FieldCondition(key="granularity", match=MatchValue(value="large")),
                        FieldCondition(key="type", match=MatchValue(value="chunk")),
                    ]
                ),
                limit=1,
            )
            for h in hits:
                results.append({**(h.payload or {}), "score": 1.0, "source": "exact"})
        return results

    def _dense_search(self, query: str, top_k: int = 20) -> list[dict]:
        vec = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        response = self.qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=vec,
            limit=top_k,
            score_threshold=0.3,
        )
        return [
            {**(p.payload or {}), "score": p.score, "source": "dense"}
            for p in response.points
        ]

    def _bm25_search(self, query: str) -> list[dict]:
        hits = self.bm25.search(query, top_k=20)
        for h in hits:
            h["source"] = "bm25"
        return hits

    async def _get_rag(self):
        """Lazily initialise one LightRAG instance and reuse it across queries."""
        if self._rag is not None:
            return self._rag
        async with self._rag_lock:
            if self._rag is None:
                from ..clients import build_lightrag

                # Must be wired with the same embedding/LLM funcs used to build
                # the graph, or querying raises "embedding_func is required".
                rag = build_lightrag(settings.lightrag_working_dir)
                await rag.initialize_storages()
                self._rag = rag
        return self._rag

    async def _graph_search(self, query: str) -> list[dict]:
        try:
            from lightrag import QueryParam

            rag = await self._get_rag()
            # only_need_context=True returns the assembled graph context WITHOUT a
            # final LLM answer-generation. We only need the text to regex article
            # refs out of, so skipping generation removes a full decode per query
            # (the graph source was the most expensive retriever).
            # enable_rerank=False: LightRAG's internal rerank defaults to on, but
            # we never configure a rerank_model_func for it (we run our own BGE +
            # LLM rerank downstream on the merged candidates). Without this flag
            # LightRAG logs "Rerank is enabled but no rerank model is configured"
            # on every query and the step is a no-op anyway.
            result = await rag.aquery(
                query,
                param=QueryParam(
                    mode=getattr(settings, "lightrag_mode", "mix"),
                    only_need_context=True,
                    enable_rerank=False,
                ),
            )
            refs = self.matcher.extract_refs(result)
            return [
                {"article_id": ref, "content_raw": result[:500], "score": 0.7, "source": "graph"}
                for ref in refs[:5]
            ]
        except Exception as exc:  # noqa: BLE001 — graph search is optional signal
            log.warning("graph_search_failed", error=str(exc))
            return []

    def _xref_expand(self, docs: list[dict], max_refs: int = 5) -> list[dict]:
        """Return new candidate docs for provisions referenced inside ``docs``.

        For each doc, extract Article/Annex refs from its ``content_raw`` text
        via ``self.matcher.extract_refs``. For each ref whose ``article_id`` is
        not already present among the input docs (and not already collected),
        fetch it with ``self._exact_match`` and tag hits ``source="xref"``,
        ``score=0.6``. Results are deduped by ``article_id`` and capped at
        ``max_refs``. The input ``docs`` list is never mutated.
        """
        if max_refs <= 0:
            return []

        existing_ids: set[str] = {d.get("article_id", "") for d in docs}
        collected: list[dict] = []
        seen_new: set[str] = set()

        for doc in docs:
            text = doc.get("content_raw", "")
            for ref in self.matcher.extract_refs(text):
                if ref in existing_ids or ref in seen_new:
                    continue
                seen_new.add(ref)
                for hit in self._exact_match([ref]):
                    hit = {**hit, "source": "xref", "score": 0.6}
                    collected.append(hit)
                if len(collected) >= max_refs:
                    return collected[:max_refs]

        return collected[:max_refs]

    def _rrf_merge(
        self,
        exact: list[dict],
        dense: list[dict],
        bm25: list[dict],
        graph: list[dict],
        xref: list[dict] | None = None,
    ) -> list[dict]:
        k = settings.rrf_k
        weights = {
            "exact": settings.rrf_weight_exact,
            "dense": settings.rrf_weight_dense,
            "bm25": settings.rrf_weight_bm25,
            "graph": settings.rrf_weight_graph,
            "xref": getattr(settings, "rrf_weight_xref", 0.5),
        }
        scores: dict[str, dict] = {}
        xref_list: list[dict] = xref if xref is not None else []

        for source_list in (exact, dense, bm25, graph, xref_list):
            for rank, doc in enumerate(source_list):
                aid = doc.get("article_id") or doc.get("content_raw", "")[:50]
                w = weights.get(doc.get("source", ""), 0.5)
                rrf = w / (k + rank + 1)
                if aid in scores:
                    scores[aid]["rrf"] += rrf
                else:
                    scores[aid] = {"doc": doc, "rrf": rrf}

        ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
        return [item["doc"] for item in ranked]

    # ------------------------------------------------------------------
    # LLM-based reasoning rerank
    # ------------------------------------------------------------------

    def _get_llm_client(self) -> Any:
        """Return the injectable LLM client, creating it lazily if needed.

        Import of OpenAI is deferred to call-time so the module can be
        imported in offline tests without network or torch dependencies.
        """
        if self.llm_rerank_client is not None:
            return self.llm_rerank_client
        from openai import OpenAI  # lazy import — not at module level

        self.llm_rerank_client = OpenAI(
            base_url=settings.sglang_base_url,
            api_key="none",
        )
        return self.llm_rerank_client

    @staticmethod
    def _build_rerank_prompt(query: str, docs: list[dict]) -> str:
        """Return a prompt asking the LLM to rank docs by direct-answer relevance.

        Snippets are intentionally large (see _LLM_RERANK_SNIPPET_CHARS): the
        rule that answers the question is often buried several paragraphs into
        the article (e.g. Art 5(1)(f) emotion-recognition prohibition; Art
        50(4) deepfake disclosure), so 300-char excerpts misled the LLM into
        picking topically-adjacent but non-normative provisions.
        """
        lines = [
            "You are a legal-citation selector for the EU AI Act.",
            "Given the QUESTION and a numbered list of PROVISIONS, rank the",
            "provisions from the one that MOST DIRECTLY ANSWERS the question to",
            "the one that least directly answers it. A provision that merely",
            "mentions the same topic is LESS relevant than one stating the actual",
            "rule, obligation, prohibition, or definition the question asks about.",
            "Output ONLY the provision numbers, comma-separated (e.g. '3,1,4,2').",
            "Do NOT explain.",
            "",
            f"QUESTION: {query}",
            "",
            "PROVISIONS:",
        ]
        for i, doc in enumerate(docs, start=1):
            aid = doc.get("article_id", f"doc-{i}")
            text = doc.get("content_raw") or ""
            snippet_chars = getattr(
                settings, "rerank_llm_snippet_chars", _LLM_RERANK_SNIPPET_CHARS
            )
            snippet = _adaptive_snippet(text, query, snippet_chars).replace("\n", " ")
            lines.append(f"{i}. [{aid}] {snippet}")
        lines.append("")
        lines.append("Direct-answer ranking (most directly first):")
        return "\n".join(lines)

    def _llm_rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """Re-order *docs* by asking the LLM which provisions actually answer *query*.

        On any exception — network error, malformed response, timeout — this
        method falls back to returning *docs* unchanged (BGE order is preserved).
        Only the first ``_LLM_RERANK_MAX_CANDIDATES`` docs are sent to keep the
        prompt small; any tail beyond that is appended after the reranked head.
        """
        if not docs:
            return docs

        head = docs[:_LLM_RERANK_MAX_CANDIDATES]
        tail = docs[_LLM_RERANK_MAX_CANDIDATES:]

        try:
            client = self._get_llm_client()
            prompt = self._build_rerank_prompt(query, head)
            resp = client.chat.completions.create(
                model=getattr(settings, "utility_model", "default"),
                temperature=0,
                max_tokens=_LLM_RERANK_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                extra_body=getattr(settings, "llm_extra_body", None),
            )
            raw = (resp.choices[0].message.content or "").strip()
            reordered = self._parse_rerank_response(raw, head)
            log.info("llm_rerank_applied", query=query[:80], ranking=raw[:80])
            return (reordered + tail)[:top_k]
        except Exception as exc:  # noqa: BLE001 — rerank is best-effort
            log.warning("llm_rerank_failed", error=str(exc))
            return docs

    @staticmethod
    def _parse_rerank_response(raw: str, docs: list[dict]) -> list[dict]:
        """Parse a comma-separated 1-based index list into a reordered doc list.

        Any index that is out-of-range or non-numeric is ignored. Docs not
        referenced by the model are appended at the end (preserving BGE order
        as a fallback for unranked items).
        """
        seen: set[int] = set()
        reordered: list[dict] = []

        for token in raw.replace(";", ",").split(","):
            token = token.strip().strip(".")
            try:
                idx = int(token) - 1  # convert 1-based to 0-based
            except ValueError:
                continue
            if 0 <= idx < len(docs) and idx not in seen:
                seen.add(idx)
                reordered.append(docs[idx])

        # Append any docs the model did not mention (preserve BGE fallback order).
        for i, doc in enumerate(docs):
            if i not in seen:
                reordered.append(doc)

        return reordered

    # ------------------------------------------------------------------
    # Multi-chunk voting rerank
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_by_article(docs: list[dict]) -> list[dict]:
        """Keep the first doc per ``article_id``, preserving input order."""
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in docs:
            aid = doc.get("article_id", "")
            if not aid or aid in seen:
                continue
            seen.add(aid)
            unique.append(doc)
        return unique

    @staticmethod
    def _blend_pool(
        rrf_ordered: list[dict],
        bge_ordered: list[dict],
        n: int,
        rrf_reserve: int,
    ) -> list[dict]:
        """Select an ``n``-chunk candidate pool that protects strong-RRF docs.

        ``rrf_ordered`` is the fused-candidate list (best RRF first, as returned
        by ``_rrf_merge``); ``bge_ordered`` is the same docs re-sorted by the BGE
        cross-encoder. The first ``rrf_reserve`` docs in RRF order are reserved
        into the pool unconditionally, then the remaining slots are filled in BGE
        order (skipping anything already reserved), up to ``n``.

        This fixes the pure-BGE pre-filter dropping a candidate the cross-encoder
        under-scores even though every other retriever ranked it first (comp_27:
        Art 62 was RRF #1 but BGE #8, so its chunks never reached the LLM vote).
        Identity is by ``id`` because BGE reranking returns the same dict objects.
        """
        pool: list[dict] = []
        seen: set[int] = set()
        for doc in rrf_ordered[: max(0, rrf_reserve)]:
            if id(doc) not in seen:
                seen.add(id(doc))
                pool.append(doc)
                if len(pool) >= n:
                    return pool
        for doc in bge_ordered:
            if id(doc) not in seen:
                seen.add(id(doc))
                pool.append(doc)
                if len(pool) >= n:
                    return pool
        return pool

    @staticmethod
    def _aggregate_chunk_votes(
        ranked_chunks: list[dict], k: int, top_k: int
    ) -> list[dict]:
        """Aggregate LLM-ranked chunks into a per-article ranking by voting.

        ``ranked_chunks`` is the LLM-ordered candidate list (best first), which
        may contain several chunks of the same article. Each chunk at 0-based
        position ``rank`` contributes ``1 / (k + rank + 1)`` to its article's
        score (RRF-style positional vote). An article is represented in the
        output by its earliest-ranked chunk, and articles are returned best
        first, capped at ``top_k``. Chunks without an ``article_id`` are skipped.

        This is the fix for "pre-rerank dedup kept the wrong chunk of the right
        article": an article now wins when *any* of its chunks ranks well, and
        several mediocre chunks can combine to beat a single strong-but-
        irrelevant one.
        """
        agg: dict[str, dict] = {}
        for rank, chunk in enumerate(ranked_chunks):
            aid = chunk.get("article_id", "")
            if not aid:
                continue
            vote = 1.0 / (k + rank + 1)
            entry = agg.get(aid)
            if entry is None:
                # First (earliest-ranked) chunk represents the article.
                agg[aid] = {"doc": {**chunk, "vote_score": vote}, "score": vote}
            else:
                entry["score"] += vote
                entry["doc"]["vote_score"] = entry["score"]

        ranked = sorted(agg.values(), key=lambda e: e["score"], reverse=True)
        return [e["doc"] for e in ranked[:top_k]]

    def _llm_rerank_vote(
        self, query: str, chunks: list[dict], top_k: int
    ) -> list[dict]:
        """LLM-rerank an *un-deduplicated* chunk pool, then vote per article.

        Unlike ``_llm_rerank`` (which reorders one chunk per article), this sends
        up to ``settings.rerank_candidate_chunks`` raw chunks — several may share
        an article — and aggregates the LLM ranking into a per-article result via
        ``_aggregate_chunk_votes``. On any error it falls back to a plain
        dedup-by-article of the input order.
        """
        if not chunks:
            return chunks

        max_chunks = getattr(settings, "rerank_candidate_chunks", 12)
        head = chunks[:max_chunks]

        try:
            client = self._get_llm_client()
            prompt = self._build_rerank_prompt(query, head)
            resp = client.chat.completions.create(
                model=getattr(settings, "utility_model", "default"),
                temperature=0,
                max_tokens=_LLM_RERANK_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                extra_body=getattr(settings, "llm_extra_body", None),
            )
            raw = (resp.choices[0].message.content or "").strip()
            ranked = self._parse_rerank_response(raw, head)
            log.info("llm_rerank_vote_applied", query=query[:80], ranking=raw[:80])
            return self._aggregate_chunk_votes(
                ranked, k=settings.rrf_k, top_k=top_k
            )
        except Exception as exc:  # noqa: BLE001 — rerank is best-effort
            log.warning("llm_rerank_vote_failed", error=str(exc))
            return self._dedup_by_article(chunks)[:top_k]
