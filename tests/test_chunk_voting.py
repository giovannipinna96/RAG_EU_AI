"""Tests for multi-chunk voting rerank.

Two units under test:

  * ``TripleRetriever._aggregate_chunk_votes`` -- pure positional-vote
    aggregation. Given chunks in LLM-ranked order, it sums 1/(k+rank) per
    ``article_id`` and returns one representative doc (the earliest-ranked
    chunk) per article, best article first. This is what lets an article win
    when several of its chunks rank well, instead of being starved by a
    pre-rerank dedup that kept the wrong chunk.

  * ``TripleRetriever._llm_rerank_vote`` -- wires the LLM call (injected stub)
    to the aggregator, with a dedup fallback on any error.

Offline: ML deps are stubbed before importing the retriever.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest import mock


def _inject_stub(name: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            sys.modules[key] = types.ModuleType(key)
    return sys.modules[name]


_bm25s_stub = _inject_stub("bm25s")
_bm25s_stub.BM25 = mock.MagicMock()  # type: ignore[attr-defined]
_bm25s_stub.tokenize = mock.MagicMock(return_value=[])  # type: ignore[attr-defined]
_stemmer_stub = _inject_stub("Stemmer")
_stemmer_stub.Stemmer = mock.MagicMock()  # type: ignore[attr-defined]
_st_stub = _inject_stub("sentence_transformers")
_st_stub.SentenceTransformer = mock.MagicMock()  # type: ignore[attr-defined]
_fe_stub = _inject_stub("FlagEmbedding")
_fe_stub.FlagReranker = mock.MagicMock()  # type: ignore[attr-defined]
_inject_stub("lightrag")

from src.retrieval.triple_retriever import TripleRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(article_id: str, content: str = "") -> dict:
    return {"article_id": article_id, "content_raw": content, "score": 0.8}


def _stub_llm_client(response_text: str) -> SimpleNamespace:
    msg = SimpleNamespace(content=response_text)
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=lambda **kw: response)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _raising_llm_client() -> SimpleNamespace:
    def _fail(**kw):
        raise RuntimeError("simulated LLM failure")

    completions = SimpleNamespace(create=_fail)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _make_retriever(llm_client=None) -> TripleRetriever:
    r = TripleRetriever.__new__(TripleRetriever)
    r.llm_rerank_client = llm_client
    return r


# ---------------------------------------------------------------------------
# _aggregate_chunk_votes -- pure aggregation
# ---------------------------------------------------------------------------


def test_aggregate_empty_returns_empty():
    assert TripleRetriever._aggregate_chunk_votes([], k=10, top_k=5) == []


def test_aggregate_one_doc_per_article():
    ranked = [
        _chunk("Article 50", "para 1"),
        _chunk("Article 19", "logs"),
        _chunk("Article 50", "para 4"),
    ]
    out = TripleRetriever._aggregate_chunk_votes(ranked, k=10, top_k=5)
    assert [d["article_id"] for d in out] == ["Article 50", "Article 19"]
    assert len({d["article_id"] for d in out}) == 2


def test_aggregate_representative_is_earliest_ranked_chunk():
    # Article 50 appears at rank 0 (deepfake para) and rank 2 (chatbots para).
    # The earliest-ranked chunk represents the article in the output.
    ranked = [
        _chunk("Article 50", "deepfake disclosure para"),
        _chunk("Article 19", "logs"),
        _chunk("Article 50", "chatbots para"),
    ]
    out = TripleRetriever._aggregate_chunk_votes(ranked, k=10, top_k=5)
    art50 = next(d for d in out if d["article_id"] == "Article 50")
    assert art50["content_raw"] == "deepfake disclosure para"


def test_aggregate_combined_votes_beat_single_top_chunk():
    # Article Y owns the single best chunk (rank 0), but Article X has two
    # mid-ranked chunks whose votes combine to outrank it.
    ranked = [
        _chunk("Article Y", "y0"),
        _chunk("Article X", "x1"),
        _chunk("Article X", "x2"),
    ]
    out = TripleRetriever._aggregate_chunk_votes(ranked, k=1, top_k=5)
    # Y = 1/(1+1) = 0.5 ; X = 1/(1+2) + 1/(1+3) = 0.583 > 0.5
    assert out[0]["article_id"] == "Article X"
    assert out[1]["article_id"] == "Article Y"


def test_aggregate_respects_top_k():
    ranked = [_chunk(f"Article {i}") for i in range(10)]
    out = TripleRetriever._aggregate_chunk_votes(ranked, k=10, top_k=3)
    assert len(out) == 3


def test_aggregate_skips_chunks_without_article_id():
    ranked = [
        {"article_id": "", "content_raw": "orphan"},
        _chunk("Article 5", "real"),
    ]
    out = TripleRetriever._aggregate_chunk_votes(ranked, k=10, top_k=5)
    assert [d["article_id"] for d in out] == ["Article 5"]


# ---------------------------------------------------------------------------
# _llm_rerank_vote -- LLM call + aggregation, with fallback
# ---------------------------------------------------------------------------


def test_vote_recovers_article_from_non_top_chunk():
    """The wrong chunk of Article 50 was kept by classic dedup; with voting the
    candidate pool holds BOTH chunks and the LLM ranks the deepfake one first,
    so Article 50 wins and is represented by its deepfake chunk."""
    chunks = [
        _chunk("Article 50", "Providers shall ensure AI systems that interact "
                             "directly with natural persons (chatbots) are disclosed."),
        _chunk("Article 19", "Automatically generated logs."),
        _chunk("Article 50", "Deployers disclosing deep fake content shall mark it "
                             "as artificially generated or manipulated."),
        _chunk("Annex I", "Union harmonisation legislation list."),
    ]
    # LLM ranks chunk 3 (deepfake) first, then chunk 1, then the rest.
    client = _stub_llm_client("3,1,2,4")
    r = _make_retriever(llm_client=client)

    out = r._llm_rerank_vote("deepfake disclosure", chunks, top_k=5)

    assert out[0]["article_id"] == "Article 50"
    assert "deep fake" in out[0]["content_raw"]


def test_vote_falls_back_to_dedup_on_llm_error():
    chunks = [
        _chunk("Article 50", "para a"),
        _chunk("Article 50", "para b"),
        _chunk("Article 19", "logs"),
    ]
    r = _make_retriever(llm_client=_raising_llm_client())

    out = r._llm_rerank_vote("q", chunks, top_k=5)

    # Fallback: distinct articles preserving input order, no crash.
    assert [d["article_id"] for d in out] == ["Article 50", "Article 19"]


def test_vote_empty_returns_empty():
    r = _make_retriever(llm_client=_stub_llm_client("1"))
    assert r._llm_rerank_vote("q", [], top_k=5) == []


# ---------------------------------------------------------------------------
# _blend_pool -- pool selection that protects strong-RRF candidates
# ---------------------------------------------------------------------------


def test_blend_pool_reserves_strong_rrf():
    """A candidate that is RRF #1 but ranked last by BGE must still enter the
    pool: the blend reserves the top RRF slots before filling by BGE.

    This is the comp_27 fix: Art 62 was RRF #1 yet the BGE cross-encoder buried
    it, so the pure-BGE pre-filter dropped it before the LLM vote could see it.
    """
    rrf = [_chunk(f"Article {i}") for i in range(6)]
    gold = rrf[0]
    bge = rrf[1:] + [gold]  # BGE puts the strong-RRF doc last
    out = TripleRetriever._blend_pool(rrf, bge, n=3, rrf_reserve=1)
    assert gold in out
    assert out[0] is gold  # reserved RRF candidate leads the pool


def test_blend_pool_fills_remaining_by_bge():
    rrf = [_chunk(f"Article {i}") for i in range(6)]
    bge = list(reversed(rrf))
    out = TripleRetriever._blend_pool(rrf, bge, n=4, rrf_reserve=1)
    # reserved: rrf[0]; remaining slots filled in BGE order (skipping rrf[0]).
    assert out[0] is rrf[0]
    assert out[1] is bge[0]


def test_blend_pool_respects_n():
    rrf = [_chunk(f"Article {i}") for i in range(10)]
    bge = list(reversed(rrf))
    out = TripleRetriever._blend_pool(rrf, bge, n=5, rrf_reserve=2)
    assert len(out) == 5


def test_blend_pool_no_duplicates():
    rrf = [_chunk(f"Article {i}") for i in range(6)]
    bge = rrf  # reserved docs are also BGE-top — must not be added twice
    out = TripleRetriever._blend_pool(rrf, bge, n=4, rrf_reserve=2)
    assert len(out) == 4
    assert len({id(d) for d in out}) == 4


def test_blend_pool_reserve_zero_is_pure_bge():
    rrf = [_chunk(f"Article {i}") for i in range(6)]
    bge = list(reversed(rrf))
    out = TripleRetriever._blend_pool(rrf, bge, n=3, rrf_reserve=0)
    assert out == bge[:3]


# ---------------------------------------------------------------------------
# _final_rerank -- the stage selector wired into retrieve()
# ---------------------------------------------------------------------------


def _make_retriever_with_reranker(llm_client, rerank_fn) -> TripleRetriever:
    r = TripleRetriever.__new__(TripleRetriever)
    r.llm_rerank_client = llm_client
    r.reranker = SimpleNamespace(rerank=rerank_fn)
    return r


def test_final_rerank_voting_path_recovers_article():
    """Flag on: candidates are NOT deduped before rerank, so both Article 50
    chunks reach the LLM and voting surfaces the deepfake one."""
    candidates = [
        _chunk("Article 50", "chatbots disclosure para"),
        _chunk("Article 19", "logs"),
        _chunk("Article 50", "deep fake artificially generated para"),
        _chunk("Annex I", "harmonisation list"),
    ]
    # BGE pre-filter: identity (return up to top_k as-is).
    rerank_fn = lambda query, docs, top_k: docs[:top_k]  # noqa: E731
    client = _stub_llm_client("3,1,2,4")  # LLM ranks the deepfake chunk first
    r = _make_retriever_with_reranker(client, rerank_fn)

    fake_settings = SimpleNamespace(
        enable_chunk_voting=True,
        enable_llm_rerank=True,
        rerank_candidate_chunks=12,
        rrf_k=60,
        utility_model="default",
    )
    with mock.patch("src.retrieval.triple_retriever.settings", fake_settings):
        out = r._final_rerank("deepfake disclosure", candidates, top_k=5)

    assert out[0]["article_id"] == "Article 50"
    assert "deep fake" in out[0]["content_raw"]


def test_final_rerank_classic_path_dedups_then_llm_reranks():
    """Flag off: candidates are deduped by article, BGE-reranked, then LLM
    reranked (existing behavior)."""
    candidates = [
        _chunk("Article 6", "high-risk classification"),
        _chunk("Article 6", "duplicate chunk of art 6"),
        _chunk("Article 50", "deepfake disclosure"),
    ]
    rerank_fn = lambda query, docs, top_k: docs[:top_k]  # noqa: E731
    client = _stub_llm_client("2,1")  # LLM promotes the 2nd deduped doc
    r = _make_retriever_with_reranker(client, rerank_fn)

    fake_settings = SimpleNamespace(
        enable_chunk_voting=False,
        enable_llm_rerank=True,
        rerank_candidate_chunks=12,
        rrf_k=60,
        utility_model="default",
    )
    with mock.patch("src.retrieval.triple_retriever.settings", fake_settings):
        out = r._final_rerank("q", candidates, top_k=5)

    # Deduped to [Article 6, Article 50]; LLM "2,1" promotes Article 50.
    assert [d["article_id"] for d in out] == ["Article 50", "Article 6"]


def test_final_rerank_blend_recovers_strong_rrf_article():
    """Voting + pool-blend: an article that is RRF #1 but BGE-buried still
    reaches the LLM vote because the blend reserves top RRF pool slots."""
    art62 = _chunk("Article 62", "measures supporting SMEs and start-ups: priority access")
    candidates = [art62] + [_chunk(f"Article {i}", f"noise {i}") for i in range(1, 12)]

    def rerank_fn(query, docs, top_k):
        # BGE buries Article 62 to the very end.
        ordered = [d for d in docs if d["article_id"] != "Article 62"]
        ordered += [d for d in docs if d["article_id"] == "Article 62"]
        return ordered[:top_k]

    client = _stub_llm_client("1")  # LLM ranks the reserved Article 62 chunk first
    r = _make_retriever_with_reranker(client, rerank_fn)

    fake_settings = SimpleNamespace(
        enable_chunk_voting=True,
        enable_pool_blend=True,
        enable_llm_rerank=True,
        rerank_candidate_chunks=4,
        pool_rrf_reserve=2,
        rrf_k=60,
        utility_model="default",
    )
    with mock.patch("src.retrieval.triple_retriever.settings", fake_settings):
        out = r._final_rerank("SME support", candidates, top_k=5)

    assert out[0]["article_id"] == "Article 62"


def test_final_rerank_pure_bge_pool_drops_strong_rrf_article():
    """Companion to the blend test: with pool-blend OFF, the same BGE-buried
    Article 62 never enters the small pure-BGE pool, so voting cannot recover
    it. This is the regression the blend fixes."""
    art62 = _chunk("Article 62", "measures supporting SMEs and start-ups: priority access")
    candidates = [art62] + [_chunk(f"Article {i}", f"noise {i}") for i in range(1, 12)]

    def rerank_fn(query, docs, top_k):
        ordered = [d for d in docs if d["article_id"] != "Article 62"]
        ordered += [d for d in docs if d["article_id"] == "Article 62"]
        return ordered[:top_k]

    client = _stub_llm_client("1,2,3,4")
    r = _make_retriever_with_reranker(client, rerank_fn)

    fake_settings = SimpleNamespace(
        enable_chunk_voting=True,
        enable_pool_blend=False,
        enable_llm_rerank=True,
        rerank_candidate_chunks=4,
        pool_rrf_reserve=2,
        rrf_k=60,
        utility_model="default",
    )
    with mock.patch("src.retrieval.triple_retriever.settings", fake_settings):
        out = r._final_rerank("SME support", candidates, top_k=5)

    assert "Article 62" not in [d["article_id"] for d in out]
