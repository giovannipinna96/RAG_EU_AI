"""Offline unit tests for SemanticCache get/set logic.

These tests validate:
  (a) Exact normalised-string match bypasses the embedding model entirely.
  (b) A high-similarity entry with a DIFFERENT ref-set is NOT returned
      (the comp_2 Article-5 vs Article-6 false-positive scenario).
  (c) A matching ref-set + high similarity IS returned.
  (d) Queries with no Article/Annex refs still work at the new 0.97 threshold.
  (e) cache_require_ref_match=False disables the ref guard.

The SentenceTransformer, FlagEmbedding, bm25s, and redis modules are all
stubbed before any src import so that no torch or ML code is loaded.
"""

from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock

# ---------------------------------------------------------------------------
# Stub heavy ML/Redis dependencies BEFORE any src import.
# Pattern mirrors tests/test_xref_expansion.py.
# ---------------------------------------------------------------------------


def _inject_stub(name: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            sys.modules[key] = types.ModuleType(key)
    return sys.modules[name]


_st_stub = _inject_stub("sentence_transformers")
_st_stub.SentenceTransformer = mock.MagicMock()  # type: ignore[attr-defined]

_fe_stub = _inject_stub("FlagEmbedding")
_fe_stub.FlagReranker = mock.MagicMock()  # type: ignore[attr-defined]

_bm25s_stub = _inject_stub("bm25s")
_bm25s_stub.BM25 = mock.MagicMock()  # type: ignore[attr-defined]
_bm25s_stub.tokenize = mock.MagicMock(return_value=[])  # type: ignore[attr-defined]

_stemmer_stub = _inject_stub("Stemmer")
_stemmer_stub.Stemmer = mock.MagicMock()  # type: ignore[attr-defined]

_lightrag_stub = _inject_stub("lightrag")

# Stub redis so SemanticCache._connect() falls back to _InMemoryBackend without
# trying to reach a real Redis server.
_redis_stub = _inject_stub("redis")
_redis_stub.from_url = mock.MagicMock(side_effect=ConnectionError("no redis"))  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Now safe to import src modules.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402  (after stubs)

from src.cache.semantic_cache import (  # noqa: E402
    SemanticCache,
    _InMemoryBackend,
    _extract_refs_safe,
    _normalise_query,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTICLE_5_QUERY = "Are AI systems intended for emotion recognition from biometric data always prohibited?"
ARTICLE_6_QUERY = "Which high-risk AI systems fall under Annex III categories?"

RESPONSE_ART5 = {"answer": "Article 5 answer", "refs": ["Article 5"]}
RESPONSE_ART6 = {"answer": "Article 6 answer", "refs": ["Article 6", "Annex III"]}


def _unit_vec(dim: int = 4, seed: int = 0) -> np.ndarray:
    """Return a deterministic unit-length vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_cache(
    *,
    threshold: float = 0.97,
    require_ref_match: bool = True,
) -> SemanticCache:
    """Build a SemanticCache backed by _InMemoryBackend with controlled settings."""
    cache = SemanticCache.__new__(SemanticCache)
    cache.threshold = threshold
    cache.require_ref_match = require_ref_match
    cache.ttl = 3600
    cache._model = None
    cache.backend = _InMemoryBackend()
    return cache


def _prime_semantic_entry(
    cache: SemanticCache,
    query: str,
    response: dict,
    vec: np.ndarray,
    refs: list[str],
) -> None:
    """Write a semantic entry directly into the backend, bypassing model.encode."""
    import hashlib

    key = cache._hash(query, [])
    cache.backend.setex(
        f"sem:{key}",
        cache.ttl,
        json.dumps({"vec": vec.tolist(), "refs": refs, "response": response}),
    )


def _inject_model(cache: SemanticCache, encode_fn) -> None:
    """Attach a stub model that calls *encode_fn* for .encode()."""
    stub = mock.MagicMock()
    stub.encode.side_effect = encode_fn
    cache._model = stub


# ---------------------------------------------------------------------------
# (a) Exact normalised-string match bypasses the embedding model
# ---------------------------------------------------------------------------


def test_exact_norm_hit_returns_response_without_model():
    """A query stored and retrieved via the normalised-exact layer never calls model.encode."""
    cache = _make_cache()
    encode_called = []

    def _encode(text, **kw):
        encode_called.append(text)
        return _unit_vec()

    _inject_model(cache, _encode)

    cache.set(ARTICLE_5_QUERY, [], RESPONSE_ART5)
    # Reset the call tracker — set() calls encode; we care about get() only.
    encode_called.clear()

    result = cache.get(ARTICLE_5_QUERY, [])

    assert result == RESPONSE_ART5
    assert encode_called == [], "model.encode must NOT be called on an exact-norm hit"


def test_exact_norm_hit_case_insensitive():
    """Normalised exact match is case-insensitive."""
    cache = _make_cache()
    _inject_model(cache, lambda text, **kw: _unit_vec())

    cache.set(ARTICLE_5_QUERY, [], RESPONSE_ART5)
    result = cache.get(ARTICLE_5_QUERY.upper(), [])
    assert result == RESPONSE_ART5


def test_exact_norm_hit_whitespace_insensitive():
    """Normalised exact match collapses extra whitespace."""
    cache = _make_cache()
    _inject_model(cache, lambda text, **kw: _unit_vec())

    cache.set(ARTICLE_5_QUERY, [], RESPONSE_ART5)
    padded = "  " + ARTICLE_5_QUERY.replace(" ", "   ") + "  "
    result = cache.get(padded, [])
    assert result == RESPONSE_ART5


# ---------------------------------------------------------------------------
# (b) Different ref-set BLOCKS a high-similarity semantic hit (comp_2 scenario)
# ---------------------------------------------------------------------------


def test_different_ref_set_blocks_high_similarity_hit():
    """A semantic entry with refs=['Article 6','Annex III'] must NOT be returned
    for a query whose extracted refs are ['Article 5'], even at similarity=1.0.

    This is the comp_2 false-positive scenario: emotion-recognition question
    (Article 5) colliding with a cached Article-6/Annex-III answer.
    """
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    base_vec = _unit_vec(seed=1)

    # Store an Article-6/Annex-III answer with a near-identical vector.
    _prime_semantic_entry(cache, ARTICLE_6_QUERY, RESPONSE_ART6, base_vec, ["Article 6", "Annex III"])

    # Query has refs=['Article 5']; model returns the *same* vector (sim=1.0).
    _inject_model(cache, lambda text, **kw: base_vec)

    result = cache.get("Are AI systems for emotion recognition prohibited under Article 5?", [])

    assert result is None, (
        "A cached Article-6/Annex-III entry must not be returned for an Article-5 query "
        "even when cosine similarity is 1.0 (ref-set guard must block it)."
    )


def test_different_ref_set_blocks_when_one_ref_differs():
    """Even one differing ref in the set is enough to block the hit."""
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    base_vec = _unit_vec(seed=2)
    _prime_semantic_entry(cache, "q1", RESPONSE_ART6, base_vec, ["Article 5", "Article 6"])

    # Query has only ['Article 5'] — not equal to ['Article 5','Article 6'].
    _inject_model(cache, lambda text, **kw: base_vec)

    result = cache.get("A question referencing only Article 5.", [])
    assert result is None


# ---------------------------------------------------------------------------
# (c) Matching ref-set + high similarity IS returned
# ---------------------------------------------------------------------------


def test_matching_ref_set_and_high_sim_returns_hit():
    """When refs match AND sim >= threshold, the cached entry is returned."""
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    base_vec = _unit_vec(seed=3)
    _prime_semantic_entry(
        cache,
        "Are prohibited AI systems listed under Article 5?",
        RESPONSE_ART5,
        base_vec,
        ["Article 5"],
    )

    # Slightly different phrasing but same Article-5 ref and near-identical vector.
    _inject_model(cache, lambda text, **kw: base_vec)  # sim = 1.0

    result = cache.get("Which practices are prohibited under Article 5?", [])
    assert result == RESPONSE_ART5


def test_matching_ref_set_low_sim_is_not_returned():
    """Even with matching refs, sim < threshold must not return a hit."""
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    stored_vec = _unit_vec(seed=4)
    _prime_semantic_entry(cache, "q", RESPONSE_ART5, stored_vec, ["Article 5"])

    # Query vector is orthogonal to stored vector → sim ≈ 0.
    query_vec = _unit_vec(seed=99)
    # Force orthogonality by nullifying the projection.
    query_vec = query_vec - np.dot(query_vec, stored_vec) * stored_vec
    query_vec = query_vec / np.linalg.norm(query_vec)

    _inject_model(cache, lambda text, **kw: query_vec)

    result = cache.get("Completely different question about Article 5.", [])
    assert result is None


# ---------------------------------------------------------------------------
# (d) No-ref queries still work at the 0.97 threshold
# ---------------------------------------------------------------------------


def test_no_ref_query_returns_high_sim_hit():
    """A query with no Article/Annex refs bypasses the ref guard and uses
    only the cosine threshold (0.97)."""
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    base_vec = _unit_vec(seed=5)
    _prime_semantic_entry(cache, "q_no_ref", {"answer": "general"}, base_vec, [])

    # Query also has no refs (plain English, no "Article N").
    _inject_model(cache, lambda text, **kw: base_vec)  # sim=1.0

    result = cache.get("What is the purpose of this regulation?", [])
    assert result == {"answer": "general"}


def test_no_ref_query_below_threshold_returns_none():
    """No-ref query with sim < threshold is not returned."""
    cache = _make_cache(threshold=0.97, require_ref_match=True)

    stored_vec = _unit_vec(seed=6)
    _prime_semantic_entry(cache, "q_no_ref", {"answer": "general"}, stored_vec, [])

    query_vec = _unit_vec(seed=77)
    query_vec = query_vec - np.dot(query_vec, stored_vec) * stored_vec
    query_vec = query_vec / np.linalg.norm(query_vec)

    _inject_model(cache, lambda text, **kw: query_vec)

    result = cache.get("Unrelated question with no article mentions.", [])
    assert result is None


# ---------------------------------------------------------------------------
# (e) cache_require_ref_match=False disables ref guard
# ---------------------------------------------------------------------------


def test_ref_guard_disabled_allows_different_ref_set_at_high_sim():
    """When require_ref_match=False, a different ref-set does not block the hit."""
    cache = _make_cache(threshold=0.97, require_ref_match=False)

    base_vec = _unit_vec(seed=7)
    _prime_semantic_entry(cache, "q", RESPONSE_ART6, base_vec, ["Article 6"])

    _inject_model(cache, lambda text, **kw: base_vec)  # sim=1.0

    result = cache.get("Article 5 question that matches at high sim.", [])
    # Without ref guard, the high-sim entry IS returned.
    assert result == RESPONSE_ART6


# ---------------------------------------------------------------------------
# (f) _normalise_query helper
# ---------------------------------------------------------------------------


def test_normalise_query_lowercases():
    assert _normalise_query("Article 5") == "article 5"


def test_normalise_query_collapses_whitespace():
    assert _normalise_query("  hello   world  ") == "hello world"


def test_normalise_query_empty_string():
    assert _normalise_query("") == ""


# ---------------------------------------------------------------------------
# (g) _extract_refs_safe does not raise on bad input
# ---------------------------------------------------------------------------


def test_extract_refs_safe_returns_list_for_normal_text():
    refs = _extract_refs_safe("See Article 5 and Annex III for details.")
    assert "Article 5" in refs
    assert "Annex III" in refs


def test_extract_refs_safe_returns_empty_for_no_refs():
    refs = _extract_refs_safe("This text has no article references at all.")
    assert refs == []


def test_extract_refs_safe_does_not_raise_on_empty_string():
    refs = _extract_refs_safe("")
    assert isinstance(refs, list)


# ---------------------------------------------------------------------------
# (h) invalidate_all clears norm: entries too
# ---------------------------------------------------------------------------


def test_invalidate_all_clears_norm_entries():
    cache = _make_cache()
    _inject_model(cache, lambda text, **kw: _unit_vec())

    cache.set(ARTICLE_5_QUERY, [], RESPONSE_ART5)
    cleared = cache.invalidate_all()
    assert cleared >= 1

    # After invalidation, exact-norm lookup returns nothing.
    _inject_model(cache, lambda text, **kw: _unit_vec())
    result = cache.get(ARTICLE_5_QUERY, [])
    assert result is None
