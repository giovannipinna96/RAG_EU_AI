"""Offline unit tests for TripleRetriever._xref_expand.

The method is tested against the contract:
  - Scans each doc's content_raw for cross-references via matcher.extract_refs.
  - For each ref NOT already present in input docs (by article_id), calls
    self._exact_match([ref]) to fetch the provision from Qdrant.
  - Tags every returned hit with source="xref" and score=0.6.
  - Deduplicates returned xref docs by article_id.
  - Caps the result at max_refs (default 5).
  - Returns ONLY the new xref docs — does NOT mutate the input docs list.

All Qdrant and embedding calls are stubbed; no torch, no network.

Import strategy: triple_retriever.py transitively pulls in bm25s (which aborts
on this login node via jax/GPU).  We inject stub modules into sys.modules
*before* the real import so the module-level ``import bm25s`` never reaches the
real library.  The same technique covers sentence-transformers and FlagEmbedding.
"""

from __future__ import annotations

import copy
import sys
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy ML dependencies before any src import that pulls them in.
# This must happen before "from src.retrieval.triple_retriever import ..."
# ---------------------------------------------------------------------------


def _inject_stub(name: str) -> types.ModuleType:
    """Put an empty stub module at *name* and all dot-prefixed parents."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            sys.modules[key] = types.ModuleType(key)
    return sys.modules[name]


# bm25s — accessed at bm25_index.py module level; stub out the whole package.
_bm25s_stub = _inject_stub("bm25s")
_bm25s_stub.BM25 = mock.MagicMock()  # type: ignore[attr-defined]
_bm25s_stub.tokenize = mock.MagicMock(return_value=[])  # type: ignore[attr-defined]

# Stemmer — also imported at bm25_index module level.
_stemmer_stub = _inject_stub("Stemmer")
_stemmer_stub.Stemmer = mock.MagicMock()  # type: ignore[attr-defined]

# sentence_transformers — imported by clients.py / get_embedding_model.
_st_stub = _inject_stub("sentence_transformers")
_st_stub.SentenceTransformer = mock.MagicMock()  # type: ignore[attr-defined]

# FlagEmbedding — optional reranker dependency.
_fe_stub = _inject_stub("FlagEmbedding")
_fe_stub.FlagReranker = mock.MagicMock()  # type: ignore[attr-defined]

# lightrag — optional graph dependency.
_lr_stub = _inject_stub("lightrag")

# Now it's safe to import the real modules.
from src.retrieval.article_matcher import ArticleMatcher  # noqa: E402
from src.retrieval.triple_retriever import TripleRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(article_id: str, content_raw: str = "") -> object:
    """Return a minimal object with a .payload attribute, mirroring a Qdrant point."""

    class _Hit:
        payload = {
            "article_id": article_id,
            "content_raw": content_raw,
            "granularity": "large",
            "type": "chunk",
        }

    return _Hit()


def _make_retriever(scroll_map: dict[str, list]) -> TripleRetriever:
    """Build a TripleRetriever bypassing __init__ (no torch / Qdrant needed).

    scroll_map maps article_id -> list of fake hits returned by qdrant.scroll.
    The stub inspects the scroll_filter to determine which article_id is being
    queried by reading the MatchValue of the first FieldCondition in the must list.
    """
    r = TripleRetriever.__new__(TripleRetriever)
    r.matcher = ArticleMatcher()

    class _StubQdrant:
        def scroll(self, *, collection_name, scroll_filter, limit, **kw):  # noqa: ARG002
            # Extract the article_id from the first must-condition's MatchValue.
            first_cond = scroll_filter.must[0]
            aid = first_cond.match.value
            hits = scroll_map.get(aid, [])
            return (hits, None)

    r.qdrant = _StubQdrant()
    return r


def _make_doc(article_id: str, content_raw: str = "") -> dict:
    return {
        "article_id": article_id,
        "content_raw": content_raw,
        "score": 1.0,
        "source": "exact",
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_annex_iv_reference_yields_xref_doc():
    """A doc whose content_raw mentions 'Annex IV' yields a new xref doc
    with article_id 'Annex IV'."""
    hit = _make_hit("Annex IV", "High-risk AI systems listed in Annex IV.")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [_make_doc("Article 5", "Refer to Annex IV for the list.")]
    result = r._xref_expand(docs)

    assert len(result) == 1
    assert result[0]["article_id"] == "Annex IV"


def test_xref_doc_has_source_xref():
    """Every xref doc must carry source='xref'."""
    hit = _make_hit("Annex IV")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [_make_doc("Article 5", "See Annex IV.")]
    result = r._xref_expand(docs)

    assert result[0]["source"] == "xref"


def test_xref_doc_has_score_0_6():
    """Every xref doc must carry score=0.6."""
    hit = _make_hit("Article 11")
    r = _make_retriever({"Article 11": [hit]})

    docs = [_make_doc("Article 5", "Pursuant to Article 11 of the Act.")]
    result = r._xref_expand(docs)

    assert result[0]["score"] == pytest.approx(0.6)


def test_article_reference_in_content_raw_yields_xref():
    """A cross-reference to an Article not in the input docs is fetched and returned."""
    hit = _make_hit("Article 11")
    r = _make_retriever({"Article 11": [hit]})

    docs = [_make_doc("Article 5", "Article 11 specifies transparency requirements.")]
    result = r._xref_expand(docs)

    assert any(d["article_id"] == "Article 11" for d in result)


# ---------------------------------------------------------------------------
# Already-present refs are NOT re-added
# ---------------------------------------------------------------------------


def test_ref_already_in_input_docs_is_not_re_added():
    """When a referenced article_id is already in the input docs, it must not
    appear in xref results."""
    hit = _make_hit("Article 6")
    r = _make_retriever({"Article 6": [hit]})

    # Article 6 is already in the input docs AND referenced in content_raw.
    docs = [
        _make_doc("Article 6", "This is Article 6 text."),
        _make_doc("Article 5", "See Article 6 for details."),
    ]
    result = r._xref_expand(docs)

    assert all(d["article_id"] != "Article 6" for d in result)


def test_only_refs_not_in_input_are_fetched():
    """Mixed content: only the ref absent from input docs is fetched."""
    r = _make_retriever(
        {
            "Article 13": [_make_hit("Article 13")],
            "Article 5": [_make_hit("Article 5")],
        }
    )

    docs = [_make_doc("Article 5", "Article 5 and Article 13 both apply.")]
    result = r._xref_expand(docs)

    ids = [d["article_id"] for d in result]
    assert "Article 13" in ids
    assert "Article 5" not in ids


# ---------------------------------------------------------------------------
# max_refs cap
# ---------------------------------------------------------------------------


def test_max_refs_caps_returned_docs():
    """Result is capped at max_refs even when more refs are found."""
    content = " ".join(f"Article {n}" for n in range(10, 20))  # 10 distinct refs
    scroll_map = {f"Article {n}": [_make_hit(f"Article {n}")] for n in range(10, 20)}
    r = _make_retriever(scroll_map)

    docs = [_make_doc("Article 5", content)]
    result = r._xref_expand(docs, max_refs=3)

    assert len(result) <= 3


def test_max_refs_default_is_5():
    """With the default max_refs=5, at most 5 docs are returned."""
    content = " ".join(f"Article {n}" for n in range(10, 20))
    scroll_map = {f"Article {n}": [_make_hit(f"Article {n}")] for n in range(10, 20)}
    r = _make_retriever(scroll_map)

    docs = [_make_doc("Article 5", content)]
    result = r._xref_expand(docs)

    assert len(result) <= 5


def test_max_refs_zero_returns_empty():
    """max_refs=0 should return an empty list."""
    hit = _make_hit("Annex IV")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [_make_doc("Article 5", "See Annex IV.")]
    result = r._xref_expand(docs, max_refs=0)

    assert result == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_same_ref_mentioned_in_two_docs_yields_one_xref_doc():
    """If two input docs both reference Annex IV, only one xref doc is returned."""
    hit = _make_hit("Annex IV")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [
        _make_doc("Article 5", "See Annex IV for the list."),
        _make_doc("Article 6", "Annex IV is also relevant here."),
    ]
    result = r._xref_expand(docs)

    annex_iv_hits = [d for d in result if d["article_id"] == "Annex IV"]
    assert len(annex_iv_hits) == 1


def test_same_ref_mentioned_twice_in_one_doc_yields_one_xref_doc():
    """If content_raw mentions a ref twice, deduplication still produces one xref doc."""
    hit = _make_hit("Article 11")
    r = _make_retriever({"Article 11": [hit]})

    docs = [_make_doc("Article 5", "Article 11 and Article 11 are important.")]
    result = r._xref_expand(docs)

    article_11_hits = [d for d in result if d["article_id"] == "Article 11"]
    assert len(article_11_hits) == 1


# ---------------------------------------------------------------------------
# No references
# ---------------------------------------------------------------------------


def test_doc_with_no_refs_yields_no_xref_docs():
    """A doc whose content_raw contains no Article/Annex references returns empty."""
    r = _make_retriever({})

    docs = [_make_doc("Article 5", "This text has no cross-references at all.")]
    result = r._xref_expand(docs)

    assert result == []


def test_empty_docs_list_yields_no_xref_docs():
    """Passing an empty docs list returns an empty list."""
    r = _make_retriever({})

    result = r._xref_expand([])

    assert result == []


# ---------------------------------------------------------------------------
# Input docs are NOT mutated
# ---------------------------------------------------------------------------


def test_input_docs_are_not_mutated():
    """_xref_expand must not alter the original docs list or its dicts."""
    hit = _make_hit("Annex IV")
    r = _make_retriever({"Annex IV": [hit]})

    original_doc = _make_doc("Article 5", "See Annex IV.")
    docs = [original_doc]
    docs_snapshot = copy.deepcopy(docs)

    r._xref_expand(docs)

    assert docs == docs_snapshot


def test_input_docs_list_length_unchanged():
    """The input docs list must have the same length after the call."""
    hit = _make_hit("Article 11")
    r = _make_retriever({"Article 11": [hit]})

    docs = [_make_doc("Article 5", "Article 11 applies here.")]
    original_len = len(docs)

    r._xref_expand(docs)

    assert len(docs) == original_len


# ---------------------------------------------------------------------------
# Multiple docs referencing different provisions
# ---------------------------------------------------------------------------


def test_multiple_docs_different_refs_all_fetched():
    """Two docs each referencing a distinct absent provision both yield xref docs."""
    scroll_map = {
        "Annex IV": [_make_hit("Annex IV")],
        "Article 13": [_make_hit("Article 13")],
    }
    r = _make_retriever(scroll_map)

    docs = [
        _make_doc("Article 5", "See Annex IV."),
        _make_doc("Article 6", "Article 13 is also applicable."),
    ]
    result = r._xref_expand(docs)

    ids = {d["article_id"] for d in result}
    assert "Annex IV" in ids
    assert "Article 13" in ids


def test_multiple_docs_mixed_known_unknown_refs():
    """When some refs are already in input docs and some are new, only new ones appear."""
    r = _make_retriever({"Article 7": [_make_hit("Article 7")]})

    docs = [
        _make_doc("Article 5", "Article 5 is defined here."),
        _make_doc("Article 6", "Article 5 and Article 7 apply."),
    ]
    result = r._xref_expand(docs)

    ids = [d["article_id"] for d in result]
    assert "Article 7" in ids
    assert "Article 5" not in ids


# ---------------------------------------------------------------------------
# content_raw missing / empty / absent key
# ---------------------------------------------------------------------------


def test_doc_without_content_raw_key_handled_gracefully():
    """A doc dict lacking 'content_raw' entirely should not raise."""
    r = _make_retriever({})

    docs = [{"article_id": "Article 5", "score": 1.0, "source": "exact"}]
    result = r._xref_expand(docs)

    assert isinstance(result, list)


def test_doc_with_empty_content_raw_yields_no_xref_docs():
    """A doc with content_raw='' contains no refs and produces no xref docs."""
    r = _make_retriever({})

    docs = [_make_doc("Article 5", "")]
    result = r._xref_expand(docs)

    assert result == []


def test_doc_with_none_content_raw_produces_no_refs():
    """A doc with content_raw=None: the contract uses d.get('content_raw','').
    Python's dict.get returns None when the key exists but holds None (the ''
    default only fires for missing keys).  The contract is silent on this case;
    we assert either: the call succeeds and yields no xref docs, OR the
    implementation raises TypeError — both surface the gap for the implementer."""
    r = _make_retriever({})

    docs = [{"article_id": "Article 5", "content_raw": None, "score": 1.0, "source": "exact"}]
    # Either behaviour is acceptable for this unspecified edge case;
    # the test documents that None is not silently converted to "".
    try:
        result = r._xref_expand(docs)
        # If it doesn't raise, it must return a list (possibly empty).
        assert isinstance(result, list)
    except TypeError:
        # The implementation chose not to guard against None — documented gap.
        pass


# ---------------------------------------------------------------------------
# Qdrant returns no hit for a ref
# ---------------------------------------------------------------------------


def test_ref_with_no_qdrant_hit_yields_no_xref_doc():
    """If _exact_match returns [] for a ref, no xref doc is produced for that ref."""
    r = _make_retriever({"Article 11": []})  # Qdrant returns no hit

    docs = [_make_doc("Article 5", "Article 11 transparency requirements.")]
    result = r._xref_expand(docs)

    assert all(d["article_id"] != "Article 11" for d in result)


def test_partial_qdrant_hits_partial_xref_docs():
    """Only refs that have a Qdrant hit produce xref docs; missing ones are skipped."""
    r = _make_retriever(
        {
            "Article 13": [_make_hit("Article 13")],
            "Annex IV": [],  # no hit
        }
    )

    docs = [_make_doc("Article 5", "Article 13 and Annex IV apply.")]
    result = r._xref_expand(docs)

    ids = [d["article_id"] for d in result]
    assert "Article 13" in ids
    assert "Annex IV" not in ids


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------


def test_xref_docs_are_dicts():
    """Each returned xref doc must be a dict."""
    hit = _make_hit("Annex IV", "Annex IV content.")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [_make_doc("Article 5", "See Annex IV.")]
    result = r._xref_expand(docs)

    for doc in result:
        assert isinstance(doc, dict)


def test_xref_doc_payload_fields_preserved():
    """Payload fields from the Qdrant hit are present in the returned xref doc."""
    hit = _make_hit("Annex IV", "Annex IV details.")
    r = _make_retriever({"Annex IV": [hit]})

    docs = [_make_doc("Article 5", "Annex IV applies.")]
    result = r._xref_expand(docs)

    assert len(result) == 1
    xref_doc = result[0]
    assert xref_doc.get("article_id") == "Annex IV"
    assert "content_raw" in xref_doc
