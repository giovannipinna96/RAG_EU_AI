"""Offline tests for BM25 hit construction (the article_id-propagation fix).

These do not need the ``ml`` extra: ``bm25s``/``Stemmer`` are imported lazily
inside ``BM25Index``, so importing ``_build_hit`` is torch/bm25s-free.
"""

from __future__ import annotations

from src.retrieval.bm25_index import _build_hit


def test_dict_record_propagates_article_id():
    hit = _build_hit({"text": "Article 50 text", "article_id": "Article 50", "granularity": "large"}, 3.2)
    assert hit["article_id"] == "Article 50"
    assert hit["content_raw"] == "Article 50 text"
    assert hit["granularity"] == "large"
    assert hit["score"] == 3.2


def test_missing_article_id_key_defaults_empty():
    hit = _build_hit({"text": "x"}, 2.0)
    assert hit["article_id"] == ""
    assert hit["content_raw"] == "x"


def test_bare_string_backward_compat():
    # Legacy index stored plain strings — no metadata available.
    hit = _build_hit("some provision text", 1.0)
    assert hit["article_id"] == ""
    assert hit["content_raw"] == "some provision text"
    assert hit["score"] == 1.0
