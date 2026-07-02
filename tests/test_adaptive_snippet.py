"""Tests for ``_adaptive_snippet`` -- the query-centered window extractor used
by ``TripleRetriever._build_rerank_prompt`` to avoid showing the LLM a fixed
prefix that misses the rule-stating paragraph.

These tests are offline -- no torch / bm25s / sentence-transformers needed.
"""

from __future__ import annotations

from src.retrieval.triple_retriever import (
    _LLM_RERANK_SNIPPET_CHARS,
    _adaptive_snippet,
)

# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_string():
    assert _adaptive_snippet("", "q", 100) == ""


def test_length_zero_returns_empty_string():
    assert _adaptive_snippet("abc def", "q", 0) == ""


def test_short_text_returned_whole_without_markers():
    text = "Article 5 - short body."
    result = _adaptive_snippet(text, "any query", 100)
    assert result == text
    assert "[...]" not in result


# ---------------------------------------------------------------------------
# No-match fallback (legacy behavior)
# ---------------------------------------------------------------------------


def test_no_keyword_match_returns_prefix_with_end_marker():
    text = "x" * 1000
    result = _adaptive_snippet(text, "nothing relates here", 200)
    # 'nothing','relates' do not appear in 'x'*1000 -> fall back to prefix
    assert result.startswith("x" * 200)
    assert result.endswith("[...]")
    assert len(result.replace("[...]", "")) == 200


def test_only_short_stopwords_in_query_falls_back():
    # All query words shorter than _LLM_RERANK_KEYWORD_MIN_LEN -> ignored
    text = "x" * 1000
    result = _adaptive_snippet(text, "is it a or", 200)
    assert result.startswith("x" * 200)
    assert result.endswith("[...]")


# ---------------------------------------------------------------------------
# Centering on a match in the middle of a long chunk
# ---------------------------------------------------------------------------


def test_match_in_middle_centers_window_and_marks_both_sides():
    # 5000-char buffer with the keyword sitting at offset 2500
    text = "a" * 2500 + "DEEPFAKE" + "b" * 2500
    result = _adaptive_snippet(text, "deepfake disclosure", length=1000)

    assert result.startswith("[...]"), "left side should be marked truncated"
    assert result.endswith("[...]"), "right side should be marked truncated"
    inner = result.removeprefix("[...]").removesuffix("[...]")
    # Window is 1000 wide, centered at 2500 -> roughly [2000..3000]
    assert len(inner) == 1000
    assert "deepfake".upper() in inner


# ---------------------------------------------------------------------------
# Match near the START -> no left marker, window starts at 0
# ---------------------------------------------------------------------------


def test_match_near_start_no_left_marker():
    text = "X" * 100 + "KEYWORD" + "Y" * 4000
    result = _adaptive_snippet(text, "keyword found", length=1000)

    assert not result.startswith("[...]"), "left side should be at chunk start"
    assert result.endswith("[...]"), "right side should be truncated"
    # The window starts at 0 (since center=100, half=500 -> start would be -400 -> 0)
    inner = result.removesuffix("[...]")
    assert inner.startswith("X" * 100 + "KEYWORD")


# ---------------------------------------------------------------------------
# Match near the END -> window shifted back, no right marker
# ---------------------------------------------------------------------------


def test_match_near_end_window_shifted_back():
    text = "Y" * 4000 + "KEYWORD" + "Z" * 100
    result = _adaptive_snippet(text, "keyword tail", length=1000)

    assert result.startswith("[...]"), "left side should be truncated"
    assert not result.endswith("[...]"), "right side should be at chunk end"
    inner = result.removeprefix("[...]")
    # Window length is exactly the requested length, anchored at text end
    assert len(inner) == 1000
    assert inner.endswith("KEYWORD" + "Z" * 100)


# ---------------------------------------------------------------------------
# Earliest match wins when multiple keywords overlap
# ---------------------------------------------------------------------------


def test_earliest_keyword_match_chosen():
    # 'biometric' at 1000, 'emotion' at 3000.  Earliest (1000) should center.
    text = "p" * 1000 + "BIOMETRIC" + "q" * 1000 + "EMOTION" + "r" * 1000
    result = _adaptive_snippet(text, "emotion recognition biometric", length=600)

    inner = result.removeprefix("[...]").removesuffix("[...]")
    assert "BIOMETRIC" in inner, "centered on earliest hit (offset 1000)"
    assert "EMOTION" not in inner, "the later match must fall outside the window"


# ---------------------------------------------------------------------------
# Smoke check: realistic Article 5 scenario from the corpus
# ---------------------------------------------------------------------------


def test_centering_on_real_article5_offset_includes_emotion_clause():
    # Mimics the actual Article 5 chunk: emotion-recognition clause at offset
    # ~2763 inside an 11k-char body.  The default snippet length (~3000 with
    # adaptive centering) must capture it.
    head = "Article 5 - Prohibited AI practices. " + ("filler-text " * 200)
    assert len(head) < 2700, "test setup: header must end before the clause"
    body = "f) emotion recognition systems in workplaces and education"
    tail = "g) biometric categorisation. " + ("more " * 1500)
    text = head + body + tail
    assert len(text) > _LLM_RERANK_SNIPPET_CHARS, "test setup: text must exceed snippet"

    result = _adaptive_snippet(
        text, "Are emotion recognition systems always prohibited?",
        length=_LLM_RERANK_SNIPPET_CHARS,
    )
    assert "emotion recognition" in result.lower()
