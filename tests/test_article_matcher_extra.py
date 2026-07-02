"""Additional edge-case tests for ArticleMatcher.extract_refs.

The existing test_retrieval.py covers the basic happy path. This file targets
boundary values: multiple refs in one sentence, case variants, text with no
refs, large article numbers, all known Annex conversions, and sub-point refs
which are collapsed to article-level.
"""

from __future__ import annotations

import pytest

from src.retrieval.article_matcher import ArticleMatcher

# ---------------------------------------------------------------------------
# Empty / whitespace
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty():
    m = ArticleMatcher()
    assert m.extract_refs("") == []


def test_whitespace_only_returns_empty():
    m = ArticleMatcher()
    assert m.extract_refs("   \n\t  ") == []


def test_text_with_no_refs_returns_empty():
    m = ArticleMatcher()
    assert m.extract_refs("This is a general statement about AI safety.") == []


# ---------------------------------------------------------------------------
# Multiple articles in one string
# ---------------------------------------------------------------------------


def test_extracts_multiple_articles_in_one_sentence():
    m = ArticleMatcher()
    result = m.extract_refs("Article 5, Article 6, and Article 10 all apply.")
    assert set(result) == {"Article 5", "Article 6", "Article 10"}


def test_extracts_article_range_separately():
    """'Article 6 and Article 7' — both extracted, no merging."""
    m = ArticleMatcher()
    result = m.extract_refs("See Article 6 and Article 7 for details.")
    assert "Article 6" in result
    assert "Article 7" in result


# ---------------------------------------------------------------------------
# Sub-point refs — collapsed to article level
# ---------------------------------------------------------------------------


def test_subpoint_ref_returns_parent_article():
    m = ArticleMatcher()
    result = m.extract_refs("Article 6.2 applies to deployers.")
    assert result == ["Article 6"]


def test_annex_subpoint_ref_returns_parent_annex():
    m = ArticleMatcher()
    result = m.extract_refs("Annex III.1 lists high-risk categories.")
    assert result == ["Annex III"]


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


def test_article_lowercase_prefix():
    m = ArticleMatcher()
    result = m.extract_refs("article 7 of the regulation")
    assert result == ["Article 7"]


def test_art_dot_abbreviation_case_variants():
    m = ArticleMatcher()
    assert m.extract_refs("ART. 5 is relevant") == ["Article 5"]
    assert m.extract_refs("Art 5 is relevant") == ["Article 5"]


def test_annex_lowercase():
    m = ArticleMatcher()
    result = m.extract_refs("annex iii applies here")
    assert result == ["Annex III"]


# ---------------------------------------------------------------------------
# Large article numbers
# ---------------------------------------------------------------------------


def test_large_article_number():
    m = ArticleMatcher()
    result = m.extract_refs("Article 113 is the final article.")
    assert result == ["Article 113"]


def test_article_zero_is_extracted():
    m = ArticleMatcher()
    # Article 0 is not a real article but the regex still captures it
    result = m.extract_refs("Article 0 is hypothetical.")
    assert result == ["Article 0"]


# ---------------------------------------------------------------------------
# Annex conversions: arabic -> roman
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arabic,roman", [
    ("1", "I"), ("2", "II"), ("3", "III"), ("4", "IV"),
    ("5", "V"), ("6", "VI"), ("7", "VII"), ("8", "VIII"),
    ("9", "IX"), ("10", "X"), ("11", "XI"), ("12", "XII"), ("13", "XIII"),
])
def test_annex_arabic_to_roman(arabic, roman):
    m = ArticleMatcher()
    result = m.extract_refs(f"Annex {arabic} applies.")
    assert result == [f"Annex {roman}"]


# ---------------------------------------------------------------------------
# Annex unknown arabic value
# ---------------------------------------------------------------------------


def test_annex_unknown_arabic_stays_as_is():
    """Annex 14 is not in INT_TO_ROMAN — the raw string is uppercased."""
    m = ArticleMatcher()
    result = m.extract_refs("Annex 14 is out of range.")
    # The regex matches but isdigit() path with missing key returns the original
    # INT_TO_ROMAN.get(14, "14") → "14", so the ref becomes "Annex 14"
    assert result == ["Annex 14"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_same_article_mentioned_multiple_times_deduped():
    m = ArticleMatcher()
    result = m.extract_refs("Article 5, Art. 5, and Article 5 again.")
    assert result.count("Article 5") == 1


def test_same_annex_arabic_and_roman_deduped():
    m = ArticleMatcher()
    result = m.extract_refs("Annex III and Annex 3 are the same.")
    assert result.count("Annex III") == 1


# ---------------------------------------------------------------------------
# Return value is always sorted
# ---------------------------------------------------------------------------


def test_result_is_sorted():
    m = ArticleMatcher()
    result = m.extract_refs("Article 10, Annex III, Article 3, Annex I")
    assert result == sorted(result)


def test_annexes_sort_before_articles_alphabetically():
    """ArticleMatcher returns sorted() — 'Annex' < 'Article' lexicographically."""
    m = ArticleMatcher()
    result = m.extract_refs("Article 5 and Annex III")
    assert result[0].startswith("Annex")
    assert result[1].startswith("Article")


# ---------------------------------------------------------------------------
# Mixed Article + Annex in one string
# ---------------------------------------------------------------------------


def test_mixed_article_and_annex_in_one_string():
    m = ArticleMatcher()
    result = m.extract_refs(
        "Pursuant to Article 6 and Annex III, the system is high-risk."
    )
    assert "Article 6" in result
    assert "Annex III" in result
    assert len(result) == 2
