"""Additional edge-case tests for ReferenceNormalizer.

The existing test_references.py covers the happy path; this file focuses on
boundary values, multiple transforms in one string, large article numbers,
upper/lower case variants, and the sort-key logic for Annex order.
"""

from __future__ import annotations

from src.generation.normalizer import ReferenceNormalizer

# ---------------------------------------------------------------------------
# Edge cases: casing and whitespace
# ---------------------------------------------------------------------------


def test_normalize_case_insensitive_article_prefix():
    n = ReferenceNormalizer()
    assert n.normalize(["article 5"]) == ["Article 5"]


def test_normalize_case_insensitive_art_abbreviation():
    n = ReferenceNormalizer()
    assert n.normalize(["ART. 10"]) == ["Article 10"]


def test_normalize_strips_leading_and_trailing_whitespace():
    n = ReferenceNormalizer()
    assert n.normalize(["  Article 5  "]) == ["Article 5"]


def test_normalize_extra_internal_space_is_not_normalised():
    """'Article  5' (double space) does NOT match and should be dropped."""
    n = ReferenceNormalizer()
    result = n.normalize(["Article  5"])
    # Not a valid canonical form after transforms
    assert "Article  5" not in result


# ---------------------------------------------------------------------------
# Large article and annex numbers
# ---------------------------------------------------------------------------


def test_normalize_large_article_number():
    n = ReferenceNormalizer()
    assert n.normalize(["Article 113"]) == ["Article 113"]


def test_normalize_article_with_large_subpoint():
    n = ReferenceNormalizer()
    assert n.normalize(["Article 50.10"]) == ["Article 50.10"]


def test_normalize_annex_beyond_known_roman():
    """Annex XIV is not in ROMAN dict; Arabic-to-Roman conversion returns the
    original string unchanged when the integer key is missing from INT_TO_ROMAN."""
    n = ReferenceNormalizer()
    # 14 is not in INT_TO_ROMAN; the result stays as "14" which won't match VALID_ANNEX
    result = n.normalize(["Annex 14"])
    # Should be dropped as unrecognised (14 not in INT_TO_ROMAN)
    assert "Annex 14" not in result


def test_normalize_valid_annex_xiii():
    n = ReferenceNormalizer()
    assert n.normalize(["Annex XIII"]) == ["Annex XIII"]


def test_normalize_annex_arabic_maps_known_range():
    """Annex 1–13 all have known Roman mappings."""
    n = ReferenceNormalizer()
    for arabic, roman in [
        ("1", "I"), ("2", "II"), ("4", "IV"), ("5", "V"),
        ("9", "IX"), ("11", "XI"), ("13", "XIII"),
    ]:
        result = n.normalize([f"Annex {arabic}"])
        assert result == [f"Annex {roman}"], f"Failed for Annex {arabic}"


# ---------------------------------------------------------------------------
# Sub-point separator variants
# ---------------------------------------------------------------------------


def test_normalize_article_slash_separator():
    n = ReferenceNormalizer()
    assert n.normalize(["Article 6/1"]) == ["Article 6.1"]


def test_normalize_article_paren_separator():
    n = ReferenceNormalizer()
    assert n.normalize(["Article 6(1)"]) == ["Article 6.1"]


def test_normalize_annex_dash_separator():
    n = ReferenceNormalizer()
    assert n.normalize(["Annex IV-3"]) == ["Annex IV.3"]


def test_normalize_annex_subpoint_already_canonical():
    n = ReferenceNormalizer()
    assert n.normalize(["Annex II.1"]) == ["Annex II.1"]


# ---------------------------------------------------------------------------
# Roman numeral article conversion
# ---------------------------------------------------------------------------


def test_normalize_article_roman_i():
    n = ReferenceNormalizer()
    assert n.normalize(["Article I"]) == ["Article 1"]


def test_normalize_article_roman_ix():
    n = ReferenceNormalizer()
    assert n.normalize(["Article IX"]) == ["Article 9"]


def test_normalize_article_roman_with_subpoint():
    n = ReferenceNormalizer()
    assert n.normalize(["Article III.2"]) == ["Article 3.2"]


# ---------------------------------------------------------------------------
# Mixed batch normalisation
# ---------------------------------------------------------------------------


def test_normalize_mixed_batch():
    n = ReferenceNormalizer()
    raw = ["Art. 5", "Annex 3", "Article III", "Annex III-2", "garbage"]
    result = n.normalize(raw)
    assert "Article 5" in result
    assert "Annex III" in result
    assert "Article 3" in result
    assert "Annex III.2" in result
    assert "garbage" not in result


def test_normalize_empty_list_returns_empty():
    n = ReferenceNormalizer()
    assert n.normalize([]) == []


def test_normalize_all_invalid_returns_empty():
    n = ReferenceNormalizer()
    result = n.normalize(["foo bar", "baz", "123", "Article", "Annex"])
    assert result == []


# ---------------------------------------------------------------------------
# Sort order: Articles before Annexes, then by number
# ---------------------------------------------------------------------------


def test_sort_articles_before_annexes():
    n = ReferenceNormalizer()
    result = n.normalize(["Annex I", "Article 1"])
    assert result[0].startswith("Article")
    assert result[1].startswith("Annex")


def test_sort_articles_numerically_not_lexicographically():
    n = ReferenceNormalizer()
    result = n.normalize(["Article 10", "Article 9", "Article 2"])
    assert result == ["Article 2", "Article 9", "Article 10"]


def test_sort_annexes_by_roman_value():
    n = ReferenceNormalizer()
    result = n.normalize(["Annex IX", "Annex II", "Annex V"])
    assert result == ["Annex II", "Annex V", "Annex IX"]


def test_sort_sub_articles_after_parent():
    n = ReferenceNormalizer()
    result = n.normalize(["Article 5.2", "Article 5", "Article 5.1"])
    assert result == ["Article 5", "Article 5.1", "Article 5.2"]


def test_sort_sub_annexes_after_parent():
    n = ReferenceNormalizer()
    result = n.normalize(["Annex III.2", "Annex III", "Annex III.1"])
    assert result == ["Annex III", "Annex III.1", "Annex III.2"]


# ---------------------------------------------------------------------------
# Deduplication across different input forms
# ---------------------------------------------------------------------------


def test_dedup_article_different_representations():
    n = ReferenceNormalizer()
    result = n.normalize(["Article 5", "Art. 5", "Art 5"])
    assert result.count("Article 5") == 1


def test_dedup_annex_arabic_and_roman():
    n = ReferenceNormalizer()
    result = n.normalize(["Annex III", "Annex 3"])
    assert result.count("Annex III") == 1
