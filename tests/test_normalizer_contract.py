"""Contract-conformance tests for ReferenceNormalizer.

These encode the competition's reference rules verbatim:

    references: list[str], minimal set, each item either
      "Article <arabic>[.<sub>]"   e.g. Article 3, Article 3.2
      "Annex   <roman>[.<sub>]"    e.g. Annex III, Annex III.2

    FORBIDDEN forms (must be coerced to the canonical form, NOT silently lost):
      Annex 3, Annex 3(2), Annex III . 2, Annex III-2,
      Article III, Article III.2, Article 3/2, Article 3(2)

The decisive property: every emitted reference matches the canonical regex, and
no recoverable input is dropped just because it arrived in a forbidden shape.
"""

from __future__ import annotations

import re

import pytest

from src.generation.normalizer import ReferenceNormalizer

CANON_ARTICLE = re.compile(r"^Article \d+(\.\d+)?$")
CANON_ANNEX = re.compile(r"^Annex [IVX]+(\.\d+)?$")


def _canon(ref: str) -> bool:
    return bool(CANON_ARTICLE.match(ref) or CANON_ANNEX.match(ref))


@pytest.fixture()
def n() -> ReferenceNormalizer:
    return ReferenceNormalizer()


# ---------------------------------------------------------------------------
# Canonical forms pass through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["Article 3", "Article 3.2", "Annex III", "Annex III.2"])
def test_canonical_forms_preserved(n, ref):
    assert n.normalize([ref]) == [ref]


# ---------------------------------------------------------------------------
# Every forbidden form in the contract must COERCE (not drop) to the canonical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # --- Annex: arabic -> roman, every separator -> dot, spaces stripped ---
        ("Annex 3", "Annex III"),
        ("Annex 3.2", "Annex III.2"),
        ("Annex 3(2)", "Annex III.2"),
        ("Annex III(2)", "Annex III.2"),
        ("Annex III . 2", "Annex III.2"),
        ("Annex III-2", "Annex III.2"),
        ("Annex III/2", "Annex III.2"),
        ("annex iii", "Annex III"),
        ("ANNEX IV", "Annex IV"),
        # --- Article: roman -> arabic, every separator -> dot, spaces stripped ---
        ("Article III", "Article 3"),
        ("Article III.2", "Article 3.2"),
        ("Article 3/2", "Article 3.2"),
        ("Article 3(2)", "Article 3.2"),
        ("Article 3 . 2", "Article 3.2"),
        ("Article 3-2", "Article 3.2"),
        ("Art. 5", "Article 5"),
        ("Art 5", "Article 5"),
        ("Art. 5(1)", "Article 5.1"),
        ("article 10", "Article 10"),
    ],
)
def test_forbidden_forms_are_coerced_not_dropped(n, raw, expected):
    result = n.normalize([raw])
    assert result == [expected], f"{raw!r} -> {result!r}, expected [{expected!r}]"


# ---------------------------------------------------------------------------
# Whitespace robustness (double / inner spaces must still resolve)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Article  6", "Article 6"),
        ("Annex   III", "Annex III"),
        ("  Article 5  ", "Article 5"),
        ("Article 6 . 2", "Article 6.2"),
    ],
)
def test_whitespace_variants_resolve(n, raw, expected):
    assert n.normalize([raw]) == [expected]


# ---------------------------------------------------------------------------
# Letter sub-points the format can't express -> reduce to nearest valid numeral
# form (keep the reference) rather than dropping it entirely.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Article 5(1)(f)", "Article 5.1"),
        ("Article 5.1.a", "Article 5.1"),
        ("Article 5(1)(a)(i)", "Article 5.1"),
        ("Article 5(a)", "Article 5"),
        ("Annex III.2(a)", "Annex III.2"),
    ],
)
def test_letter_subpoints_reduced_to_valid_numeral_form(n, raw, expected):
    assert n.normalize([raw]) == [expected]


# ---------------------------------------------------------------------------
# Safety: valid roman annexes ending in letters must NOT be mangled by the
# letter-stripping logic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["Annex I", "Annex V", "Annex IX", "Annex XIII", "Annex IV"])
def test_roman_annexes_not_mangled(n, ref):
    assert n.normalize([ref]) == [ref]


# ---------------------------------------------------------------------------
# Output invariant: whatever survives, ALWAYS matches the canonical regex and
# contains none of the forbidden separator characters.
# ---------------------------------------------------------------------------


def test_every_output_is_canonical_for_a_messy_batch(n):
    messy = [
        "Annex 3", "Annex 3(2)", "Annex III . 2", "Annex III-2",
        "Article III", "Article III.2", "Article 3/2", "Article 3(2)",
        "Art. 5", "article 10", "Article 5(1)(f)", "Annex IV",
        "garbage", "Section 3", "Recital 9", "",
    ]
    for ref in n.normalize(messy):
        assert _canon(ref), f"non-canonical output leaked: {ref!r}"
        assert not re.search(r"[()/]", ref)
        assert " - " not in ref and "-" not in ref
        assert " . " not in ref
        assert "  " not in ref


def test_forbidden_forms_never_appear_verbatim(n):
    forbidden = ["Annex 3", "Annex 3(2)", "Annex III . 2", "Annex III-2",
                 "Article III", "Article III.2", "Article 3/2", "Article 3(2)"]
    out = n.normalize(forbidden)
    for bad in forbidden:
        assert bad not in out
