"""Offline tests for the regex Article/Annex matcher."""

from __future__ import annotations

from src.retrieval.article_matcher import ArticleMatcher


def test_extracts_articles():
    m = ArticleMatcher()
    assert m.extract_refs("See Article 5 and Article 6.") == ["Article 5", "Article 6"]


def test_extracts_article_with_subpoint():
    m = ArticleMatcher()
    # Sub-point is matched by the regex but the canonical ref is the article id.
    assert m.extract_refs("pursuant to Article 6.2") == ["Article 6"]


def test_extracts_annex_roman_and_arabic():
    m = ArticleMatcher()
    assert m.extract_refs("Annex III lists the systems") == ["Annex III"]
    assert m.extract_refs("Annex 3 lists the systems") == ["Annex III"]


def test_handles_abbreviation():
    m = ArticleMatcher()
    assert m.extract_refs("Art. 5 prohibits") == ["Article 5"]


def test_deduplicates_and_sorts():
    m = ArticleMatcher()
    assert m.extract_refs("Article 6, Annex III, Article 6") == ["Annex III", "Article 6"]
