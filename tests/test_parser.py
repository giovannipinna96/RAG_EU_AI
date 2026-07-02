"""Offline tests for the EU AI Act HTML parser."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.parser import EUAIActParser

SAMPLE_HTML = """
<html><body>
<p>Article 1</p>
<p>Subject matter</p>
<p>1. This Regulation lays down rules on artificial intelligence.</p>
<p>2. It applies to providers and deployers.</p>
<p>Article 2</p>
<p>Scope</p>
<p>1. This Regulation applies to providers placing systems on the market.</p>
<p>ANNEX I</p>
<p>List of Union harmonisation legislation</p>
<p>ANNEX III</p>
<p>High-risk AI systems by use case area</p>
</body></html>
"""


def _write(tmp_path: Path) -> str:
    f = tmp_path / "act.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")
    return str(f)


def test_parses_articles_and_annexes(tmp_path):
    parser = EUAIActParser()
    nodes = parser.parse(_write(tmp_path))

    articles = [n for n in nodes if n.article_type == "article"]
    annexes = [n for n in nodes if n.article_type == "annex"]

    assert {a.article_id for a in articles} == {"Article 1", "Article 2"}
    assert {a.article_id for a in annexes} == {"Annex I", "Annex III"}


def test_extracts_numbered_paragraphs(tmp_path):
    parser = EUAIActParser()
    nodes = parser.parse(_write(tmp_path))

    art1 = next(n for n in nodes if n.article_id == "Article 1")
    refs = [p["ref"] for p in art1.paragraphs]
    assert "Article 1.1" in refs
    assert "Article 1.2" in refs


def test_word_count_property(tmp_path):
    parser = EUAIActParser()
    nodes = parser.parse(_write(tmp_path))
    assert all(n.word_count > 0 for n in nodes)
