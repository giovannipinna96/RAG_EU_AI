"""Offline tests for the Hugging Face dataset source adapter.

Builds a tiny in-memory parquet (no network) and verifies ArticleNode mapping.
Skipped automatically if pyarrow (the `data` extra) is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from src.ingestion.hf_loader import HFDatasetLoader  # noqa: E402

COLUMNS = [
    "language", "chunk_type", "text", "article_no",
    "paragraph_no", "annex_no", "annex_section", "recital_no",
]


def _row(**kw):
    base = dict.fromkeys(COLUMNS)
    base.update(kw)
    return base


def _make_parquet(tmp_path: Path) -> str:
    rows = [
        _row(language="en", chunk_type="article_full", article_no=5,
             text="Article 5 — Prohibited practices\n\n1. The following are prohibited."),
        _row(language="en", chunk_type="paragraph", article_no=5, paragraph_no=1,
             text="The following are prohibited."),
        _row(language="en", chunk_type="paragraph", article_no=5, paragraph_no=2,
             text="There are exceptions for medical purposes."),
        _row(language="en", chunk_type="annex_item", annex_no="III", annex_section="s1",
             text="Annex III — High-risk AI systems\n\n1. Biometrics."),
        _row(language="en", chunk_type="annex_item", annex_no="?", annex_section="s1",
             text="unparseable annex item"),
        _row(language="en", chunk_type="recital", recital_no=1, text="A recital, ignored."),
        # Other-language row that must be filtered out.
        _row(language="fr", chunk_type="article_full", article_no=5, text="Article 5 — Interdit"),
    ]
    # Union of all keys across rows for a stable schema.
    keys = sorted({k for r in rows for k in r})
    table = pa.table({k: [r.get(k) for r in rows] for k in keys})
    dest = tmp_path / "chunks.parquet"
    pq.write_table(table, dest)
    return str(dest)


def test_builds_articles_with_paragraphs(tmp_path):
    nodes = HFDatasetLoader(_make_parquet(tmp_path)).load()
    art = next(n for n in nodes if n.article_id == "Article 5")
    assert art.article_type == "article"
    assert art.title == "Prohibited practices"
    assert [p["ref"] for p in art.paragraphs] == ["Article 5.1", "Article 5.2"]


def test_builds_annex_and_skips_unparseable(tmp_path):
    nodes = HFDatasetLoader(_make_parquet(tmp_path)).load()
    annex_ids = {n.article_id for n in nodes if n.article_type == "annex"}
    assert annex_ids == {"Annex III"}  # the "?" annex is dropped


def test_filters_language(tmp_path):
    nodes = HFDatasetLoader(_make_parquet(tmp_path), language="en").load()
    # Only one Article 5 node despite the French duplicate.
    assert sum(1 for n in nodes if n.article_id == "Article 5") == 1
