"""Reference-format tests. The normalizer tests are fully offline."""

from __future__ import annotations

import re

import pytest

from src.generation.normalizer import ReferenceNormalizer

VALID_ARTICLE = re.compile(r"^Article \d+(\.\d+)?$")
VALID_ANNEX = re.compile(r"^Annex [IVX]+(\.\d+)?$")


def test_normalizer_formats():
    n = ReferenceNormalizer()
    assert n.normalize(["Art. 5"]) == ["Article 5"]
    assert n.normalize(["Article III"]) == ["Article 3"]
    assert n.normalize(["Annex 3"]) == ["Annex III"]
    assert n.normalize(["Annex III-2"]) == ["Annex III.2"]
    assert n.normalize(["Article 3/2"]) == ["Article 3.2"]
    assert n.normalize(["Article 3(2)"]) == ["Article 3.2"]
    assert n.normalize(["invalid reference"]) == []


def test_normalizer_deduplication():
    n = ReferenceNormalizer()
    assert n.normalize(["Article 5", "Art. 5", "Article 5"]) == ["Article 5"]


def test_normalizer_sorting():
    n = ReferenceNormalizer()
    result = n.normalize(["Annex III", "Article 5", "Article 3", "Annex I"])
    assert result == ["Article 3", "Article 5", "Annex I", "Annex III"]


def test_normalizer_subpoint_sorting():
    n = ReferenceNormalizer()
    result = n.normalize(["Article 6.2", "Article 6", "Article 6.1"])
    assert result == ["Article 6", "Article 6.1", "Article 6.2"]


@pytest.mark.integration
def test_api_references_valid(client, competition_questions):
    """Every reference returned by the live API must match the canonical format."""
    for q in competition_questions:
        resp = client.post("/answer", json=[{"role": "user", "content": q}])
        assert resp.status_code == 200
        for ref in resp.json()["references"]:
            assert VALID_ARTICLE.match(ref) or VALID_ANNEX.match(ref), (
                f"Invalid reference format: '{ref}' for question: {q[:60]}"
            )
