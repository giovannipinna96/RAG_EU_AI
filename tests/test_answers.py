"""Answer-quality tests against known ground truth (requires the full stack)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


GROUND_TRUTH = [
    {
        "question": "What is the definition of an AI system?",
        "must_reference": ["Article 3"],
        "answer_must_contain": ["machine-based", "system"],
    },
    {
        "question": "What AI practices are prohibited?",
        "must_reference": ["Article 5"],
        "answer_must_contain": ["prohibit"],
    },
    {
        "question": "What does Annex III list?",
        "must_reference": ["Annex III"],
        "answer_must_contain": ["high-risk"],
    },
]


@pytest.mark.parametrize("tc", GROUND_TRUTH, ids=[tc["question"][:40] for tc in GROUND_TRUTH])
def test_answer_quality(client, tc):
    resp = client.post("/answer", json=[{"role": "user", "content": tc["question"]}])
    data = resp.json()

    for expected_ref in tc["must_reference"]:
        assert any(r.startswith(expected_ref) for r in data["references"]), (
            f"Missing reference '{expected_ref}' in {data['references']}"
        )

    answer_lower = data["answer"].lower()
    for keyword in tc["answer_must_contain"]:
        assert keyword.lower() in answer_lower, (
            f"Answer missing keyword '{keyword}': {data['answer'][:100]}"
        )

    sentences = [s.strip() for s in data["answer"].split(".") if s.strip()]
    assert len(sentences) <= 6, f"Answer too long: {len(sentences)} sentences"
