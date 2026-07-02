"""Multi-turn conversation tests (requires the full stack)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_pronoun_resolution(client):
    """The system should resolve 'that' to Article 5 from the previous turn."""
    conv = [{"role": "user", "content": "What does Article 5 establish?"}]
    r1 = client.post("/answer", json=conv)
    assert r1.status_code == 200

    conv.append({"role": "assistant", "content": r1.json()["answer"]})
    conv.append({"role": "user", "content": "Are there exceptions to that?"})

    r2 = client.post("/answer", json=conv)
    assert r2.status_code == 200
    assert "Article 5" in str(r2.json()["references"])


def test_topic_switch(client):
    """The system should handle a topic change within a conversation."""
    conv = [{"role": "user", "content": "What is the definition of an AI system?"}]
    r1 = client.post("/answer", json=conv)

    conv.append({"role": "assistant", "content": r1.json()["answer"]})
    conv.append({"role": "user", "content": "What are the penalties for non-compliance?"})

    r2 = client.post("/answer", json=conv)
    assert r2.status_code == 200
    assert len(r2.json()["answer"]) > 20
