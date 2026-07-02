"""Input-robustness tests for the /answer endpoint.

The competition says the API receives an "OpenAI/LiteLLM standard" conversation
history. In practice that arrives in one of two shapes:

* a bare message array (the rules' own example):   [{"role","content"}, ...]
* the OpenAI chat-completions body wrapper:         {"messages": [ ... ]}

Both MUST be accepted, extra top-level fields (model, temperature, ...) and
extra per-message fields ignored. Malformed bodies must fail with 4xx, never
crash. These tests are fully offline: the pure helper needs no services, and
the endpoint tests stub the cache so no LLM/embedding/Qdrant call is made.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.main import Message, _extract_messages

# ---------------------------------------------------------------------------
# _extract_messages — pure normalisation of the two accepted shapes
# ---------------------------------------------------------------------------


def test_bare_array_accepted():
    msgs = _extract_messages([{"role": "user", "content": "hi"}])
    assert [m.role for m in msgs] == ["user"]
    assert msgs[0].content == "hi"


def test_messages_wrapper_accepted():
    msgs = _extract_messages({"messages": [{"role": "user", "content": "hi"}]})
    assert [m.role for m in msgs] == ["user"]
    assert msgs[0].content == "hi"


def test_wrapper_with_extra_top_level_fields_ignored():
    payload = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "stream": False,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
    }
    msgs = _extract_messages(payload)
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1].content == "q2"


def test_extra_per_message_fields_ignored():
    msgs = _extract_messages(
        [{"role": "user", "content": "hi", "name": "bob", "tool_calls": []}]
    )
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_bare_and_wrapper_yield_identical_messages():
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    a = _extract_messages(conv)
    b = _extract_messages({"messages": conv})
    assert [(m.role, m.content) for m in a] == [(m.role, m.content) for m in b]


@pytest.mark.parametrize(
    "bad",
    [
        [],                                   # empty array
        {},                                   # no messages key
        {"messages": []},                     # empty messages
        {"messages": "not a list"},           # messages not a list
        "a string",                           # not array/object
        42,                                   # not array/object
        None,                                 # null
        {"messages": [{"role": "user"}]},     # message missing content
        {"messages": [{"content": "x"}]},     # message missing role
        [{"role": "user"}],                   # bare array, missing content
        ["just a string"],                    # bare array, item not an object
    ],
)
def test_malformed_bodies_raise_422(bad):
    with pytest.raises(HTTPException) as ei:
        _extract_messages(bad)
    assert ei.value.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint — both shapes reach the handler (cache stubbed → no real pipeline)
# ---------------------------------------------------------------------------


class _StubCache:
    """Returns a canned answer on get() so /answer short-circuits at the cache
    layer, exercising only input parsing — no LLM/embedding/Qdrant required."""

    canned = {"reasoning": "", "answer": "stub answer", "references": ["Article 3"]}

    def get(self, query, history):
        return dict(self.canned)

    def set(self, query, history, result):
        pass


@pytest.fixture
def stub_client(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import main

    monkeypatch.setattr(main.components, "_cache", _StubCache())
    return TestClient(main.app)


def test_endpoint_accepts_bare_array(stub_client):
    r = stub_client.post("/answer", json=[{"role": "user", "content": "q"}])
    assert r.status_code == 200
    assert r.json()["answer"] == "stub answer"


def test_endpoint_accepts_messages_wrapper(stub_client):
    r = stub_client.post("/answer", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub answer"


def test_endpoint_accepts_wrapper_with_extra_fields(stub_client):
    body = {"model": "x", "temperature": 0, "messages": [{"role": "user", "content": "q"}]}
    r = stub_client.post("/answer", json=body)
    assert r.status_code == 200


def test_endpoint_rejects_empty_array(stub_client):
    r = stub_client.post("/answer", json=[])
    assert r.status_code == 422


def test_endpoint_rejects_object_without_messages(stub_client):
    r = stub_client.post("/answer", json={"foo": "bar"})
    assert r.status_code == 422


def test_endpoint_rejects_non_json(stub_client):
    r = stub_client.post(
        "/answer", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400


def test_endpoint_rejects_last_message_not_user(stub_client):
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    r = stub_client.post("/answer", json=conv)
    assert r.status_code == 400


def test_endpoint_wrapper_last_message_not_user_rejected(stub_client):
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    r = stub_client.post("/answer", json=body)
    assert r.status_code == 400


def test_message_model_is_role_content():
    m = Message(role="user", content="x")
    assert m.role == "user"
    assert m.content == "x"
