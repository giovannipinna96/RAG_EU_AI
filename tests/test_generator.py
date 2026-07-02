"""Offline unit tests for the Generator class.

Tests cover:
- _parse_json with valid JSON, invalid JSON, None, empty string, partial keys
- _build_context deduplication and truncation
- generate() integration with normalizer via an injected stub OpenAI client
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.generation.generator import Generator

# ---------------------------------------------------------------------------
# Stub OpenAI client helpers
# ---------------------------------------------------------------------------


def _stub_client(content: str):
    """Return a SimpleNamespace that mimics openai.OpenAI for Generator."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=lambda **kw: response)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _stub_client_with_reasoning(content: str, reasoning: str):
    """Stub whose message also carries a native `reasoning_content` field,
    mimicking gemma-4 / llama.cpp reasoning models."""
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=lambda **kw: response)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


# ---------------------------------------------------------------------------
# _parse_json tests
# ---------------------------------------------------------------------------


def test_parse_json_valid_returns_dict():
    payload = {"answer": "yes", "references": ["Article 5"], "reasoning": "r"}
    result = Generator._parse_json(json.dumps(payload))
    assert result == payload


def test_parse_json_invalid_json_returns_empty_dict():
    result = Generator._parse_json("{not valid json")
    assert result == {}


def test_parse_json_none_returns_empty_dict():
    result = Generator._parse_json(None)
    assert result == {}


def test_parse_json_empty_string_returns_empty_dict():
    result = Generator._parse_json("")
    assert result == {}


def test_parse_json_whitespace_only_returns_empty_dict():
    result = Generator._parse_json("   ")
    assert result == {}


def test_parse_json_valid_but_empty_object():
    result = Generator._parse_json("{}")
    assert result == {}


def test_parse_json_missing_all_keys():
    result = Generator._parse_json('{"unexpected": "key"}')
    assert isinstance(result, dict)
    assert "unexpected" in result


def test_parse_json_partial_keys_returns_available_data():
    result = Generator._parse_json('{"answer": "yes"}')
    assert result.get("answer") == "yes"
    assert "references" not in result


def test_parse_json_strips_markdown_fence():
    # gemma-4 / llama.cpp wraps JSON in a ```json ... ``` fence.
    fenced = '```json\n{"answer": "yes", "references": ["Article 5"]}\n```'
    result = Generator._parse_json(fenced)
    assert result.get("answer") == "yes"
    assert result.get("references") == ["Article 5"]


def test_parse_json_strips_bare_fence():
    fenced = '```\n{"answer": "ok"}\n```'
    assert Generator._parse_json(fenced).get("answer") == "ok"


def test_parse_json_extracts_object_amid_prose():
    # Fallback: object survives even with stray prose around it.
    messy = 'Here is the answer:\n{"answer": "a", "references": []}\nDone.'
    result = Generator._parse_json(messy)
    assert result.get("answer") == "a"


# ---------------------------------------------------------------------------
# _build_context tests
# ---------------------------------------------------------------------------


def test_build_context_single_chunk():
    gen = Generator(client=_stub_client("{}"))
    chunks = [{"article_id": "Article 5", "content_raw": "Some text about AI."}]
    ctx = gen._build_context(chunks)
    assert "--- Article 5 ---" in ctx
    assert "Some text about AI." in ctx


def test_build_context_deduplicates_by_article_id():
    gen = Generator(client=_stub_client("{}"))
    chunks = [
        {"article_id": "Article 5", "content_raw": "First chunk."},
        {"article_id": "Article 5", "content_raw": "Duplicate chunk should be dropped."},
        {"article_id": "Article 6", "content_raw": "Another article."},
    ]
    ctx = gen._build_context(chunks)
    # "Article 5" header appears exactly once
    assert ctx.count("--- Article 5 ---") == 1
    assert "Duplicate chunk should be dropped." not in ctx


def test_build_context_preserves_order_of_distinct_articles():
    gen = Generator(client=_stub_client("{}"))
    chunks = [
        {"article_id": "Article 6", "content_raw": "Art 6 text."},
        {"article_id": "Article 5", "content_raw": "Art 5 text."},
    ]
    ctx = gen._build_context(chunks)
    pos6 = ctx.index("--- Article 6 ---")
    pos5 = ctx.index("--- Article 5 ---")
    assert pos6 < pos5


def test_build_context_truncates_content_to_1500_chars():
    gen = Generator(client=_stub_client("{}"))
    long_text = "x" * 3000
    chunks = [{"article_id": "Article 7", "content_raw": long_text}]
    ctx = gen._build_context(chunks)
    # The truncated portion is 1500 chars; the header adds overhead
    assert "x" * 1500 in ctx
    assert "x" * 1501 not in ctx


def test_build_context_empty_chunks_returns_empty_string():
    gen = Generator(client=_stub_client("{}"))
    assert gen._build_context([]) == ""


def test_build_context_chunk_missing_article_id():
    gen = Generator(client=_stub_client("{}"))
    chunks = [{"content_raw": "No article id here."}]
    ctx = gen._build_context(chunks)
    # Missing article_id defaults to empty string — should still include content
    assert "No article id here." in ctx


def test_build_context_chunk_missing_content_raw():
    gen = Generator(client=_stub_client("{}"))
    chunks = [{"article_id": "Article 99"}]
    ctx = gen._build_context(chunks)
    assert "--- Article 99 ---" in ctx


# ---------------------------------------------------------------------------
# generate() integration: normalizer receives LLM references
# ---------------------------------------------------------------------------


def test_generate_normalizes_references():
    """References in non-canonical form from the LLM get normalized."""
    payload = {"answer": "Prohibited.", "reasoning": "r", "references": ["Art. 5", "Annex 3"]}
    gen = Generator(client=_stub_client(json.dumps(payload)))
    result = gen.generate(
        history=[{"role": "user", "content": "What is prohibited?"}],
        chunks=[{"article_id": "Article 5", "content_raw": "Prohibited AI practices."}],
    )
    assert "Article 5" in result["references"]
    assert "Annex III" in result["references"]


def test_generate_deduplicates_references():
    payload = {
        "answer": "See Article 5.",
        "reasoning": "r",
        "references": ["Article 5", "Art. 5", "Article 5"],
    }
    gen = Generator(client=_stub_client(json.dumps(payload)))
    result = gen.generate(
        history=[{"role": "user", "content": "q"}],
        chunks=[],
    )
    assert result["references"].count("Article 5") == 1


def test_generate_drops_invalid_references():
    payload = {
        "answer": "Answer.",
        "reasoning": "r",
        "references": ["not a reference", "completely invalid"],
    }
    gen = Generator(client=_stub_client(json.dumps(payload)))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["references"] == []


def test_generate_empty_llm_response_returns_safe_defaults():
    gen = Generator(client=_stub_client("{}"))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["answer"] == ""
    assert result["reasoning"] == ""
    assert result["references"] == []


def test_generate_returns_required_keys():
    payload = {"answer": "A.", "reasoning": "r", "references": ["Article 6"]}
    gen = Generator(client=_stub_client(json.dumps(payload)))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert set(result.keys()) == {"answer", "reasoning", "references"}


def test_generate_parses_fenced_answer_from_gemma():
    """gemma-4 returns the JSON answer wrapped in a markdown fence."""
    body = '{"answer": "Prohibited.", "reasoning": "r", "references": ["Article 5"]}'
    fenced = f"```json\n{body}\n```"
    gen = Generator(client=_stub_client(fenced))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["answer"] == "Prohibited."
    assert "Article 5" in result["references"]


def test_generate_falls_back_to_native_reasoning_content():
    """When the JSON omits `reasoning`, the model's native reasoning_content
    surfaces so the reasoning is never silently dropped."""
    payload = '{"answer": "A.", "references": ["Article 6"]}'
    gen = Generator(client=_stub_client_with_reasoning(payload, "model chain of thought"))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["reasoning"] == "model chain of thought"
    assert result["answer"] == "A."


def test_generate_prefers_structured_reasoning_over_native():
    payload = '{"answer": "A.", "reasoning": "structured", "references": []}'
    gen = Generator(client=_stub_client_with_reasoning(payload, "native cot"))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["reasoning"] == "structured"


def test_generate_invalid_json_from_llm_returns_safe_defaults():
    gen = Generator(client=_stub_client("I am not json!"))
    result = gen.generate(history=[{"role": "user", "content": "q"}], chunks=[])
    assert result["references"] == []
    assert result["answer"] == ""
