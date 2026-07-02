"""Comprehensive offline unit tests for SACChunker, ProcessedChunk, and DOC_FINGERPRINT.

All tests inject a stub client — no network, no torch, no services required.
Run with:  uv run --no-sync pytest tests/test_chunker_suite.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.ingestion.chunker import DOC_FINGERPRINT, ProcessedChunk, SACChunker
from src.ingestion.parser import ArticleNode

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _Stub:
    """OpenAI-compatible stub that returns a fixed context string."""

    def __init__(self, content: str = "Ctx.") -> None:
        self._c = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **k: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=self._c))]
                )
            )
        )


class _RaisingStub:
    """Stub whose create() always raises — triggers fallback path."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("LLM unavailable")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._raise))

    def _raise(self, **kwargs):
        raise self._exc


# ---------------------------------------------------------------------------
# ArticleNode factories
# ---------------------------------------------------------------------------

_WORD_400 = " ".join(f"word{i}" for i in range(401))
_WORD_300 = " ".join(f"word{i}" for i in range(300))
_WORD_600 = " ".join(f"word{i}" for i in range(600))
_WORD_1000 = " ".join(f"word{i}" for i in range(1000))


def _article(
    *,
    article_id: str = "Article 6",
    article_type: str = "article",
    number: str = "6",
    title: str = "Some title",
    full_text: str = "Article 6\nSome title\n1. Provision text.",
    paragraphs: list[dict] | None = None,
) -> ArticleNode:
    return ArticleNode(
        article_id=article_id,
        article_type=article_type,
        number=number,
        title=title,
        full_text=full_text,
        paragraphs=paragraphs if paragraphs is not None else [],
    )


def _para(num: str, ref: str, text: str) -> dict:
    return {"num": num, "ref": ref, "text": text}


def _article_with_paragraphs(n: int, article_num: str = "10") -> ArticleNode:
    paras = [
        _para(str(i), f"Article {article_num}.{i}", f"Paragraph {i} text.") for i in range(1, n + 1)
    ]
    return _article(
        article_id=f"Article {article_num}",
        number=article_num,
        full_text=f"Article {article_num}\nTitle\n"
        + "\n".join(f"{i}. Paragraph {i} text." for i in range(1, n + 1)),
        paragraphs=paras,
    )


def _long_no_para_article(word_count: int = 500, article_num: str = "99") -> ArticleNode:
    text = " ".join(f"w{i}" for i in range(word_count))
    return _article(
        article_id=f"Article {article_num}",
        number=article_num,
        full_text=text,
        paragraphs=[],
    )


def _annex_node(num: str = "III", word_count: int = 100) -> ArticleNode:
    text = " ".join(f"a{i}" for i in range(word_count))
    return ArticleNode(
        article_id=f"Annex {num}",
        article_type="annex",
        number=num,
        title=f"Annex {num}",
        full_text=text,
        paragraphs=[],
    )


# ---------------------------------------------------------------------------
# 1. DOC_FINGERPRINT constant
# ---------------------------------------------------------------------------


class TestDocFingerprint:
    def test_contains_regulation_number(self):
        assert "2024/1689" in DOC_FINGERPRINT

    def test_contains_document_tag(self):
        assert "[DOCUMENT]" in DOC_FINGERPRINT

    def test_contains_article_count(self):
        assert "113" in DOC_FINGERPRINT

    def test_contains_annex_count(self):
        assert "13 Annexes" in DOC_FINGERPRINT

    def test_contains_eu_ai_act(self):
        assert "EU AI Act" in DOC_FINGERPRINT

    def test_contains_risk_classification(self):
        assert "risk" in DOC_FINGERPRINT.lower()

    def test_is_non_empty_string(self):
        assert isinstance(DOC_FINGERPRINT, str) and len(DOC_FINGERPRINT) > 0

    def test_mentions_chapters(self):
        assert "13 Chapters" in DOC_FINGERPRINT

    def test_mentions_penalties(self):
        assert "penalties" in DOC_FINGERPRINT.lower()

    def test_mentions_providers(self):
        assert "providers" in DOC_FINGERPRINT.lower()


# ---------------------------------------------------------------------------
# 2. ProcessedChunk dataclass
# ---------------------------------------------------------------------------


class TestProcessedChunk:
    def _make(self, **kwargs) -> ProcessedChunk:
        defaults = {
            "content": "[DOCUMENT]: ...\n[CONTEXT]: ctx\n\nfull text",
            "content_raw": "full text",
            "article_id": "Article 1",
            "article_type": "article",
            "paragraph_refs": [],
            "granularity": "large",
            "title": "Subject matter",
            "context": "ctx",
        }
        defaults.update(kwargs)
        return ProcessedChunk(**defaults)

    def test_all_fields_accessible(self):
        c = self._make()
        assert c.content and c.content_raw and c.article_id

    def test_default_title_is_empty_string(self):
        c = ProcessedChunk(
            content="x",
            content_raw="x",
            article_id="Article 1",
            article_type="article",
            paragraph_refs=[],
            granularity="large",
        )
        assert c.title == ""

    def test_default_context_is_empty_string(self):
        c = ProcessedChunk(
            content="x",
            content_raw="x",
            article_id="Article 1",
            article_type="article",
            paragraph_refs=[],
            granularity="large",
        )
        assert c.context == ""

    def test_granularity_values_accepted(self):
        for g in ("large", "small"):
            c = self._make(granularity=g)
            assert c.granularity == g

    def test_paragraph_refs_list_is_preserved(self):
        refs = ["Article 6.1", "Article 6.2"]
        c = self._make(paragraph_refs=refs)
        assert c.paragraph_refs == refs

    def test_article_type_stored(self):
        c = self._make(article_type="annex")
        assert c.article_type == "annex"


# ---------------------------------------------------------------------------
# 3. SACChunker instantiation
# ---------------------------------------------------------------------------


class TestSACChunkerInit:
    def test_accepts_injected_client(self):
        stub = _Stub()
        chunker = SACChunker(client=stub)
        assert chunker.client is stub

    def test_chunk_all_returns_list(self):
        chunker = SACChunker(client=_Stub())
        result = chunker.chunk_all([])
        assert isinstance(result, list)

    def test_empty_node_list_returns_empty(self):
        chunker = SACChunker(client=_Stub())
        assert chunker.chunk_all([]) == []


# ---------------------------------------------------------------------------
# 4. Large chunk structure
# ---------------------------------------------------------------------------


class TestLargeChunk:
    @pytest.fixture
    def node(self):
        return _article_with_paragraphs(3)

    @pytest.fixture
    def large(self, node):
        return next(
            c
            for c in SACChunker(client=_Stub("MyCtx")).chunk_all([node])
            if c.granularity == "large"
        )

    def test_exactly_one_large_per_node(self, node):
        chunks = SACChunker(client=_Stub()).chunk_all([node])
        assert sum(1 for c in chunks if c.granularity == "large") == 1

    def test_content_starts_with_doc_fingerprint(self, large):
        assert large.content.startswith(DOC_FINGERPRINT)

    def test_content_contains_context_tag(self, large):
        assert "[CONTEXT]:" in large.content

    def test_context_string_in_content(self, large):
        assert "MyCtx" in large.content

    def test_content_raw_equals_full_text(self, node, large):
        assert large.content_raw == node.full_text

    def test_article_id_set_correctly(self, node, large):
        assert large.article_id == node.article_id

    def test_article_type_set_correctly(self, node, large):
        assert large.article_type == node.article_type

    def test_title_set_correctly(self, node, large):
        assert large.title == node.title

    def test_context_field_equals_stub_return(self, large):
        assert large.context == "MyCtx"

    def test_paragraph_refs_are_all_refs_from_node(self, node, large):
        expected = [p["ref"] for p in node.paragraphs]
        assert large.paragraph_refs == expected

    def test_large_granularity_label(self, large):
        assert large.granularity == "large"

    def test_full_text_included_after_context(self, node, large):
        # full_text must appear after the context preamble
        _, _, body = large.content.partition("[CONTEXT]:")
        assert node.full_text in body


# ---------------------------------------------------------------------------
# 5. Small chunks — nodes WITH paragraphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_paras", [1, 2, 5, 10, 20])
def test_small_count_equals_paragraph_count(n_paras: int):
    node = _article_with_paragraphs(n_paras)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    small = [c for c in chunks if c.granularity == "small"]
    assert len(small) == n_paras


@pytest.mark.parametrize("n_paras", [1, 3, 7])
def test_total_chunk_count_with_paragraphs(n_paras: int):
    node = _article_with_paragraphs(n_paras)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    # 1 large + n_paras small
    assert len(chunks) == 1 + n_paras


def test_small_chunk_paragraph_ref_is_single_element():
    node = _article_with_paragraphs(4)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    for c in smalls:
        assert len(c.paragraph_refs) == 1


def test_small_chunk_refs_match_node_paragraphs():
    node = _article_with_paragraphs(3, "20")
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    small_refs = [c.paragraph_refs[0] for c in chunks if c.granularity == "small"]
    expected = [p["ref"] for p in node.paragraphs]
    assert small_refs == expected


def test_small_chunk_content_raw_equals_para_text():
    node = _article_with_paragraphs(2)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    for c, para in zip(smalls, node.paragraphs, strict=False):
        assert c.content_raw == para["text"]


def test_small_chunk_content_starts_with_fingerprint():
    node = _article_with_paragraphs(2)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    for c in smalls:
        assert c.content.startswith(DOC_FINGERPRINT)


def test_small_chunk_content_contains_context_tag():
    node = _article_with_paragraphs(2)
    smalls = [
        c for c in SACChunker(client=_Stub("X")).chunk_all([node]) if c.granularity == "small"
    ]
    for c in smalls:
        assert "[CONTEXT]:" in c.content


def test_small_chunk_content_contains_para_ref_bracket():
    node = _article_with_paragraphs(3, "7")
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c, para in zip(smalls, node.paragraphs, strict=False):
        assert f"[{para['ref']}]" in c.content


def test_small_chunk_article_id_matches_node():
    node = _article_with_paragraphs(2, "42")
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.article_id == "Article 42"


def test_small_chunk_article_type_matches_node():
    node = _article_with_paragraphs(1)
    small = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"
    )
    assert small.article_type == "article"


def test_small_chunk_context_field_matches_stub():
    node = _article_with_paragraphs(1)
    small = next(
        c
        for c in SACChunker(client=_Stub("CtxValue")).chunk_all([node])
        if c.granularity == "small"
    )
    assert small.context == "CtxValue"


def test_small_chunk_title_matches_node():
    node = _article_with_paragraphs(1)
    small = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"
    )
    assert small.title == node.title


# ---------------------------------------------------------------------------
# 6. Sliding-window split — no paragraphs, >400 words
# ---------------------------------------------------------------------------


def _expected_window_count(word_count: int, step: int = 250, window: int = 300) -> int:
    """Mirrors the range(0, len(words), step) loop — last block may be shorter."""
    return math.ceil(word_count / step)


@pytest.mark.parametrize(
    "word_count,step,window",
    [
        (401, 250, 300),  # just over threshold: 2 windows
        (500, 250, 300),  # 2 windows
        (600, 250, 300),  # 3 windows (indices 0, 250, 500)
        (750, 250, 300),  # 3 windows (indices 0, 250, 500)
        (1000, 250, 300),  # 4 windows
        (1200, 250, 300),  # 5 windows
    ],
)
def test_sliding_window_small_count(word_count: int, step: int, window: int):
    node = _long_no_para_article(word_count)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    assert len(smalls) == _expected_window_count(word_count, step, window)


def test_no_sliding_window_when_short_no_para():
    """<= 400 words and no paragraphs → only the large chunk."""
    node = _long_no_para_article(400)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert all(c.granularity == "large" for c in chunks)
    assert len(chunks) == 1


def test_no_sliding_window_at_exactly_400_words():
    node = _long_no_para_article(400)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    assert smalls == []


def test_sliding_window_chunks_start_with_fingerprint():
    node = _long_no_para_article(500)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.content.startswith(DOC_FINGERPRINT)


def test_sliding_window_chunks_have_empty_paragraph_refs():
    node = _long_no_para_article(500)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.paragraph_refs == []


def test_sliding_window_content_raw_is_word_block():
    node = _long_no_para_article(500)
    words = node.full_text.split()
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    expected_raws = [" ".join(words[j : j + 300]) for j in range(0, len(words), 250)]
    actual_raws = [c.content_raw for c in smalls]
    assert actual_raws == expected_raws


def test_sliding_window_first_block_words_correct():
    node = _long_no_para_article(501)
    words = node.full_text.split()
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    assert smalls[0].content_raw == " ".join(words[:300])


def test_sliding_window_second_block_starts_at_250():
    node = _long_no_para_article(600)
    words = node.full_text.split()
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    assert smalls[1].content_raw == " ".join(words[250:550])


def test_sliding_window_granularity_label():
    node = _long_no_para_article(500)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.granularity == "small"


def test_sliding_window_article_id_preserved():
    node = _long_no_para_article(500, "77")
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.article_id == "Article 77"


def test_sliding_window_article_type_preserved():
    node = _long_no_para_article(500)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert c.article_type == "article"


# ---------------------------------------------------------------------------
# 7. Paragraphs take precedence over sliding window
# ---------------------------------------------------------------------------


def test_paragraphs_suppress_sliding_window_even_when_long():
    """A node with both many words AND paragraphs should produce paragraph smalls only."""
    long_text = " ".join(f"w{i}" for i in range(800))
    paras = [_para("1", "Article 3.1", "Para 1."), _para("2", "Article 3.2", "Para 2.")]
    node = _article(full_text=long_text, paragraphs=paras, article_id="Article 3", number="3")
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    assert len(smalls) == 2
    assert all(c.paragraph_refs != [] for c in smalls)


# ---------------------------------------------------------------------------
# 8. Annex-type nodes
# ---------------------------------------------------------------------------


def test_annex_large_chunk_has_annex_article_type():
    node = _annex_node("I")
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.article_type == "annex"


def test_annex_large_chunk_article_id():
    node = _annex_node("III")
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.article_id == "Annex III"


def test_annex_short_no_small_chunks():
    """Short annex (<= 400 words) with no paragraphs → only large chunk."""
    node = _annex_node("II", word_count=100)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert len(chunks) == 1
    assert chunks[0].granularity == "large"


def test_annex_long_triggers_sliding_window():
    node = _annex_node("IV", word_count=500)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    assert len(smalls) > 0


def test_annex_large_content_starts_with_fingerprint():
    node = _annex_node("V")
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.content.startswith(DOC_FINGERPRINT)


def test_annex_large_content_raw_equals_full_text():
    node = _annex_node("VI", word_count=50)
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.content_raw == node.full_text


# ---------------------------------------------------------------------------
# 9. Empty title node
# ---------------------------------------------------------------------------


def test_empty_title_node_large_chunk_produced():
    node = _article(
        title="",
        full_text="Article 1\n\n1. Something.",
        paragraphs=[_para("1", "Article 1.1", "Something.")],
    )
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert any(c.granularity == "large" for c in chunks)


def test_empty_title_stored_in_chunk():
    node = _article(
        title="",
        full_text="Article 1\n\n1. Something.",
        paragraphs=[_para("1", "Article 1.1", "Something.")],
    )
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.title == ""


def test_fallback_context_with_empty_title():
    """When title is empty, fallback should be '<article_id>: ' (article_id + colon + space)."""
    node = _article(article_id="Article 99", title="", full_text="x")
    ctx = SACChunker(client=_RaisingStub())._generate_context(node)
    assert ctx.startswith("Article 99:")


# ---------------------------------------------------------------------------
# 10. Context generation and fallback
# ---------------------------------------------------------------------------


def test_generate_context_returns_stub_content():
    node = _article_with_paragraphs(1)
    chunker = SACChunker(client=_Stub("Specific context."))
    ctx = chunker._generate_context(node)
    assert ctx == "Specific context."


def test_generate_context_strips_whitespace():
    stub = _Stub("  padded  ")
    chunker = SACChunker(client=stub)
    ctx = chunker._generate_context(_article())
    assert ctx == "padded"


def test_generate_context_fallback_on_runtime_error():
    node = _article(article_id="Article 10", title="Risk classification")
    ctx = SACChunker(client=_RaisingStub(RuntimeError("down")))._generate_context(node)
    assert ctx == "Article 10: Risk classification"


def test_generate_context_fallback_on_value_error():
    node = _article(article_id="Article 11", title="Compliance")
    ctx = SACChunker(client=_RaisingStub(ValueError("bad")))._generate_context(node)
    assert ctx == "Article 11: Compliance"


def test_generate_context_fallback_on_exception():
    node = _article(article_id="Annex I", title="Annex I", article_type="annex")
    ctx = SACChunker(client=_RaisingStub(Exception("any")))._generate_context(node)
    assert ctx == "Annex I: Annex I"


def test_fallback_context_appears_in_large_chunk_content():
    node = _article(article_id="Article 3", title="Definitions")
    large = next(
        c for c in SACChunker(client=_RaisingStub()).chunk_all([node]) if c.granularity == "large"
    )
    assert "Article 3: Definitions" in large.content


def test_fallback_context_stored_in_context_field():
    node = _article(article_id="Article 3", title="Definitions")
    large = next(
        c for c in SACChunker(client=_RaisingStub()).chunk_all([node]) if c.granularity == "large"
    )
    assert large.context == "Article 3: Definitions"


def test_fallback_does_not_raise():
    node = _article()
    # Should not propagate any exception.
    chunks = SACChunker(client=_RaisingStub()).chunk_all([node])
    assert len(chunks) >= 1


def test_llm_called_once_per_node():
    """Context generation result is shared across all chunks of the same node."""
    calls = []

    class _CountingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            calls.append(1)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Ctx"))]
            )

    node = _article_with_paragraphs(5)
    SACChunker(client=_CountingStub()).chunk_all([node])
    assert len(calls) == 1


def test_llm_called_once_per_node_multi_node():
    calls = []

    class _CountingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            calls.append(1)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Ctx"))]
            )

    nodes = [_article_with_paragraphs(2, str(i)) for i in range(1, 6)]
    SACChunker(client=_CountingStub()).chunk_all(nodes)
    assert len(calls) == 5


def test_context_shared_across_large_and_small():
    """All chunks for a node carry the same context string."""
    node = _article_with_paragraphs(3)
    chunks = SACChunker(client=_Stub("SharedCtx")).chunk_all([node])
    for c in chunks:
        assert c.context == "SharedCtx"


# ---------------------------------------------------------------------------
# 11. Multiple nodes — totals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "para_counts",
    [
        [1, 1],
        [0, 3],
        [5, 5, 5],
        [0, 0, 0],  # all short no-para: only larges
        [2, 0, 4],
    ],
)
def test_total_large_count_equals_node_count(para_counts: list[int]):
    nodes = [
        _article_with_paragraphs(n, str(i + 1))
        if n > 0
        else _article(article_id=f"Article {i + 1}", number=str(i + 1))
        for i, n in enumerate(para_counts)
    ]
    chunks = SACChunker(client=_Stub()).chunk_all(nodes)
    assert sum(1 for c in chunks if c.granularity == "large") == len(nodes)


def test_two_nodes_with_paragraphs_chunk_count():
    nodes = [_article_with_paragraphs(3, "1"), _article_with_paragraphs(2, "2")]
    chunks = SACChunker(client=_Stub()).chunk_all(nodes)
    # 2 large + 3 + 2 small = 7
    assert len(chunks) == 7


def test_nodes_produce_independent_article_ids():
    nodes = [_article_with_paragraphs(1, "10"), _article_with_paragraphs(1, "20")]
    chunks = SACChunker(client=_Stub()).chunk_all(nodes)
    ids = {c.article_id for c in chunks}
    assert "Article 10" in ids and "Article 20" in ids


def test_order_large_before_smalls_per_node():
    """Large chunk is produced before the small chunks for the same node."""
    node = _article_with_paragraphs(3)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert chunks[0].granularity == "large"
    for c in chunks[1:]:
        assert c.granularity == "small"


# ---------------------------------------------------------------------------
# 12. Content enrichment details
# ---------------------------------------------------------------------------


def test_large_content_has_newline_between_fingerprint_and_context():
    node = _article_with_paragraphs(1)
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    # format: DOC_FINGERPRINT + "\n[CONTEXT]: ..."
    assert f"{DOC_FINGERPRINT}\n[CONTEXT]:" in large.content


def test_large_content_context_section_format():
    node = _article_with_paragraphs(1)
    large = next(
        c for c in SACChunker(client=_Stub("TheCtx")).chunk_all([node]) if c.granularity == "large"
    )
    assert "[CONTEXT]: TheCtx" in large.content


def test_small_content_para_ref_bracket_format():
    paras = [_para("1", "Article 5.1", "Para text here.")]
    node = _article(
        article_id="Article 5",
        number="5",
        paragraphs=paras,
        full_text="Article 5\n1. Para text here.",
    )
    small = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"
    )
    assert "[Article 5.1]: Para text here." in small.content


def test_content_raw_never_contains_fingerprint_for_paragraphs():
    node = _article_with_paragraphs(2)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert "[DOCUMENT]" not in c.content_raw


def test_content_raw_never_contains_fingerprint_for_large():
    node = _article_with_paragraphs(1)
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    # content_raw is the raw full_text without the DOC_FINGERPRINT prefix
    assert large.content_raw == node.full_text
    assert not large.content_raw.startswith("[DOCUMENT]")


def test_content_raw_never_contains_context_tag_for_paragraphs():
    node = _article_with_paragraphs(2)
    smalls = [c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "small"]
    for c in smalls:
        assert "[CONTEXT]:" not in c.content_raw


def test_content_raw_never_contains_context_tag_for_large():
    node = _article_with_paragraphs(1)
    large = next(
        c for c in SACChunker(client=_Stub()).chunk_all([node]) if c.granularity == "large"
    )
    assert "[CONTEXT]:" not in large.content_raw


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------


def test_node_with_single_long_word_no_split():
    """A node whose full_text is one very long token (<= 400 words total) → no small."""
    text = "x" * 5000  # 1 word, 5000 chars
    node = _article(full_text=text, paragraphs=[])
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert len(chunks) == 1 and chunks[0].granularity == "large"


def test_node_full_text_exactly_401_words_triggers_window():
    text = " ".join(["w"] * 401)
    node = _article(full_text=text, paragraphs=[])
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    smalls = [c for c in chunks if c.granularity == "small"]
    assert len(smalls) >= 1


def test_many_paragraphs_node():
    node = _article_with_paragraphs(50)
    chunks = SACChunker(client=_Stub()).chunk_all([node])
    assert sum(1 for c in chunks if c.granularity == "small") == 50


def test_chunk_all_with_mix_of_node_types():
    nodes = [
        _article_with_paragraphs(2, "1"),
        _long_no_para_article(500, "2"),
        _annex_node("I", 50),
    ]
    chunks = SACChunker(client=_Stub()).chunk_all(nodes)
    larges = [c for c in chunks if c.granularity == "large"]
    assert len(larges) == 3


def test_none_context_from_stub_becomes_empty_string():
    """If stub returns None as content, _generate_context should return ''."""

    class _NoneStub:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **k: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
                    )
                )
            )

    node = _article()
    ctx = SACChunker(client=_NoneStub())._generate_context(node)
    assert ctx == ""


def test_generate_context_passes_article_id_in_prompt():
    """The LLM call kwargs should mention the article_id in the user message."""
    captured = {}

    class _CapturingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._capture))

        def _capture(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    node = _article(article_id="Article 42", title="My Title")
    SACChunker(client=_CapturingStub())._generate_context(node)
    user_content = captured["messages"][0]["content"]
    assert "Article 42" in user_content


def test_generate_context_passes_title_in_prompt():
    captured = {}

    class _CapturingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._capture))

        def _capture(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    node = _article(article_id="Article 1", title="Unique Title XYZ")
    SACChunker(client=_CapturingStub())._generate_context(node)
    user_content = captured["messages"][0]["content"]
    assert "Unique Title XYZ" in user_content


def test_generate_context_temperature_zero():
    """The LLM call should use temperature=0."""
    captured = {}

    class _CapturingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._capture))

        def _capture(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    SACChunker(client=_CapturingStub())._generate_context(_article())
    assert captured.get("temperature") == 0


def test_generate_context_max_tokens_100():
    captured = {}

    class _CapturingStub:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._capture))

        def _capture(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    SACChunker(client=_CapturingStub())._generate_context(_article())
    assert captured.get("max_tokens") == 100
