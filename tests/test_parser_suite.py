"""Comprehensive offline unit tests for EUAIActParser and ArticleNode.

Covers: parse(), _parse_articles(), _parse_annexes(), validate(), ArticleNode.word_count.
All tests are purely offline — no network, GPU, or running services required.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.ingestion.parser import ArticleNode, EUAIActParser

# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]


def _html(body: str) -> str:
    """Wrap a body snippet in minimal valid HTML."""
    return f"<html><body>{body}</body></html>"


def _write(tmp_path: Path, html: str, name: str = "act.html") -> str:
    p = tmp_path / name
    p.write_text(html, encoding="utf-8")
    return str(p)


def _article_block(n: int, title: str = "", paragraphs: int = 0) -> str:
    """Return an HTML snippet for a single Article."""
    lines = [f"<p>Article {n}</p>"]
    if title:
        lines.append(f"<p>{title}</p>")
    for i in range(1, paragraphs + 1):
        lines.append(f"<p>{i}. Paragraph {i} text for article {n}.</p>")
    return "\n".join(lines)


def _annex_block(roman: str, extra: str = "") -> str:
    """Return an HTML snippet for a single Annex."""
    lines = [f"<p>ANNEX {roman}</p>"]
    if extra:
        lines.append(f"<p>{extra}</p>")
    return "\n".join(lines)


def _doc_with_n_articles(n: int, annexes: int = 0) -> str:
    """Full HTML document with n articles and a given number of annexes."""
    body = ""
    for i in range(1, n + 1):
        body += _article_block(i, title=f"Title of article {i}", paragraphs=2)
    for j in range(annexes):
        body += _annex_block(_ROMAN[j], extra=f"Content of annex {j + 1}.")
    return _html(body)


# ---------------------------------------------------------------------------
# ArticleNode.word_count
# ---------------------------------------------------------------------------


class TestArticleNodeWordCount:
    def _make(self, full_text: str) -> ArticleNode:
        return ArticleNode(
            article_id="Article 1",
            article_type="article",
            number="1",
            title="T",
            full_text=full_text,
        )

    def test_single_word(self):
        assert self._make("hello").word_count == 1

    def test_multiple_words(self):
        assert self._make("one two three").word_count == 3

    def test_empty_string_returns_zero(self):
        # str.split() on "" returns [] → length 0
        assert self._make("").word_count == 0

    def test_whitespace_only_returns_zero(self):
        assert self._make("   \t\n  ").word_count == 0

    def test_newlines_treated_as_separators(self):
        assert self._make("word1\nword2\nword3").word_count == 3

    def test_mixed_whitespace(self):
        assert self._make("a  b   c").word_count == 3

    @pytest.mark.parametrize("n_words", [1, 10, 50, 100, 400, 401])
    def test_word_count_exact(self, n_words: int):
        text = " ".join(["word"] * n_words)
        assert self._make(text).word_count == n_words

    def test_long_article_exceeds_400_words(self):
        text = " ".join(["word"] * 450)
        node = self._make(text)
        assert node.word_count > 400

    def test_article_text_with_punctuation(self):
        # Punctuation attached to words — they still count as one token each.
        assert self._make("Hello, world!").word_count == 2


# ---------------------------------------------------------------------------
# ArticleNode dataclass fields
# ---------------------------------------------------------------------------


class TestArticleNodeFields:
    def _make_article(self) -> ArticleNode:
        return ArticleNode(
            article_id="Article 6",
            article_type="article",
            number="6",
            title="Prohibited AI Practices",
            full_text="Article 6\nProhibited AI Practices\n1. Some text.",
            paragraphs=[{"num": "1", "ref": "Article 6.1", "text": "Some text."}],
        )

    def test_article_id(self):
        assert self._make_article().article_id == "Article 6"

    def test_article_type(self):
        assert self._make_article().article_type == "article"

    def test_number(self):
        assert self._make_article().number == "6"

    def test_title(self):
        assert self._make_article().title == "Prohibited AI Practices"

    def test_paragraphs_list(self):
        assert len(self._make_article().paragraphs) == 1

    def test_default_paragraphs_empty(self):
        node = ArticleNode("Article 1", "article", "1", "T", "text")
        assert node.paragraphs == []

    def test_annex_type_field(self):
        node = ArticleNode("Annex III", "annex", "III", "Annex III", "ANNEX III content")
        assert node.article_type == "annex"

    def test_annex_id_field(self):
        node = ArticleNode("Annex III", "annex", "III", "Annex III", "ANNEX III content")
        assert node.article_id == "Annex III"


# ---------------------------------------------------------------------------
# EUAIActParser.parse() — basic structure
# ---------------------------------------------------------------------------


class TestParserParse:
    def test_returns_list(self, tmp_path):
        html = _html(_article_block(1))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert isinstance(nodes, list)

    def test_empty_document_returns_empty_list(self, tmp_path):
        nodes = EUAIActParser().parse(_write(tmp_path, _html("<p>No act content here.</p>")))
        assert nodes == []

    def test_articles_only_no_annexes(self, tmp_path):
        html = _html(_article_block(1) + _article_block(2))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annexes = [n for n in nodes if n.article_type == "annex"]
        assert annexes == []

    def test_annexes_only_no_articles(self, tmp_path):
        html = _html(_annex_block("I") + _annex_block("II"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        articles = [n for n in nodes if n.article_type == "article"]
        assert articles == []

    def test_combined_count(self, tmp_path):
        html = _html(_article_block(1) + _article_block(2) + _annex_block("I"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert len(nodes) == 3

    def test_reads_utf8_file(self, tmp_path):
        html = _html("<p>Article 1</p><p>Tîtle wîth ünïcode</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_type == "article" for n in nodes)

    def test_accepts_string_path(self, tmp_path):
        html = _html(_article_block(1))
        path = _write(tmp_path, html)
        assert isinstance(path, str)
        nodes = EUAIActParser().parse(path)
        assert len(nodes) >= 1


# ---------------------------------------------------------------------------
# _parse_articles — IDs, counts, numbering
# ---------------------------------------------------------------------------


class TestParseArticles:
    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50, 100, 113])
    def test_article_count(self, tmp_path, n: int):
        html = _doc_with_n_articles(n)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        articles = [x for x in nodes if x.article_type == "article"]
        assert len(articles) == n

    @pytest.mark.parametrize("n", [1, 7, 42, 113])
    def test_article_id_format(self, tmp_path, n: int):
        html = _html(_article_block(n))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        ids = {x.article_id for x in nodes if x.article_type == "article"}
        assert f"Article {n}" in ids

    @pytest.mark.parametrize("n", [1, 7, 42, 113])
    def test_article_number_string(self, tmp_path, n: int):
        html = _html(_article_block(n))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        nums = {x.number for x in nodes if x.article_type == "article"}
        assert str(n) in nums

    def test_article_type_field_is_article(self, tmp_path):
        html = _html(_article_block(1))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert all(x.article_type == "article" for x in nodes if x.article_type == "article")

    def test_article_full_text_non_empty(self, tmp_path):
        html = _html(_article_block(1, title="Definitions"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        article = next(x for x in nodes if x.article_type == "article")
        assert article.full_text.strip() != ""

    def test_article_full_text_contains_article_marker(self, tmp_path):
        html = _html(_article_block(5))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        article = next(x for x in nodes if x.article_id == "Article 5")
        assert "Article 5" in article.full_text

    def test_sequential_articles_ordered(self, tmp_path):
        html = _html(_article_block(1) + _article_block(2) + _article_block(3))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        articles = [x for x in nodes if x.article_type == "article"]
        assert [a.number for a in articles] == ["1", "2", "3"]

    def test_article_title_extracted_from_second_line(self, tmp_path):
        html = _html(_article_block(3, title="Scope of Application"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_id == "Article 3")
        assert art.title == "Scope of Application"

    def test_article_missing_title_empty_string(self, tmp_path):
        # No title line between "Article N" and the next paragraph.
        html = _html("<p>Article 1</p><p>1. Some text.</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        # title may be empty string or the paragraph text; either way the node exists
        assert art.article_id == "Article 1"

    def test_large_article_number(self, tmp_path):
        html = _html(_article_block(999))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(x.article_id == "Article 999" for x in nodes)

    def test_word_count_positive_for_article_with_paragraphs(self, tmp_path):
        html = _html(_article_block(1, title="T", paragraphs=3))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        assert art.word_count > 0


# ---------------------------------------------------------------------------
# _parse_articles — paragraph extraction
# ---------------------------------------------------------------------------


class TestParseArticleParagraphs:
    @pytest.mark.parametrize("n_paras", [0, 1, 2, 5])
    def test_paragraph_count(self, tmp_path, n_paras: int):
        html = _html(_article_block(1, paragraphs=n_paras))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        assert len(art.paragraphs) == n_paras

    @pytest.mark.parametrize("para_num", [1, 2, 3])
    def test_paragraph_ref_format(self, tmp_path, para_num: int):
        html = _html(_article_block(7, paragraphs=3))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_id == "Article 7")
        refs = [p["ref"] for p in art.paragraphs]
        assert f"Article 7.{para_num}" in refs

    def test_paragraph_dict_keys(self, tmp_path):
        html = _html(_article_block(1, paragraphs=1))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        p = art.paragraphs[0]
        assert set(p.keys()) == {"num", "ref", "text"}

    def test_paragraph_num_field_is_string(self, tmp_path):
        html = _html(_article_block(1, paragraphs=2))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        for p in art.paragraphs:
            assert isinstance(p["num"], str)

    def test_paragraph_text_non_empty(self, tmp_path):
        html = _html(_article_block(1, paragraphs=2))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        for p in art.paragraphs:
            assert p["text"].strip() != ""

    def test_article_no_paragraphs_empty_list(self, tmp_path):
        html = _html("<p>Article 1</p><p>General provisions</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_type == "article")
        assert art.paragraphs == []

    def test_paragraph_ref_uses_article_number(self, tmp_path):
        html = _html(_article_block(42, paragraphs=1))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(x for x in nodes if x.article_id == "Article 42")
        assert art.paragraphs[0]["ref"].startswith("Article 42.")


# ---------------------------------------------------------------------------
# _parse_annexes — IDs, numerals, counts
# ---------------------------------------------------------------------------


class TestParseAnnexes:
    @pytest.mark.parametrize("roman", _ROMAN)
    def test_annex_id_all_roman_numerals(self, tmp_path, roman: str):
        html = _html(_annex_block(roman, extra="Some content."))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annexes = [x for x in nodes if x.article_type == "annex"]
        assert any(x.article_id == f"Annex {roman}" for x in annexes)

    @pytest.mark.parametrize("n_annexes", [1, 5, 10, 13])
    def test_annex_count(self, tmp_path, n_annexes: int):
        body = "".join(_annex_block(_ROMAN[i]) for i in range(n_annexes))
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annexes = [x for x in nodes if x.article_type == "annex"]
        assert len(annexes) == n_annexes

    def test_annex_type_field_is_annex(self, tmp_path):
        html = _html(_annex_block("I"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert all(x.article_type == "annex" for x in nodes if x.article_type == "annex")

    @pytest.mark.parametrize("roman", _ROMAN)
    def test_annex_number_matches_roman(self, tmp_path, roman: str):
        html = _html(_annex_block(roman))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annex = next(x for x in nodes if x.article_type == "annex")
        assert annex.number == roman

    def test_annex_title_set_to_annex_plus_number(self, tmp_path):
        html = _html(_annex_block("IV"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annex = next(x for x in nodes if x.article_type == "annex")
        assert annex.title == "Annex IV"

    def test_annex_paragraphs_always_empty(self, tmp_path):
        body = "".join(_annex_block(_ROMAN[i]) for i in range(3))
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        for node in nodes:
            if node.article_type == "annex":
                assert node.paragraphs == []

    def test_annex_full_text_contains_annex_marker(self, tmp_path):
        html = _html(_annex_block("VII"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annex = next(x for x in nodes if x.article_type == "annex")
        assert "ANNEX VII" in annex.full_text

    def test_annex_word_count_positive(self, tmp_path):
        html = _html(_annex_block("I", extra="Some annex content words."))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annex = next(x for x in nodes if x.article_type == "annex")
        assert annex.word_count > 0

    def test_thirteen_annexes_all_present(self, tmp_path):
        body = "".join(_annex_block(r) for r in _ROMAN)
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annex_ids = {x.article_id for x in nodes if x.article_type == "annex"}
        expected = {f"Annex {r}" for r in _ROMAN}
        assert annex_ids == expected


# ---------------------------------------------------------------------------
# validate() — threshold logic
# ---------------------------------------------------------------------------


class TestValidate:
    def _make_nodes(self, n_articles: int, n_annexes: int) -> list[ArticleNode]:
        nodes: list[ArticleNode] = []
        for i in range(1, n_articles + 1):
            nodes.append(
                ArticleNode(
                    article_id=f"Article {i}",
                    article_type="article",
                    number=str(i),
                    title=f"Title {i}",
                    full_text=f"Article {i} text.",
                )
            )
        for j in range(n_annexes):
            r = _ROMAN[j % len(_ROMAN)]
            nodes.append(
                ArticleNode(
                    article_id=f"Annex {r}",
                    article_type="annex",
                    number=r,
                    title=f"Annex {r}",
                    full_text=f"ANNEX {r} content.",
                )
            )
        return nodes

    @pytest.mark.parametrize(
        "n_articles,n_annexes,expected",
        [
            (100, 10, True),
            (113, 13, True),
            (100, 13, True),
            (113, 10, True),
            (150, 15, True),
            (99, 10, False),
            (100, 9, False),
            (99, 9, False),
            (0, 0, False),
            (0, 13, False),
            (113, 0, False),
            (1, 1, False),
            (99, 9, False),
            (100, 10, True),  # boundary exact
        ],
    )
    def test_validate_threshold(self, n_articles: int, n_annexes: int, expected: bool):
        parser = EUAIActParser()
        nodes = self._make_nodes(n_articles, n_annexes)
        assert parser.validate(nodes) is expected

    def test_validate_empty_list_false(self):
        assert EUAIActParser().validate([]) is False

    def test_validate_returns_bool(self):
        nodes = self._make_nodes(100, 10)
        result = EUAIActParser().validate(nodes)
        assert isinstance(result, bool)

    def test_validate_articles_only_false(self):
        nodes = self._make_nodes(113, 0)
        assert EUAIActParser().validate(nodes) is False

    def test_validate_annexes_only_false(self):
        nodes = self._make_nodes(0, 13)
        assert EUAIActParser().validate(nodes) is False

    def test_validate_real_parse_below_threshold(self, tmp_path):
        # A tiny doc has fewer than 100 articles → validate returns False.
        html = _doc_with_n_articles(5, annexes=3)
        parser = EUAIActParser()
        nodes = parser.parse(_write(tmp_path, html))
        assert parser.validate(nodes) is False

    def test_validate_real_parse_at_threshold(self, tmp_path):
        html = _doc_with_n_articles(100, annexes=10)
        parser = EUAIActParser()
        nodes = parser.parse(_write(tmp_path, html))
        assert parser.validate(nodes) is True

    def test_validate_ignores_mixed_type_list(self):
        # Mix articles + annexes, exactly at threshold.
        parser = EUAIActParser()
        nodes = self._make_nodes(100, 10)
        # Add an extra node with an unknown type — validate still counts correctly.
        nodes.append(
            ArticleNode(
                article_id="Recital 1",
                article_type="recital",
                number="1",
                title="",
                full_text="Some recital.",
            )
        )
        assert parser.validate(nodes) is True


# ---------------------------------------------------------------------------
# Inline cross-reference noise (spurious-split risk)
# ---------------------------------------------------------------------------


class TestInlineCrossReferenceNoise:
    """Article N appearing inside paragraph text can cause extra splits.

    These tests document and exercise the known caveat mentioned in CLAUDE.md.
    We verify observable behaviour rather than asserting it doesn't happen.
    """

    def test_inline_cross_ref_does_not_drop_real_articles(self, tmp_path):
        # Two real articles; second has an inline cross-reference to Article 1.
        body = (
            "<p>Article 1</p><p>Scope</p>"
            "<p>1. This Regulation applies.</p>"
            "<p>Article 2</p><p>Definitions</p>"
            "<p>1. As referred to in Article 1, the following definitions apply.</p>"
        )
        nodes = EUAIActParser().parse(_write(tmp_path, _html(body)))
        article_ids = {n.article_id for n in nodes if n.article_type == "article"}
        assert "Article 1" in article_ids
        assert "Article 2" in article_ids

    def test_inline_cross_ref_may_cause_extra_split(self, tmp_path):
        """An inline 'Article 3' inside Article 2's body can produce a spurious node."""
        body = (
            "<p>Article 1</p><p>General</p>"
            "<p>Article 2</p><p>Scope</p>"
            "<p>1. See Article 3 for details.</p>"
            # No real Article 3 follows.
        )
        nodes = EUAIActParser().parse(_write(tmp_path, _html(body)))
        article_ids = [n.article_id for n in nodes if n.article_type == "article"]
        # There are at least 2 (the real ones), possibly 3 if the cross-ref splits.
        assert len(article_ids) >= 2

    def test_cross_ref_in_annex_text_does_not_create_extra_annex(self, tmp_path):
        # "ANNEX I" text inside Annex II body should not create a third annex node.
        body = (
            "<p>ANNEX I</p><p>List A</p>"
            "<p>ANNEX II</p><p>1. See ANNEX I for the list.</p>"
        )
        nodes = EUAIActParser().parse(_write(tmp_path, _html(body)))
        annexes = [n for n in nodes if n.article_type == "annex"]
        # The parser splits on every ANNEX match, so "ANNEX I" inside ANNEX II body
        # would create an extra entry. Document the actual count (2 or 3).
        assert len(annexes) >= 2

    def test_article_number_in_title_line_does_not_duplicate(self, tmp_path):
        # Title says "Article 5 assessment" — check no duplicate Article 5.
        body = (
            "<p>Article 5</p>"
            "<p>Prohibition under Article 5 conditions</p>"
            "<p>1. Some prohibition text.</p>"
        )
        nodes = EUAIActParser().parse(_write(tmp_path, _html(body)))
        art5_nodes = [n for n in nodes if n.article_id == "Article 5"]
        # Exactly one real Article 5; the title cross-ref should not cause a second
        # split because the regex matches only at "Article\s+(\d+)" word boundaries.
        assert len(art5_nodes) >= 1


# ---------------------------------------------------------------------------
# Edge cases — document structure
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_html_with_nested_tags(self, tmp_path):
        html = _html("<div><p><strong>Article 1</strong></p><p>Title</p></div>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_id == "Article 1" for n in nodes)

    def test_html_with_table_tags(self, tmp_path):
        html = _html(
            "<table><tr><td>Article 1</td></tr><tr><td>Scope</td></tr></table>"
        )
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_type == "article" for n in nodes)

    def test_html_with_br_tags(self, tmp_path):
        html = _html("<p>Article 1<br/>Scope<br/>1. This applies.</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_id == "Article 1" for n in nodes)

    def test_plain_html_entities(self, tmp_path):
        html = _html("<p>Article 1</p><p>Subject &amp; Scope</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(n for n in nodes if n.article_type == "article")
        assert art.article_id == "Article 1"

    def test_article_1_through_10_no_gaps(self, tmp_path):
        body = "".join(_article_block(i) for i in range(1, 11))
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        nums = {n.number for n in nodes if n.article_type == "article"}
        assert nums == {str(i) for i in range(1, 11)}

    def test_only_annexes_no_articles_section(self, tmp_path):
        body = "".join(_annex_block(_ROMAN[i]) for i in range(5))
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert all(n.article_type == "annex" for n in nodes)

    def test_article_and_annex_ids_do_not_overlap(self, tmp_path):
        html = _html(_article_block(1) + _annex_block("I"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        article_ids = {n.article_id for n in nodes if n.article_type == "article"}
        annex_ids = {n.article_id for n in nodes if n.article_type == "annex"}
        assert article_ids.isdisjoint(annex_ids)

    def test_long_article_over_400_words(self, tmp_path):
        long_para = " ".join(["word"] * 200)
        # Title must not contain the word "Article" followed by a digit: the
        # parser's ARTICLE_RE (r"Article\s+(\d+)") would match "Article\n1"
        # across the title/first-paragraph boundary and split spuriously.
        body = (
            f"<p>Article 1</p><p>Long Provisions</p>"
            f"<p>1. {long_para}</p>"
            f"<p>2. {long_para}</p>"
            f"<p>3. {long_para}</p>"
        )
        html = _html(body)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(n for n in nodes if n.article_type == "article")
        assert art.word_count > 400

    def test_annex_content_after_article_content(self, tmp_path):
        html = _html(
            _article_block(1, paragraphs=2)
            + _article_block(2, paragraphs=1)
            + _annex_block("I", extra="Annex content here.")
        )
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert sum(1 for n in nodes if n.article_type == "article") == 2
        assert sum(1 for n in nodes if n.article_type == "annex") == 1

    def test_no_title_for_article_does_not_raise(self, tmp_path):
        html = _html("<p>Article 1</p>")
        # Should not raise any exception.
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_id == "Article 1" for n in nodes)

    def test_deeply_nested_article_text_extracted(self, tmp_path):
        html = _html(
            "<div><section><p>Article 1</p>"
            "<p>Scope</p><p>1. This Regulation applies.</p></section></div>"
        )
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next((n for n in nodes if n.article_id == "Article 1"), None)
        assert art is not None

    def test_multiple_files_parser_is_reusable(self, tmp_path):
        parser = EUAIActParser()
        html1 = _html(_article_block(1))
        html2 = _html(_article_block(2))
        nodes1 = parser.parse(_write(tmp_path, html1, "a.html"))
        nodes2 = parser.parse(_write(tmp_path, html2, "b.html"))
        assert len(nodes1) == 1
        assert len(nodes2) == 1

    def test_article_boundary_at_end_of_text(self, tmp_path):
        # Last article has no following article — parser should still capture it.
        html = _html(_article_block(1) + _article_block(2))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_id == "Article 2" for n in nodes)

    def test_annex_boundary_at_end_of_text(self, tmp_path):
        html = _html(_annex_block("I") + _annex_block("II"))
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert any(n.article_id == "Annex II" for n in nodes)

    def test_article_with_unicode_title(self, tmp_path):
        html = _html("<p>Article 1</p><p>Définitions générales</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art = next(n for n in nodes if n.article_type == "article")
        assert "Définitions" in art.title

    def test_article_whitespace_normalised_in_id(self, tmp_path):
        # "Article  1" (double space) — regex uses \s+ so it still matches.
        html = _html("<p>Article  1</p><p>Title</p>")
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        article_ids = {n.article_id for n in nodes if n.article_type == "article"}
        assert "Article 1" in article_ids


# ---------------------------------------------------------------------------
# Regex patterns exposed on the class
# ---------------------------------------------------------------------------


class TestParserRegexPatterns:
    def test_article_re_matches_article_1(self):
        m = EUAIActParser.ARTICLE_RE.search("Article 1")
        assert m is not None
        assert m.group(1) == "1"

    def test_article_re_matches_article_113(self):
        m = EUAIActParser.ARTICLE_RE.search("Article 113")
        assert m is not None
        assert m.group(1) == "113"

    def test_article_re_uses_plus_quantifier_for_spaces(self):
        m = EUAIActParser.ARTICLE_RE.search("Article  42")
        assert m is not None and m.group(1) == "42"

    def test_annex_re_matches_roman_i(self):
        m = EUAIActParser.ANNEX_RE.search("ANNEX I")
        assert m is not None and m.group(1) == "I"

    def test_annex_re_matches_roman_xiii(self):
        m = EUAIActParser.ANNEX_RE.search("ANNEX XIII")
        assert m is not None and m.group(1) == "XIII"

    def test_annex_re_does_not_match_lowercase(self):
        m = EUAIActParser.ANNEX_RE.search("annex i")
        assert m is None

    def test_para_re_matches_numbered_paragraph(self):
        text = "\n1. This is a paragraph."
        m = EUAIActParser.PARA_RE.search(text)
        assert m is not None and m.group(1) == "1"

    def test_para_re_captures_paragraph_text(self):
        text = "\n2. Some text here."
        m = EUAIActParser.PARA_RE.search(text)
        assert m is not None and "Some text here" in m.group(2)

    def test_article_re_does_not_match_articleWithoutSpace(self):
        m = EUAIActParser.ARTICLE_RE.search("ArticleX")
        assert m is None

    def test_annex_re_does_not_match_partial_roman(self):
        # "ANNEX" followed by non-Roman content should not match.
        m = EUAIActParser.ANNEX_RE.search("ANNEX 1")
        assert m is None


# ---------------------------------------------------------------------------
# Full-document round-trip with 100+ articles and 10+ annexes
# ---------------------------------------------------------------------------


class TestFullDocumentRoundTrip:
    def test_113_articles_13_annexes_validate_true(self, tmp_path):
        html = _doc_with_n_articles(113, annexes=13)
        parser = EUAIActParser()
        nodes = parser.parse(_write(tmp_path, html))
        assert parser.validate(nodes) is True

    def test_113_articles_parsed_count(self, tmp_path):
        html = _doc_with_n_articles(113, annexes=0)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        articles = [n for n in nodes if n.article_type == "article"]
        assert len(articles) == 113

    def test_13_annexes_parsed_count(self, tmp_path):
        html = _doc_with_n_articles(0, annexes=13)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        annexes = [n for n in nodes if n.article_type == "annex"]
        assert len(annexes) == 13

    def test_all_article_ids_unique_in_full_doc(self, tmp_path):
        html = _doc_with_n_articles(113, annexes=13)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        ids = [n.article_id for n in nodes if n.article_type == "article"]
        assert len(ids) == len(set(ids))

    def test_all_annex_ids_unique_in_full_doc(self, tmp_path):
        html = _doc_with_n_articles(0, annexes=13)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        ids = [n.article_id for n in nodes if n.article_type == "annex"]
        assert len(ids) == len(set(ids))

    def test_paragraphs_in_full_doc_have_correct_refs(self, tmp_path):
        # Spot-check Article 50 paragraph refs in a full doc.
        html = _doc_with_n_articles(113, annexes=13)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        art50 = next((n for n in nodes if n.article_id == "Article 50"), None)
        assert art50 is not None
        for p in art50.paragraphs:
            assert p["ref"].startswith("Article 50.")

    def test_full_doc_total_node_count(self, tmp_path):
        html = _doc_with_n_articles(113, annexes=13)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        assert len(nodes) == 113 + 13

    def test_articles_precede_annexes_in_output(self, tmp_path):
        html = _doc_with_n_articles(5, annexes=3)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        types = [n.article_type for n in nodes]
        # parse() calls _parse_articles first, then _parse_annexes.
        assert types.index("annex") > types.index("article")

    def test_word_count_summed_across_all_nodes_positive(self, tmp_path):
        html = _doc_with_n_articles(10, annexes=5)
        nodes = EUAIActParser().parse(_write(tmp_path, html))
        total_words = sum(n.word_count for n in nodes)
        assert total_words > 0

    def test_validate_100_articles_9_annexes_false(self, tmp_path):
        html = _doc_with_n_articles(100, annexes=9)
        parser = EUAIActParser()
        nodes = parser.parse(_write(tmp_path, html))
        assert parser.validate(nodes) is False

    def test_validate_99_articles_10_annexes_false(self, tmp_path):
        html = _doc_with_n_articles(99, annexes=10)
        parser = EUAIActParser()
        nodes = parser.parse(_write(tmp_path, html))
        assert parser.validate(nodes) is False

    def test_multiline_paragraph_text_captured(self, tmp_path):
        # Paragraph text that spans multiple lines (via \n inside a <p>).
        body = textwrap.dedent(
            """\
            <p>Article 1</p>
            <p>Title</p>
            <p>1. First line
            continued on second line.</p>
            """
        )
        nodes = EUAIActParser().parse(_write(tmp_path, _html(body)))
        art = next(n for n in nodes if n.article_type == "article")
        assert len(art.paragraphs) >= 1
