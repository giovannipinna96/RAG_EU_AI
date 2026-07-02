"""Parse the EU AI Act HTML (EUR-Lex) into structured :class:`ArticleNode` objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)


@dataclass
class ArticleNode:
    """A single Article or Annex of the regulation."""

    article_id: str  # "Article 6" or "Annex III"
    article_type: str  # "article" or "annex"
    number: str  # "6" or "III"
    title: str
    full_text: str
    paragraphs: list[dict] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class EUAIActParser:
    """Regex/structure-based parser for the EUR-Lex plain-text rendering."""

    ARTICLE_RE = re.compile(r"Article\s+(\d+)")
    ANNEX_RE = re.compile(r"ANNEX\s+([IVX]+)")
    PARA_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)", re.DOTALL)

    def parse(self, file_path: str) -> list[ArticleNode]:
        html = Path(file_path).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)

        nodes: list[ArticleNode] = []
        nodes.extend(self._parse_articles(text))
        nodes.extend(self._parse_annexes(text))
        log.info("parsed_document", articles=len(nodes), source=file_path)
        return nodes

    def _parse_articles(self, text: str) -> list[ArticleNode]:
        splits = list(self.ARTICLE_RE.finditer(text))
        nodes: list[ArticleNode] = []

        for i, match in enumerate(splits):
            num = match.group(1)
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()

            lines = content.split("\n", 2)
            title = lines[1].strip() if len(lines) > 1 else ""

            paragraphs = [
                {
                    "num": m.group(1),
                    "ref": f"Article {num}.{m.group(1)}",
                    "text": m.group(2).strip(),
                }
                for m in self.PARA_RE.finditer(content)
            ]

            nodes.append(
                ArticleNode(
                    article_id=f"Article {num}",
                    article_type="article",
                    number=num,
                    title=title,
                    full_text=content,
                    paragraphs=paragraphs,
                )
            )
        return nodes

    def _parse_annexes(self, text: str) -> list[ArticleNode]:
        splits = list(self.ANNEX_RE.finditer(text))
        nodes: list[ArticleNode] = []

        for i, match in enumerate(splits):
            num = match.group(1)
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()

            nodes.append(
                ArticleNode(
                    article_id=f"Annex {num}",
                    article_type="annex",
                    number=num,
                    title=f"Annex {num}",
                    full_text=content,
                    paragraphs=[],
                )
            )
        return nodes

    def validate(self, nodes: list[ArticleNode]) -> bool:
        """Sanity-check the parse: the Act has 113 Articles and 13 Annexes."""
        articles = [n for n in nodes if n.article_type == "article"]
        annexes = [n for n in nodes if n.article_type == "annex"]
        log.info("parse_validation", articles=len(articles), annexes=len(annexes))
        return len(articles) >= 100 and len(annexes) >= 10
