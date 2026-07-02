"""Source adapter for the `jeroenherczeg/eu-ai-act` Hugging Face dataset.

That dataset ships the EU AI Act already parsed from EUR-Lex Formex XML into
clean, structured chunks (parquet). We map it straight onto :class:`ArticleNode`
so the rest of the pipeline (chunker → indexer → retriever → generator) runs
unchanged — no regex HTML parsing needed for this source.

Relevant columns: ``language``, ``chunk_type`` (``article_full`` | ``paragraph``
| ``annex_item`` | ``recital``), ``text``, ``article_no``, ``paragraph_no``,
``annex_no`` (Roman), ``annex_section``.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from .parser import ArticleNode

log = structlog.get_logger(__name__)

HF_REPO = "jeroenherczeg/eu-ai-act"
PARQUET_FILE = "ai_act_chunks.parquet"


class HFDatasetLoader:
    def __init__(self, parquet_path: str, language: str = "en") -> None:
        self.parquet_path = parquet_path
        self.language = language

    def load(self) -> list[ArticleNode]:
        import pyarrow.parquet as pq

        rows = pq.read_table(self.parquet_path).to_pylist()
        rows = [r for r in rows if r.get("language") == self.language]

        nodes = self._build_articles(rows) + self._build_annexes(rows)
        articles = sum(1 for n in nodes if n.article_type == "article")
        annexes = sum(1 for n in nodes if n.article_type == "annex")
        log.info("hf_dataset_loaded", language=self.language, articles=articles, annexes=annexes)
        return nodes

    def _build_articles(self, rows: list[dict]) -> list[ArticleNode]:
        full = {r["article_no"]: r for r in rows if r["chunk_type"] == "article_full"}
        paras: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            if r["chunk_type"] == "paragraph" and r.get("article_no") is not None:
                paras[r["article_no"]].append(r)

        nodes: list[ArticleNode] = []
        for num in sorted(full):
            text = full[num]["text"] or ""
            paragraphs = [
                {
                    "num": str(p["paragraph_no"]),
                    "ref": f"Article {num}.{p['paragraph_no']}",
                    "text": (p["text"] or "").strip(),
                }
                for p in sorted(paras.get(num, []), key=lambda p: p["paragraph_no"] or 0)
            ]
            nodes.append(
                ArticleNode(
                    article_id=f"Article {num}",
                    article_type="article",
                    number=str(num),
                    title=self._title(text),
                    full_text=text,
                    paragraphs=paragraphs,
                )
            )
        return nodes

    def _build_annexes(self, rows: list[dict]) -> list[ArticleNode]:
        sections: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["chunk_type"] == "annex_item":
                annex = r.get("annex_no")
                if annex and annex != "?":  # skip the single unparseable item
                    sections[annex].append(r)

        nodes: list[ArticleNode] = []
        for annex in sorted(sections, key=self._roman_key):
            parts = sorted(sections[annex], key=lambda r: r.get("annex_section") or "")
            full_text = "\n\n".join((p["text"] or "").strip() for p in parts)
            nodes.append(
                ArticleNode(
                    article_id=f"Annex {annex}",
                    article_type="annex",
                    number=annex,
                    title=f"Annex {annex}",
                    full_text=full_text,
                    paragraphs=[],
                )
            )
        return nodes

    @staticmethod
    def _title(text: str) -> str:
        first_line = text.split("\n", 1)[0]
        if " — " in first_line:
            return first_line.split(" — ", 1)[1].strip().strip("`").strip()
        return first_line.strip()

    _ROMAN = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
        "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    }

    def _roman_key(self, roman: str) -> int:
        return self._ROMAN.get(roman, 99)
