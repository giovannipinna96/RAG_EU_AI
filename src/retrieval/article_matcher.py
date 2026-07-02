"""Regex-based Article/Annex reference extractor.

Pure-Python and dependency-free, so it is safe to import anywhere (including
offline unit tests).
"""

from __future__ import annotations

import re


class ArticleMatcher:
    ARTICLE_RE = re.compile(r"(?:Article|Art\.?)\s+(\d+)(?:\.(\d+))?", re.IGNORECASE)
    ANNEX_RE = re.compile(r"Annex\s+([IVX]+|\d+)(?:\.(\d+))?", re.IGNORECASE)

    ROMAN = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
        "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    }
    INT_TO_ROMAN = {v: k for k, v in ROMAN.items()}

    def extract_refs(self, text: str) -> list[str]:
        """Return the sorted set of Article/Annex references mentioned in ``text``."""
        refs: set[str] = set()
        for m in self.ARTICLE_RE.finditer(text):
            refs.add(f"Article {m.group(1)}")
        for m in self.ANNEX_RE.finditer(text):
            num = m.group(1)
            num = self.INT_TO_ROMAN.get(int(num), num) if num.isdigit() else num.upper()
            refs.add(f"Annex {num}")
        return sorted(refs)
