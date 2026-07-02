"""Regex post-processor enforcing the competition's reference format.

Canonical forms:
    Article <arabic>[.<sub>]      e.g. "Article 6", "Article 6.2"
    Annex   <roman>[.<sub>]       e.g. "Annex III", "Annex III.2"

Anything that cannot be coerced to a valid form is dropped. The module is pure
Python and dependency-free.
"""

from __future__ import annotations

import re


class ReferenceNormalizer:
    ROMAN = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
        "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    }
    INT_TO_ROMAN = {v: k for k, v in ROMAN.items()}

    VALID_ARTICLE = re.compile(r"^Article \d+(\.\d+)?$")
    VALID_ANNEX = re.compile(r"^Annex [IVX]+(\.\d+)?$")

    def normalize(self, references: list[str]) -> list[str]:
        normalized: set[str] = set()
        for ref in references:
            clean = self._apply_transforms(ref.strip())
            if self.VALID_ARTICLE.match(clean) or self.VALID_ANNEX.match(clean):
                normalized.add(clean)
        return sorted(normalized, key=self._sort_key)

    def _apply_transforms(self, ref: str) -> str:
        # Whitespace: collapse runs of spaces and strip spaces around the
        # sub-point dot ("Annex III . 2" -> "Annex III.2", "Article  6" -> "Article 6").
        ref = re.sub(r"\s+", " ", ref).strip()
        ref = re.sub(r"\s*\.\s*", ".", ref)
        # Canonicalise the leading keyword's case ("article"/"annex" -> "Article"/"Annex").
        ref = re.sub(r"^\s*article\b", "Article", ref, flags=re.I)
        ref = re.sub(r"^\s*annex\b", "Annex", ref, flags=re.I)
        # "Art. N" / "Art N" -> "Article N"
        ref = re.sub(r"\bArt\.?\s*(\d+)", r"Article \1", ref, flags=re.I)
        # Numeric sub-point separators -> dot, for BOTH Article and Annex and any
        # numeral kind: "Article 3/2", "Article 3(2)", "Article 3-2", "Annex 3(2)",
        # "Annex III-2", "Annex III/2" -> "...3.2" / "...III.2". The trailing "\)?"
        # absorbs the closing paren of "(2)". Applied left-to-right one group at a
        # time so the FIRST numeric group of "Article 5(1)(f)" becomes ".1".
        sep = re.compile(r"(Article\s+\d+|Annex\s+(?:[IVX]+|\d+))\s*[/().-]\s*(\d+)\)?", re.I)
        prev = None
        while prev != ref:
            prev = ref
            ref = sep.sub(r"\1.\2", ref, count=1)
        # Uppercase a Roman numeral following "Annex " ("annex iii" -> "Annex III").
        m = re.match(r"Annex\s+([ivx]+)(\.\d+)?$", ref, re.I)
        if m:
            ref = f"Annex {m.group(1).upper()}" + (m.group(2) or "")
        # "Article ROMAN" -> "Article arabic"
        m = re.match(r"Article\s+([IVX]+)(?:\.(\d+))?$", ref, re.I)
        if m and m.group(1).upper() in self.ROMAN:
            num = self.ROMAN[m.group(1).upper()]
            ref = f"Article {num}" + (f".{m.group(2)}" if m.group(2) else "")
        # "Annex arabic" -> "Annex ROMAN"
        m = re.match(r"Annex\s+(\d+)(?:\.(\d+))?$", ref, re.I)
        if m:
            roman = self.INT_TO_ROMAN.get(int(m.group(1)), m.group(1))
            ref = f"Annex {roman}" + (f".{m.group(2)}" if m.group(2) else "")
        # The competition format expresses only "<keyword> <num>[.<num>]"; a
        # trailing letter sub-point ("Article 5.1(f)", "Article 5.1.a") cannot be
        # encoded, so peel trailing "(x)"/".x" groups to keep the nearest valid
        # numeral form instead of dropping the reference. Guarded to a single
        # letter after "." or "(...)" so roman annexes (e.g. "Annex IX") are safe.
        prev = None
        while prev != ref:
            prev = ref
            ref = re.sub(r"(?:\([a-z]\)|\.[a-z])$", "", ref, flags=re.I)
        return ref

    def _sort_key(self, ref: str) -> tuple[int, int, int]:
        sub_match = re.search(r"\.(\d+)$", ref)
        sub = int(sub_match.group(1)) if sub_match else 0
        if ref.startswith("Article"):
            n = re.search(r"\d+", ref)
            return (0, int(n.group()) if n else 0, sub)
        roman = ref.split()[1].split(".")[0]
        return (1, self.ROMAN.get(roman.upper(), 99), sub)
