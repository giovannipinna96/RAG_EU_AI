"""Query-centered snippet extraction.

Pure-Python and dependency-free, so it is safe to import anywhere — the
retriever's LLM-rerank prompt, the generator's context builder, and offline
unit tests all use it without pulling torch.
"""

from __future__ import annotations

import re

# Drop query stopwords shorter than this when picking the centering keyword.
KEYWORD_MIN_LEN = 4


def adaptive_snippet(text: str, query: str, length: int) -> str:
    """Return up to *length* chars from *text*, centered on the best query match.

    Picks the earliest in-text occurrence of any query content-word
    (``\\w+`` of length >= ``KEYWORD_MIN_LEN``) and centers a *length*-char
    window around it. If the text already fits in *length* chars, returns the
    whole text. If no query word matches, falls back to ``text[:length]``
    (legacy behavior). Window boundaries that hit the chunk boundary are
    shifted to keep the full *length* whenever possible. Truncation is marked
    with ``[...]`` so the LLM sees the context is partial. Centering is purely
    lexical -- no embeddings, no tokenizer.
    """
    if length <= 0 or not text:
        return ""
    if len(text) <= length:
        return text

    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= KEYWORD_MIN_LEN]
    text_lower = text.lower()
    hits = [text_lower.find(w) for w in words]
    hits = [h for h in hits if h >= 0]
    if not hits:
        return text[:length] + "[...]"

    center = min(hits)  # earliest match -- usually the start of the relevant paragraph
    half = length // 2
    start = max(0, center - half)
    end = min(len(text), start + length)
    start = max(0, end - length)  # shift back if we hit the right boundary

    snippet = text[start:end]
    prefix = "[...]" if start > 0 else ""
    suffix = "[...]" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"
