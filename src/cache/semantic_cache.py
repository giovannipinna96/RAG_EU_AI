"""Two-level response cache: exact hash + cosine-similarity semantic match.

Primary backend is Redis. If Redis is unreachable at startup the cache falls
back to an in-process store so the API keeps serving (without cross-process
sharing). Cosine matching catches rephrasings of an already-answered question.

Guard layers (outermost to innermost):
  1. Exact-match (SHA-256 of query+history) — zero false-positive risk.
  2. Ref-scoped semantic match — the incoming query's Article/Annex ref-set
     must equal the stored entry's ref-set before cosine similarity is checked
     (prevents Article-5 queries from hitting Article-6 cached answers).
     If ``cache_require_ref_match`` is False, or a query has no refs, the
     ref-set guard is skipped and pure cosine similarity is used.
  3. Cosine threshold — only entries above ``cache_similarity_threshold``
     (default 0.97) are returned.
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata

import numpy as np
import structlog

from ..clients import get_embedding_model
from ..config import settings

log = structlog.get_logger(__name__)


def _normalise_query(query: str) -> str:
    """Return a case-folded, whitespace-normalised, unicode-NFC form of *query*.

    Used as the exact-match key so minor formatting differences (extra spaces,
    mixed case) still collide on the exact layer without touching semantics.
    """
    nfc = unicodedata.normalize("NFC", query)
    return " ".join(nfc.lower().split())


def _extract_refs_safe(query: str) -> list[str]:
    """Return Article/Annex refs from *query*; return [] on any error."""
    try:
        from ..retrieval.article_matcher import ArticleMatcher  # local import — offline-safe

        return ArticleMatcher().extract_refs(query)
    except Exception:  # noqa: BLE001
        return []


class _InMemoryBackend:
    """Minimal TTL key/value store mirroring the subset of Redis we use."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def _expired(self, key: str) -> bool:
        item = self._store.get(key)
        return item is not None and item[0] < time.time()

    def get(self, key: str):
        if key not in self._store or self._expired(key):
            self._store.pop(key, None)
            return None
        return self._store[key][1]

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (time.time() + ttl, value)

    def scan_iter(self, pattern: str, count: int = 100):
        prefix = pattern.rstrip("*")
        for key in list(self._store):
            if key.startswith(prefix) and not self._expired(key):
                yield key

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self._store.pop(key, None) is not None:
                removed += 1
        return removed


class SemanticCache:
    def __init__(self) -> None:
        self.threshold = settings.cache_similarity_threshold
        self.require_ref_match = settings.cache_require_ref_match
        self.ttl = settings.cache_ttl_seconds
        self._model = None
        self.backend = self._connect()

    def _connect(self):
        try:
            import redis

            client = redis.from_url(settings.redis_url)
            client.ping()
            log.info("cache_backend", backend="redis", url=settings.redis_url)
            return client
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_unavailable_using_memory_cache", error=str(exc))
            return _InMemoryBackend()

    @property
    def model(self):
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    def get(self, query: str, history: list[dict]) -> dict | None:
        # Layer 1: exact normalised-string match — no embedding needed, no
        # false-positive risk.  Stored under a separate "norm:" namespace so it
        # is always checked first and never confused with the SHA-256 exact key.
        norm_key = _normalise_query(query)
        norm_hash = hashlib.sha256(norm_key.encode()).hexdigest()[:16]
        cached_exact_norm = self.backend.get(f"norm:{norm_hash}")
        if cached_exact_norm:
            log.debug("cache_hit", layer="exact_norm")
            return json.loads(cached_exact_norm)

        # Legacy exact hash (query+history) kept for back-compat with existing
        # entries that were stored before this refactor.
        key = self._hash(query, history)
        cached = self.backend.get(f"exact:{key}")
        if cached:
            log.debug("cache_hit", layer="exact_hash")
            return json.loads(cached)

        # Layer 2+3: ref-scoped cosine similarity.
        q_refs = _extract_refs_safe(query)
        q_vec = self.model.encode(query, normalize_embeddings=True)
        for k in self.backend.scan_iter("sem:*", count=100):
            try:
                data = json.loads(self.backend.get(k))
                # Ref-set guard: when enabled and the query has refs, the stored
                # entry must carry the SAME ref-set.  This prevents legally
                # distinct questions from colliding (e.g. comp_2 Article-5 vs
                # Article-6).  If the stored entry pre-dates this feature
                # (no "refs" key), we skip it conservatively.
                if self.require_ref_match and q_refs:
                    stored_refs = data.get("refs")
                    if stored_refs is None or sorted(stored_refs) != sorted(q_refs):
                        continue
                sim = float(np.dot(q_vec, np.array(data["vec"])))
                if sim >= self.threshold:
                    log.debug("cache_hit", layer="semantic", sim=round(sim, 4))
                    return data["response"]
            except Exception:  # noqa: BLE001 — skip malformed/expired entries
                continue
        return None

    def set(self, query: str, history: list[dict], response: dict) -> None:
        # Write the normalised-exact key.
        norm_key = _normalise_query(query)
        norm_hash = hashlib.sha256(norm_key.encode()).hexdigest()[:16]
        self.backend.setex(f"norm:{norm_hash}", self.ttl, json.dumps(response))

        # Write the legacy exact hash.
        key = self._hash(query, history)
        self.backend.setex(f"exact:{key}", self.ttl, json.dumps(response))

        # Write the semantic entry with the ref-set embedded.
        refs = _extract_refs_safe(query)
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        self.backend.setex(
            f"sem:{key}",
            self.ttl,
            json.dumps({"vec": vec, "refs": refs, "response": response}),
        )

    def invalidate_all(self) -> int:
        keys = (
            list(self.backend.scan_iter("norm:*"))
            + list(self.backend.scan_iter("exact:*"))
            + list(self.backend.scan_iter("sem:*"))
        )
        return self.backend.delete(*keys) if keys else 0

    def _hash(self, query: str, history: list[dict]) -> str:
        content = query + "".join(m.get("content", "") for m in history)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
