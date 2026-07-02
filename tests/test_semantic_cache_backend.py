"""Offline unit tests for _InMemoryBackend TTL/expiry/scan/delete behaviour.

The SemanticCache.get/set methods load a SentenceTransformer (torch) so they
are NOT tested here. We only test the _InMemoryBackend directly.
"""

from __future__ import annotations

import time

from src.cache.semantic_cache import _InMemoryBackend

# ---------------------------------------------------------------------------
# Basic get/setex
# ---------------------------------------------------------------------------


def test_setex_and_get_returns_value():
    b = _InMemoryBackend()
    b.setex("k1", 60, "hello")
    assert b.get("k1") == "hello"


def test_get_missing_key_returns_none():
    b = _InMemoryBackend()
    assert b.get("nonexistent") is None


def test_setex_overwrite_updates_value():
    b = _InMemoryBackend()
    b.setex("k", 60, "first")
    b.setex("k", 60, "second")
    assert b.get("k") == "second"


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_get_returns_none_after_ttl_expires():
    b = _InMemoryBackend()
    # Set a 1-second TTL; manipulate internal expiry directly to avoid sleeping
    b._store["k"] = (time.time() - 1, "expired_value")  # already past
    assert b.get("k") is None


def test_expired_entry_is_evicted_on_access():
    b = _InMemoryBackend()
    b._store["k"] = (time.time() - 1, "gone")
    b.get("k")  # triggers eviction
    assert "k" not in b._store


def test_non_expired_entry_is_not_evicted():
    b = _InMemoryBackend()
    b.setex("k", 3600, "alive")
    b.get("k")
    assert "k" in b._store


def test_expired_returns_true_for_past_expiry():
    b = _InMemoryBackend()
    b._store["k"] = (time.time() - 0.001, "old")
    assert b._expired("k") is True


def test_expired_returns_false_for_future_expiry():
    b = _InMemoryBackend()
    b._store["k"] = (time.time() + 3600, "fresh")
    assert b._expired("k") is False


def test_expired_returns_false_for_missing_key():
    b = _InMemoryBackend()
    assert b._expired("no_such_key") is False


# ---------------------------------------------------------------------------
# scan_iter
# ---------------------------------------------------------------------------


def test_scan_iter_returns_matching_prefix():
    b = _InMemoryBackend()
    b.setex("exact:abc", 60, "v1")
    b.setex("exact:def", 60, "v2")
    b.setex("sem:abc", 60, "v3")
    results = list(b.scan_iter("exact:*"))
    assert set(results) == {"exact:abc", "exact:def"}


def test_scan_iter_excludes_expired():
    b = _InMemoryBackend()
    b._store["exact:old"] = (time.time() - 1, "expired")
    b.setex("exact:new", 60, "alive")
    results = list(b.scan_iter("exact:*"))
    assert "exact:old" not in results
    assert "exact:new" in results


def test_scan_iter_empty_store_yields_nothing():
    b = _InMemoryBackend()
    assert list(b.scan_iter("exact:*")) == []


def test_scan_iter_no_match_yields_nothing():
    b = _InMemoryBackend()
    b.setex("sem:abc", 60, "v")
    assert list(b.scan_iter("exact:*")) == []


def test_scan_iter_wildcard_pattern_matches_all():
    b = _InMemoryBackend()
    b.setex("exact:a", 60, "1")
    b.setex("sem:b", 60, "2")
    # Pattern "" (empty after rstrip("*") with "*") matches everything
    results = list(b.scan_iter("*"))
    assert set(results) == {"exact:a", "sem:b"}


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_existing_key_returns_one():
    b = _InMemoryBackend()
    b.setex("k", 60, "v")
    assert b.delete("k") == 1


def test_delete_removes_key():
    b = _InMemoryBackend()
    b.setex("k", 60, "v")
    b.delete("k")
    assert b.get("k") is None


def test_delete_missing_key_returns_zero():
    b = _InMemoryBackend()
    assert b.delete("does_not_exist") == 0


def test_delete_multiple_keys():
    b = _InMemoryBackend()
    b.setex("k1", 60, "a")
    b.setex("k2", 60, "b")
    assert b.delete("k1", "k2") == 2
    assert b.get("k1") is None
    assert b.get("k2") is None


def test_delete_mix_of_existing_and_missing():
    b = _InMemoryBackend()
    b.setex("k1", 60, "a")
    result = b.delete("k1", "k_missing")
    assert result == 1


def test_delete_no_keys_returns_zero():
    b = _InMemoryBackend()
    assert b.delete() == 0


# ---------------------------------------------------------------------------
# invalidate_all style: set then scan then delete
# ---------------------------------------------------------------------------


def test_full_invalidation_flow():
    b = _InMemoryBackend()
    b.setex("exact:aa", 60, "r1")
    b.setex("exact:bb", 60, "r2")
    b.setex("sem:aa", 60, "r3")

    keys = list(b.scan_iter("exact:*")) + list(b.scan_iter("sem:*"))
    removed = b.delete(*keys) if keys else 0

    assert removed == 3
    assert list(b.scan_iter("exact:*")) == []
    assert list(b.scan_iter("sem:*")) == []
