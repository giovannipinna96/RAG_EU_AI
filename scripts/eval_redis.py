"""Validate the SemanticCache against a REAL Redis backend.

Offline unit tests use the in-process ``_InMemoryBackend`` mock, which returns
``str`` and cannot expose Redis-specific bugs (real redis-py returns ``bytes``,
``scan_iter`` yields ``bytes`` keys, ``setex`` arg order, etc.). This script
runs the actual ``SemanticCache`` against a live Redis (started by
``slurm/redis_eval.slurm``), with the embedding model STUBBED so no torch / GPU
is needed — it tests the *infrastructure*, not the embeddings.

Exit code 0 = all checks pass; 1 = at least one failure.
"""

from __future__ import annotations

import sys

import numpy as np

from src.cache.semantic_cache import SemanticCache, _InMemoryBackend

# --- stub embedding model -------------------------------------------------
# Deterministic, torch-free. "emotion" queries map to one unit vector, "fine"
# queries to an orthogonal one — so paraphrases collide (cos≈1) and unrelated
# topics don't (cos≈0).
_VEC_EMOTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_VEC_FINE = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _FakeModel:
    def encode(self, text: str, normalize_embeddings: bool = True):
        v = _VEC_FINE if "fine" in text.lower() or "penalt" in text.lower() else _VEC_EMOTION
        return v.copy()


_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    cache = SemanticCache()

    # 1. THE key infra check: did we actually connect to Redis, not fall back?
    is_redis = not isinstance(cache.backend, _InMemoryBackend)
    check(
        "connect_to_real_redis",
        is_redis,
        f"backend={type(cache.backend).__name__}"
        + ("" if is_redis else " (fell back to in-memory — Redis unreachable!)"),
    )
    if not is_redis:
        print("\nRedis backend not active; aborting infra eval.")
        return 1

    cache.invalidate_all()  # clean slate
    cache._model = _FakeModel()

    # 2. exact-norm round trip through real Redis (bytes -> json.loads)
    q = "What fine applies to prohibited practices?"
    resp = {"answer": "Up to EUR 35m", "references": ["Article 99"], "format_ok": True}
    cache.set(q, [], resp)
    got = cache.get(q, [])
    check("exact_norm_roundtrip", got == resp, f"got={got}")

    # 3. raw value is bytes on real Redis, yet handled correctly
    import hashlib

    norm_hash = hashlib.sha256(" ".join(q.lower().split()).encode()).hexdigest()[:16]
    raw = cache.backend.get(f"norm:{norm_hash}")
    check("redis_returns_bytes", isinstance(raw, (bytes, bytearray)), f"type={type(raw).__name__}")

    # 4. case / whitespace insensitivity on the norm layer
    got2 = cache.get("  WHAT   fine APPLIES to prohibited practices?  ", [])
    check("norm_case_whitespace_insensitive", got2 == resp)

    # 5. semantic layer: same ref-set + high similarity -> HIT
    cache.invalidate_all()
    stored_q = "Are emotion recognition systems prohibited under Article 5?"
    stored_resp = {"answer": "With exceptions", "references": ["Article 5"], "format_ok": True}
    cache.set(stored_q, [], stored_resp)
    para = "Is emotion recognition banned per Article 5?"  # same ref {Article 5}, cos≈1
    got3 = cache.get(para, [])
    check("semantic_hit_same_refs", got3 == stored_resp, f"got={got3}")

    # 6. comp_2 scenario: high similarity but DIFFERENT ref-set -> MISS
    diff_ref = "Are emotion recognition systems high-risk under Article 50?"  # {Article 50}
    got4 = cache.get(diff_ref, [])
    check("ref_scope_blocks_wrong_article", got4 is None, f"got={got4}")

    # 7. invalidate_all sweeps every namespace on real Redis
    removed = cache.invalidate_all()
    after = cache.get(stored_q, [])
    check("invalidate_all_clears", after is None, f"removed={removed}")

    # 8. persistence: a fresh client (new connection) sees a written entry
    cache.set(q, [], resp)
    fresh = SemanticCache()
    fresh._model = _FakeModel()
    got5 = fresh.get(q, [])
    check("persists_across_clients", got5 == resp, f"got={got5}")
    cache.invalidate_all()

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{'='*48}\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
