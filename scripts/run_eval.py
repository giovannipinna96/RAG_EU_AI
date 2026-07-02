"""Evaluate the system against eval/test_set.json.

Measures reference recall, keyword recall, format compliance, conciseness, and
latency, then writes eval/results.json.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx

API = os.environ.get("RAG_API_URL", "http://localhost:8000/answer")
VALID_REF = re.compile(r"^(Article \d+(\.\d+)?|Annex [IVX]+(\.\d+)?)$")


def _ref_matches(found: str, expected: str) -> bool:
    """A returned reference satisfies an expected gold ref.

    Matches the exact ref or a sub-point of it ("Article 5" -> "Article 5.2"),
    but NOT a different number that merely shares a prefix ("Article 5" must not
    match "Article 50"). A bare ``startswith`` produced that false positive.
    """
    return found == expected or found.startswith(expected + ".")


def run() -> None:
    test_set = json.loads(Path("eval/test_set.json").read_text())
    # Heavy configs (LLM rerank + chunk voting + LightRAG mix) can exceed 30s per
    # query; a ReadTimeout there would abort the whole eval. Configurable.
    client = httpx.Client(timeout=float(os.environ.get("EVAL_TIMEOUT", "120")))
    results = []

    for tc in test_set:
        start = time.time()
        resp = client.post(API, json=[{"role": "user", "content": tc["question"]}])
        latency = time.time() - start

        # Guard non-200 responses so one API error doesn't crash the whole eval.
        if resp.status_code != 200:
            print(f"ERROR {tc['id']}: HTTP {resp.status_code} — {resp.text[:120]}")
            results.append({
                "id": tc["id"], "latency_s": round(latency, 2), "format_ok": False,
                "ref_recall": 0.0, "keyword_recall": 0.0, "concise": False,
                "refs": [], "answer_preview": f"HTTP {resp.status_code}",
            })
            continue
        data = resp.json()

        format_ok = all(VALID_REF.match(r) for r in data["references"])

        refs_found = sum(
            1
            for exp in tc["expected_refs"]
            if any(_ref_matches(r, exp) for r in data["references"])
        )
        ref_recall = refs_found / len(tc["expected_refs"]) if tc["expected_refs"] else 1.0

        answer_lower = data["answer"].lower()
        kw_found = sum(1 for kw in tc["keywords"] if kw.lower() in answer_lower)
        kw_recall = kw_found / len(tc["keywords"]) if tc["keywords"] else 1.0

        sentences = len([s for s in data["answer"].split(".") if s.strip()])
        concise = sentences <= 4

        results.append(
            {
                "id": tc["id"],
                "latency_s": round(latency, 2),
                "format_ok": format_ok,
                "ref_recall": round(ref_recall, 2),
                "keyword_recall": round(kw_recall, 2),
                "concise": concise,
                "refs": data["references"],
                "answer_preview": data["answer"][:120],
            }
        )

        status = "PASS" if format_ok and ref_recall >= 0.5 and concise else "FAIL"
        print(
            f"{status} {tc['id']}: latency={latency:.2f}s refs={data['references']} "
            f"format={format_ok} ref_recall={ref_recall:.0%} kw_recall={kw_recall:.0%}"
        )

    print(f"\n{'=' * 50}")
    if not results:
        print("No results (empty test set).")
        return
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_ref = sum(r["ref_recall"] for r in results) / len(results)
    avg_kw = sum(r["keyword_recall"] for r in results) / len(results)
    print(f"Avg latency:    {avg_latency:.2f}s")
    print(f"Ref recall:     {avg_ref:.0%}")
    print(f"Keyword recall: {avg_kw:.0%}")
    print(f"All formats OK: {all(r['format_ok'] for r in results)}")
    print(f"All concise:    {all(r['concise'] for r in results)}")

    Path("eval/results.json").write_text(json.dumps(results, indent=2))
    print("\nResults saved to eval/results.json")


if __name__ == "__main__":
    run()
