"""Load test: latency and error rate under concurrent requests.

Hits a *running* API at RAG_API_URL (default http://localhost:8000/answer).
Marked integration; skipped unless RUN_INTEGRATION=1.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integration

API = os.environ.get("RAG_API_URL", "http://localhost:8000/answer")
QUESTIONS = [
    "What is a high-risk AI system?",
    "What does Article 5 prohibit?",
    "What transparency obligations exist?",
    "What are the penalties for non-compliance?",
    "Does the AI Act apply to open source?",
]


async def test_sequential_latency():
    async with httpx.AsyncClient(timeout=10) as client:
        for q in QUESTIONS:
            start = time.time()
            resp = await client.post(API, json=[{"role": "user", "content": q}])
            elapsed = time.time() - start
            assert resp.status_code == 200, f"Failed: {q}"
            assert elapsed < 5, f"Too slow ({elapsed:.1f}s): {q}"


async def test_concurrent_load():
    async with httpx.AsyncClient(timeout=15) as client:
        tasks = [
            client.post(API, json=[{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}])
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        errors = [r for r in results if r.status_code != 200]
        assert len(errors) == 0, f"{len(errors)} requests failed"
