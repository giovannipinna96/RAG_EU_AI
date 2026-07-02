"""Pre-compute answers for ~300 predictable questions to warm the Redis cache."""

from __future__ import annotations

import os
import time

import httpx

API = os.environ.get("RAG_API_URL", "http://localhost:8000/answer")

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]

QUESTIONS: list[str] = []

# Per-article questions (113 articles).
QUESTIONS += [f"What does Article {i} of the EU AI Act establish?" for i in range(1, 114)]

# Per-annex questions (13 annexes).
QUESTIONS += [f"What does Annex {r} of the EU AI Act cover?" for r in ROMAN]

# Thematic and competition-style questions.
QUESTIONS += [
    "What AI practices are prohibited under the EU AI Act?",
    "What is the definition of an AI system?",
    "What is a high-risk AI system?",
    "What are the requirements for high-risk AI systems?",
    "What transparency obligations exist?",
    "What are the penalties for non-compliance?",
    "Does the AI Act apply to open source models?",
    "Who qualifies as a provider under the AI Act?",
    "Who qualifies as a deployer under the AI Act?",
    "What are the obligations for general-purpose AI?",
    "What is the role of the AI Office?",
    "When does the AI Act enter into force?",
    "What conformity assessment procedures are required?",
    "What are the obligations for importers of AI systems?",
    "How does the AI Act classify risk levels?",
    "Does the technical documentation of a high-risk AI system require to provide "
    "specifications regarding the required hardware?",
    "What must be included in technical documentation for high-risk AI?",
    "Are AI systems intended for emotion recognition from biometric data always prohibited?",
    "When is emotion recognition in AI prohibited?",
    "What exceptions exist for emotion recognition AI?",
    "Is an AI that transcribes doctor-patient conversations prohibited?",
    "How does the AI Act classify medical transcription AI?",
    "Is medical transcription AI high-risk under Annex III?",
    "How do Article 6 and Annex III relate to each other?",
    "What is the relationship between prohibited practices and high-risk systems?",
    "How do providers and deployers obligations differ?",
]


def warmup() -> None:
    total = len(QUESTIONS)
    success = 0
    start = time.time()
    print(f"Warming up cache with {total} questions...")

    with httpx.Client(timeout=30) as client:
        for i, q in enumerate(QUESTIONS):
            try:
                resp = client.post(API, json=[{"role": "user", "content": q}])
                if resp.status_code == 200:
                    success += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  Error: {q[:50]}... — {exc}")

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{total} ({success} OK) — {rate:.1f} q/s — ~{remaining:.0f}s left")

    print(f"\nDone: {success}/{total} cached in {time.time() - start:.0f}s")


if __name__ == "__main__":
    warmup()
