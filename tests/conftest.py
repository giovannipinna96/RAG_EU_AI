"""Shared pytest fixtures and the integration-skip gate.

Offline unit tests (normalizer, parser, article matcher) always run. Tests that
need the full stack (SGLang + Qdrant + Redis) are marked ``integration`` and are
skipped unless ``RUN_INTEGRATION=1`` is set in the environment.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test — set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def sample_query():
    return [{"role": "user", "content": "What is the definition of an AI system?"}]


@pytest.fixture
def competition_questions():
    return [
        "Does the technical documentation of a high-risk AI system require to provide "
        "specifications regarding the required hardware?",
        "Are AI systems intended for emotion recognition from biometric data always prohibited?",
        "Is an AI that transcribes doctor-patient conversations prohibited? Or is it high-risk "
        "as per the use cases of Annex III of the AI Act?",
    ]
