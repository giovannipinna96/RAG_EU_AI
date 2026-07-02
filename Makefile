# Convenience targets. All Python goes through uv (no pip/poetry/conda).
# Heavy ML/GPU work belongs in SLURM jobs (see slurm/), not the login node.

.PHONY: help setup setup-ml lint format type test test-all eval warmup download clean

help:
	@echo "setup     - create venv + install core/dev deps (login-node safe)"
	@echo "setup-ml  - add embedding/reranker extras (GPU node only)"
	@echo "lint      - ruff check"
	@echo "format    - ruff format + autofix"
	@echo "type      - mypy on src"
	@echo "test      - offline unit tests"
	@echo "test-all  - full suite incl. integration (needs running stack)"
	@echo "download  - fetch EU AI Act HTML"
	@echo "eval      - run evaluation against RAG_API_URL"
	@echo "warmup    - warm the response cache"

setup:
	uv sync

setup-ml:
	uv sync --extra ml

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

type:
	uv run mypy src

test:
	uv run pytest -q

test-all:
	RUN_INTEGRATION=1 uv run pytest -q

download:
	uv run python scripts/download.py

eval:
	uv run python scripts/run_eval.py

warmup:
	uv run python scripts/warmup_cache.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
