# CLAUDE.md — EU AI Act RAG

Development notes and conventions for this project. Read before making changes.

## What this is

A production-grade RAG system answering EU AI Act (Reg. 2024/1689) questions with
cited Article/Annex references, for the regenold Q&A competition. Spec lives in
`README_EU_AI_RAG.md` and `IMPLEMENTATION_GUIDE.md`; the HPC environment is
described in `demetra_hpc_guide.md`.

## Hard constraints (do not violate)

- **Package management is `uv` only** — `uv sync`, `uv run`, `uv pip`, `uv venv`.
  Never `pip`/`poetry`/`conda`.
- **This session runs on the Demetra login node.** No heavy compute here.
  Anything needing a GPU, model downloads, or long runs goes through SLURM
  (`slurm/*.slurm`) on compute nodes.
- **Docker is unavailable.** `Dockerfile` / `docker-compose.yml` are kept as
  portability deliverables only. The real execution path is Apptainer + SLURM.
- **Git identity:** the configured user is `giovannipinna96`. Do not author or
  reference "Claude" in commits.

## Environment

- System Python is 3.9; the project needs 3.11+. `uv` provisions its own 3.11
  (`uv python install 3.11`) — the venv is `.venv/`.
- Apptainer is available as a module: `module load apptainer/1.1.9-gcc-13.2.0-i4ns3xh`
  (not on the login PATH by default).
- GPUs (A100 40GB) are on `lovelace-01/-02` and `babbage`, reached via SLURM
  `--gres=gpu:1` on partitions `lovelace` / `Main`.

## Dependency layout (pyproject.toml)

- **core** (`[project.dependencies]`) — no torch; installs anywhere incl. the
  login node. Enough to import the app and run offline tests.
- **`ml` extra** — sentence-transformers, FlagEmbedding, bm25s, PyStemmer,
  lightrag-hku. Pulls torch → GPU compute node only (`uv sync --extra ml`).
- **`gpu` extra** — sglang. CUDA-only, A100 node.
- **`dev` group** — pytest, ruff, mypy (installed by default).

## Common commands

```bash
make setup        # uv sync (core + dev) — safe on login node
make test         # offline unit tests (integration auto-skipped)
make lint         # ruff check
make format       # ruff format + --fix
make type         # mypy src
make test-all     # RUN_INTEGRATION=1 — needs the full running stack
```

Run anything ad hoc with `uv run <cmd>`.

## Testing model

- Offline unit tests (parser, chunker w/ stub client, article matcher,
  normalizer) run anywhere and gate every change. 15 tests, no services needed.
- Integration tests are marked `@pytest.mark.integration` and **skipped unless
  `RUN_INTEGRATION=1`**. They need SGLang + Qdrant + Redis + a running API.
  `test_load.py` hits `RAG_API_URL` (default `http://localhost:8000/answer`).

## Architecture notes for editors

- `src/config.py` — single `settings` object (pydantic-settings v2, reads `.env`).
- `src/clients.py` — shared Qdrant client factory (honours `QDRANT_LOCAL_PATH`
  for embedded mode) and a process-cached SentenceTransformer loader. Use these
  instead of re-instantiating clients.
- `src/api/main.py` — components are **lazily initialised** via the `_Components`
  registry so importing the app never loads torch or opens connections. Keep it
  that way; tests rely on it.
- `src/cache/semantic_cache.py` — Redis with a transparent in-process fallback
  when Redis is unreachable.
- Ingestion/retrieval modules import `sentence_transformers`/`FlagEmbedding`/
  `bm25s` only inside functions or at module level guarded by the `ml` extra —
  never import them from code paths exercised by offline tests.

## Running on Demetra

See `slurm/README.md` for the full orchestration (sglang → infra → ingest →
api → warmup → eval), plus the single-node and embedded/no-infra shortcuts.

## Test corpus (Hugging Face mirror)

EUR-Lex blocks scripted downloads (HTTP 202 + empty body, anti-bot). Use the HF
mirror `jeroenherczeg/eu-ai-act` instead — the real Act, pre-parsed, ~548 KB,
reachable from Demetra:

```bash
uv sync --extra data                         # pyarrow + huggingface-hub
uv run python scripts/download_hf.py         # -> data/raw/ai_act_chunks.parquet
uv run python scripts/ingest_hf.py           # needs ml extra + SGLang (GPU node)
```

`src/ingestion/hf_loader.py` adapts the parquet → `ArticleNode`, so the chunker/
indexer/retriever/generator are unchanged. On Demetra: `sbatch slurm/ingest_hf.slurm`.
The regex `download.py`/`ingest.py` path remains for real EUR-Lex HTML if egress
is fixed. Note: the loader yields 12 annexes (upstream labels one annex `?`).

## Known caveats

- The parser is regex-over-full-text; inline "Article N" cross-references can
  create spurious splits. `EUAIActParser.validate()` guards the real document
  (≥100 articles, ≥10 annexes).
- `slurm/sglang_server.slurm` uses placeholder model IDs from the guide — verify
  the Gemma/EAGLE model paths on Hugging Face before a long GPU run.
- Apptainer/SLURM scripts could not be executed from the login node; validate
  them on a compute node before relying on them.
