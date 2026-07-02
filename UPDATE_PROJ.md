# UPDATE_PROJ.md — Implementation Log

Running log of implementation milestones. Newest first.

---

## 2026-05-27 — SGLang + EAGLE3 working (the guide's technique) + eval scorer fix

The guide's intended serving stack (`--speculative-algorithm EAGLE3`) now runs on
this node — job 54737, model **Qwen2.5-14B-Instruct** + draft
`ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat`.

**How the CUDA wall was cleared:** driver is CUDA 12.4 (r550); the prebuilt SGLang
*containers* moved to CUDA 13 (won't run), but the **`cu129` image
(`lmsysorg/sglang:v0.5.6.post2-cu129-amd64`) runs via CUDA minor-version compat** —
probe confirmed `torch 2.9.1+cu129 cuda 12.9 avail True` on the 12.4 driver. So the
flashinfer-from-pip dead-end is sidestepped entirely: the container ships flashinfer
+ sgl-kernel pre-built. New script `slurm/sglang_eagle.slurm` (build SIF once →
CUDA-compat probe → EAGLE3 serve → reuse indices → API → eval, all fail-loud).
SIF build is ~45 min one-time (huge image, NFS squashfs); reused thereafter.

**Quality jump from the 14B:** eval ref_recall rose materially vs the 7B-AWQ —
comp_1 went **0% → 100%** (now cites **Annex IV**, the gold ref it kept missing).
Latency ~1.24s avg (EAGLE3 active: draft cuda graphs captured); EAGLE's gain is
modest here because answers are short (<~150 tok), so the win is the 14B's citation
quality, not raw speed.

**Eval scorer bug found + fixed (`scripts/run_eval.py`):** reference matching used
a bare `startswith`, so gold "Article 5" spuriously matched the returned
"Article 50" → comp_2 reported a false 100%. Added `_ref_matches()` (exact or
sub-point `Article 5.2`, never `Article 50`). Corrected 14B ref_recall: comp_1 100%,
comp_2 0%, comp_3 50% → **~50%** (vs 17% with 7B-AWQ). This fix matters for ALL
future tuning — the old scorer would have hidden real misses.

---

## 2026-05-26 — SGLang-native feasibility probe (re: CUDA 12.4 + EAGLE)

User found sources stating SGLang supports CUDA 12.4 + A100 (correct), challenging
the earlier "needs 12.8" note. Investigated installing SGLang **natively via uv**
(no container) to unlock the guide's `--speculative-algorithm EAGLE` path, which
vLLM 0.6.6 could not run (draft-model vocab mismatch; n-gram engine AssertionError).

**Confirmed facts (node lovelace-01):**
- Driver **550.54.15 → CUDA 12.4** max. The A100s are 80GB PCIe (not 40GB as I'd noted).
- Earlier "SGLang needs 12.8" was **wrong**: the logged SGLang failures were disk
  ("no space left on device", pre-BIG_STORE) and slow NFS image unpack — not a CUDA
  error. Corrected the comment in `00_common.sh`.

**Probe (`slurm/sglang_probe.slurm`, jobs 54696/54697):**
- Added a `gpu-native` extra (`sglang>=0.4.0`, no `[all]`) so `uv sync` manages it
  (a bare `uv pip install` gets wiped by `uv run`'s implicit re-sync — that was the
  first probe's `ModuleNotFoundError: sglang`).
- `uv sync --extra ml --extra gpu-native` resolves & installs **sglang 0.4.9.post3**
  fine (venv is py3.13 on BIG_STORE). torch resolved to **2.7.1 (bundles cu126)** —
  runs on the 12.4 driver via CUDA minor-version compat (the vLLM AWQ API already
  proved cu126 embeddings work here).
- Import then failed on `pybase64` → core `sglang` is too thin; serving needs the
  `[srt]` extra.

**The actual blocker — flashinfer:**
- `sglang[srt]` requires `sgl-kernel==0.2.6.post1` (✅ `cp39-abi3` wheel on PyPI, works)
  **and** `flashinfer_python==0.2.7.post1` (❌ **source-only on PyPI**, no wheel).
- flashinfer 0.2.x is JIT/source — needs `nvcc` on the node to build, or a prebuilt
  wheel from flashinfer's index (the old `flashinfer.ai/whl/cuXXX/torchY` paths 404).
- So `sglang[srt]` cannot install from plain PyPI on a bare node.

**Verdict:** SGLang-native is *possible* but blocked behind flashinfer sourcing
(nvcc build or correct wheel index) + an EAGLE draft head for our Qwen target (the
guide's EAGLE checkpoint is Gemma-specific). 2-3 more uncertain iterations. The
vLLM **AWQ 7B** path is working, fast (~1.1s/query, warm), and reliable, so SGLang+
EAGLE is parked as a documented option, not a blocker. `gpu` (sglang[all], full/
flashinfer) and `gpu-native` (core, Triton-backend) extras both kept in pyproject.

---

## 2026-05-26 — First graph-populated run + measured performance (job 53834)

Ran `single_node.slurm` with **Qwen2.5-7B-Instruct** and **`SKIP_LIGHTRAG=0`** —
the first end-to-end run with the LightRAG knowledge graph actually built
(778 nodes / 674 edges; 718 entities, 674 relations) and queried at retrieval time.

**Test-suite housekeeping:** fixed 3 failing swarm-generated tests (all bad
assumptions, not code bugs) → **694 passed / 8 skipped**, lint clean.
- `test_generator_suite`: counted `---` (two per `--- Article N ---` header) → count `--- Article`.
- `test_parser_suite`: fixture title "Long Article" + numbered para tripped the
  documented `ARTICLE_RE` inline cross-ref split → renamed title to "Long Provisions".
- `test_query_engine_suite`: score-1 complex query makes 2 LLM calls (complexity
  + sub-query decomposition), not 1.

**Eval (eval/test_set.json, 3 questions):**
| metric | result |
|---|---|
| Avg latency | 6.88s (comp_1 7.1s, comp_3 13.5s complex/decomposed, comp_2 0.02s = semantic-cache hit) |
| Format compliance | 100% |
| Conciseness | 100% |
| Keyword recall | 89% (answers substantively on-topic) |
| **Reference recall** | **0%** — the real gap |

**Key finding — citation grounding is the bottleneck, not generation.** The 7B
fixed the empty-answer problem (comp_2 from job 53833); answers are now fluent and
well-formatted. But final `references` come from the **LLM's own JSON**
(`generator.py:41` normalizes `raw["references"]`), not the retrieved set, so the
model cites plausible-but-wrong adjacent provisions: Article 11 instead of Annex IV
(comp_1), Article 6 instead of Article 5 (comp_2/3), and even **drops Annex III when
the question names it explicitly** (comp_3). comp_2's 0.02s confirms the semantic
cache matches across paraphrases (smoke query → comp_2).

**Next fix (highest leverage for the competition, which scores on cited refs):**
ground references in retrieval — always inject exact-match/explicit-ref article_ids
into the final reference list, and tighten the prompt to cite only from provided
context. Do NOT trust LLM free-choice citations alone.

---

## 2026-05-26 — Fix LightRAG graph retrieval (embedding_func)

`_graph_search` constructed `LightRAG(working_dir=...)` with no funcs, so querying
raised "embedding_func is required for vector storage" (caught → graph source
silently returned []). Added a shared `build_lightrag()` factory in
`src/clients.py` (deferred ML imports) that wires the SGLang LLM func + local
embedding func, and used it in BOTH `triple_retriever._get_rag` (query) and
`LightRAGIndexer.build_graph` (ingest) so they can't drift. Graph retrieval now
works once the graph is built — re-ingest WITHOUT `SKIP_LIGHTRAG=1` to populate it.
Verified: ruff + mypy clean, 141 passed / 8 skipped, offline imports still safe.

---

## 2026-05-26 — ruflo swarm: tests + script review, and fixes

Ran a 2-agent ruflo swarm (`ruflo-testgen:tester` + `ruflo-core:reviewer`).

**Tests** — added 5 files (`test_normalizer_extra`, `test_article_matcher_extra`,
`test_generator`, `test_query_engine`, `test_semantic_cache_backend`). Suite went
**18 → 141 passing** (8 integration skipped). ruff + mypy clean.

**Review findings fixed:**
- **C-1** `single_node.slurm`: health-waits fell through silently — the prior run
  (53829) ingested against a never-ready SGLang, producing a fallback-context
  index. Added `exit 1` guards after both SGLang and API health loops.
- **H-1** SGLang `apptainer exec docker://…` re-unpacked the multi-GB image to NFS
  every run (>20 min, 85k xattr warnings). Now builds a reusable `sglang.sif`
  once (`apptainer build`) and `exec`s the SIF. Applied to `sglang_server.slurm` too.
- **C-2** `generator._build_context` collapsed all BM25-only hits (no article_id)
  into one. Now only dedups chunks that carry an article_id. (triple_retriever
  already guarded this.)
- **C-3** `api/main.py`: wrapped the blocking cache/query/generate calls in
  `asyncio.to_thread` so the event loop stays free under concurrent load.
- **H-2/H-3** `run_eval.py`: guard non-200 responses and an empty test set.
- **M-3** `warmup_cache.py`: `httpx.Client` now used as a context manager.
- Normalizer improved to canonicalise lowercase `article`/`annex` prefixes and
  uppercase annex Roman numerals (resolves a new tester assertion + robustness).

**Deferred (need a re-ingest decision or larger refactor):** M-1 (infra `wait -n`
crash detection), M-2 (BM25 missing-index graceful 500), M-4 (hf_loader None
paragraph_no sort), L-1/L-2/L-3.

---

## 2026-05-26 — Single-node, no-infra runner

- `slurm/single_node.slurm`: full pipeline on one A100 with **embedded Qdrant +
  in-process cache** (no containers). Launches SGLang (Apptainer) → ingest → API
  → smoke query → eval on localhost; sequences ingest before API (embedded Qdrant
  is single-writer). Tunables: `MODEL` (default small ungated `Qwen2.5-1.5B-Instruct`),
  `MEM_FRACTION` (0.55), `SKIP_LIGHTRAG` (1), `KEEP_API` (1).
- `SKIP_LIGHTRAG=1` support added to both `ingest.py` and `ingest_hf.py`.
- **Verified embedded Qdrant** (login node, no torch): collection create,
  payload-filtered `scroll` (exact match), and `query_points` (dense) all work in
  local file mode — the exact ops the indexer/retriever use.

---

## 2026-05-26 — Test corpus via Hugging Face (EUR-Lex 202 workaround)

EUR-Lex returns **HTTP 202 + empty body** (async renderer / anti-bot), so
`download.py` was silently writing a 0-byte file. Added a reliable alternative
source and hardened the original downloader.

- **Source adapter** `src/ingestion/hf_loader.py` (`HFDatasetLoader`) maps the
  `jeroenherczeg/eu-ai-act` parquet (real Act, parsed from EUR-Lex Formex XML,
  ~548 KB) directly onto `ArticleNode` — chunker/indexer/retriever/generator run
  unchanged. Verified on the real file: **113 articles, 12 annexes** (upstream
  labels one annex `?`, dropped); Article 5 = "Prohibited AI practices" (8 paras),
  Article 3 = "Definitions". Eval-set targets (Art 3/5, Annex III/IV) all present.
- `scripts/download_hf.py` (HF Hub download) + `scripts/ingest_hf.py` (HF-source
  ingestion) + `slurm/ingest_hf.slurm`.
- `download.py` now polls EUR-Lex, validates payload size, and fails loudly with
  a pointer to the HF mirror instead of writing an empty file.
- New `data` extra (pyarrow + huggingface-hub; login-node safe, no torch).
- `tests/test_hf_loader.py` (3 offline tests, synthetic parquet).
- **Verified on login node:** download from HF CDN OK; loader OK on real data;
  **parse→chunk end-to-end on real corpus** (4 nodes → 25 chunks) with graceful
  SGLang-offline fallback. Suite now **18 passed / 8 skipped**, ruff + mypy clean.

---

## 2026-05-26 — Initial full scaffold

Implemented the complete codebase from `IMPLEMENTATION_GUIDE.md`, adapted to the
Demetra HPC environment (uv + Apptainer + SLURM; no Docker, no pip).

### Project setup
- `pyproject.toml` for **uv** with layered deps: core (no torch, login-node safe),
  `ml` extra (embeddings/reranker/bm25/lightrag), `gpu` extra (sglang), `dev`
  group (pytest/ruff/mypy). Ruff + mypy + pytest configured.
- Provisioned **Python 3.11.14** via `uv python install`; created `.venv`.
- `uv sync` installed core + dev cleanly (no torch on login node). ✅
- `.env.example`, `.gitignore`, `Makefile`.

### Source (`src/`)
- `config.py` — pydantic-settings v2; added `qdrant_local_path` (embedded mode)
  and `bm25_index_dir`.
- `clients.py` *(new)* — shared Qdrant client factory (honours embedded mode) +
  process-cached SentenceTransformer loader; deferred ML imports.
- `ingestion/`: `parser.py`, `chunker.py` (DI-friendly client), `hype.py`,
  `indexer.py` (Qdrant + BM25 + LightRAG; uses shared clients).
- `retrieval/`: `article_matcher.py`, `reranker.py`, `bm25_index.py`,
  `query_engine.py`, `triple_retriever.py` (LightRAG instance reused across
  queries, not re-init per call).
- `generation/`: `prompts.py`, `normalizer.py` (removed dead `TRANSFORMS` list;
  fixed sub-point sort ordering), `generator.py` (robust JSON parse).
- `cache/semantic_cache.py` — Redis **with in-process fallback** when unreachable.
- `observability/metrics.py` — Prometheus counters/histograms.
- `api/main.py` — FastAPI; **lazy component init** so import is cheap; added
  `/health` and `/cache/invalidate`; `/metrics` mounted.

### Scripts (`scripts/`)
- `download.py`, `ingest.py`, `warmup_cache.py`, `run_eval.py`
  (`run_eval`/`warmup` honour `RAG_API_URL`).

### Tests (`tests/`)
- Offline (always run): `test_parser.py`, `test_chunker.py` (stub LLM client),
  `test_retrieval.py` (matcher), `test_references.py` (normalizer).
- Integration (marked, skipped unless `RUN_INTEGRATION=1`): `test_answers.py`,
  `test_multiturn.py`, `test_load.py`, `test_references::test_api_references_valid`.
- `conftest.py` gates integration tests via `RUN_INTEGRATION`.
- **Result: 15 passed, 8 skipped.** ✅  Ruff: **all checks passed.** ✅

### Infrastructure
- **Docker deliverables** (not run): `Dockerfile`, `docker-compose.yml`,
  `infra/prometheus.yml`, `infra/grafana/dashboards/rag_dashboard.json`.
- **Apptainer + SLURM** (Demetra execution path): `slurm/00_common.sh`,
  `infra.slurm`, `sglang_server.slurm`, `ingest.slurm`, `api_server.slurm`,
  and `slurm/README.md` with orchestration order + single-node / embedded
  shortcuts.

### Hardening beyond the guide
- Migrated dense search to the **qdrant-client 1.18 API** (`query_points(...).points`;
  the guide's `.search()` was removed upstream). Guarded nullable payloads.
- Fixed `asyncio.gather(return_exceptions=True)` result handling to narrow on
  `BaseException` (correctness + type-safety).

### Verified (login node, core+dev deps only)
- `uv run pytest` → **15 passed / 8 skipped**.
- `uv run ruff check src tests scripts` → **All checks passed**.
- `uv run mypy src` → **Success: no issues found in 24 source files**.
- `from src.api.main import app` imports with all routes and **no torch/network**.

### Deferred to compute nodes (cannot run on login node)
- `uv sync --extra ml/--extra gpu`, model downloads, ingestion, SGLang serving,
  Apptainer container pulls, and integration tests — all via `slurm/`.

### Open follow-ups
- Validate Apptainer service launch on a compute node (image entrypoints).
- Confirm real Gemma-27B-AWQ + EAGLE draft model IDs before a long GPU run.
- Run ingestion against live EUR-Lex HTML and tune RRF weights on eval results.
