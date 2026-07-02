# EU AI Act RAG System

A production-grade Retrieval-Augmented Generation system that answers questions
about the EU AI Act (Regulation 2024/1689) with cited references, built for
the [regenold EU AI Act Q&A Competition](https://regenold.com/landing/ai/eu-ai-act-competition).

---

## What this project does

This system receives a natural language question about the EU AI Act and returns
a concise, professionally worded answer with exact references to the relevant
Articles and Annexes.

**Input** (from regenold's benchmark):
```json
[
  {"role": "user", "content": "Are AI systems for emotion recognition always prohibited?"}
]
```

**Output**:
```json
{
  "reasoning": "Article 5(1)(f) prohibits emotion recognition in workplace and education, but Article 5(2) provides exceptions for medical and safety purposes.",
  "answer": "AI systems for emotion recognition from biometric data are not universally prohibited. Pursuant to Article 5(1)(f), such systems are prohibited in workplace and educational settings. However, Article 5(2) provides exceptions where deployment serves medical or safety purposes.",
  "references": ["Article 5.1", "Article 5.2"]
}
```

The system handles multi-turn conversations, meaning follow-up questions like
"Are there exceptions to that?" are resolved using the conversation context.

---

## Implementation status

The full pipeline is implemented and runs end-to-end on Demetra (SGLang +
EAGLE3 on an 80GB A100, embedded Qdrant/BM25/LightRAG, in-process or Redis
cache). Three retrieval/cache fixes landed after the first end-to-end eval:

- **Semantic cache → 3 layers + ref-scoping.** Normalised-exact, hash-exact,
  then ref-scoped cosine (threshold raised 0.95 → 0.97). Kills false-positive
  hits between legally distinct questions.
- **BM25 carries `article_id`.** The lexical index previously stored bare
  strings, so BM25 hits had no `article_id` and the gold provision never
  reached the candidate set. Both build and search sides now use dict records.
- **LLM rerank with adaptive snippets.** A second-stage LLM reorders BGE
  candidates by legal relevance. Snippets are centered on the query match
  (the rule-stating clause is often deep in a long article — e.g. Art 5(1)(f),
  Art 50(4)) instead of a fixed prefix. Falls back to BGE order on any error.

Validation on the offline suite is green (780 unit tests). The end-to-end eval
set is 30 questions (`eval/test_set.json`); the on-cluster ref/keyword-recall
numbers are pending the next GPU run after these fixes. A multi-chunk voting
rerank is designed but not implemented (`docs/voting_rerank_design.md`),
gated on whether the adaptive-snippet fix alone clears the eval.

Not yet exercised: the full-stack server mode (Qdrant server + Redis +
Prometheus + Grafana via `infra.slurm`) and the `RUN_INTEGRATION=1` suite.

---

## Why this project exists

### The competition

Regenold GmbH runs a benchmark competition (May–June 2026) that evaluates AI
systems on their ability to answer questions about the EU AI Act faithfully.
Systems are scored on six metrics plus a bonus:

| Metric | What it measures |
|---|---|
| Answer correctness (strict + loose) | Is the answer factually correct? |
| Reference correctness (strict + loose) | Are the cited Articles/Annexes right? |
| Answer conciseness | 1–4 sentences, no fluff |
| Reference conciseness | Minimum necessary set of citations |
| Regulatory tone | Formal, professional language |
| Latency | Time between question and answer |
| **Bonus**: Multi-turn | All of the above within a conversation |

### Why not just use ChatGPT?

A general-purpose LLM hallucinates references, over-cites, gives verbose answers,
and cannot guarantee the specific reference format the competition requires
(e.g., "Article 6.2" not "Art. 6(2)"). This system solves all of that by
grounding every answer in the actual text of the regulation and post-processing
every reference through a strict normalizer.

### Why this architecture matters

This project demonstrates end-to-end AI system design: data ingestion,
knowledge graph construction, hybrid retrieval, cross-encoder reranking,
optimized LLM inference, semantic caching, observability, and automated
evaluation. Every component uses the best available technology as of May 2026.

---

## Architecture overview

The system has two pipelines that run independently.

### Pipeline A — Ingestion (offline, runs once, ~20 minutes)

Transforms the EU AI Act PDF into a searchable index.

```
EU AI Act (HTML from EUR-Lex)
     │
     ▼
HTML Parser ──────────── Extracts 113 Articles + 13 Annexes
     │
     ▼
SAC Chunker ──────────── Creates LARGE (full article) and SMALL (paragraph)
     │                    chunks, each prefixed with a document fingerprint
     │                    and a positional context sentence
     ▼
HyPE Generator ───────── For each chunk, pre-generates 3-5 hypothetical
     │                    questions that chunk would answer
     ▼
Indexer
  ├── Qdrant ──────────── Dense vectors (BGE embeddings) + keyword indices
  ├── BM25S ───────────── In-memory lexical search index
  └── LightRAG ────────── Knowledge graph of entity relationships
     │
     ▼
Cache Warmup ──────────── Pre-computes ~300 predictable Q&A pairs in Redis
```

### Pipeline B — Runtime (online, per query, ~800ms without cache)

Processes each incoming question through seven stages.

```
POST /answer
     │
     ▼
Semantic Cache (3-layer) ── Hit? Return in ~5ms. Miss? Continue.
     │                         norm-exact → hash-exact → ref-scoped cosine (≥0.97)
     ▼
Query Engine
  ├── Multi-turn resolver ── Rewrites query resolving pronouns (LLM, 50ms)
  ├── Article matcher ────── Extracts "Article N" / "Annex N" via regex (0ms)
  ├── Complexity detector ── Linguistic patterns + LLM fallback (0-50ms)
  └── Sub-query decomposer ─ Splits complex questions into 2-3 parts (80ms)
     │
     ▼
Triple Retriever (parallel with asyncio.gather)
  ├── Dense search ───────── BGE embedding → Qdrant cosine similarity
  ├── BM25S ──────────────── Keyword matching; hits carry article_id metadata
  ├── LightRAG ───────────── Knowledge graph traversal (mode="mix")
  ├── Exact match ────────── Qdrant payload filter on article_id
  └── Xref expansion ─────── Pulls provisions cited inside retrieved chunks
     │
     ▼
RRF Fusion + dedup ────────── Reciprocal Rank Fusion with per-source weights
     │
     ▼
BGE Reranker v2-m3 ────────── Cross-encoder rescoring → top candidates
     │
     ▼
LLM Rerank (optional) ─────── Reorders by legal relevance using adaptive
     │                         query-centered snippets; falls back to BGE order
     ▼
SGLang + EAGLE3 ───────────── Qwen2.5-14B-Instruct, temp=0, JSON mode, 300 tokens
     │                         RadixAttention prefix caching + speculative decoding
     ▼
Reference Normalizer ──────── Regex: "Art. 5" → "Article 5", "Annex 3" → "Annex III"
     │
     ▼
Cache Write ───────────────── Saves for future similar queries (TTL 1 hour)
     │
     ▼
JSON Response
```

---

## Tech stack

Every component is free and open source except for cloud hosting.

| Component | Technology | Why this one |
|---|---|---|
| LLM inference | SGLang + EAGLE3 | Automatic prefix caching, speculative decoding, structured JSON output |
| LLM model | Qwen2.5-14B-Instruct | Validated production target on a single 80GB A100; paired EAGLE3 draft (`ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat`) |
| Embeddings | BGE-large-en-v1.5 | Local, free, strong on MTEB benchmarks |
| Vector DB | Qdrant | Rust-based, fast filtering, built-in payload indices (server or embedded on-disk) |
| Keyword search | BM25S | 500x faster than rank-bm25, pure Python; index stores article_id per doc |
| Knowledge graph | LightRAG | Graph RAG for legal documents, mode="mix"; internal rerank disabled (we rerank downstream) |
| Reranker | BGE-reranker-v2-m3 + LLM | Cross-encoder first pass, then an optional LLM rerank by legal relevance |
| Cache | Redis (+ in-process fallback) | 3-layer semantic cache: norm-exact → hash-exact → ref-scoped cosine |
| API framework | FastAPI | Async, automatic OpenAPI docs, Pydantic validation |
| Monitoring | Prometheus + Grafana | Industry standard, free |
| LLM tracing | LangSmith (free tier) | Traces every LLM call for debugging |
| Containerization | Docker Compose | Single command to start all infrastructure |

---

## How to build it

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- NVIDIA A100 GPU with CUDA 12.x
- ~50GB disk space

### Step 1: Clone and set up

```bash
git clone https://github.com/yourname/eu-ai-act-rag.git
cd eu-ai-act-rag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install "sglang[all]" "bm25s[full]" "lightrag-hku[api]"
```

### Step 2: Configure

```bash
cp .env.example .env
# Edit .env with your settings (all defaults work for local development)
```

### Step 3: Start infrastructure

```bash
docker compose up -d qdrant redis prometheus grafana
```

### Step 4: Start the LLM server

```bash
python -m sglang.launch_server \
  --model-path google/gemma-3-27b-it \
  --quantization awq \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmsys/sglang-EAGLE-gemma-3-27b-it \
  --port 8899 \
  --dtype float16 \
  --mem-fraction-static 0.85
```

Wait until you see "Server is ready" in the terminal.

### Step 5: Download and ingest

```bash
# Download the EU AI Act from EUR-Lex
python scripts/download.py

# Parse, chunk, embed, index (takes ~15-20 minutes)
python scripts/ingest.py
```

This will:
- Parse the HTML into 113 Articles and 13 Annexes
- Create LARGE and SMALL chunks with SAC enrichment
- Generate HyPE questions for each chunk
- Index everything into Qdrant (vectors + keyword indices)
- Build a BM25S in-memory index
- Construct a LightRAG knowledge graph

### Step 6: Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Step 7: Warm up the cache

```bash
python scripts/warmup_cache.py
```

Pre-computes ~300 answers and stores them in Redis. Any matching or
semantically similar question from regenold will be served in 5ms.

### Step 8: Verify

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '[{"role":"user","content":"What is the definition of an AI system?"}]'
```

---

## How to use it

### API endpoint

**POST /answer**

Send a conversation history in OpenAI/LiteLLM format. The last message must
have `role: "user"`.

```bash
# Single question
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '[{"role":"user","content":"What AI practices are prohibited?"}]'

# Multi-turn conversation
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '[
    {"role":"user","content":"What does Article 5 establish?"},
    {"role":"assistant","content":"Article 5 establishes the list of prohibited AI practices..."},
    {"role":"user","content":"Are there any exceptions?"}
  ]'
```

### Response format

```json
{
  "reasoning": "Internal chain-of-thought (not scored by the competition)",
  "answer": "1-4 sentences in formal regulatory language",
  "references": ["Article 5.1", "Annex III"]
}
```

Reference format rules (enforced by the normalizer):
- Articles use Arabic numerals: `Article 6` or `Article 6.2`
- Annexes use Roman numerals: `Annex III` or `Annex III.2`
- Sub-points use a dot separator: `Article 6.2` (not `6(2)` or `6/2`)

### Other endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/answer` | POST | Main Q&A endpoint (competition contract) |
| `/health` | GET | Component health check |
| `/metrics` | GET | Prometheus metrics |
| `/cache/invalidate` | POST | Clear all cached responses |

### Running tests

```bash
# All tests
pytest tests/ -v

# Only reference format tests (no server needed)
pytest tests/test_references.py -v

# Only integration tests (needs full stack)
pytest tests/test_answers.py tests/test_multiturn.py -v

# Load test
pytest tests/test_load.py -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Running the evaluation

```bash
python scripts/run_eval.py
```

This tests the system against the three example questions from the competition
and measures reference recall, keyword recall, format compliance, conciseness,
and latency. Results are saved to `eval/results.json`.

### Monitoring

| Service | URL | Credentials |
|---|---|---|
| API health | http://localhost:8000/health | — |
| Qdrant dashboard | http://localhost:6333/dashboard | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

Key Grafana panels to watch:
- **Latency P95**: should be under 3 seconds without cache
- **Cache hit rate**: should be 40-60% after warmup
- **Error rate**: should be 0%
- **Reference count**: typically 1-3 per response

---

## How to submit to the competition

1. Make your API publicly accessible (use ngrok, a cloud VM, or Google Cloud Run)
2. Verify the endpoint works from outside your network:
   ```bash
   curl -X POST https://your-public-url/answer \
     -H "Content-Type: application/json" \
     -d '[{"role":"user","content":"What is the definition of an AI system?"}]'
   ```
3. Send the URL to regenold at the contact on their website
4. They will send questions to your endpoint and evaluate the responses

---

## Performance

Measured on NVIDIA A100 80GB with Qwen2.5-14B-Instruct + EAGLE3, on the
18-question eval (job 54797, before the adaptive-snippet rerank fix):

| Metric | Value |
|---|---|
| Avg latency (cache cleared before eval) | ~1.71s |
| Cache hit | ~5ms |
| Reference recall | 86% |
| Keyword recall | 78% |
| Format compliance | 100% |

These numbers predate the LLM-rerank snippet fix and the 30-question eval set;
they will be re-measured on the next GPU run. Re-run with
`uv run python scripts/run_eval.py` against a live API.

---

## Key design decisions

### Why HyPE instead of HyDE?

HyDE generates a hypothetical document at query time, adding 100-300ms of
latency and risking hallucination. HyPE pre-generates hypothetical questions
at indexing time, achieving the same retrieval improvement with zero runtime
cost. On a small, structured corpus like the EU AI Act, this is strictly better.

### Why a two-level complexity detector?

Linguistic pattern matching handles 75% of queries in 0ms. The LLM fallback
only fires for the remaining 25% of ambiguous cases (~50ms). This saves an
average of 35ms per query compared to always using the LLM.

### Why four retrieval sources?

Each source catches what the others miss:
- **Dense search** finds semantically similar content even with different wording
- **BM25** catches exact term matches ("conformity assessment") that dense search
  ranks lower
- **LightRAG** finds relationships between articles that neither dense nor BM25 see
  (e.g., "Article 6 references Annex III")
- **Exact match** guarantees that explicitly mentioned articles are always retrieved

### Why Redis semantic cache?

The competition measures latency. With ~300 pre-computed answers, 40-60% of
questions are served from cache in 5ms. The semantic matching is ref-scoped
cosine similarity ≥ 0.97 (raised from 0.95 to stop legally distinct questions
colliding), behind two exact-match layers — see Implementation status.

### Why a reference normalizer?

The LLM occasionally outputs "Art. 5" or "Article III" or "Annex 3". Each
format error costs points on the reference correctness metric. The regex
post-processor catches and corrects 100% of known format variants.

---

## Project structure

```
eu-ai-act-rag/
├── src/
│   ├── config.py              # All settings from .env
│   ├── ingestion/
│   │   ├── parser.py          # HTML → ArticleNode[]
│   │   ├── chunker.py         # SAC + multi-granularity chunking
│   │   ├── hype.py            # Pre-generate hypothetical questions
│   │   └── indexer.py         # Qdrant + BM25S + LightRAG indexing
│   ├── retrieval/
│   │   ├── query_engine.py    # Adaptive: resolver + matcher + detector + decomposer
│   │   ├── triple_retriever.py # 4 sources in parallel → RRF → reranker
│   │   ├── bm25_index.py      # BM25S in-memory search
│   │   ├── article_matcher.py # Regex Article/Annex extractor
│   │   └── reranker.py        # BGE cross-encoder
│   ├── generation/
│   │   ├── generator.py       # SGLang client, JSON mode
│   │   ├── prompts.py         # System prompt tuned for competition metrics
│   │   └── normalizer.py      # Reference format post-processor
│   ├── cache/
│   │   └── semantic_cache.py  # Redis exact + cosine similarity cache
│   ├── observability/
│   │   └── metrics.py         # Prometheus counters and histograms
│   └── api/
│       └── main.py            # FastAPI server
├── scripts/
│   ├── download.py            # Fetch EU AI Act from EUR-Lex
│   ├── ingest.py              # Run full ingestion pipeline
│   ├── warmup_cache.py        # Pre-compute ~300 answers
│   └── run_eval.py            # Evaluate against test set
├── tests/                     # pytest suite
├── eval/                      # Test set and results
├── infra/                     # Prometheus config, Grafana dashboards
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## License

MIT
