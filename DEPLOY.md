# Deploying the EU AI Act RAG with Docker Compose

A portable, single-node deployment for **any machine with Docker** (no GPU
required). The API runs on CPU; the embedding + reranker models and the prebuilt
retrieval stores (embedded Qdrant + BM25 + LightRAG) are baked into the image.
The **LLM stays external** — you point the app at an OpenAI-compatible endpoint.

> On Demetra this path is unused (Docker is unavailable there); the API runs via
> `uv` inside a SLURM job. This is the portability deliverable.

## What ships in the package

| Component | Where | Notes |
|-----------|-------|-------|
| API (FastAPI) | image `eu-ai-act-rag` | CPU; browser test page at `/`, docs at `/docs` |
| Embedding + reranker | baked into image | BGE-large + bge-reranker-v2-m3, downloaded at build |
| Qdrant | baked into image (`data/qdrant`) | **embedded on-disk**, no server |
| BM25 + LightRAG | baked into image | prebuilt indices |
| Redis | compose service | persistent semantic cache (named volume) |
| **LLM** | **external** | you provide it via `SGLANG_BASE_URL` |

## ⚠️ Transfer the folder, not a `git clone`

The prebuilt stores `data/qdrant/`, `bm25_index/`, `lightrag_data/` are
**gitignored**. A fresh `git clone` will NOT contain them and the image would be
built with empty indices (broken retrieval). Copy the **working directory**:

```bash
# from the machine that has this repo (e.g. Demetra), to the target:
rsync -av --exclude .venv --exclude .git \
  /u/<you>/phd_projects/RAG_EU_AI/  user@target:/opt/eu-ai-act-rag/
# or: tar czf rag.tgz --exclude=.venv --exclude=.git RAG_EU_AI && scp ...
```

Confirm on the target that these are non-empty before building:
`data/qdrant/`, `bm25_index/`, `lightrag_data/`.

## Steps on the target machine

Requires Docker + the Compose plugin, and **internet at build time** (to fetch
CPU torch and the model weights). Runtime needs neither.

```bash
cd /opt/eu-ai-act-rag

# 1. Configure — the only line you must edit is the LLM endpoint.
cp .env.docker.example .env.docker
$EDITOR .env.docker            # set SGLANG_BASE_URL=http://<your-llm>:PORT/v1

# 2. Build + start (first build ~several minutes: torch + BGE weights).
docker compose up --build -d

# 3. Watch it come up (models load on CPU at startup).
docker compose logs -f api     # wait for "Application startup complete"
curl http://localhost:8000/health          # -> {"status":"ok"}
```

Then open **http://localhost:8000/** in a browser for the test page, or POST:

```bash
curl -s http://localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '[{"role":"user","content":"What are the transparency obligations for deepfakes?"}]'
```

## The LLM endpoint (`SGLANG_BASE_URL`)

The lab gemma server `http://172.30.42.129:8080/v1` is reachable **only from the
Demetra login node**. From any other machine you must point `SGLANG_BASE_URL` at
an LLM you can actually reach — for example:

- a local llama.cpp / vLLM server on the same host (`http://host.docker.internal:8080/v1`
  on Docker Desktop, or the host IP on Linux);
- a hosted OpenAI-compatible API;
- an SSH tunnel to the lab server.

Set `LLM_MODEL` / `UTILITY_MODEL` to the id that endpoint expects. If the LLM is
unreachable, the API still starts and serves cached queries, but fresh `/answer`
calls will error (and the startup warm-up will wait on the LLM's timeout).

## Operating

```bash
docker compose ps
docker compose logs -f api
docker compose down            # stop (Redis cache persists in the named volume)
docker compose down -v         # stop and wipe the Redis cache volume
```

## Tuning (edit `.env.docker`, then `docker compose up -d`)

- `LIGHTRAG_MODE=naive` — drop the per-query LLM keyword call (a bit faster, less recall).
- `ENABLE_CHUNK_VOTING=true` — higher reference recall, but ~+11 s/query.
- `LLM_ENABLE_THINKING=true` — native chain-of-thought (~2.6× latency).
- `CACHE_TTL_SECONDS` — semantic-cache lifetime (default 24 h).

## Notes / limits

- **CPU only.** A fresh (uncached) query spends a few extra seconds on CPU
  embedding + reranking vs a GPU node. For a GPU host, a CUDA image is a
  separate build (not included here).
- **No LLM in the box.** The gemma MoE is too large to ship for CPU; keeping the
  LLM external is deliberate.
- Only one `api` replica: the embedded Qdrant uses a single-writer file lock.
