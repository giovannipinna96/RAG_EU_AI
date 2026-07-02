# EU AI Act RAG — portable API image.
#
# CPU-only, self-contained: embedding + reranker weights and the prebuilt
# retrieval stores (embedded Qdrant + BM25 + LightRAG) are baked in, so at
# runtime the container needs NO GPU and NO HuggingFace access. The LLM is
# external — point SGLANG_BASE_URL (in .env.docker) at an OpenAI-compatible
# endpoint reachable from the host. See DEPLOY.md.
#
# On Demetra this path is unused: the API runs via uv inside a SLURM job
# (slurm/api_tunnel.slurm). Docker is a portability deliverable only.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# curl is needed by the compose healthcheck; uv drives the installs.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# 1) CPU-only torch FIRST, from the pytorch CPU index. This satisfies the ml
#    extra's transitive torch dependency, so uv never pulls the ~2 GB CUDA wheel
#    from PyPI — the image stays small and free of CUDA runtime libraries.
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu torch

# 2) Dependency layer (cached unless pyproject/src change). ml extra brings the
#    embedder, reranker, bm25s and lightrag; torch is already satisfied above.
COPY pyproject.toml README_EU_AI_RAG.md ./
COPY src ./src
RUN uv pip install --system --no-cache ".[ml]"

# 3) Pre-download the models named in .env.docker so runtime is offline-capable.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')" \
    && python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"

# 4) Prebuilt stores (data/qdrant, bm25_index, lightrag_data) + the rest of the
#    app. .dockerignore keeps envs, caches and the .venv out of the context.
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
