# EU AI Act RAG — Complete Implementation Guide

> Step-by-step guide to build, test, deploy, monitor, and evaluate
> the system for the regenold EU AI Act Q&A Competition.

> **Divergences from the as-built system.** This guide is the original build
> spec; some example commands and snippets use placeholders that the shipped
> code has since refined. Where they differ, the code and
> `README_EU_AI_RAG.md` ("Implementation status") are authoritative:
> - **LLM model:** validated target is `Qwen2.5-14B-Instruct` + the matching
>   EAGLE3 draft (see `slurm/sglang_eagle.slurm`), not the `gemma-3-27b`
>   placeholder used in the launch examples below.
> - **Semantic cache:** 3 layers (norm-exact → hash-exact → ref-scoped cosine),
>   threshold `0.97` (not `0.95`), with `CACHE_REQUIRE_REF_MATCH`.
> - **BM25 index:** stores `article_id` per document (was bare strings).
> - **Rerank:** BGE cross-encoder followed by an optional LLM rerank that uses
>   adaptive query-centered snippets (`ENABLE_LLM_RERANK`).

---

## Table of contents

1. Prerequisites & environment setup
2. Project scaffolding
3. Configuration
4. Ingestion pipeline (offline)
5. Runtime pipeline (online)
6. API server
7. Docker & docker-compose
8. Testing
9. Monitoring & observability
10. Evaluation framework
11. Cache warmup strategy
12. Deployment checklist

---

## 1. Prerequisites & environment setup

### Hardware
- NVIDIA A100 80GB (for SGLang + EAGLE)
- 32GB+ system RAM
- 50GB+ disk space

### Software
- Python 3.11+
- Docker & Docker Compose
- CUDA 12.x + cuDNN
- Git

### Step 1: Create the project

```bash
mkdir eu-ai-act-rag && cd eu-ai-act-rag
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
# Create pyproject.toml (see section 2), then:
pip install -e ".[dev]" --break-system-packages

# Install SGLang separately (GPU-specific)
pip install "sglang[all]"

# Install BM25S
pip install "bm25s[full]"

# Install LightRAG (HKUDS version, not the other one)
pip install "lightrag-hku[api]"
```

---

## 2. Project scaffolding

### Directory structure

```
eu-ai-act-rag/
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── hype.py
│   │   └── indexer.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_engine.py
│   │   ├── triple_retriever.py
│   │   ├── bm25_index.py
│   │   ├── article_matcher.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── normalizer.py
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── semantic_cache.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py
│
├── scripts/
│   ├── download.py
│   ├── ingest.py
│   ├── warmup_cache.py
│   └── run_eval.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_references.py
│   ├── test_answers.py
│   ├── test_multiturn.py
│   └── test_load.py
│
├── eval/
│   ├── test_set.json
│   └── report_template.md
│
├── infra/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── rag_dashboard.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── lightrag_data/
├── bm25_index/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

### pyproject.toml

```toml
[project]
name = "eu-ai-act-rag"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    # API
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    # Retrieval
    "qdrant-client>=1.12.0",
    "bm25s[full]>=0.2.0",
    "lightrag-hku>=1.0.0",
    "FlagEmbedding>=1.3.0",
    "sentence-transformers>=3.3.0",
    # Cache
    "redis>=5.2.0",
    # LLM client (OpenAI-compatible, talks to SGLang)
    "openai>=1.50.0",
    # Ingestion
    "beautifulsoup4>=4.12.0",
    "lxml>=5.3.0",
    "httpx>=0.27.0",
    # Observability
    "prometheus-client>=0.21.0",
    "structlog>=24.4.0",
    # Utilities
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "ruff>=0.7.0",
    "mypy>=1.12.0",
]
```

### .env.example

```bash
# LLM (SGLang server)
SGLANG_BASE_URL=http://localhost:8899/v1
LLM_MODEL=default
UTILITY_MODEL=default

# Embedding (local model)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIM=1024

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=eu_ai_act

# Redis
REDIS_URL=redis://localhost:6379/0

# LightRAG
LIGHTRAG_WORKING_DIR=./lightrag_data

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Cache
CACHE_TTL_SECONDS=3600
CACHE_SIMILARITY_THRESHOLD=0.97
CACHE_REQUIRE_REF_MATCH=true

# RRF weights
RRF_WEIGHT_EXACT=3.0
RRF_WEIGHT_DENSE=1.0
RRF_WEIGHT_BM25=0.6
RRF_WEIGHT_GRAPH=0.9
RRF_K=60
```

### .gitignore

```
.venv/
__pycache__/
*.pyc
.env
data/raw/
lightrag_data/
bm25_index/
*.egg-info/
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

---

## 3. Configuration

### src/config.py

```python
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    sglang_base_url: str = "http://localhost:8899/v1"
    llm_model: str = "default"
    utility_model: str = "default"

    # Embedding
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "eu_ai_act"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LightRAG
    lightrag_working_dir: str = "./lightrag_data"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    enable_llm_rerank: bool = True

    # Cache
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.97
    cache_require_ref_match: bool = True

    # RRF
    rrf_weight_exact: float = 3.0
    rrf_weight_dense: float = 1.0
    rrf_weight_bm25: float = 0.6
    rrf_weight_graph: float = 0.9
    rrf_k: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
```

---

## 4. Ingestion pipeline

Run once. Downloads, parses, chunks, indexes everything.

### scripts/download.py

```python
"""Download the EU AI Act from EUR-Lex."""
import httpx
from pathlib import Path


def download():
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    dest = Path("data/raw/eu_ai_act.html")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Already downloaded: {dest}")
        return

    print("Downloading EU AI Act from EUR-Lex...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    print(f"Saved to {dest} ({len(resp.text) // 1024} KB)")


if __name__ == "__main__":
    download()
```

### src/ingestion/parser.py

```python
"""Parse EU AI Act HTML into structured ArticleNodes."""
import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from pathlib import Path


@dataclass
class ArticleNode:
    article_id: str          # "Article 6" or "Annex III"
    article_type: str        # "article" or "annex"
    number: str              # "6" or "III"
    title: str
    full_text: str
    paragraphs: list[dict] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class EUAIActParser:
    ARTICLE_RE = re.compile(r"Article\s+(\d+)")
    ANNEX_RE = re.compile(r"ANNEX\s+([IVX]+)")
    PARA_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)", re.DOTALL)

    def parse(self, file_path: str) -> list[ArticleNode]:
        html = Path(file_path).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)

        nodes = []
        nodes.extend(self._parse_articles(text))
        nodes.extend(self._parse_annexes(text))
        return nodes

    def _parse_articles(self, text: str) -> list[ArticleNode]:
        # Find all article boundaries
        splits = list(self.ARTICLE_RE.finditer(text))
        nodes = []

        for i, match in enumerate(splits):
            num = match.group(1)
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()

            # Extract title (first line after "Article N")
            lines = content.split("\n", 2)
            title = lines[1].strip() if len(lines) > 1 else ""

            # Extract numbered paragraphs
            paragraphs = [
                {"num": m.group(1), "ref": f"Article {num}.{m.group(1)}", "text": m.group(2).strip()}
                for m in self.PARA_RE.finditer(content)
            ]

            nodes.append(ArticleNode(
                article_id=f"Article {num}",
                article_type="article",
                number=num,
                title=title,
                full_text=content,
                paragraphs=paragraphs,
            ))
        return nodes

    def _parse_annexes(self, text: str) -> list[ArticleNode]:
        splits = list(self.ANNEX_RE.finditer(text))
        nodes = []

        for i, match in enumerate(splits):
            num = match.group(1)
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()

            nodes.append(ArticleNode(
                article_id=f"Annex {num}",
                article_type="annex",
                number=num,
                title=f"Annex {num}",
                full_text=content,
                paragraphs=[],
            ))
        return nodes

    def validate(self, nodes: list[ArticleNode]) -> bool:
        articles = [n for n in nodes if n.article_type == "article"]
        annexes = [n for n in nodes if n.article_type == "annex"]
        print(f"Parsed: {len(articles)} articles, {len(annexes)} annexes")
        return len(articles) >= 100 and len(annexes) >= 10
```

### src/ingestion/chunker.py

```python
"""
SAC Chunker: Structure-aware + Summary-Augmented + Multi-granularity.
Produces LARGE (full article) and SMALL (paragraph-level) chunks.
"""
from dataclasses import dataclass, field
from openai import OpenAI
from .parser import ArticleNode
from ..config import settings


DOC_FINGERPRINT = (
    "[DOCUMENT]: EU AI Act — Regulation (EU) 2024/1689 of 13 June 2024. "
    "Contains 113 Articles across 13 Chapters and 13 Annexes (I-XIII). "
    "Covers risk classification of AI systems, obligations for providers "
    "and deployers, prohibited practices, conformity assessment, and penalties."
)


@dataclass
class ProcessedChunk:
    content: str          # SAC-enriched text (for embedding)
    content_raw: str      # original text (for BM25 and reranker)
    article_id: str
    article_type: str
    paragraph_refs: list[str]
    granularity: str      # "large" or "small"
    title: str = ""
    context: str = ""


class SACChunker:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.sglang_base_url,
            api_key="none",
        )

    def chunk_all(self, nodes: list[ArticleNode]) -> list[ProcessedChunk]:
        chunks = []
        for i, node in enumerate(nodes):
            print(f"  Chunking {node.article_id} ({i+1}/{len(nodes)})...")
            context = self._generate_context(node)

            # LARGE chunk: full article
            large_content = f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n{node.full_text}"
            chunks.append(ProcessedChunk(
                content=large_content,
                content_raw=node.full_text,
                article_id=node.article_id,
                article_type=node.article_type,
                paragraph_refs=[p["ref"] for p in node.paragraphs],
                granularity="large",
                title=node.title,
                context=context,
            ))

            # SMALL chunks: one per paragraph
            if node.paragraphs:
                for para in node.paragraphs:
                    small_content = f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n[{para['ref']}]: {para['text']}"
                    chunks.append(ProcessedChunk(
                        content=small_content,
                        content_raw=para["text"],
                        article_id=node.article_id,
                        article_type=node.article_type,
                        paragraph_refs=[para["ref"]],
                        granularity="small",
                        title=node.title,
                        context=context,
                    ))
            elif node.word_count > 400:
                # Split long articles without numbered paragraphs
                words = node.full_text.split()
                for j in range(0, len(words), 250):
                    block = " ".join(words[j:j + 300])
                    small_content = f"{DOC_FINGERPRINT}\n[CONTEXT]: {context}\n\n{block}"
                    chunks.append(ProcessedChunk(
                        content=small_content,
                        content_raw=block,
                        article_id=node.article_id,
                        article_type=node.article_type,
                        paragraph_refs=[],
                        granularity="small",
                        title=node.title,
                        context=context,
                    ))

        large = sum(1 for c in chunks if c.granularity == "large")
        small = sum(1 for c in chunks if c.granularity == "small")
        print(f"Created {len(chunks)} chunks ({large} large, {small} small)")
        return chunks

    def _generate_context(self, node: ArticleNode) -> str:
        """Contextual Retrieval: 1-2 sentences about what this article covers."""
        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": (
                        f"In 1-2 sentences, explain what {node.article_id} "
                        f"('{node.title}') covers in the EU AI Act. "
                        f"Be specific.\n\nText: {node.full_text[:2000]}"
                    ),
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"{node.article_id}: {node.title}"
```

### src/ingestion/hype.py

```python
"""
HyPE: Hypothetical Prompt Embeddings.
Pre-generates questions for each chunk at indexing time.
"""
from openai import OpenAI
from .chunker import ProcessedChunk
from ..config import settings


class HyPEGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.sglang_base_url,
            api_key="none",
        )

    def generate_questions(self, chunk: ProcessedChunk, n: int = 5) -> list[str]:
        """Generate n hypothetical questions this chunk would answer."""
        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.3,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate exactly {n} diverse questions that the following "
                        f"EU AI Act provision would answer. Include yes/no questions, "
                        f"what/how questions, and scenario-based questions.\n"
                        f"Output ONLY the questions, one per line, no numbering.\n\n"
                        f"[{chunk.article_id}]:\n{chunk.content_raw[:1500]}"
                    ),
                }],
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:n]
        except Exception:
            return []
```

### src/ingestion/indexer.py

```python
"""Index chunks into Qdrant (vectors + BM25) and LightRAG (knowledge graph)."""
import uuid
import asyncio
import numpy as np
import bm25s
import Stemmer
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
)
from sentence_transformers import SentenceTransformer
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from .chunker import ProcessedChunk
from .hype import HyPEGenerator
from ..config import settings


class QdrantIndexer:
    def __init__(self):
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.embed_model = SentenceTransformer(settings.embedding_model)
        self.hype = HyPEGenerator()

    def create_collection(self):
        coll = settings.qdrant_collection
        if self.qdrant.collection_exists(coll):
            self.qdrant.delete_collection(coll)

        self.qdrant.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        # Keyword index on article_id (for exact match)
        self.qdrant.create_payload_index(coll, "article_id", PayloadSchemaType.KEYWORD)
        self.qdrant.create_payload_index(coll, "granularity", PayloadSchemaType.KEYWORD)
        self.qdrant.create_payload_index(coll, "type", PayloadSchemaType.KEYWORD)
        print(f"Collection '{coll}' created")

    def index_chunks(self, chunks: list[ProcessedChunk]):
        """Index original chunks + HyPE questions."""
        all_points = []

        for i, chunk in enumerate(chunks):
            # Original chunk embedding
            vec = self.embed_model.encode(chunk.content, normalize_embeddings=True)
            all_points.append(self._make_point(chunk, vec.tolist(), "chunk"))

            # HyPE questions (only for LARGE chunks to save time)
            if chunk.granularity == "large":
                questions = self.hype.generate_questions(chunk, n=5)
                for q in questions:
                    q_vec = self.embed_model.encode(q, normalize_embeddings=True)
                    all_points.append(self._make_point(chunk, q_vec.tolist(), "hype", hype_question=q))

            if (i + 1) % 20 == 0:
                print(f"  Embedded {i+1}/{len(chunks)} chunks...")

        # Batch upsert
        BATCH = 64
        for i in range(0, len(all_points), BATCH):
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=all_points[i:i + BATCH],
            )
        print(f"Indexed {len(all_points)} points ({len(chunks)} chunks + HyPE questions)")

    def _make_point(self, chunk, vector, point_type, hype_question=""):
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "article_id": chunk.article_id,
                "article_type": chunk.article_type,
                "content_raw": chunk.content_raw,
                "title": chunk.title,
                "context": chunk.context,
                "paragraph_refs": chunk.paragraph_refs,
                "granularity": chunk.granularity,
                "type": point_type,
                "hype_question": hype_question,
            },
        )


class BM25Indexer:
    def __init__(self):
        self.stemmer = Stemmer.Stemmer("english")

    def build_index(self, chunks: list[ProcessedChunk]):
        """Build BM25S in-memory index and save to disk."""
        corpus = [c.content_raw for c in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        save_dir = Path("bm25_index")
        save_dir.mkdir(exist_ok=True)
        retriever.save(str(save_dir), corpus=corpus)
        print(f"BM25 index saved to {save_dir}")


class LightRAGIndexer:
    async def build_graph(self, full_text: str):
        """Index the full EU AI Act text into LightRAG knowledge graph."""
        from lightrag.utils import wrap_embedding_func_with_attrs

        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                settings.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=settings.sglang_base_url,
                api_key="none",
                **kwargs,
            )

        @wrap_embedding_func_with_attrs(embedding_dim=settings.embedding_dim, max_token_size=8192)
        async def embed_func(texts: list[str]) -> np.ndarray:
            model = SentenceTransformer(settings.embedding_model)
            return model.encode(texts, normalize_embeddings=True)

        rag = LightRAG(
            working_dir=settings.lightrag_working_dir,
            llm_model_func=llm_func,
            embedding_func=embed_func,
        )
        await rag.initialize_storages()

        print("Indexing into LightRAG (this may take 5-15 minutes)...")
        await rag.ainsert(full_text)
        await rag.finalize_storages()
        print("LightRAG knowledge graph built")
```

### scripts/ingest.py

```python
"""Run the full ingestion pipeline."""
import asyncio
from src.ingestion.parser import EUAIActParser
from src.ingestion.chunker import SACChunker
from src.ingestion.indexer import QdrantIndexer, BM25Indexer, LightRAGIndexer


async def main():
    # 1. Parse
    print("=== PARSING ===")
    parser = EUAIActParser()
    nodes = parser.parse("data/raw/eu_ai_act.html")
    assert parser.validate(nodes), "Parsing failed validation"

    # 2. Chunk (SAC + multi-granularity + contextual enrichment)
    print("\n=== CHUNKING ===")
    chunker = SACChunker()
    chunks = chunker.chunk_all(nodes)

    # 3. Index into Qdrant (vectors + HyPE questions)
    print("\n=== INDEXING QDRANT ===")
    qdrant_indexer = QdrantIndexer()
    qdrant_indexer.create_collection()
    qdrant_indexer.index_chunks(chunks)

    # 4. Build BM25 index
    print("\n=== BUILDING BM25 INDEX ===")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(chunks)

    # 5. Build LightRAG knowledge graph
    print("\n=== BUILDING KNOWLEDGE GRAPH ===")
    full_text = "\n\n".join(n.full_text for n in nodes)
    lightrag_indexer = LightRAGIndexer()
    await lightrag_indexer.build_graph(full_text)

    print(f"\n{'='*50}")
    print(f"INGESTION COMPLETE")
    print(f"  Nodes parsed:     {len(nodes)}")
    print(f"  Chunks created:   {len(chunks)}")
    print(f"  Qdrant indexed:   OK")
    print(f"  BM25 index:       OK")
    print(f"  LightRAG graph:   OK")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Runtime pipeline

### src/retrieval/article_matcher.py

```python
"""Regex-based Article/Annex reference extractor."""
import re


class ArticleMatcher:
    ARTICLE_RE = re.compile(r"(?:Article|Art\.?)\s+(\d+)(?:\.(\d+))?", re.IGNORECASE)
    ANNEX_RE = re.compile(r"Annex\s+([IVX]+|\d+)(?:\.(\d+))?", re.IGNORECASE)

    ROMAN = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12,"XIII":13}
    INT_TO_ROMAN = {v: k for k, v in ROMAN.items()}

    def extract_refs(self, text: str) -> list[str]:
        refs = set()
        for m in self.ARTICLE_RE.finditer(text):
            ref = f"Article {m.group(1)}"
            refs.add(ref)
        for m in self.ANNEX_RE.finditer(text):
            num = m.group(1)
            if num.isdigit():
                num = self.INT_TO_ROMAN.get(int(num), num)
            refs.add(f"Annex {num}")
        return sorted(refs)
```

### src/retrieval/query_engine.py

```python
"""
Adaptive query engine with:
1. Multi-turn resolver
2. Article exact matcher
3. Two-level complexity detector
4. Conditional sub-query decomposition
"""
from dataclasses import dataclass, field
from openai import OpenAI
from .article_matcher import ArticleMatcher
from ..config import settings


@dataclass
class ProcessedQuery:
    original_query: str
    resolved_query: str
    explicit_refs: list[str]
    sub_queries: list[str] | None = None
    is_complex: bool = False


COMPLEXITY_SIGNALS = [
    " or ", " versus ", " compared to ", " differ",
    " because ", " resulted in ", " caused by ",
    " relationship between ", " how do ", " how does ",
]


class QueryEngine:
    def __init__(self):
        self.client = OpenAI(base_url=settings.sglang_base_url, api_key="none")
        self.matcher = ArticleMatcher()

    def process(self, query: str, history: list[dict]) -> ProcessedQuery:
        # 1. Multi-turn resolution
        resolved = self._resolve_multi_turn(query, history)

        # 2. Extract explicit references
        refs = self.matcher.extract_refs(resolved)
        for msg in history:
            if msg["role"] == "user":
                refs.extend(self.matcher.extract_refs(msg["content"]))
        refs = sorted(set(refs))

        # 3. Complexity detection (two-level)
        is_complex = self._detect_complexity(resolved)

        # 4. Sub-query decomposition (only if complex)
        sub_queries = None
        if is_complex:
            sub_queries = self._decompose(resolved)

        return ProcessedQuery(
            original_query=query,
            resolved_query=resolved,
            explicit_refs=refs,
            sub_queries=sub_queries,
            is_complex=is_complex,
        )

    def _resolve_multi_turn(self, query: str, history: list[dict]) -> str:
        if len(history) <= 1:
            return query
        try:
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rewrite the last question to be fully self-contained. "
                        "Resolve all pronouns and references.\n\n"
                        f"Conversation:\n"
                        + "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-6:])
                        + "\n\nRewritten question:"
                    ),
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return query

    def _detect_complexity(self, query: str) -> bool:
        # Level 1: Linguistic patterns (0ms)
        q_lower = query.lower()
        signal_count = sum(1 for s in COMPLEXITY_SIGNALS if s in q_lower)
        ref_count = len(self.matcher.extract_refs(query))
        question_marks = query.count("?")

        score = signal_count + (1 if ref_count >= 2 else 0) + (1 if question_marks >= 2 else 0)

        if score == 0:
            return False
        if score >= 2:
            return True

        # Level 2: LLM classify (only if uncertain, ~50ms)
        try:
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": (
                        "Classify as SIMPLE or COMPLEX.\n"
                        "SIMPLE: one fact, one article, direct answer.\n"
                        "COMPLEX: multiple articles, comparison, multi-step reasoning.\n\n"
                        f"Question: {query}\nClassification:"
                    ),
                }],
            )
            return "COMPLEX" in resp.choices[0].message.content.upper()
        except Exception:
            return False

    def _decompose(self, query: str) -> list[str]:
        try:
            resp = self.client.chat.completions.create(
                model=settings.utility_model,
                temperature=0,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        "Break this EU AI Act question into 2-3 simpler sub-questions.\n"
                        "Output only the sub-questions, one per line.\n\n"
                        f"Question: {query}"
                    ),
                }],
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:3]
        except Exception:
            return [query]
```

### src/retrieval/bm25_index.py

```python
"""BM25S in-memory search."""
import bm25s
import Stemmer


class BM25Index:
    def __init__(self, index_dir: str = "bm25_index"):
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25.load(index_dir, load_corpus=True)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(tokens, k=top_k, sorted=True)

        hits = []
        for doc_text, score in zip(results[0], scores[0]):
            if score > 0:
                hits.append({"content_raw": str(doc_text), "score": float(score)})
        return hits
```

### src/retrieval/reranker.py

```python
"""BGE cross-encoder reranker."""
from FlagEmbedding import FlagReranker
from ..config import settings


class BGEReranker:
    def __init__(self):
        self.model = FlagReranker(settings.reranker_model, use_fp16=True)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        pairs = [[query, doc.get("content_raw", "")] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

### src/retrieval/triple_retriever.py

```python
"""
Triple Retriever: Dense + BM25 + LightRAG + Exact Match.
All sources run in parallel with asyncio.gather.
"""
import asyncio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from lightrag import LightRAG, QueryParam

from .query_engine import ProcessedQuery
from .bm25_index import BM25Index
from .article_matcher import ArticleMatcher
from .reranker import BGEReranker
from ..config import settings


class TripleRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.embed_model = SentenceTransformer(settings.embedding_model)
        self.bm25 = BM25Index()
        self.matcher = ArticleMatcher()
        self.reranker = BGEReranker()

    async def retrieve(self, pq: ProcessedQuery, top_k: int = 5) -> list[dict]:
        # Run all sources in parallel
        exact_task = asyncio.to_thread(self._exact_match, pq.explicit_refs)
        dense_task = asyncio.to_thread(self._dense_search, pq.resolved_query)
        bm25_task = asyncio.to_thread(self._bm25_search, pq.resolved_query)
        graph_task = self._graph_search(pq.resolved_query)

        results = await asyncio.gather(
            exact_task, dense_task, bm25_task, graph_task,
            return_exceptions=True,
        )

        exact = results[0] if not isinstance(results[0], Exception) else []
        dense = results[1] if not isinstance(results[1], Exception) else []
        bm25 = results[2] if not isinstance(results[2], Exception) else []
        graph = results[3] if not isinstance(results[3], Exception) else []

        # If complex query, also search sub-queries
        if pq.sub_queries:
            sub_tasks = []
            for sq in pq.sub_queries:
                sub_tasks.append(asyncio.to_thread(self._dense_search, sq))
                sub_tasks.append(asyncio.to_thread(self._bm25_search, sq))
            sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)
            for r in sub_results:
                if not isinstance(r, Exception):
                    dense.extend(r)

        # RRF fusion
        candidates = self._rrf_merge(exact, dense, bm25, graph)

        # Deduplicate by article_id
        seen = set()
        unique = []
        for doc in candidates:
            aid = doc.get("article_id", "")
            if aid not in seen:
                seen.add(aid)
                unique.append(doc)

        # Rerank
        return self.reranker.rerank(pq.resolved_query, unique, top_k=top_k)

    def _exact_match(self, refs: list[str]) -> list[dict]:
        results = []
        for ref in refs:
            hits, _ = self.qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="article_id", match=MatchValue(value=ref)),
                    FieldCondition(key="granularity", match=MatchValue(value="large")),
                    FieldCondition(key="type", match=MatchValue(value="chunk")),
                ]),
                limit=1,
            )
            for h in hits:
                results.append({**h.payload, "score": 1.0, "source": "exact"})
        return results

    def _dense_search(self, query: str, top_k: int = 20) -> list[dict]:
        vec = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        hits = self.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=vec,
            limit=top_k,
            score_threshold=0.3,
        )
        return [{**h.payload, "score": h.score, "source": "dense"} for h in hits]

    def _bm25_search(self, query: str) -> list[dict]:
        hits = self.bm25.search(query, top_k=20)
        for h in hits:
            h["source"] = "bm25"
        return hits

    async def _graph_search(self, query: str) -> list[dict]:
        try:
            rag = LightRAG(working_dir=settings.lightrag_working_dir)
            await rag.initialize_storages()
            result = await rag.aquery(query, param=QueryParam(mode="mix"))
            await rag.finalize_storages()

            # Extract article references from LightRAG's text response
            refs = self.matcher.extract_refs(result)
            return [
                {"article_id": ref, "content_raw": result[:500], "score": 0.7, "source": "graph"}
                for ref in refs[:5]
            ]
        except Exception:
            return []

    def _rrf_merge(self, exact, dense, bm25, graph) -> list[dict]:
        k = settings.rrf_k
        weights = {
            "exact": settings.rrf_weight_exact,
            "dense": settings.rrf_weight_dense,
            "bm25": settings.rrf_weight_bm25,
            "graph": settings.rrf_weight_graph,
        }
        scores = {}

        for source_list in [exact, dense, bm25, graph]:
            for rank, doc in enumerate(source_list):
                aid = doc.get("article_id", doc.get("content_raw", "")[:50])
                w = weights.get(doc.get("source", ""), 0.5)
                rrf = w / (k + rank + 1)

                if aid in scores:
                    scores[aid]["rrf"] += rrf
                else:
                    scores[aid] = {"doc": doc, "rrf": rrf}

        ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
        return [item["doc"] for item in ranked]
```

### src/generation/prompts.py

```python
"""System prompt optimized for the competition metrics."""

SYSTEM_PROMPT = """You are a regulatory expert on the EU AI Act (Regulation 2024/1689).

RULES:
1. Answer in 1-4 sentences maximum. Be concise and direct.
2. Use formal regulatory language. Say "pursuant to" not "according to".
3. Base every claim on the provided provisions. If the answer is not in the
   context, say: "This specific point is not addressed in the provided provisions."
4. Never invent information not present in the context.

REFERENCE RULES:
1. Cite the MINIMUM necessary set of references.
2. Articles use Arabic numerals: "Article 6" or "Article 6.2".
3. Annexes use Roman numerals: "Annex III" or "Annex III.2".
4. NEVER use "Art.", "Article III", "Annex 3", "Article 3/2", or "Annex III-2".

OUTPUT: Return ONLY valid JSON:
{"reasoning":"brief internal reasoning","answer":"1-4 sentences","references":["Article 6","Annex III"]}"""
```

### src/generation/normalizer.py

```python
"""Regex post-processor for reference format compliance."""
import re


class ReferenceNormalizer:
    ROMAN = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12,"XIII":13}
    INT_TO_ROMAN = {v: k for k, v in ROMAN.items()}

    VALID_ARTICLE = re.compile(r"^Article \d+(\.\d+)?$")
    VALID_ANNEX = re.compile(r"^Annex [IVX]+(\.\d+)?$")

    TRANSFORMS = [
        (re.compile(r"Art\.?\s*(\d+)(?:\.(\d+))?", re.I), lambda m: f"Article {m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")),
        (re.compile(r"Article\s+(I{1,3}V?|VI{0,3}|IX|XI{0,3})(?:\.(\d+))?", re.I), None),  # handled below
        (re.compile(r"Annex\s+(\d+)(?:\.(\d+))?", re.I), None),  # handled below
        (re.compile(r"(Article\s+\d+)[/()](\d+)[)]?", re.I), lambda m: f"{m.group(1)}.{m.group(2)}"),
        (re.compile(r"(Annex\s+[IVX]+)-(\d+)", re.I), lambda m: f"{m.group(1)}.{m.group(2)}"),
    ]

    def normalize(self, references: list[str]) -> list[str]:
        normalized = set()
        for ref in references:
            clean = ref.strip()
            clean = self._apply_transforms(clean)
            if self.VALID_ARTICLE.match(clean) or self.VALID_ANNEX.match(clean):
                normalized.add(clean)
        return sorted(normalized, key=self._sort_key)

    def _apply_transforms(self, ref: str) -> str:
        # Art. N → Article N
        ref = re.sub(r"Art\.?\s*(\d+)", r"Article \1", ref, flags=re.I)
        # Article ROMAN → Article arabic
        m = re.match(r"Article\s+([IVX]+)(?:\.(\d+))?$", ref, re.I)
        if m and m.group(1).upper() in self.ROMAN:
            num = self.ROMAN[m.group(1).upper()]
            ref = f"Article {num}" + (f".{m.group(2)}" if m.group(2) else "")
        # Annex arabic → Annex ROMAN
        m = re.match(r"Annex\s+(\d+)(?:\.(\d+))?$", ref, re.I)
        if m:
            roman = self.INT_TO_ROMAN.get(int(m.group(1)), m.group(1))
            ref = f"Annex {roman}" + (f".{m.group(2)}" if m.group(2) else "")
        # Separators: / and () → .
        ref = re.sub(r"(Article\s+\d+)[/()](\d+)[)]?", r"\1.\2", ref, flags=re.I)
        ref = re.sub(r"(Annex\s+[IVX]+)-(\d+)", r"\1.\2", ref, flags=re.I)
        return ref

    def _sort_key(self, ref: str) -> tuple:
        if ref.startswith("Article"):
            n = re.search(r"\d+", ref)
            return (0, int(n.group()) if n else 0)
        return (1, self._roman_to_int(ref.split()[1].split(".")[0]))

    def _roman_to_int(self, s: str) -> int:
        return self.ROMAN.get(s.upper(), 99)
```

### src/generation/generator.py

```python
"""LLM generation via SGLang."""
import json
from openai import OpenAI
from .prompts import SYSTEM_PROMPT
from .normalizer import ReferenceNormalizer
from ..config import settings


class Generator:
    def __init__(self):
        self.client = OpenAI(base_url=settings.sglang_base_url, api_key="none")
        self.normalizer = ReferenceNormalizer()

    def generate(self, history: list[dict], chunks: list[dict]) -> dict:
        context = self._build_context(chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"RELEVANT PROVISIONS:\n\n{context}"},
            *history,
        ]

        resp = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )

        raw = json.loads(resp.choices[0].message.content)
        raw["references"] = self.normalizer.normalize(raw.get("references", []))

        return {
            "reasoning": raw.get("reasoning", ""),
            "answer": raw.get("answer", ""),
            "references": raw["references"],
        }

    def _build_context(self, chunks: list[dict]) -> str:
        seen = set()
        parts = []
        for c in chunks:
            aid = c.get("article_id", "")
            if aid in seen:
                continue
            seen.add(aid)
            text = c.get("content_raw", "")[:1500]
            parts.append(f"--- {aid} ---\n{text}")
        return "\n\n".join(parts)
```

### src/cache/semantic_cache.py

```python
"""Redis semantic cache with exact + cosine similarity matching."""
import json
import hashlib
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from ..config import settings


class SemanticCache:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url)
        self.model = SentenceTransformer(settings.embedding_model)
        self.threshold = settings.cache_similarity_threshold
        self.ttl = settings.cache_ttl_seconds

    def get(self, query: str, history: list[dict]) -> dict | None:
        # Level 1: exact hash
        key = self._hash(query, history)
        cached = self.redis.get(f"exact:{key}")
        if cached:
            return json.loads(cached)

        # Level 2: semantic similarity
        q_vec = self.model.encode(query, normalize_embeddings=True)
        for k in self.redis.scan_iter("sem:*", count=100):
            try:
                data = json.loads(self.redis.get(k))
                stored_vec = np.array(data["vec"])
                sim = float(np.dot(q_vec, stored_vec))
                if sim >= self.threshold:
                    return data["response"]
            except Exception:
                continue
        return None

    def set(self, query: str, history: list[dict], response: dict):
        key = self._hash(query, history)
        self.redis.setex(f"exact:{key}", self.ttl, json.dumps(response))

        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        self.redis.setex(f"sem:{key}", self.ttl, json.dumps({"vec": vec, "response": response}))

    def invalidate_all(self) -> int:
        keys = list(self.redis.scan_iter("exact:*")) + list(self.redis.scan_iter("sem:*"))
        if keys:
            return self.redis.delete(*keys)
        return 0

    def _hash(self, query: str, history: list[dict]) -> str:
        content = query + "".join(m.get("content", "") for m in history)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### src/observability/metrics.py

```python
"""Prometheus metrics."""
from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_requests_total", "Total requests", ["status"])
LATENCY = Histogram("rag_latency_seconds", "End-to-end latency", buckets=[0.1, 0.3, 0.5, 1, 2, 3, 5, 10])
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
RETRIEVAL_SOURCE = Counter("rag_retrieval_source", "Retrieval results by source", ["source"])
REFERENCE_COUNT = Histogram("rag_reference_count", "References per response", buckets=[0, 1, 2, 3, 4, 5, 10])
```

---

## 6. API server

### src/api/main.py

```python
"""FastAPI server — competition endpoint."""
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from ..cache.semantic_cache import SemanticCache
from ..retrieval.query_engine import QueryEngine
from ..retrieval.triple_retriever import TripleRetriever
from ..generation.generator import Generator
from ..observability.metrics import REQUESTS, LATENCY, CACHE_HITS, REFERENCE_COUNT

app = FastAPI(title="EU AI Act Q&A", version="1.0.0")
app.mount("/metrics", make_asgi_app())

cache = SemanticCache()
query_engine = QueryEngine()
retriever = TripleRetriever()
generator = Generator()


class Message(BaseModel):
    role: str
    content: str


class AnswerResponse(BaseModel):
    reasoning: str
    answer: str
    references: list[str]


@app.post("/answer", response_model=AnswerResponse)
async def answer(conversation: list[Message]):
    start = time.time()

    if not conversation or conversation[-1].role != "user":
        raise HTTPException(400, "Last message must have role='user'")

    history = [m.model_dump() for m in conversation]
    query = conversation[-1].content

    try:
        # 1. Cache check
        cached = cache.get(query, history)
        if cached:
            CACHE_HITS.inc()
            LATENCY.observe(time.time() - start)
            REQUESTS.labels(status="cache_hit").inc()
            return AnswerResponse(**cached)

        # 2. Query engine
        pq = query_engine.process(query, history)

        # 3. Triple retrieval + reranking
        chunks = await retriever.retrieve(pq, top_k=5)

        # 4. Generation + normalization
        result = generator.generate(history, chunks)

        # 5. Cache write
        cache.set(query, history, result)

        # 6. Metrics
        REQUESTS.labels(status="success").inc()
        LATENCY.observe(time.time() - start)
        REFERENCE_COUNT.observe(len(result["references"]))

        return AnswerResponse(**result)

    except Exception as e:
        REQUESTS.labels(status="error").inc()
        LATENCY.observe(time.time() - start)
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 7. Docker & docker-compose

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Pre-download embedding model and reranker
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
RUN python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"

COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      qdrant: { condition: service_healthy }
      redis: { condition: service_healthy }
    volumes:
      - ./src:/app/src
      - ./lightrag_data:/app/lightrag_data
      - ./bm25_index:/app/bm25_index

  qdrant:
    image: qdrant/qdrant:v1.14.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  qdrant_data:
  redis_data:
```

### infra/prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "rag-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

---

## 8. Testing

### tests/conftest.py

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_query():
    return [{"role": "user", "content": "What is the definition of an AI system?"}]


@pytest.fixture
def competition_questions():
    return [
        "Does the technical documentation of a high-risk AI system require to provide specifications regarding the required hardware?",
        "Are AI systems intended for emotion recognition from biometric data always prohibited?",
        "Is an AI that transcribes doctor-patient conversations prohibited? Or is it high-risk as per the use cases of Annex III of the AI Act?",
    ]
```

### tests/test_references.py

```python
"""Test that every response has valid reference format."""
import re
import pytest

VALID_ARTICLE = re.compile(r"^Article \d+(\.\d+)?$")
VALID_ANNEX = re.compile(r"^Annex [IVX]+(\.\d+)?$")


def test_normalizer_formats():
    from src.generation.normalizer import ReferenceNormalizer
    n = ReferenceNormalizer()

    assert n.normalize(["Art. 5"]) == ["Article 5"]
    assert n.normalize(["Article III"]) == ["Article 3"]
    assert n.normalize(["Annex 3"]) == ["Annex III"]
    assert n.normalize(["Annex III-2"]) == ["Annex III.2"]
    assert n.normalize(["Article 3/2"]) == ["Article 3.2"]
    assert n.normalize(["Article 3(2)"]) == ["Article 3.2"]
    assert n.normalize(["invalid reference"]) == []


def test_normalizer_deduplication():
    from src.generation.normalizer import ReferenceNormalizer
    n = ReferenceNormalizer()

    result = n.normalize(["Article 5", "Art. 5", "Article 5"])
    assert result == ["Article 5"]


def test_normalizer_sorting():
    from src.generation.normalizer import ReferenceNormalizer
    n = ReferenceNormalizer()

    result = n.normalize(["Annex III", "Article 5", "Article 3", "Annex I"])
    assert result == ["Article 3", "Article 5", "Annex I", "Annex III"]


def test_api_references_valid(client, competition_questions):
    """Every reference from the API must match valid format."""
    for q in competition_questions:
        resp = client.post("/answer", json=[{"role": "user", "content": q}])
        assert resp.status_code == 200

        data = resp.json()
        for ref in data["references"]:
            assert VALID_ARTICLE.match(ref) or VALID_ANNEX.match(ref), \
                f"Invalid reference format: '{ref}' for question: {q[:60]}"
```

### tests/test_answers.py

```python
"""Test answer quality against known ground truth."""
import pytest


GROUND_TRUTH = [
    {
        "question": "What is the definition of an AI system?",
        "must_reference": ["Article 3"],
        "answer_must_contain": ["machine-based", "system"],
    },
    {
        "question": "What AI practices are prohibited?",
        "must_reference": ["Article 5"],
        "answer_must_contain": ["prohibit"],
    },
    {
        "question": "What does Annex III list?",
        "must_reference": ["Annex III"],
        "answer_must_contain": ["high-risk"],
    },
]


@pytest.mark.parametrize("tc", GROUND_TRUTH, ids=[tc["question"][:40] for tc in GROUND_TRUTH])
def test_answer_quality(client, tc):
    resp = client.post("/answer", json=[{"role": "user", "content": tc["question"]}])
    data = resp.json()

    # Check references
    for expected_ref in tc["must_reference"]:
        assert any(r.startswith(expected_ref) for r in data["references"]), \
            f"Missing reference '{expected_ref}' in {data['references']}"

    # Check answer content
    answer_lower = data["answer"].lower()
    for keyword in tc["answer_must_contain"]:
        assert keyword.lower() in answer_lower, \
            f"Answer missing keyword '{keyword}': {data['answer'][:100]}"

    # Check conciseness (1-4 sentences)
    sentences = [s.strip() for s in data["answer"].split(".") if s.strip()]
    assert len(sentences) <= 6, f"Answer too long: {len(sentences)} sentences"
```

### tests/test_multiturn.py

```python
"""Test multi-turn conversation handling."""


def test_pronoun_resolution(client):
    """System should resolve 'it' and 'that' from previous turns."""
    conv = [
        {"role": "user", "content": "What does Article 5 establish?"},
    ]
    r1 = client.post("/answer", json=conv)
    assert r1.status_code == 200

    conv.append({"role": "assistant", "content": r1.json()["answer"]})
    conv.append({"role": "user", "content": "Are there exceptions to that?"})

    r2 = client.post("/answer", json=conv)
    assert r2.status_code == 200
    # The system should understand "that" = Article 5 prohibitions
    assert "Article 5" in str(r2.json()["references"])


def test_topic_switch(client):
    """System should handle topic changes in conversation."""
    conv = [
        {"role": "user", "content": "What is the definition of an AI system?"},
    ]
    r1 = client.post("/answer", json=conv)

    conv.append({"role": "assistant", "content": r1.json()["answer"]})
    conv.append({"role": "user", "content": "What are the penalties for non-compliance?"})

    r2 = client.post("/answer", json=conv)
    assert r2.status_code == 200
    assert len(r2.json()["answer"]) > 20
```

### tests/test_load.py

```python
"""Load test: verify latency and error rate under concurrent requests."""
import asyncio
import time
import httpx
import pytest

API = "http://localhost:8000/answer"
QUESTIONS = [
    "What is a high-risk AI system?",
    "What does Article 5 prohibit?",
    "What transparency obligations exist?",
    "What are the penalties for non-compliance?",
    "Does the AI Act apply to open source?",
]


@pytest.mark.asyncio
async def test_sequential_latency():
    """Each query should respond within 5 seconds."""
    async with httpx.AsyncClient(timeout=10) as client:
        for q in QUESTIONS:
            start = time.time()
            resp = await client.post(API, json=[{"role": "user", "content": q}])
            elapsed = time.time() - start
            assert resp.status_code == 200, f"Failed: {q}"
            assert elapsed < 5, f"Too slow ({elapsed:.1f}s): {q}"


@pytest.mark.asyncio
async def test_concurrent_load():
    """5 concurrent requests should all succeed."""
    async with httpx.AsyncClient(timeout=15) as client:
        tasks = [
            client.post(API, json=[{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}])
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        errors = [r for r in results if r.status_code != 200]
        assert len(errors) == 0, f"{len(errors)} requests failed"
```

### Run all tests

```bash
# Unit tests (no server needed for normalizer/parser tests)
pytest tests/test_references.py -v

# Integration tests (needs docker compose up + SGLang running)
pytest tests/test_answers.py tests/test_multiturn.py -v

# Load tests (needs full stack running)
pytest tests/test_load.py -v

# All with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 9. Monitoring & observability

### What Prometheus tracks

- `rag_requests_total{status}` — counter per success/error/cache_hit
- `rag_latency_seconds` — histogram of end-to-end latency
- `rag_cache_hits_total` — how many queries served from cache
- `rag_reference_count` — distribution of references per response

### Grafana setup

1. Open `http://localhost:3000` (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Import dashboard or create panels:

Key panels to create:
- Latency P50/P95/P99 over time
- Request rate (req/min)
- Cache hit rate (%)
- Error rate (%)
- Reference count distribution

### Useful Prometheus queries

```promql
# P95 latency
histogram_quantile(0.95, rate(rag_latency_seconds_bucket[5m]))

# Cache hit rate
rate(rag_cache_hits_total[5m]) / rate(rag_requests_total[5m])

# Error rate
rate(rag_requests_total{status="error"}[5m]) / rate(rag_requests_total[5m])
```

---

## 10. Evaluation framework

### eval/test_set.json

```json
[
  {
    "id": "comp_1",
    "question": "Does the technical documentation of a high-risk AI system require to provide specifications regarding the required hardware?",
    "expected_refs": ["Annex IV"],
    "keywords": ["hardware", "technical documentation"]
  },
  {
    "id": "comp_2",
    "question": "Are AI systems intended for emotion recognition from biometric data always prohibited?",
    "expected_refs": ["Article 5"],
    "keywords": ["emotion recognition", "not always", "exception"]
  },
  {
    "id": "comp_3",
    "question": "Is an AI that transcribes doctor-patient conversations prohibited? Or is it high-risk as per the use cases of Annex III of the AI Act?",
    "expected_refs": ["Article 5", "Annex III"],
    "keywords": ["not prohibited", "high-risk"]
  }
]
```

### scripts/run_eval.py

```python
"""Evaluate the system against the test set."""
import json
import time
import httpx
import re

API = "http://localhost:8000/answer"
VALID_REF = re.compile(r"^(Article \d+(\.\d+)?|Annex [IVX]+(\.\d+)?)$")


def run():
    test_set = json.load(open("eval/test_set.json"))
    client = httpx.Client(timeout=30)
    results = []

    for tc in test_set:
        start = time.time()
        resp = client.post(API, json=[{"role": "user", "content": tc["question"]}])
        latency = time.time() - start
        data = resp.json()

        # Check reference format
        format_ok = all(VALID_REF.match(r) for r in data["references"])

        # Check expected references
        refs_found = sum(
            1 for exp in tc["expected_refs"]
            if any(r.startswith(exp) for r in data["references"])
        )
        ref_recall = refs_found / len(tc["expected_refs"]) if tc["expected_refs"] else 1.0

        # Check keywords in answer
        answer_lower = data["answer"].lower()
        kw_found = sum(1 for kw in tc["keywords"] if kw.lower() in answer_lower)
        kw_recall = kw_found / len(tc["keywords"]) if tc["keywords"] else 1.0

        # Conciseness
        sentences = len([s for s in data["answer"].split(".") if s.strip()])
        concise = sentences <= 4

        result = {
            "id": tc["id"],
            "latency_s": round(latency, 2),
            "format_ok": format_ok,
            "ref_recall": round(ref_recall, 2),
            "keyword_recall": round(kw_recall, 2),
            "concise": concise,
            "refs": data["references"],
            "answer_preview": data["answer"][:120],
        }
        results.append(result)

        status = "PASS" if format_ok and ref_recall >= 0.5 and concise else "FAIL"
        print(f"{status} {tc['id']}: latency={latency:.2f}s refs={data['references']} "
              f"format={format_ok} ref_recall={ref_recall:.0%} kw_recall={kw_recall:.0%}")

    # Summary
    print(f"\n{'='*50}")
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_ref = sum(r["ref_recall"] for r in results) / len(results)
    avg_kw = sum(r["keyword_recall"] for r in results) / len(results)
    all_format = all(r["format_ok"] for r in results)
    all_concise = all(r["concise"] for r in results)

    print(f"Avg latency:      {avg_latency:.2f}s")
    print(f"Ref recall:       {avg_ref:.0%}")
    print(f"Keyword recall:   {avg_kw:.0%}")
    print(f"All formats OK:   {all_format}")
    print(f"All concise:      {all_concise}")

    json.dump(results, open("eval/results.json", "w"), indent=2)
    print("\nResults saved to eval/results.json")


if __name__ == "__main__":
    run()
```

---

## 11. Cache warmup strategy

### scripts/warmup_cache.py

```python
"""Pre-compute answers for ~300 predictable questions."""
import httpx
import time

API = "http://localhost:8000/answer"

QUESTIONS = []

# Per-article questions (113 articles)
for i in range(1, 114):
    QUESTIONS.append(f"What does Article {i} of the EU AI Act establish?")

# Per-annex questions (13 annexes)
ROMAN = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII"]
for r in ROMAN:
    QUESTIONS.append(f"What does Annex {r} of the EU AI Act cover?")

# Thematic questions
QUESTIONS.extend([
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
    # Competition example questions + variants
    "Does the technical documentation of a high-risk AI system require to provide specifications regarding the required hardware?",
    "What must be included in technical documentation for high-risk AI?",
    "Are AI systems intended for emotion recognition from biometric data always prohibited?",
    "When is emotion recognition in AI prohibited?",
    "What exceptions exist for emotion recognition AI?",
    "Is an AI that transcribes doctor-patient conversations prohibited?",
    "How does the AI Act classify medical transcription AI?",
    "Is medical transcription AI high-risk under Annex III?",
    # Cross-article questions
    "How do Article 6 and Annex III relate to each other?",
    "What is the relationship between prohibited practices and high-risk systems?",
    "How do providers and deployers obligations differ?",
])


def warmup():
    client = httpx.Client(timeout=30)
    total = len(QUESTIONS)
    success = 0

    print(f"Warming up cache with {total} questions...")
    start = time.time()

    for i, q in enumerate(QUESTIONS):
        try:
            resp = client.post(API, json=[{"role": "user", "content": q}])
            if resp.status_code == 200:
                success += 1
        except Exception as e:
            print(f"  Error: {q[:50]}... — {e}")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  {i+1}/{total} ({success} OK) — {rate:.1f} q/s — ~{remaining:.0f}s remaining")

    print(f"\nDone: {success}/{total} cached in {time.time()-start:.0f}s")


if __name__ == "__main__":
    warmup()
```

---

## 12. Deployment checklist

### Startup sequence

```bash
# 1. Start infrastructure
docker compose up -d qdrant redis prometheus grafana

# 2. Start SGLang on A100
python -m sglang.launch_server \
  --model-path google/gemma-3-27b-it \
  --quantization awq \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmsys/sglang-EAGLE-gemma-3-27b-it \
  --port 8899 \
  --dtype float16 \
  --mem-fraction-static 0.85

# 3. Wait for SGLang to be ready
until curl -s http://localhost:8899/health | grep -q "ok"; do sleep 2; done

# 4. Download EU AI Act
python scripts/download.py

# 5. Run ingestion (parse + chunk + index + graph)
python scripts/ingest.py

# 6. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 7. Warm up cache
python scripts/warmup_cache.py

# 8. Run evaluation
python scripts/run_eval.py

# 9. Run tests
pytest tests/ -v

# 10. Verify health
curl http://localhost:8000/health
```

### Pre-submission verification

```bash
# Simulate what regenold will send
curl -X POST http://YOUR_IP:8000/answer \
  -H "Content-Type: application/json" \
  -d '[{"role":"user","content":"Are AI systems for emotion recognition always prohibited?"}]'

# Expected: JSON with reasoning, answer (1-4 sentences), references (Article/Annex format)
```

### Monitoring URLs

- API: http://localhost:8000/health
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Qdrant dashboard: http://localhost:6333/dashboard
