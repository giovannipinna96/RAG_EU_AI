"""Centralised application settings, loaded from environment / `.env`.

A single `settings` instance is imported across the codebase. Values are
validated by pydantic-settings v2 at import time.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the EU AI Act RAG system."""

    # --- LLM (OpenAI-compatible server: SGLang, or llama.cpp for gemma-4) ---
    sglang_base_url: str = "http://localhost:8899/v1"
    llm_model: str = "default"
    utility_model: str = "default"
    # gemma-4 (and other reasoning models) stream their chain-of-thought in a
    # separate `reasoning_content` field BEFORE the visible answer. With thinking
    # enabled a small max_tokens budget is spent entirely on reasoning and
    # `content` comes back empty (finish_reason="length"). For the competition we
    # disable thinking — the answer JSON is complete and latency drops ~3x
    # (~9.7s vs ~29s on gemma-4-31B-Q8) — and keep a comfortable decode budget.
    llm_max_tokens: int = 512
    llm_enable_thinking: bool = False

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # --- Qdrant ---
    # When ``qdrant_local_path`` is set it takes precedence and the client runs
    # in embedded on-disk mode (no server required) — useful on HPC nodes.
    qdrant_url: str = "http://localhost:6333"
    qdrant_local_path: str | None = None
    qdrant_collection: str = "eu_ai_act"

    # --- Redis (semantic cache) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- LightRAG ---
    lightrag_working_dir: str = "./lightrag_data"
    # Query mode for the graph retrieval source. "mix"/"local"/"global" each make
    # an LLM keyword-extraction call to the generation model per query — the
    # single most expensive step in the pipeline (~1 extra gemma round-trip).
    # "naive" is pure vector search over the graph's chunk store: no LLM call, so
    # it keeps LightRAG as a cheap extra recall source while cutting the latency.
    # Trade-off: naive loses the entity/relation graph traversal (cross-article
    # links); dense + BM25 + xref-expansion + exact-match still cover most of it.
    lightrag_mode: str = "mix"

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    # Second-stage LLM rerank over the BGE-ranked candidates. The LLM reorders
    # by legal relevance using adaptive query-centered snippets; on any error it
    # falls back to the BGE order. Disable to use BGE ranking alone.
    enable_llm_rerank: bool = True
    # Multi-chunk voting rerank: skip the pre-rerank per-article dedup, LLM-rank
    # an un-deduped pool of `rerank_candidate_chunks` chunks, then aggregate the
    # ranking into a per-article result by positional voting. Fixes "dedup kept
    # the wrong chunk of the right article". See docs/voting_rerank_design.md.
    # Default on: the A/B eval (jobs 56944 off vs 56945 on, 30Q) raised ref
    # recall 85% -> 92% and recovered comp_11 (deepfake) + comp_1, at +0.5s.
    enable_chunk_voting: bool = True
    rerank_candidate_chunks: int = 12
    # Pool-blend (voting mode only): instead of selecting the LLM-vote pool by
    # pure BGE score, reserve the top `pool_rrf_reserve` RRF candidates first,
    # then fill the rest by BGE. Stops the cross-encoder from burying a doc that
    # every other retriever ranked first (comp_27: Art 62 RRF #1 but BGE #8).
    # Default off pending the A/B on the 30-question eval — regression risk on
    # the currently-passing cases. See docs/voting_rerank_design.md.
    enable_pool_blend: bool = False
    pool_rrf_reserve: int = 4
    # Per-candidate snippet size in the LLM rerank / voting prompt. Large by
    # default so a rule buried deep in an article still reaches the LLM, but it
    # dominates the rerank prefill cost (candidates × snippet). Shrink it (and/or
    # rerank_candidate_chunks) for a cheaper "voting light".
    rerank_llm_snippet_chars: int = 3000

    # --- BM25 ---
    bm25_index_dir: str = "./bm25_index"

    # --- Semantic cache ---
    cache_ttl_seconds: int = 3600
    # 0.97 is deliberately strict: legal Q&A requires near-identical phrasing
    # before a cached answer is reused.  0.95 caused false positives between
    # legally distinct questions (e.g. Article 5 vs Article 6 collision in eval).
    cache_similarity_threshold: float = 0.97
    # When True, the ref-set extracted from the query must match the stored
    # ref-set exactly before a semantic hit is eligible (two-layer guard).
    cache_require_ref_match: bool = True

    # --- Reciprocal Rank Fusion ---
    rrf_weight_exact: float = 3.0
    rrf_weight_dense: float = 1.0
    rrf_weight_bm25: float = 0.6
    rrf_weight_graph: float = 0.9
    # Cross-reference expansion: provisions cited inside retrieved chunks are
    # pulled in as extra candidates with this fusion weight.
    rrf_weight_xref: float = 0.5
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_extra_body(self) -> dict:
        """Extra OpenAI request-body params shared by every LLM call.

        ``chat_template_kwargs.enable_thinking`` toggles the model's native
        chain-of-thought (llama.cpp / gemma-4). Kept off by default so no call
        is starved of decode budget by a long reasoning prefix. Passed via the
        OpenAI SDK ``extra_body`` (non-standard field) on every ``create``.
        """
        return {"chat_template_kwargs": {"enable_thinking": self.llm_enable_thinking}}


settings = Settings()
