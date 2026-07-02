"""FastAPI server — the competition endpoint contract.

Components (cache, query engine, retriever, generator) are initialised lazily on
first use rather than at import time, so the module can be imported cheaply (for
tests and tooling) without loading embedding weights or opening connections.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, ValidationError

from ..observability.metrics import CACHE_HITS, LATENCY, REFERENCE_COUNT, REQUESTS

log = structlog.get_logger(__name__)

app = FastAPI(title="EU AI Act Q&A", version="1.0.0")
app.mount("/metrics", make_asgi_app())


class _Components:
    """Lazy singletons for the heavy runtime objects."""

    def __init__(self) -> None:
        self._cache = None
        self._query_engine = None
        self._retriever = None
        self._generator = None

    @property
    def cache(self):
        if self._cache is None:
            from ..cache.semantic_cache import SemanticCache

            self._cache = SemanticCache()
        return self._cache

    @property
    def query_engine(self):
        if self._query_engine is None:
            from ..retrieval.query_engine import QueryEngine

            self._query_engine = QueryEngine()
        return self._query_engine

    @property
    def retriever(self):
        if self._retriever is None:
            from ..retrieval.triple_retriever import TripleRetriever

            self._retriever = TripleRetriever()
        return self._retriever

    @property
    def generator(self):
        if self._generator is None:
            from ..generation.generator import Generator

            self._generator = Generator()
        return self._generator


components = _Components()


class Message(BaseModel):
    role: str
    content: str


class AnswerResponse(BaseModel):
    reasoning: str
    answer: str
    references: list[str]


def _extract_messages(payload: object) -> list[Message]:
    """Normalise either accepted input shape into a list of messages.

    The competition specifies an "OpenAI/LiteLLM standard" conversation history.
    In practice that arrives either as a bare message array
    ``[{"role","content"}, ...]`` (the rules' own example) or wrapped as the
    OpenAI chat-completions body ``{"messages": [...]}``. We accept both. Extra
    top-level fields (model, temperature, ...) and extra per-message fields are
    ignored. Anything else fails with 422 rather than crashing.
    """
    if isinstance(payload, dict):
        raw = payload.get("messages")
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = None
    if not isinstance(raw, list) or not raw:
        raise HTTPException(
            422,
            "Body must be a non-empty message array or an object with a non-empty "
            "'messages' array",
        )
    try:
        return [Message.model_validate(m) for m in raw]
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid message in conversation: {exc.errors()}") from exc


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: Request) -> AnswerResponse:
    start = time.time()

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 — any malformed/non-JSON body
        raise HTTPException(400, "Request body must be valid JSON") from exc

    conversation = _extract_messages(payload)
    if conversation[-1].role != "user":
        raise HTTPException(400, "Last message must have role='user'")

    history = [m.model_dump() for m in conversation]
    query = conversation[-1].content

    try:
        # Each component call below is synchronous (embedding inference, OpenAI
        # HTTP, Redis I/O), so off-load to a thread to keep the event loop free
        # under concurrent load. Retrieval is already async.
        # 1. Cache check.
        cached = await asyncio.to_thread(components.cache.get, query, history)
        if cached:
            CACHE_HITS.inc()
            REQUESTS.labels(status="cache_hit").inc()
            LATENCY.observe(time.time() - start)
            return AnswerResponse(**cached)

        # 2. Adaptive query processing.
        pq = await asyncio.to_thread(components.query_engine.process, query, history)

        # 3. Hybrid retrieval + reranking.
        chunks = await components.retriever.retrieve(pq, top_k=5)

        # 4. Generation + reference normalisation. Pass the query's explicit refs
        # so citations stay grounded in retrieval, not the LLM's free choice.
        result = await asyncio.to_thread(
            components.generator.generate, history, chunks, pq.explicit_refs
        )

        # 5. Cache write.
        await asyncio.to_thread(components.cache.set, query, history, result)

        # 6. Metrics.
        REQUESTS.labels(status="success").inc()
        LATENCY.observe(time.time() - start)
        REFERENCE_COUNT.observe(len(result["references"]))

        return AnswerResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        REQUESTS.labels(status="error").inc()
        LATENCY.observe(time.time() - start)
        log.error("answer_failed", error=str(exc))
        raise HTTPException(500, str(exc)) from exc


@app.on_event("startup")
async def _warmup() -> None:
    """Pre-load heavy models and open the stores at boot so no *request* pays the
    cold-start cost. Runs one dummy pass through the real pipeline (query engine →
    retriever → generator), which loads the BGE embedding + reranker weights,
    opens Qdrant/BM25/LightRAG, and primes the LLM connection. Any failure (e.g.
    the LLM tunnel not yet ready) is logged but never blocks startup — the app
    still serves and warms lazily on first use. The cache is intentionally not
    written, so the throwaway query leaves no entry behind.
    """
    t0 = time.time()
    warm_history = [{"role": "user", "content": "What is the definition of an AI system?"}]
    try:
        pq = await asyncio.to_thread(
            components.query_engine.process, warm_history[-1]["content"], warm_history
        )
        chunks = await components.retriever.retrieve(pq, top_k=3)
        await asyncio.to_thread(
            components.generator.generate, warm_history, chunks, pq.explicit_refs
        )
        _ = components.cache  # force the cache backend + its embedder to load too
        log.info("warmup_complete", seconds=round(time.time() - t0, 1))
    except Exception as exc:  # noqa: BLE001 — warmup is best-effort
        log.warning("warmup_failed", error=str(exc), seconds=round(time.time() - t0, 1))


_TEST_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EU AI Act Q&A — test</title>
<style>
 body{font:16px/1.5 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a1a2e}
 h1{font-size:1.3rem} textarea{width:100%;box-sizing:border-box;padding:.6rem;font-size:1rem}
 button{margin-top:.6rem;padding:.5rem 1.2rem;font-size:1rem;cursor:pointer}
 .box{margin-top:1rem;padding:1rem;border:1px solid #ccc;border-radius:8px;background:#fafafa;white-space:pre-wrap}
 .k{color:#555;font-weight:600} .refs span{display:inline-block;background:#e8eefc;border-radius:4px;padding:.1rem .5rem;margin:.15rem}
 .muted{color:#888;font-size:.85rem}
</style></head><body>
<h1>EU AI Act Q&amp;A — browser test</h1>
<p class="muted">Type a question about the EU AI Act (Reg. 2024/1689). Multi-turn not shown here; this sends a single user turn to <code>POST /answer</code>.</p>
<textarea id="q" rows="3">What are the transparency obligations for deepfakes?</textarea>
<button id="go" onclick="ask()">Ask</button>
<div id="out"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out'), btn=document.getElementById('go');
  btn.disabled=true; out.innerHTML='<div class="box muted">…querying (first fresh query ~10s)…</div>';
  const t0=performance.now();
  try{
    const r=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify([{role:'user',content:q}])});
    const dt=((performance.now()-t0)/1000).toFixed(1);
    if(!r.ok){out.innerHTML='<div class="box">HTTP '+r.status+' — '+(await r.text())+'</div>';return;}
    const d=await r.json();
    const refs=(d.references||[]).map(x=>'<span>'+x+'</span>').join(' ')||'<span class="muted">none</span>';
    out.innerHTML='<div class="box"><div class="k">ANSWER</div>'+ (d.answer||'') +
      '<div class="k" style="margin-top:.8rem">REFERENCES</div><div class="refs">'+refs+'</div>'+
      '<div class="k" style="margin-top:.8rem">REASONING</div><span class="muted">'+(d.reasoning||'')+'</span>'+
      '<div class="muted" style="margin-top:.8rem">latency '+dt+'s</div></div>';
  }catch(e){out.innerHTML='<div class="box">error: '+e+'</div>';}
  finally{btn.disabled=false;}
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')ask();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def test_page() -> str:
    """Minimal same-origin browser test UI for POST /answer (no CORS setup)."""
    return _TEST_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/cache/invalidate")
def cache_invalidate() -> dict:
    removed = components.cache.invalidate_all()
    return {"status": "ok", "invalidated": removed}
