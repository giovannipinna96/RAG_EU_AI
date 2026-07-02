"""Prometheus metrics exposed at /metrics."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_requests_total", "Total requests", ["status"])
LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end latency",
    buckets=[0.1, 0.3, 0.5, 1, 2, 3, 5, 10],
)
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
RETRIEVAL_SOURCE = Counter("rag_retrieval_source", "Retrieval results by source", ["source"])
REFERENCE_COUNT = Histogram(
    "rag_reference_count",
    "References per response",
    buckets=[0, 1, 2, 3, 4, 5, 10],
)
