"""
Custom Prometheus metrics beyond what prometheus-fastapi-instrumentator
gives you automatically (request count/latency). These are the numbers
the Grafana "cost savings" / hit-rate dashboard is built on.
"""
from prometheus_client import Counter, Histogram

cache_requests_total = Counter(
    "semantic_cache_requests_total",
    "Total chat completion requests received by the proxy",
    ["provider"],
)

cache_hits_total = Counter(
    "semantic_cache_hits_total",
    "Total requests served from the semantic cache",
    ["provider"],
)

cache_misses_total = Counter(
    "semantic_cache_misses_total",
    "Total requests forwarded upstream (cache miss)",
    ["provider"],
)

cache_similarity_score = Histogram(
    "semantic_cache_similarity_score",
    "Similarity score of the nearest cached neighbor found for each request "
    "(useful for tuning the similarity threshold: shows the hit/near-miss "
    "distribution).",
    buckets=(0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0),
)

estimated_cost_saved_usd_total = Counter(
    "semantic_cache_estimated_cost_saved_usd_total",
    "Estimated cumulative USD saved by serving cache hits instead of "
    "calling the upstream provider (rough estimate, see "
    "EST_COST_PER_REQUEST_USD).",
)

upstream_errors_total = Counter(
    "semantic_cache_upstream_errors_total",
    "Total errors returned by the upstream provider",
    ["provider"],
)

request_latency_seconds = Histogram(
    "semantic_cache_request_latency_seconds",
    "End-to-end request latency, labeled by whether it was served from "
    "cache or forwarded upstream. This is the source for the 'cache hits "
    "are Nx faster' comparison panel.",
    ["cache_status"],  # "hit" | "miss"
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)

