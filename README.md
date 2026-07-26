# Semantic Caching Layer for LLMs 🚀

A drop-in middleware caching service that sits between your application and
your LLM provider(s), detects semantically similar requests that have
already been answered, and serves cached responses instantly — cutting
latency to near-zero and reducing API costs.

Supports **OpenAI** and **Groq** as chat-completion upstreams, routed
per-request by model name (or an explicit `"provider"` field). Embeddings
for the semantic cache always go through OpenAI's `text-embedding-3-small`,
since Groq doesn't offer an embeddings endpoint.

## 🌟 Key Features

* **Drop-In Proxy:** Mirrors the OpenAI Chat Completions API contract
  (`/v1/chat/completions`) exactly — change your `base_url` and you're done.
* **Dual Provider Routing:** Chat completions route to OpenAI or Groq based
  on the model name (Groq's `llama3-*`, `mixtral-*`, `gemma*` families are
  detected automatically), or force it with `"provider": "openai"|"groq"`
  in the request body.
* **Semantic Vector Search:** Redis Stack + cosine similarity KNN. Only the
  final user turn is embedded; the full prior conversation is folded into
  the cache partition key so unrelated conversations can never collide.
* **Threshold Tuner:** `/api/tuning/analysis` replays recent similarity
  scores to show the hit-rate-vs-threshold tradeoff so you can pick a
  number with data instead of guessing.
* **Dynamic TTL:** Regex/word-boundary classifier assigns a short TTL to
  time-sensitive queries, a long TTL to stable/factual ones, and can
  disable caching entirely for "right now" style queries.
* **Observability:** Langfuse tracing, Prometheus metrics (hit rate,
  similarity distribution, estimated cost saved), Grafana dashboard.
* **Streaming:** Cache hits stream instantly; cache misses stream from the
  provider to the client while safely reassembling the full response
  (byte/char-boundary-safe) for caching in the background.

## 🛠️ Tech Stack

* Python 3.12 · FastAPI · Redis Stack · OpenAI Embeddings
* OpenAI & Groq (chat completions) · Langfuse · Helicone
* Prometheus & Grafana · Docker & Docker Compose

## 🚀 How to Run Locally

### Prerequisites
* Docker and Docker Compose
* An [OpenAI API key](https://platform.openai.com/api-keys) (required —
  used for embeddings regardless of chat provider)
* Optionally, a [Groq API key](https://console.groq.com/keys)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd semantic-caching-layer
   ```

2. **Set environment variables** (create a `.env` file at the repo root):
   ```bash
   OPENAI_API_KEY="sk-your-openai-api-key"
   GROQ_API_KEY="gsk-your-groq-api-key"          # optional
   DEFAULT_PROVIDER="openai"                       # or "groq"

   # Optional: observability
   LANGFUSE_PUBLIC_KEY="pk-lf-..."
   LANGFUSE_SECRET_KEY="sk-lf-..."
   HELICONE_API_KEY="sk-helicone-..."

   # Optional: tuning
   SIMILARITY_THRESHOLD=0.95
   EST_COST_PER_REQUEST_USD=0.002
   ```

3. **Start everything:**
   ```bash
   docker-compose up --build
   ```
   This spins up:
   * **FastAPI Proxy** on `localhost:8000` (health check at `/health`)
   * **Grafana** on `localhost:3000`
   * **Prometheus** on `localhost:9090`
   * **Redis Stack** on `localhost:6379` (RedisInsight on `:8001`)

### Routing to Groq vs OpenAI

The proxy inspects `model` in the request body:

```jsonc
// -> routed to Groq automatically
{"model": "llama3-8b-8192", "messages": [...]}

// -> routed to OpenAI automatically
{"model": "gpt-4o-mini", "messages": [...]}

// -> force a provider explicitly
{"model": "some-custom-model", "provider": "groq", "messages": [...]}
```

## 🎛️ Threshold Tuning

```bash
curl http://localhost:8000/api/tuning                 # read current threshold
curl -X POST "http://localhost:8000/api/tuning?threshold=0.93"   # update it
curl http://localhost:8000/api/tuning/analysis         # hit-rate curve from live traffic
```

## 🧪 Load Testing

```bash
python load_test.py --requests 2000 --concurrency 20 --model llama3-8b-8192
```
Prints hit rate, cache-hit vs cache-miss latency (avg/p50/p95), the observed
speedup, and an estimated dollar savings figure.

## 🌐 How it works under the hood

1. **Conversation parsing:** the system prompt and every message except the
   final user turn form a *conversation prefix*; only the final user
   message is embedded. The prefix (plus system prompt, temperature,
   max_tokens, model) is hashed into a `partition_key` — two requests only
   share a cache partition if all of that matches exactly.
2. **Embedding:** the final user message is embedded via
   `text-embedding-3-small` (1536-dim).
3. **Vector search:** Redis Stack finds the nearest neighbor *within* the
   same `partition_key` using cosine similarity.
4. **Hit/miss:** similarity ≥ threshold → instant cached response. Otherwise
   the request is routed to OpenAI or Groq (optionally via Helicone for
   OpenAI), traced in Langfuse, and the response is cached asynchronously
   with a TTL from the classifier — only if the upstream call fully
   succeeded.

## Changelog highlights (bug fixes from the original prototype)

* Multi-turn conversations no longer get flattened into one blob — only the
  latest user turn is embedded, and the full prior conversation is part of
  the cache partition key, so unrelated chats can no longer collide on a
  cache hit.
* SSE stream reassembly is now buffer-safe across chunk/character
  boundaries (previously could silently truncate or corrupt what got
  cached).
* Redis text fields are decoded safely (`decode_responses=False` was
  leaking `bytes` into JSON responses on cache hits).
* Failed embeddings no longer get cached as zero-vectors (which could
  cause spurious future hits); the request just bypasses the cache.
* `/api/tuning` update is now a `POST` instead of a side-effecting `GET`,
  with input validation.
* `CORSMiddleware` no longer combines `allow_origins=["*"]` with
  `allow_credentials=True` (invalid per spec, silently rejected by
  browsers).
* Added `/health` for container/orchestrator healthchecks.
* Removed an unused `sentence-transformers`/`torch` dependency that was
  bloating the Docker image and never actually used for embeddings.
* Dockerfile now runs as a non-root user and declares a `HEALTHCHECK`.
* `render.yaml` now provisions `OPENAI_API_KEY` (previously missing, so
  cache-miss requests would fail with 401 on Render even though only
  `GROQ_API_KEY` was wired up).
