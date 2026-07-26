import asyncio
import codecs
import time
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from langfuse import Langfuse
from prometheus_fastapi_instrumentator import Instrumentator

from src.redis_client import redis_client
from src.cache_service import cache_service
from src.embedding_service import embedding_service
from src.ttl_classifier import ttl_classifier
from src.config import settings
from src.providers import resolve_upstream, build_headers, ProviderError
from src import metrics

langfuse = None
if settings.langfuse_public_key and settings.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client.setup_index(embedding_service.dimension)
    # One shared client with a real connection pool, reused across every
    # request. Creating a new httpx.AsyncClient() per-request (the previous
    # approach) opens a fresh connection pool each time -- under sustained
    # concurrent load this churns through sockets/ports fast and shows up
    # as intermittent connection errors, especially on Windows.
    app.state.upstream_client = httpx.AsyncClient(
        timeout=settings.upstream_timeout_seconds,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    yield
    await app.state.upstream_client.aclose()
    if langfuse:
        langfuse.flush()


app = FastAPI(title="Semantic Caching Layer", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# NOTE: allow_origins=["*"] + allow_credentials=True is rejected by browsers
# (and technically invalid per the CORS spec). If you need cookies/auth
# headers from a browser client, set CORS_ALLOWED_ORIGINS to an explicit
# list instead of "*". Since this proxy is typically called
# server-to-server (you just swap base_url), credentials are off by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_conversation(messages: list) -> tuple[str, list, str]:
    """
    Split an OpenAI-style messages array into:
      - system_prompt: concatenation of all system messages
      - conversation_prefix: every message before the final user message
        (this is what must match exactly for a cache hit)
      - query_text: the content of the final user message (this is what
        gets embedded and semantically matched)

    Only the final user message is embedded -- earlier drafts concatenated
    every user turn in the whole conversation into one blob, which both
    diluted the embedding's meaning and let unrelated conversations collide
    whenever their latest message happened to be similar.
    """
    system_prompt = " ".join(m["content"] for m in messages if m.get("role") == "system").strip()

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        # No user message at all -- nothing sensible to cache/embed.
        return system_prompt, messages, ""

    query_text = str(messages[last_user_idx].get("content", "")).strip()
    conversation_prefix = [m for i, m in enumerate(messages) if i != last_user_idx and m.get("role") != "system"]
    return system_prompt, conversation_prefix, query_text


def _make_chat_response(trace_id: str, model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{trace_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


def _make_chat_chunk(trace_id: str, model: str, content: str | None, finish_reason: str | None) -> dict:
    delta = {"content": content} if content is not None else {}
    return {
        "id": f"chatcmpl-{trace_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def stream_upstream_and_cache(
    resp: httpx.Response,
    background_tasks: BackgroundTasks,
    query_text: str,
    embedding_bytes: bytes | None,
    partition_key: str,
    ttl: int,
):
    """
    Stream SSE bytes straight through to the client while reassembling the
    full text for caching.

    Important: httpx `aiter_bytes()` yields arbitrary byte boundaries, not
    line boundaries and not even guaranteed UTF-8 character boundaries. The
    original version decoded each raw chunk independently, which could
    split a multi-byte character or a "data: {...}" line across two chunks
    and silently drop/corrupt part of the cached response (errors were
    swallowed). This version uses an incremental UTF-8 decoder and buffers
    partial lines across chunks.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    line_buffer = ""
    full_response_text = ""

    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
            line_buffer += decoder.decode(chunk)

            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                line = line.strip("\r")
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    data_json = json.loads(line[6:])
                    choices = data_json.get("choices") or []
                    if choices and "delta" in choices[0]:
                        full_response_text += choices[0]["delta"].get("content", "") or ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    finally:
        # Flush any trailing bytes held by the incremental decoder.
        line_buffer += decoder.decode(b"", final=True)

    if full_response_text:
        background_tasks.add_task(
            cache_service.insert_cache,
            user_prompt=query_text,
            response=full_response_text,
            embedding_bytes=embedding_bytes,
            partition_key=partition_key,
            ttl=ttl,
        )


def _safe_langfuse_trace(**kwargs):
    if not langfuse:
        return
    try:
        langfuse.trace(**kwargs)
    except Exception as e:
        # Observability must never take down the actual proxy request.
        print(f"[langfuse] trace() failed, continuing without it: {e}")


def _safe_langfuse_generation(**kwargs):
    if not langfuse:
        return
    try:
        langfuse.generation(**kwargs)
    except Exception as e:
        print(f"[langfuse] generation() failed, continuing without it: {e}")


@app.get("/health")
async def health():
    ok = redis_client.ping()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "redis": ok},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    messages = body.get("messages", [])
    model = body.get("model", settings.model_name)
    temperature = body.get("temperature", 1.0)
    max_tokens = body.get("max_tokens")
    stream = body.get("stream", False)
    explicit_provider = body.get("provider")  # optional override: "openai" | "groq"

    try:
        upstream_url, api_key, provider = resolve_upstream(model, explicit_provider)
    except ProviderError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    metrics.cache_requests_total.labels(provider=provider).inc()

    system_prompt, conversation_prefix, query_text = _extract_conversation(messages)

    trace_id = str(uuid.uuid4())
    _safe_langfuse_trace(
        id=trace_id,
        name="chat_completion",
        input=query_text,
        metadata={"model": model, "temperature": temperature, "provider": provider},
    )

    start_time = time.time()
    # check_cache does CPU-bound embedding (sentence-transformers, when using
    # the local backend) plus a synchronous Redis call. Neither yields to the
    # event loop, so calling this directly would serialize ALL concurrent
    # requests on Uvicorn's single event loop thread -- under load, later
    # requests back up and eventually hit client-side read timeouts even
    # though the server is technically still "working". Running it in a
    # worker thread lets multiple requests actually progress concurrently.
    cache_result = await asyncio.to_thread(
        cache_service.check_cache,
        query_text=query_text,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        conversation_prefix=conversation_prefix,
    )

    if cache_result["hit"]:
        cached_response = cache_result["response"]
        metrics.cache_hits_total.labels(provider=provider).inc()
        metrics.estimated_cost_saved_usd_total.inc(settings.est_cost_per_request_usd)
        metrics.request_latency_seconds.labels(cache_status="hit").observe(time.time() - start_time)

        if langfuse:
            _safe_langfuse_generation(
                trace_id=trace_id,
                name="semantic_cache",
                model="cache",
                input=query_text,
                output=cached_response,
                start_time=start_time,
                end_time=time.time(),
                metadata={"similarity": cache_result.get("similarity")},
            )

        if stream:
            def cache_streamer():
                yield f"data: {json.dumps(_make_chat_chunk(trace_id, model, cached_response, None))}\n\n".encode("utf-8")
                yield f"data: {json.dumps(_make_chat_chunk(trace_id, model, None, 'stop'))}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                cache_streamer(),
                media_type="text/event-stream",
                headers={"X-Cache-Status": "HIT", "X-Cache-Similarity": f"{cache_result.get('similarity', 0):.4f}"},
            )
        else:
            return Response(
                content=json.dumps(_make_chat_response(trace_id, model, cached_response)),
                media_type="application/json",
                headers={"X-Cache-Status": "HIT", "X-Cache-Similarity": f"{cache_result.get('similarity', 0):.4f}"},
            )

    # --- Cache miss: forward upstream ---
    metrics.cache_misses_total.labels(provider=provider).inc()
    headers = build_headers(provider, api_key)
    ttl = ttl_classifier.classify(query_text)

    client: httpx.AsyncClient = request.app.state.upstream_client

    if stream:
        req = client.build_request("POST", upstream_url, headers=headers, json=body)
        resp = await client.send(req, stream=True)
        metrics.request_latency_seconds.labels(cache_status="miss").observe(time.time() - start_time)
        if resp.status_code >= 400:
            metrics.upstream_errors_total.labels(provider=provider).inc()
        return StreamingResponse(
            stream_upstream_and_cache(
                resp,
                background_tasks,
                query_text,
                cache_result.get("embedding_bytes"),
                cache_result["partition_key"],
                ttl,
            ),
            media_type="text/event-stream",
            headers={"X-Cache-Status": "MISS", "X-TTL-Seconds": str(ttl), "X-Provider": provider},
        )
    else:
        resp = await client.post(upstream_url, headers=headers, json=body)
        metrics.request_latency_seconds.labels(cache_status="miss").observe(time.time() - start_time)
        if resp.status_code >= 400:
            metrics.upstream_errors_total.labels(provider=provider).inc()
            return Response(content=resp.text, status_code=resp.status_code, media_type="application/json")

        data = resp.json()
        content = ""
        if data.get("choices"):
            content = data["choices"][0]["message"]["content"]
            # Only cache complete, successful responses.
            background_tasks.add_task(
                cache_service.insert_cache,
                user_prompt=query_text,
                response=content,
                embedding_bytes=cache_result.get("embedding_bytes"),
                partition_key=cache_result["partition_key"],
                ttl=ttl,
            )

        if langfuse:
            _safe_langfuse_generation(
                trace_id=trace_id,
                name=f"{provider}_upstream",
                model=model,
                input=query_text,
                output=content,
                start_time=start_time,
                end_time=time.time(),
            )

        return Response(
            content=resp.text,
            media_type="application/json",
            headers={"X-Cache-Status": "MISS", "X-TTL-Seconds": str(ttl), "X-Provider": provider},
        )


@app.get("/api/tuning")
async def get_tuning():
    return {"current_threshold": settings.similarity_threshold}


@app.post("/api/tuning")
async def set_tuning(threshold: float):
    if not 0.0 <= threshold <= 1.0:
        return JSONResponse(status_code=400, content={"error": "threshold must be between 0.0 and 1.0"})
    settings.similarity_threshold = threshold
    return {"message": "Threshold updated", "current_threshold": settings.similarity_threshold}


@app.get("/api/tuning/analysis")
async def tuning_analysis(sample_limit: int = 5000):
    """
    Shows the hit-rate-vs-threshold tradeoff using recently observed
    similarity scores, so you can pick a threshold with actual data instead
    of guessing. This is the 'similarity threshold tuner' talking point
    from the project spec.
    """
    samples = cache_service.get_similarity_samples(limit=sample_limit)
    if not samples:
        return {"sample_count": 0, "curve": []}

    candidate_thresholds = [round(0.80 + i * 0.01, 2) for i in range(21)]  # 0.80 .. 1.00
    curve = []
    for t in candidate_thresholds:
        hits = sum(1 for s in samples if s >= t)
        curve.append({"threshold": t, "hit_rate": round(hits / len(samples), 4)})

    return {"sample_count": len(samples), "curve": curve}


# NOTE: /metrics is already registered by Instrumentator().expose(app) above.
# Custom counters/histograms defined in src/metrics.py live on the same
# default prometheus_client registry, so they show up there automatically --
# no separate route needed.
