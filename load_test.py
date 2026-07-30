"""
Load test for the semantic caching proxy.

Sends a mix of unique and repeated/semantically-similar prompts and reports
hit rate convergence, latency (cache hit vs miss), and an estimated cost
savings figure -- the numbers used for the portfolio headline.

Usage:
    python load_test.py                # default: 200 requests
    python load_test.py --requests 2000
    python load_test.py --requests 2000 --concurrency 20
"""
import argparse
import asyncio
import os
import random
import statistics
import time
import httpx

PROXY_URL = "http://localhost:8000/v1/chat/completions"
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")  # only needed if the server has PROXY_API_KEY set

# Base prompt "clusters" -- each cluster is semantically similar phrasings
# of the same underlying question, so repeated sampling should converge
# toward a high hit rate after the first request in each cluster.
CLUSTERS = [
    [
        "What is the capital of France?",
        "Can you tell me France's capital?",
        "Which city is the capital of France?",
        "France's capital city is what?",
    ],
    [
        "Explain quantum computing in simple terms.",
        "What is quantum computing simply explained?",
        "Could you clarify quantum computing for a beginner?",
        "Give me a simple explanation of quantum computing.",
    ],
    [
        "How do I reverse a linked list in Python?",
        "Reverse a linked list, Python example please.",
        "Show me Python code to reverse a linked list.",
    ],
    [
        "What's a good recipe for chocolate chip cookies?",
        "How do I make chocolate chip cookies?",
        "Give me a chocolate chip cookie recipe.",
    ],
]

EST_COST_PER_REQUEST_USD = 0.002  # keep in sync with server EST_COST_PER_REQUEST_USD


def sample_prompt() -> str:
    cluster = random.choice(CLUSTERS)
    return random.choice(cluster)


async def send_request(client: httpx.AsyncClient, prompt: str, model: str):
    start = time.time()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
    }
    headers = {"Authorization": f"Bearer {PROXY_API_KEY}"} if PROXY_API_KEY else {}
    try:
        response = await client.post(PROXY_URL, json=payload, headers=headers, timeout=30.0)
        latency = time.time() - start
        cache_status = response.headers.get("X-Cache-Status", "UNKNOWN")
        return cache_status, latency
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


async def worker(client, queue, results, model):
    while True:
        try:
            prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        status, latency = await send_request(client, prompt, model)
        results.append((status, latency))


async def run(total_requests: int, concurrency: int, model: str):
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(total_requests):
        queue.put_nowait(sample_prompt())

    results: list[tuple[str, float]] = []
    async with httpx.AsyncClient() as client:
        workers = [asyncio.create_task(worker(client, queue, results, model)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    summarize(results, total_requests)


def summarize(results: list[tuple[str, float]], total_requests: int):
    hits = [lat for status, lat in results if status == "HIT"]
    misses = [lat for status, lat in results if status == "MISS"]
    errors = [status for status, _ in results if status.startswith("ERROR")]
    other = [status for status, _ in results if status not in ("HIT", "MISS") and not status.startswith("ERROR")]

    print("\n--- Load Test Summary ---")
    print(f"Total requests sent: {total_requests}")
    print(f"Cache hits:   {len(hits)}")
    print(f"Cache misses: {len(misses)}")
    print(f"Errors:       {len(errors)}")
    if other:
        print(f"Other/unrecognized status: {len(other)}")

    if errors:
        sample = {}
        for e in errors:
            sample[e] = sample.get(e, 0) + 1
        print("\nSample error messages (message: count):")
        for msg, count in sorted(sample.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  [{count}x] {msg}")

    if other:
        sample = {}
        for o in other:
            sample[o] = sample.get(o, 0) + 1
        print("\nSample 'other' statuses (status: count):")
        for status, count in sorted(sample.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  [{count}x] {status}")

    if hits or misses:
        hit_rate = len(hits) / (len(hits) + len(misses)) * 100
        print(f"Hit rate: {hit_rate:.1f}%")

    def pct(data, p):
        if not data:
            return float("nan")
        data_sorted = sorted(data)
        idx = min(len(data_sorted) - 1, int(len(data_sorted) * p))
        return data_sorted[idx]

    if hits:
        print(f"\nCache HIT latency  -> avg: {statistics.mean(hits)*1000:.1f}ms  "
              f"p50: {pct(hits,0.5)*1000:.1f}ms  p95: {pct(hits,0.95)*1000:.1f}ms")
    if misses:
        print(f"Cache MISS latency -> avg: {statistics.mean(misses)*1000:.1f}ms  "
              f"p50: {pct(misses,0.5)*1000:.1f}ms  p95: {pct(misses,0.95)*1000:.1f}ms")

    if hits and misses:
        speedup = statistics.mean(misses) / statistics.mean(hits)
        print(f"\nCache hits were ~{speedup:.1f}x faster than misses on this run.")

    est_savings = len(hits) * EST_COST_PER_REQUEST_USD
    print(f"Estimated cost saved: ${est_savings:.4f} "
          f"(at ${EST_COST_PER_REQUEST_USD:.4f}/request -- adjust to your real per-request cost)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Model name to send")
    args = parser.parse_args()

    print(f"Starting load test: {args.requests} requests, concurrency={args.concurrency}, model={args.model}")
    asyncio.run(run(args.requests, args.concurrency, args.model))
