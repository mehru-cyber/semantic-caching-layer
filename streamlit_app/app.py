"""
Interactive chat UI for the Semantic Caching Layer.

Talks to the FastAPI proxy exactly like any other client would -- it's not
special-cased, just a normal caller of /v1/chat/completions -- and surfaces
the X-Cache-Status / X-Cache-Similarity / X-Provider headers so you can
*see* the cache working in real time instead of reading curl output.
"""
import os
import time

import requests
import streamlit as st

DEFAULT_PROXY_URL = os.getenv("PROXY_BASE_URL", "http://api:8000")

MODEL_OPTIONS = {
    "Groq: openai/gpt-oss-20b (fast, small)": "openai/gpt-oss-20b",
    "Groq: openai/gpt-oss-120b (fast, larger)": "openai/gpt-oss-120b",
    "Groq: llama-4-scout": "llama-4-scout-17b-16e-instruct",
    "OpenAI: gpt-4o-mini": "gpt-4o-mini",
}

st.set_page_config(page_title="Semantic Cache Demo", page_icon="\U0001F9E0", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": ..., "content": ..., "meta": {...}}]
if "stats" not in st.session_state:
    st.session_state.stats = {"requests": 0, "hits": 0, "misses": 0, "errors": 0}

# ---------------------------------------------------------------------------
# Sidebar: connection + model + live stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F9E0 Semantic Cache")
    st.caption("Chat UI for the semantic caching proxy")

    proxy_url = st.text_input("Proxy base URL", value=DEFAULT_PROXY_URL)
    proxy_api_key = st.text_input(
        "Proxy API key (only if PROXY_API_KEY is set on the server)",
        value=os.getenv("PROXY_API_KEY", ""),
        type="password",
    )
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model = MODEL_OPTIONS[model_label]
    system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=80)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.divider()
    st.subheader("Session stats")

    stats = st.session_state.stats
    total = stats["requests"]
    hit_rate = (stats["hits"] / total * 100) if total else 0.0
    est_saved = stats["hits"] * 0.002  # keep in sync with EST_COST_PER_REQUEST_USD

    c1, c2 = st.columns(2)
    c1.metric("Requests", total)
    c2.metric("Hit rate", f"{hit_rate:.0f}%")
    c1.metric("Hits", stats["hits"])
    c2.metric("Misses", stats["misses"])
    st.metric("Est. cost saved", f"${est_saved:.4f}")

    auth_headers = {"Authorization": f"Bearer {proxy_api_key}"} if proxy_api_key else {}

    try:
        r = requests.get(f"{proxy_url}/api/tuning", headers=auth_headers, timeout=3)
        if r.ok:
            st.caption(f"Current similarity threshold: **{r.json().get('current_threshold')}**")
    except requests.RequestException:
        st.caption("\u26A0\uFE0F Can't reach proxy /api/tuning right now.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Chat")
st.caption(
    "Ask something, then ask a reworded version of the same question -- "
    "watch it come back as a cache HIT instead of calling the model again."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            if meta["cache_status"] == "HIT":
                st.caption(
                    f"\u26A1 Cache HIT \u00b7 similarity {meta.get('similarity', '?')} \u00b7 "
                    f"{meta['latency_ms']:.0f}ms"
                )
            elif meta["cache_status"] == "MISS":
                st.caption(
                    f"\U0001F310 Cache MISS \u00b7 provider: {meta.get('provider', '?')} \u00b7 "
                    f"{meta['latency_ms']:.0f}ms"
                )
            elif meta["cache_status"] == "ERROR":
                st.caption(f"\u274C Error: {meta.get('error')}")

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("\u23F3 thinking...")
        start = time.time()
        try:
            resp = requests.post(f"{proxy_url}/v1/chat/completions", json=payload, headers=auth_headers, timeout=60)
            latency_ms = (time.time() - start) * 1000
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            cache_status = resp.headers.get("X-Cache-Status", "UNKNOWN")
            similarity = resp.headers.get("X-Cache-Similarity")
            provider = resp.headers.get("X-Provider")

            placeholder.markdown(content)
            meta = {"cache_status": cache_status, "latency_ms": latency_ms, "similarity": similarity, "provider": provider}

            st.session_state.stats["requests"] += 1
            if cache_status == "HIT":
                st.session_state.stats["hits"] += 1
                st.caption(f"\u26A1 Cache HIT \u00b7 similarity {similarity} \u00b7 {latency_ms:.0f}ms")
            elif cache_status == "MISS":
                st.session_state.stats["misses"] += 1
                st.caption(f"\U0001F310 Cache MISS \u00b7 provider: {provider} \u00b7 {latency_ms:.0f}ms")

            st.session_state.messages.append({"role": "assistant", "content": content, "meta": meta})
        except requests.RequestException as e:
            latency_ms = (time.time() - start) * 1000
            error_text = str(e)
            placeholder.markdown(f"\u274C Request failed: {error_text}")
            st.session_state.stats["requests"] += 1
            st.session_state.stats["errors"] += 1
            st.session_state.messages.append(
                {"role": "assistant", "content": f"\u274C Request failed: {error_text}", "meta": {"cache_status": "ERROR", "latency_ms": latency_ms, "error": error_text}}
            )

    st.rerun()
