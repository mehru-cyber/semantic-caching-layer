"""
Upstream LLM provider routing.

Both OpenAI and Groq expose an OpenAI-compatible /chat/completions contract,
so the proxy can forward the client's body largely untouched -- the only
thing that changes per-provider is the base URL and the auth header.

Embeddings (for the semantic cache) always go through OpenAI regardless of
which provider answers the chat request -- Groq does not offer an
embeddings endpoint.
"""
from src.config import settings

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
HELICONE_OPENAI_URL = "https://oai.hconeai.com/v1/chat/completions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Known Groq-hosted model families. Not exhaustive -- Groq's catalog changes
# fairly often (they deprecated llama-3.1-8b-instant and
# llama-3.3-70b-versatile on 2026-06-17, for example), so prefix-matching is
# a best-effort fallback. When in doubt, pass `"provider": "groq"` explicitly
# in the request body to force routing -- this is strongly recommended for
# the "openai/gpt-oss-*" family below, since the name alone looks like it
# should route to real OpenAI.
GROQ_MODEL_PREFIXES = (
    "llama3-",
    "llama-3",
    "llama-4-",
    "mixtral-",
    "gemma",
    "gemma2-",
    "whisper-",  # groq also serves whisper, not relevant here but harmless
    "qwen/",
    "openai/gpt-oss",  # Groq-hosted OSS models -- NOT the real OpenAI API despite the name
)


class ProviderError(Exception):
    pass


def resolve_upstream(model: str, explicit_provider: str | None = None) -> tuple[str, str, str]:
    """
    Decide which upstream to call for a given model.

    Returns (url, api_key, provider_name).
    Raises ProviderError if the resolved provider has no API key configured.
    """
    provider = explicit_provider or ("groq" if _looks_like_groq_model(model) else settings.default_provider)

    if provider == "groq":
        if not settings.groq_api_key:
            raise ProviderError("GROQ_API_KEY is not configured on this server.")
        return GROQ_CHAT_URL, settings.groq_api_key, "groq"

    # default: openai
    if not settings.openai_api_key:
        raise ProviderError("OPENAI_API_KEY is not configured on this server.")

    url = OPENAI_CHAT_URL
    if settings.helicone_api_key:
        url = HELICONE_OPENAI_URL
    return url, settings.openai_api_key, "openai"


def _looks_like_groq_model(model: str) -> bool:
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) for p in GROQ_MODEL_PREFIXES)


def build_headers(provider: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openai" and settings.helicone_api_key:
        headers["Helicone-Auth"] = f"Bearer {settings.helicone_api_key}"
        # We run our own semantic cache layer; don't double-cache in Helicone.
        headers["Helicone-Cache-Enabled"] = "false"
    return headers
