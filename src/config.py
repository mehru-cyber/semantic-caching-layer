import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Redis ---
    redis_host: str = os.getenv("REDIS_HOST", "localhost").strip()
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_password: str = os.getenv("REDIS_PASSWORD", "").strip()

    # --- Providers ---
    # OpenAI is required regardless of which chat provider is used, because
    # embeddings are always generated with OpenAI's text-embedding-3-small.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()

    # Which backend generates the vectors used for semantic cache matching.
    # "local" (default): free, offline, sentence-transformers, 384-dim.
    # "openai": text-embedding-3-small, 1536-dim, costs a small amount per
    #           request and requires OPENAI_API_KEY.
    # NOTE: these two produce different vector dimensions -- switching
    # requires clearing existing Redis cache data (docker-compose down -v).
    # .strip() guards against stray whitespace/tabs from copy-pasting env
    # vars into a dashboard (e.g. Railway) -- a real failure mode seen in
    # practice, not just theoretical.
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local").strip()

    # Default provider when a model name doesn't match a known Groq model
    # and the request doesn't specify one explicitly. "openai" or "groq".
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "openai").strip()

    # --- Observability ---
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    # Langfuse's docs currently call this LANGFUSE_BASE_URL; older SDK/docs
    # versions called it LANGFUSE_HOST. We read both and pass it to the
    # client explicitly rather than relying on the installed SDK version to
    # pick the right one up implicitly. Leave unset for Langfuse Cloud (EU).
    langfuse_host: str = os.getenv("LANGFUSE_BASE_URL", "") or os.getenv("LANGFUSE_HOST", "")
    helicone_api_key: str = os.getenv("HELICONE_API_KEY", "")

    # --- Cache tuning ---
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.95))
    # How many recent (similarity, hit) samples to retain for the
    # threshold-tuning analysis endpoint.
    similarity_log_size: int = int(os.getenv("SIMILARITY_LOG_SIZE", 5000))
    # Best-effort lock TTL (seconds) used to reduce duplicate cache writes
    # from concurrent identical cache-miss requests.
    insert_lock_ttl_seconds: int = int(os.getenv("INSERT_LOCK_TTL_SECONDS", 5))

    # --- TTL tiers (seconds) ---
    ttl_short_seconds: int = int(os.getenv("TTL_SHORT_SECONDS", 3600))       # time-sensitive
    ttl_long_seconds: int = int(os.getenv("TTL_LONG_SECONDS", 86400))        # stable/factual
    # 0 disables caching entirely for queries classified as "disabled"
    # (e.g. live scores, breaking news right now).

    # --- Cost estimation (for the Grafana "cost savings" headline metric) ---
    # Rough blended per-request cost used only to estimate savings; override
    # per deployment. Not meant to be exact billing.
    est_cost_per_request_usd: float = float(os.getenv("EST_COST_PER_REQUEST_USD", 0.002))

    # --- Misc ---
    model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    upstream_timeout_seconds: float = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", 60.0))

    # Optional gate on /v1/chat/completions and /api/tuning*. If unset
    # (default), the proxy is fully open -- fine for local dev, NOT fine for
    # a public URL with real provider keys behind it. When set, callers must
    # send `Authorization: Bearer <this value>` (same header shape as the
    # OpenAI SDK already uses, so clients need no special handling).
    proxy_api_key: str = os.getenv("PROXY_API_KEY", "").strip()


settings = Settings()

