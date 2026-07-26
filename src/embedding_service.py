from src.config import settings


class EmbeddingError(Exception):
    """Raised when we can't get a real embedding. Callers should treat this
    as 'cache unavailable for this request' rather than silently caching a
    zero vector (which would create spurious matches against other failed
    embeddings)."""


class _OpenAIBackend:
    dimension = 1536
    model_name = "text-embedding-3-small"

    def __init__(self):
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY is required for the 'openai' embedding backend."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        try:
            response = self._client_or_raise().embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding request failed: {e}") from e


class _LocalBackend:
    """
    Free, offline embeddings via sentence-transformers. No API key, no
    per-request cost, no network call -- runs entirely on CPU in the
    container.

    Tradeoffs vs. the OpenAI backend (text-embedding-3-small, 1536-dim):
      - Lower dimensional (384) and generally weaker semantic separation,
        so expect somewhat more false MISSes on paraphrased queries.
      - The model weights are baked into the Docker image at build time
        (see Dockerfile) so there's no first-request download delay.

    IMPORTANT: dimension differs from the OpenAI backend (384 vs 1536).
    Redis Stack's vector index has a fixed dimension set at creation time,
    so switching EMBEDDING_BACKEND requires clearing existing cache data:
        docker-compose down -v && docker-compose up --build
    """
    dimension = 384
    model_name = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None

    def _model_or_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        try:
            vector = self._model_or_load().encode(text, normalize_embeddings=True)
            return vector.tolist()
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}") from e


class EmbeddingService:
    """
    Wraps whichever backend is selected via EMBEDDING_BACKEND ('openai' or
    'local'). The rest of the app only ever calls get_embedding() /
    .dimension -- it doesn't need to know which backend is active.
    """

    def __init__(self):
        backend_name = (settings.embedding_backend or "local").lower()
        if backend_name == "openai":
            self._backend = _OpenAIBackend()
        elif backend_name == "local":
            self._backend = _LocalBackend()
        else:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND '{backend_name}' -- expected 'openai' or 'local'."
            )
        self.backend_name = backend_name

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def get_embedding(self, text: str) -> list[float]:
        return self._backend.embed(text)


embedding_service = EmbeddingService()
