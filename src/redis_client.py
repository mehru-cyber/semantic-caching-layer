import redis
from redis.commands.search.field import VectorField, TagField, TextField

# redis-py renamed this module from `indexDefinition` (camelCase) to
# `index_definition` (snake_case) around v6.0. Try the current path first,
# fall back to the old one so this works across installed versions.
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ModuleNotFoundError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from src.config import settings


class RedisClient:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            decode_responses=False,  # required so vector bytes survive intact
        )

    def setup_index(self, dimension: int = 1536):
        try:
            self.r.ft("prompt_idx").info()
            print("Index 'prompt_idx' already exists.")
        except redis.exceptions.ResponseError:
            schema = (
                TagField("partition_key"),  # system_prompt+temp+max_tokens+model+conversation hash
                TextField("prompt"),
                TextField("response"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            self.r.ft("prompt_idx").create_index(
                schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
            )
            print("Created index 'prompt_idx'.")

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


def decode_field(value) -> str:
    """redis-py with decode_responses=False returns bytes for TEXT/TAG
    fields too (not just the vector). Normalize safely to str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


redis_client = RedisClient()
