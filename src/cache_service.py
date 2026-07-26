import hashlib
import json
import time
import uuid
import numpy as np
from redis.commands.search.query import Query

from src.redis_client import redis_client, decode_field
from src.embedding_service import embedding_service, EmbeddingError
from src.config import settings
from src import metrics

SIMILARITY_LOG_KEY = "cache:similarity_log"


class CacheService:
    def _generate_partition_key(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int | None,
        model: str,
        conversation_prefix: list,
    ) -> str:
        """
        Deterministic hash of everything that must match EXACTLY for two
        requests to be allowed to share a cache entry:
          - system prompt
          - temperature / max_tokens / model
          - the full prior conversation turns (everything except the final
            user message, which is what gets semantically matched)

        Including the conversation prefix is what prevents two unrelated
        conversations that happen to end in a similar last message (e.g.
        "yes, continue" or "what about the second one?") from colliding.
        """
        max_t = max_tokens if max_tokens is not None else 0
        temp = temperature if temperature is not None else 1.0
        prefix_json = json.dumps(conversation_prefix, sort_keys=True, ensure_ascii=False)

        key_content = f"{system_prompt}|{temp}|{max_t}|{model}|{prefix_json}"
        return hashlib.sha256(key_content.encode("utf-8")).hexdigest()

    def check_cache(
        self,
        query_text: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int | None,
        model: str,
        conversation_prefix: list,
    ) -> dict:
        partition_key = self._generate_partition_key(
            system_prompt, temperature, max_tokens, model, conversation_prefix
        )

        try:
            embedding = embedding_service.get_embedding(query_text)
        except EmbeddingError as e:
            # Can't compute a vector -> we can't use the cache for this
            # request. Fail open (treat as a miss) rather than caching a
            # zero-vector that would create spurious future matches.
            print(f"[cache_service] embedding unavailable, bypassing cache: {e}")
            return {
                "hit": False,
                "embedding_bytes": None,
                "partition_key": partition_key,
                "cache_unavailable": True,
            }

        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        query_str = f"(@partition_key:{{{partition_key}}})=>[KNN 1 @embedding $vec AS score]"
        q = (
            Query(query_str)
            .sort_by("score")
            .return_fields("score", "response", "prompt")
            .dialect(2)
        )

        try:
            res = redis_client.r.ft("prompt_idx").search(q, query_params={"vec": embedding_bytes})
            if res.docs:
                doc = res.docs[0]
                distance = float(doc.score)
                similarity = 1.0 - distance
                self._log_similarity(similarity)
                metrics.cache_similarity_score.observe(similarity)

                if similarity >= settings.similarity_threshold:
                    return {
                        "hit": True,
                        "similarity": similarity,
                        "response": decode_field(doc.response),
                        "cached_prompt": decode_field(doc.prompt),
                        "embedding_bytes": embedding_bytes,
                        "partition_key": partition_key,
                    }
        except Exception as e:
            print(f"Error querying Redis cache: {e}")

        return {
            "hit": False,
            "embedding_bytes": embedding_bytes,
            "partition_key": partition_key,
        }

    def insert_cache(
        self,
        user_prompt: str,
        response: str,
        embedding_bytes: bytes | None,
        partition_key: str,
        ttl: int = 86400,
    ):
        if ttl <= 0:
            return  # classified as "do not cache"
        if not embedding_bytes:
            return  # embedding wasn't available at check-time

        # Best-effort dedup lock: if another concurrent request for the same
        # partition+prompt is already inserting, skip. Not a strict
        # correctness guarantee (two different-but-similar prompts can still
        # race), just cuts down on obvious duplicate inserts under load.
        lock_key = f"cache:lock:{partition_key}:{hashlib.sha256(user_prompt.encode('utf-8')).hexdigest()}"
        try:
            acquired = redis_client.r.set(lock_key, b"1", nx=True, ex=settings.insert_lock_ttl_seconds)
            if not acquired:
                return
        except Exception as e:
            print(f"Error acquiring insert lock (continuing anyway): {e}")

        doc_id = f"cache:{uuid.uuid4()}"
        mapping = {
            "partition_key": partition_key,
            "prompt": user_prompt,
            "response": response,
            "embedding": embedding_bytes,
        }

        try:
            pipe = redis_client.r.pipeline()
            pipe.hset(doc_id, mapping=mapping)
            pipe.expire(doc_id, ttl)
            pipe.execute()
        except Exception as e:
            print(f"Error inserting into Redis cache: {e}")

    def _log_similarity(self, similarity: float):
        """Keep a capped rolling log of similarity scores so the threshold
        tuning endpoint can show a hit-rate-vs-threshold curve without
        needing a separate analytics store."""
        try:
            pipe = redis_client.r.pipeline()
            pipe.lpush(SIMILARITY_LOG_KEY, json.dumps({"s": similarity, "t": time.time()}))
            pipe.ltrim(SIMILARITY_LOG_KEY, 0, settings.similarity_log_size - 1)
            pipe.execute()
        except Exception as e:
            print(f"Error logging similarity sample: {e}")

    def get_similarity_samples(self, limit: int = 5000) -> list[float]:
        try:
            raw = redis_client.r.lrange(SIMILARITY_LOG_KEY, 0, limit - 1)
            samples = []
            for item in raw:
                try:
                    samples.append(json.loads(decode_field(item))["s"])
                except Exception:
                    continue
            return samples
        except Exception as e:
            print(f"Error reading similarity log: {e}")
            return []


cache_service = CacheService()
