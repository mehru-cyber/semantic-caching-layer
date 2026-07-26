import re
from src.config import settings

# Queries about things that are true "right now, this instant" and change
# fast enough that even a 1-hour TTL is stale -> don't cache at all.
DISABLED_PATTERNS = [
    r"\bright now\b",
    r"\blive score\b",
    r"\bbreaking news\b",
    r"\bhappening now\b",
    r"\bcurrent(ly)? trading at\b",
]

# Time-sensitive but not instant-by-instant -> short TTL.
SHORT_TTL_KEYWORDS = [
    "today", "tonight", "now", "current", "currently",
    "weather", "forecast",
    "stock", "stock price", "share price", "market cap",
    "news", "headline",
    "score", "schedule",
    "latest", "this week", "this morning",
]

_word_re_cache: dict[str, re.Pattern] = {}


def _word_pattern(word: str) -> re.Pattern:
    if word not in _word_re_cache:
        _word_re_cache[word] = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return _word_re_cache[word]


class TTLClassifier:
    def classify(self, text: str) -> int:
        """Returns a TTL in seconds. 0 means 'do not cache this at all'."""
        if not text:
            return settings.ttl_long_seconds

        for pattern in DISABLED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 0

        for keyword in SHORT_TTL_KEYWORDS:
            if _word_pattern(keyword).search(text):
                return settings.ttl_short_seconds

        return settings.ttl_long_seconds


ttl_classifier = TTLClassifier()
