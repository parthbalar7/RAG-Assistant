"""
core/query_cache.py — Semantic Query Cache

LRU in-memory cache keyed by query embedding. On every query:
  1. Embed the query (fast — 384-dim vector)
  2. Matrix-multiply against all cached query vectors
  3. If max cosine similarity >= threshold → return cached answer instantly

Zero LLM calls, zero retrieval, ~50 ms for a cache hit vs 3-5 s for a full pipeline run.

Cache is per-user, capped at MAX_ENTRIES (LRU eviction).
Invalidated automatically when the user's index changes (ingest / delete / clear).
Entries older than `settings.query_cache_ttl_hours` are skipped and evicted lazily.

Callers should gate BOTH lookup and store on `is_cache_eligible(query)` — short
queries and anaphoric follow-ups ("what does it return?") depend on conversation
context and must never be served from (or minted into) a cross-conversation cache.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.93  # cosine sim above this → cache hit
MAX_ENTRIES = 200  # LRU cap per user

_WORD_RE = re.compile(r"[a-z0-9']+")

# Function words that carry no retrievable content on their own
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "by",
        "with",
        "and",
        "or",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "whose",
        "whom",
        "do",
        "does",
        "did",
        "done",
        "can",
        "could",
        "would",
        "should",
        "shall",
        "will",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "we",
        "me",
        "my",
        "your",
        "his",
        "her",
        "our",
        "us",
        "them",
        "from",
        "as",
        "if",
        "then",
        "than",
        "so",
        "about",
        "into",
        "over",
        "under",
        "between",
        "there",
        "here",
        "please",
        "tell",
        "show",
        "give",
        "explain",
    ]
)

# Pronouns whose referent lives in the conversation, not the query text —
# a cached answer minted in another conversation would resolve them wrongly.
_ANAPHORIC = frozenset({"it", "that", "this", "they", "those", "these"})

# Tokens that flip or narrow a query's meaning while barely moving its MiniLM
# embedding — opposite-qualifier paraphrases score above the 0.93 threshold.
_QUALIFIER_TOKENS = frozenset({"not", "without", "except", "latest", "never", "no"})


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def is_cache_eligible(query: str) -> bool:
    """
    Whether a query is safe to serve from / store into the semantic cache.
    False for queries with fewer than 3 content words or containing anaphoric
    pronouns — both depend on conversation context the cache doesn't have.
    """
    words = _tokens(query)
    if any(w in _ANAPHORIC for w in words):
        return False
    content = [w for w in words if w not in _STOPWORDS]
    return len(content) >= 3


def _qualifier_signature(text: str) -> frozenset[str]:
    sig = {w for w in _tokens(text) if w in _QUALIFIER_TOKENS}
    # Contractions ("don't", "isn't") negate too
    if any(w.endswith("n't") for w in _tokens(text)):
        sig.add("not")
    return frozenset(sig)


class QueryCache:
    def __init__(self, max_entries: int = MAX_ENTRIES, threshold: float = SIMILARITY_THRESHOLD):
        self.max_entries = max_entries
        self.threshold = threshold
        # OrderedDict preserves insertion order for LRU eviction (last = most recent)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        # Pre-built matrix for fast batch similarity — rebuilt on every store/evict
        self._vecs: np.ndarray | None = None  # shape (N, 384)
        self._keys: list[str] = []  # ordered parallel to _vecs rows

    # ── internal ──────────────────────────────────────────────────────────────

    def _rebuild_matrix(self):
        if not self._cache:
            self._vecs = None
            self._keys = []
            return
        self._keys = list(self._cache.keys())
        self._vecs = np.array([self._cache[k]["vec"] for k in self._keys], dtype=np.float32)

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(self, query_vec: np.ndarray, query_text: str = "") -> dict | None:
        """
        Check if a semantically similar query is cached.
        Returns the cache entry dict (answer, sources, hits, matched_query, sim)
        or None on a miss. Expired entries are evicted lazily; when *query_text*
        is provided, candidates whose stored query differs on negation/qualifier
        tokens (not/without/except/latest/never/no) are rejected despite passing
        the cosine threshold.
        """
        if self._vecs is None or len(self._keys) == 0:
            return None

        qv = self._normalise(np.array(query_vec, dtype=np.float32))

        # _vecs rows are pre-normalised at store time → plain dot product = cosine sim
        sims = self._vecs @ qv  # (N,)

        ttl_seconds = settings.query_cache_ttl_hours * 3600.0  # 0 disables TTL
        query_sig = _qualifier_signature(query_text) if query_text else None
        now = time.time()
        evicted = False
        result = None

        # Scan candidates best-first: an expired or qualifier-mismatched best hit
        # shouldn't mask a fresh, compatible second-best one.
        for idx in np.argsort(sims)[::-1]:
            best_sim = float(sims[idx])
            if best_sim < self.threshold:
                break

            key = self._keys[idx]
            entry = self._cache.get(key)
            if entry is None:
                continue

            if ttl_seconds > 0 and now - entry.get("ts", 0.0) > ttl_seconds:
                del self._cache[key]
                evicted = True
                continue

            if query_sig is not None and _qualifier_signature(entry.get("query_text", "")) != query_sig:
                logger.info(
                    "Query cache qualifier mismatch: rejecting '{}' for '{}'".format(
                        entry.get("query_text", "")[:60], query_text[:60]
                    )
                )
                continue

            # Touch for LRU
            self._cache.move_to_end(key)
            logger.info("Query cache HIT: sim={:.3f} for '{}'".format(best_sim, entry.get("query_text", "")[:60]))
            result = {**entry, "sim": round(best_sim, 4)}
            break

        if evicted:
            self._rebuild_matrix()
        return result

    def store(
        self,
        query_vec: np.ndarray,
        query_text: str,
        answer: str,
        sources: list,
        hits: list,
    ):
        """Store a successfully answered query in the cache."""
        qv = self._normalise(np.array(query_vec, dtype=np.float32))
        # Key: first 16 hex chars of the vec's SHA-256 (stable, collision-free for our scale)
        key = hashlib.sha256(qv.tobytes()).hexdigest()[:16]

        self._cache[key] = {
            "vec": qv,
            "query_text": query_text,
            "answer": answer,
            "sources": sources,
            "hits": hits,
            "ts": time.time(),
        }
        self._cache.move_to_end(key)

        # Evict oldest entries when over limit
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        self._rebuild_matrix()
        logger.info(f"Query cache stored: {len(self._cache)} entries total")

    def clear(self):
        """Invalidate the entire cache (call on ingest / delete / index clear)."""
        self._cache.clear()
        self._vecs = None
        self._keys = []
        logger.info("Query cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)


# ── per-user singletons ───────────────────────────────────────────────────────

_user_caches: dict[str, QueryCache] = {}


def get_user_cache(uid: str) -> QueryCache:
    if uid not in _user_caches:
        _user_caches[uid] = QueryCache()
    return _user_caches[uid]


def invalidate_user_cache(uid: str):
    """Call whenever the user's document index changes."""
    if uid in _user_caches:
        _user_caches[uid].clear()
