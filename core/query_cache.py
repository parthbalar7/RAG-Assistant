"""
core/query_cache.py — Semantic Query Cache

LRU in-memory cache keyed by query embedding. On every query:
  1. Embed the query (fast — 384-dim vector)
  2. Matrix-multiply against all cached query vectors
  3. If max cosine similarity >= threshold → return cached answer instantly

Zero LLM calls, zero retrieval, ~50 ms for a cache hit vs 3-5 s for a full pipeline run.

Cache is per-user, capped at MAX_ENTRIES (LRU eviction).
Invalidated automatically when the user's index changes (ingest / delete / clear).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.93   # cosine sim above this → cache hit
MAX_ENTRIES = 200              # LRU cap per user


class QueryCache:
    def __init__(self, max_entries: int = MAX_ENTRIES, threshold: float = SIMILARITY_THRESHOLD):
        self.max_entries = max_entries
        self.threshold = threshold
        # OrderedDict preserves insertion order for LRU eviction (last = most recent)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        # Pre-built matrix for fast batch similarity — rebuilt on every store/evict
        self._vecs: Optional[np.ndarray] = None   # shape (N, 384)
        self._keys: list[str] = []                # ordered parallel to _vecs rows

    # ── internal ──────────────────────────────────────────────────────────────

    def _rebuild_matrix(self):
        if not self._cache:
            self._vecs = None
            self._keys = []
            return
        self._keys = list(self._cache.keys())
        self._vecs = np.array(
            [self._cache[k]["vec"] for k in self._keys], dtype=np.float32
        )

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(self, query_vec: np.ndarray) -> Optional[dict]:
        """
        Check if a semantically similar query is cached.
        Returns the cache entry dict (answer, sources, hits, matched_query, sim)
        or None on a miss.
        """
        if self._vecs is None or len(self._keys) == 0:
            return None

        qv = self._normalise(np.array(query_vec, dtype=np.float32))

        # _vecs rows are pre-normalised at store time → plain dot product = cosine sim
        sims = self._vecs @ qv                          # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.threshold:
            return None

        best_key = self._keys[best_idx]
        # Touch for LRU
        self._cache.move_to_end(best_key)
        entry = self._cache[best_key]
        logger.info("Query cache HIT: sim={:.3f} for '{}'".format(
            best_sim, entry.get("query_text", "")[:60]))
        return {**entry, "sim": round(best_sim, 4)}

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
        logger.info("Query cache stored: {} entries total".format(len(self._cache)))

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
