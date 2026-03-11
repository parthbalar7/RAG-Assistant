"""
core/embed_cache.py — Disk-backed Embedding Cache

Maps SHA-256(chunk_text) → float32[384] vector on disk as a compressed numpy archive.

On ingest, embed_texts() checks this cache first. Only texts whose hash is absent
get sent to the SentenceTransformer model. New embeddings are merged in and saved.

For a 200-file project (~5000 chunks):
  - First ingest: full encoding run, cache built (writes ~7 MB .npz)
  - Re-ingest unchanged project: 0 model calls, ~1 s total
  - Re-ingest after editing 2 files: only the changed chunks are re-embedded
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/embed_cache.npz")


class EmbedCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._lock = threading.Lock()
        # sha256_hex -> np.ndarray float32 shape (dim,)
        self._store: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not self.path.exists():
            logger.info("Embed cache: no existing cache at {}".format(self.path))
            return
        try:
            data = np.load(str(self.path), allow_pickle=False)
            self._store = {k: data[k] for k in data.files}
            logger.info("Embed cache loaded: {} entries from {}".format(
                len(self._store), self.path))
        except Exception as e:
            logger.warning("Embed cache load failed (starting fresh): {}".format(e))
            self._store = {}

    def save(self):
        """Flush new entries to disk. No-op if nothing changed."""
        if not self._dirty:
            return
        with self._lock:
            if not self._dirty:   # double-check after acquiring lock
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(self.path), **self._store)
                self._dirty = False
                logger.info("Embed cache saved: {} entries ({})".format(
                    len(self._store), self.path))
            except Exception as e:
                logger.warning("Embed cache save failed: {}".format(e))

    # ── cache operations ──────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return cached embedding for *text*, or None on a miss."""
        return self._store.get(self._key(text))

    def put(self, text: str, vec: np.ndarray):
        """Store an embedding. Thread-safe."""
        key = self._key(text)
        with self._lock:
            self._store[key] = np.array(vec, dtype=np.float32)
            self._dirty = True

    def put_batch(self, texts: list[str], vecs: np.ndarray):
        """Store multiple embeddings at once (faster than repeated put())."""
        with self._lock:
            for text, vec in zip(texts, vecs):
                self._store[self._key(text)] = np.array(vec, dtype=np.float32)
            self._dirty = True

    @property
    def size(self) -> int:
        return len(self._store)


# ── singleton ─────────────────────────────────────────────────────────────────

_embed_cache: Optional[EmbedCache] = None
_embed_cache_lock = threading.Lock()


def get_embed_cache() -> EmbedCache:
    global _embed_cache
    if _embed_cache is None:
        with _embed_cache_lock:
            if _embed_cache is None:
                _embed_cache = EmbedCache()
    return _embed_cache
