"""
core/embed_cache.py — Disk-backed Embedding Cache

Maps SHA-256(chunk_text) → float32 vector on disk as a compressed numpy archive.
Cache files are per-model (data/embed_cache_{sha256(model)[:8]}.npz) so switching
RAG_EMBEDDING_MODEL never mixes vectors of different dimensions.

On ingest, embed_texts() checks this cache first. Only texts whose hash is absent
get sent to the SentenceTransformer model. New embeddings are merged in and saved.

For a 200-file project (~5000 chunks):
  - First ingest: full encoding run, cache built (writes ~7 MB .npz)
  - Re-ingest unchanged project: 0 model calls, ~1 s total
  - Re-ingest after editing 2 files: only the changed chunks are re-embedded
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Model whose vectors historically lived in the unsuffixed cache file
LEGACY_DEFAULT_MODEL = "all-MiniLM-L6-v2"
LEGACY_CACHE_PATH = Path("data/embed_cache.npz")

# Full-archive rewrites are expensive on the retrieval hot path: save() only
# writes once this many new entries accumulate; flush() (also atexit) forces it.
FLUSH_THRESHOLD = 32


def _cache_path_for_model(model_name: str) -> Path:
    suffix = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return Path(f"data/embed_cache_{suffix}.npz")


class EmbedCache:
    def __init__(self, path: Path | None = None, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.path = path or _cache_path_for_model(self.model_name)
        self._lock = threading.Lock()
        # sha256_hex -> np.ndarray float32 shape (dim,)
        self._store: dict[str, np.ndarray] = {}
        self._pending = 0  # entries added since the last disk write
        self._load()
        atexit.register(self.flush)  # persist any sub-threshold tail on clean shutdown

    # ── persistence ───────────────────────────────────────────────────────────

    def _sweep_orphan_tmps(self):
        """Best-effort removal of temp files left behind by a mid-save kill:
        the current '*.npz.tmp' pattern and the old mkstemp 'tmp*.tmp.npz' one."""
        cache_dir = self.path.parent
        if not cache_dir.exists():
            return
        for pattern in ("*.npz.tmp", "tmp*.tmp.npz"):
            for orphan in cache_dir.glob(pattern):
                with contextlib.suppress(OSError):
                    orphan.unlink()
                    logger.info(f"Removed orphaned embed-cache temp file: {orphan}")

    def _load(self):
        self._sweep_orphan_tmps()
        load_path = self.path
        if not load_path.exists() and self.model_name == LEGACY_DEFAULT_MODEL and LEGACY_CACHE_PATH.exists():
            # Pre-versioning cache written by the historical default model; saves
            # go to the per-model path, leaving the legacy file untouched.
            load_path = LEGACY_CACHE_PATH
        if not load_path.exists():
            logger.info(f"Embed cache: no existing cache at {load_path}")
            return
        try:
            data = np.load(str(load_path), allow_pickle=False)
            self._store = {k: data[k] for k in data.files}
            logger.info(f"Embed cache loaded: {len(self._store)} entries from {load_path}")
        except Exception as e:
            logger.warning(f"Embed cache load failed (starting fresh): {e}")
            self._store = {}

    def save(self):
        """Write to disk once enough new entries accumulated (>= FLUSH_THRESHOLD).
        Cheap no-op below the threshold — retrieval-path callers can invoke this
        per query without triggering a full archive rewrite each time. Pending
        entries are never lost: flush() forces the write and runs via atexit."""
        self._write_if_pending(FLUSH_THRESHOLD)

    def flush(self):
        """Write any pending entries to disk immediately. No-op if nothing changed."""
        self._write_if_pending(1)

    def _write_if_pending(self, min_pending: int):
        if self._pending < min_pending:
            return
        with self._lock:
            if self._pending < min_pending:  # re-check after acquiring lock
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.path.with_name(self.path.name + ".tmp")
                try:
                    # Write through an open handle: given a *path* without ".npz",
                    # np.savez_compressed appends the extension and writes elsewhere
                    with open(tmp_path, "wb") as fh:
                        np.savez_compressed(fh, **self._store)
                    os.replace(tmp_path, self.path)
                except Exception:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)
                    raise
                self._pending = 0
                logger.info(f"Embed cache saved: {len(self._store)} entries ({self.path})")
            except Exception as e:
                logger.warning(f"Embed cache save failed: {e}")

    # ── cache operations ──────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Return cached embedding for *text*, or None on a miss."""
        return self._store.get(self._key(text))

    def put(self, text: str, vec: np.ndarray):
        """Store an embedding. Thread-safe."""
        key = self._key(text)
        with self._lock:
            self._store[key] = np.array(vec, dtype=np.float32)
            self._pending += 1

    def put_batch(self, texts: list[str], vecs: np.ndarray):
        """Store multiple embeddings at once (faster than repeated put())."""
        with self._lock:
            for text, vec in zip(texts, vecs):
                self._store[self._key(text)] = np.array(vec, dtype=np.float32)
                self._pending += 1

    @property
    def size(self) -> int:
        return len(self._store)


# ── singleton ─────────────────────────────────────────────────────────────────

_embed_cache: EmbedCache | None = None
_embed_cache_lock = threading.Lock()


def get_embed_cache() -> EmbedCache:
    global _embed_cache
    if _embed_cache is None:
        with _embed_cache_lock:
            if _embed_cache is None:
                _embed_cache = EmbedCache()
    return _embed_cache
