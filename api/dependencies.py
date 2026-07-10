"""
Shared FastAPI dependencies and application-level singletons.
"""

import threading

from core.retriever import VectorStore

# ── Per-user vector store management ──

_user_stores: dict[str, VectorStore] = {}
_user_stores_lock = threading.Lock()


def get_user_store(uid: str) -> VectorStore:
    """Get or create a user-scoped vector store (thread-safe)."""
    with _user_stores_lock:
        if uid not in _user_stores:
            _user_stores[uid] = VectorStore(collection_name=f"docs_{uid}")
        return _user_stores[uid]
