"""
Shared helper functions used across multiple routers.
Extracted from server.py to eliminate code duplication.
"""

import logging
import threading
import time
from typing import Optional

from api import database as db
from config import settings
from core.generator import Message
from core.memory import MemoryContext, get_memory_store, get_style_rules, retrieve_memories

logger = logging.getLogger(__name__)

# -- User activity tracking (feeds the idle-maintenance sweep in api/server.py) --
# Timestamps (not booleans) so a write landing mid-sweep stays newer than
# last_maintenance and re-qualifies the user instead of being lost.

_user_activity: dict[str, dict] = {}
_activity_lock = threading.Lock()


def mark_user_activity(uid: str, wrote: bool = False) -> None:
    """Record that a user was active now; wrote=True also records a memory write."""
    now = time.time()
    with _activity_lock:
        entry = _user_activity.setdefault(uid, {"last_seen": 0.0, "last_write": 0.0, "last_maintenance": 0.0})
        entry["last_seen"] = now
        if wrote:
            entry["last_write"] = now


def get_maintenance_candidates(idle_threshold_s: float) -> list[str]:
    """User ids idle for at least idle_threshold_s with memory writes newer than their last sweep."""
    now = time.time()
    with _activity_lock:
        return [
            uid
            for uid, entry in _user_activity.items()
            if now - entry["last_seen"] >= idle_threshold_s and entry["last_write"] > entry["last_maintenance"]
        ]


def mark_maintenance_done(uid: str, started_at: float) -> None:
    """Record a completed maintenance sweep (stamped with its start time, not its end)."""
    with _activity_lock:
        entry = _user_activity.get(uid)
        if entry is not None:
            entry["last_maintenance"] = started_at


def format_sources(hits: list[dict]) -> list[dict]:
    """Convert raw retrieval hits into API-friendly source dicts."""
    return [
        {
            "file": h["metadata"].get("document_path", ""),
            "lines": f"{h['metadata'].get('start_line', '?')}-{h['metadata'].get('end_line', '?')}",
            "language": h["metadata"].get("language", ""),
            "score": round(h.get("rerank_score", h.get("score", 0)), 4),
            "search_type": h.get("search_type", ""),
            "preview": h["content"][:200],
        }
        for h in hits
    ]


def convert_history(raw_list: list[dict] | None) -> list[Message] | None:
    """Convert raw conversation history dicts into Message objects."""
    if not raw_list:
        return None
    return [Message(role=m["role"], content=m["content"]) for m in raw_list]


def safe_retrieve_memories(uid: str, query: str) -> Optional["MemoryContext"]:
    """Retrieve memories with graceful error handling. Returns None on failure."""
    if not settings.memory_enabled:
        return None
    try:
        mark_user_activity(uid)
        mem_store = get_memory_store(uid)
        memory_ctx = retrieve_memories(mem_store, query)
        if memory_ctx.count > 0:
            logger.info("Retrieved %d memories for query", memory_ctx.count)
        try:
            style_rules = get_style_rules(uid)
            if style_rules:
                block = "[Style rules]\n" + style_rules
                memory_ctx.formatted = block + ("\n\n" + memory_ctx.formatted if memory_ctx.formatted else "")
        except Exception as e:
            logger.warning("Style-rules injection failed (non-fatal): %s", e)
        return memory_ctx
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
        return None


def safe_process_memories(uid: str, query: str, answer: str, session_id: str) -> None:
    """Extract and store memories from a Q&A turn. Silently fails."""
    if not (settings.memory_enabled and settings.memory_auto_extract):
        return
    try:
        from core.memory import process_turn_memories

        process_turn_memories(uid, query, answer, session_id)
    except Exception as e:
        logger.warning("Memory extraction failed: %s", e)


def process_memories_background(uid: str, query: str, answer: str, session_id: str) -> None:
    """Run memory extraction after the response path has been released."""
    if not (settings.memory_enabled and settings.memory_auto_extract):
        return
    mark_user_activity(uid, wrote=True)
    threading.Thread(
        target=safe_process_memories,
        args=(uid, query, answer, session_id),
        daemon=True,
    ).start()


def auto_title_session(session_id: str | None, query: str, is_first: bool) -> str | None:
    """Auto-set session title from the first query. Returns title if set, else None."""
    if is_first and session_id:
        title = query[:50] + ("..." if len(query) > 50 else "")
        db.update_session_title(session_id, title)
        return title
    return None


def save_messages(
    session_id: str | None, query: str, answer: str, sources: list | None = None, metadata: dict | None = None
) -> None:
    """Persist user query and assistant response to the session."""
    if session_id:
        db.add_message(session_id, "user", query)
        db.add_message(session_id, "assistant", answer, sources, metadata)
