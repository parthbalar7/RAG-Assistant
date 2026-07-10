"""
Memory-Augmented Conversation System — Token-Optimized.

Token savings vs naive approach:
1. Uses Haiku (not Sonnet) for extraction → 10-15x cheaper
2. Extracts every N turns, not every turn → 3x fewer extraction calls
3. Skips trivial Q&A (short answers, greetings) → ~30% fewer calls
4. Truncates Q&A before sending to extractor → ~40% fewer input tokens
5. Compact memory format (no verbose labels) → smaller prompt injection
6. History summarization → prevents history from growing unbounded
7. Deduplicates memories before storing → fewer redundant embeddings

Architecture:
1. After every Nth conversation turn, extract key facts → store as "memory fragments"
2. Each fragment gets embedded (local model, free) and stored in ChromaDB
3. Before generating a response, retrieve top-K memories via embedding similarity
4. Inject compact memory context into the system prompt

Memory Types:
- fact: Specific fact (e.g., "Project uses FastAPI with PostgreSQL")
- pref: User preference (e.g., "User prefers Python over JavaScript")
- decision: Important decision made
- insight: Pattern or insight discovered
- summary: Condensed summary of a full chat session
"""

import json
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

MEMORY_TYPES = ["fact", "pref", "decision", "insight", "summary"]

# Recency decay constant for composite memory ranking: e-folding time of one week,
# so a memory untouched for 168h contributes ~37% of full recency score.
RECENCY_DECAY_HOURS = 168.0

# Write-reconciliation band on search()'s (1+cos)/2 similarity scale.
# Chosen band: [0.775, 0.92] — the roadmap's raw-cosine 0.55 floor maps to
# (1+0.55)/2 = 0.775 here, and the 0.92 ceiling is the pre-existing exact-dup
# skip threshold (raw cosine 0.84). Below the floor the memories are unrelated
# and the write is a plain ADD.
DUP_SIM_THRESHOLD = 0.92
RECONCILE_SIM_FLOOR = 0.775

# Soft-invalidation exists for short-term reversibility, not forever: fragments whose
# invalid_at is older than this are hard-deleted by run_maintenance so superseded and
# archived rows cannot accumulate and crowd live memories out of search()'s over-fetch.
INVALID_PURGE_DAYS = 30

# Legacy type mapping for backward compat
_TYPE_ALIASES = {
    "conversation_summary": "summary",
    "key_fact": "fact",
    "user_preference": "pref",
}


@dataclass
class MemoryFragment:
    content: str
    memory_type: str
    source_session_id: str = ""
    source_query: str = ""
    importance: float = 0.5
    tags: list = field(default_factory=list)
    created_at: float = 0.0
    fragment_id: str = ""

    def __post_init__(self):
        if not self.fragment_id:
            self.fragment_id = "mem_" + str(uuid.uuid4())[:10]
        if not self.created_at:
            self.created_at = time.time()
        # Normalize legacy types
        self.memory_type = _TYPE_ALIASES.get(self.memory_type, self.memory_type)

    def to_dict(self):
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_session_id": self.source_session_id,
            "source_query": self.source_query,
            "importance": self.importance,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d):
        return MemoryFragment(
            fragment_id=d.get("fragment_id", ""),
            content=d.get("content", ""),
            memory_type=d.get("memory_type", "fact"),
            source_session_id=d.get("source_session_id", ""),
            source_query=d.get("source_query", ""),
            importance=d.get("importance", 0.5),
            tags=d.get("tags", []),
            created_at=d.get("created_at", 0),
        )


@dataclass
class MemoryContext:
    """Retrieved memories formatted for injection into the LLM prompt."""

    fragments: list
    formatted: str
    count: int = 0
    retrieval_ms: float = 0.0


# -- Embedding (reuses local model from retriever — FREE, no API cost) --


def _embed_texts(texts):
    from core.retriever import embed_texts

    return embed_texts(texts)


# -- Per-User Write/Maintenance Locks --

_user_locks = {}
_user_locks_guard = threading.Lock()


def _get_user_lock(user_id):
    """Per-user reentrant lock serializing memory writes (add_fragment, add_fragments,
    consolidate_memories) against run_maintenance. RLock so nested paths — add_fragments
    -> add_fragment, run_maintenance -> consolidate_memories -> add_fragment — reacquire
    safely on the same thread."""
    with _user_locks_guard:
        lock = _user_locks.get(user_id)
        if lock is None:
            lock = _user_locks[user_id] = threading.RLock()
        return lock


# -- Memory Store (ChromaDB) --


class MemoryStore:
    """Persistent memory store using a dedicated ChromaDB collection."""

    def __init__(self, user_id="default", persist_dir=None):
        self.user_id = user_id
        persist_dir = persist_dir or settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection_name = "memories_{}".format(user_id[:20].replace("-", "_"))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._turn_counter = 0  # Tracks turns for interval-based extraction
        logger.info(f"MemoryStore ready for '{user_id}': {self.collection.count()} fragments")

    @property
    def count(self):
        return self.collection.count()

    def increment_turn(self):
        """Increment turn counter. Returns True if extraction should run."""
        self._turn_counter += 1
        return self._turn_counter % settings.memory_extract_interval == 0

    def add_fragment(self, fragment, reconcile=True):
        """Store a single memory fragment, reconciling against the closest existing memory (Mem0-style).

        reconcile=False skips the dedup/reconcile pass (and its LLM round trip) and stores
        the fragment as an authoritative plain ADD — used for manual, user-typed adds.
        """
        with _get_user_lock(self.user_id):
            if reconcile and self.count > 0:
                existing = self.search(fragment.content, top_k=1)
                if existing:
                    top = existing[0]
                    sim = top.get("similarity", 0)
                    if sim > DUP_SIM_THRESHOLD:
                        logger.info(f"Skipping duplicate memory: {fragment.content[:50]}")
                        return None
                    if sim >= RECONCILE_SIM_FLOOR:
                        op, new_content = _reconcile_op(fragment.content, top["content"])
                        if op == "NOOP":
                            logger.info(f"Reconcile NOOP, skipping: {fragment.content[:50]}")
                            return None
                        if op == "DELETE":
                            self.invalidate_fragment(top["fragment_id"])
                            logger.info(f"Reconcile DELETE: invalidated {top['fragment_id']}")
                            return None
                        if op == "UPDATE":
                            if new_content:
                                fragment.content = new_content
                            self.invalidate_fragment(top["fragment_id"], superseded_by=fragment.fragment_id)
                            logger.info(f"Reconcile UPDATE: {top['fragment_id']} superseded by {fragment.fragment_id}")
                        # ADD (or reconciliation fail-open) falls through to a plain store

            embedding = _embed_texts([fragment.content])[0]
            self.collection.upsert(
                ids=[fragment.fragment_id],
                embeddings=[embedding],
                documents=[fragment.content],
                metadatas=[
                    {
                        "memory_type": fragment.memory_type,
                        "source_session_id": fragment.source_session_id,
                        "source_query": fragment.source_query[:150],
                        "importance": fragment.importance,
                        "tags": json.dumps(fragment.tags),
                        "created_at": fragment.created_at,
                        "last_accessed": fragment.created_at,
                        "access_count": 0,
                        "user_id": self.user_id,
                    }
                ],
            )
            logger.info(f"Stored memory: {fragment.fragment_id} [{fragment.memory_type}]")
            return fragment.fragment_id

    def add_fragments(self, fragments):
        """Store multiple fragments with deduplication. Serialized against run_maintenance."""
        if not fragments:
            return []
        stored = []
        with _get_user_lock(self.user_id):
            for f in fragments:
                fid = self.add_fragment(f)
                if fid:
                    stored.append(fid)
        logger.info(f"Stored {len(stored)}/{len(fragments)} memory fragments (deduped)")
        return stored

    def invalidate_fragment(self, fragment_id, superseded_by="", archived=False):
        """Graphiti-style soft invalidation — never hard-deletes, so mistakes stay reversible
        and invalidated fragments remain inspectable in the memory browser via get_all()."""
        meta = {"invalid_at": time.time()}
        if superseded_by:
            meta["superseded_by"] = superseded_by
        if archived:
            meta["archived"] = True
        try:
            self.collection.update(ids=[fragment_id], metadatas=[meta])
            return True
        except Exception as e:
            logger.error(f"Soft-invalidate {fragment_id} failed: {e}")
            return False

    def search(self, query, top_k=5, memory_type=None, min_importance=0.0, created_range=None):
        """Retrieve relevant memory fragments. created_range is an optional (lo, hi) epoch-seconds filter."""
        if self.count == 0:
            return []
        query_embedding = _embed_texts([query])[0]
        conditions = []
        if memory_type:
            conditions.append({"memory_type": memory_type})
        if created_range:
            lo, hi = created_range
            conditions.append({"created_at": {"$gte": lo}})
            conditions.append({"created_at": {"$lte": hi}})
        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif conditions:
            where_filter = {"$and": conditions}
        # Over-fetch: soft-invalidated and procedural fragments are post-filtered below,
        # so top_k results must survive the filter even when many neighbors around a
        # frequently-updated topic are superseded/archived rows.
        fetch_k = min(top_k * 6 + 10, self.count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        fragments = []
        if results["documents"] and results["documents"][0]:
            for fid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if len(fragments) >= top_k:
                    break
                # Superseded/archived and procedural fragments never surface in normal
                # search; fragments lacking the invalid_at key are treated as valid.
                if meta.get("invalid_at") or fid.startswith("proc_"):
                    continue
                similarity = 1 - (dist / 2)
                if similarity < 0.2:
                    continue
                importance = meta.get("importance", 0.5)
                if importance < min_importance:
                    continue
                fragments.append(
                    {
                        "fragment_id": fid,
                        "content": doc,
                        "memory_type": meta.get("memory_type", "fact"),
                        "source_session_id": meta.get("source_session_id", ""),
                        "source_query": meta.get("source_query", ""),
                        "importance": importance,
                        "tags": json.loads(meta.get("tags", "[]")),
                        "created_at": meta.get("created_at", 0),
                        "last_accessed": meta.get("last_accessed", meta.get("created_at", 0)),
                        "access_count": meta.get("access_count", 0),
                        "similarity": round(similarity, 4),
                    }
                )
        return fragments

    def get_all(self, limit=100):
        """Return all stored memories."""
        if self.count == 0:
            return []
        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=limit,
        )
        fragments = []
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            fragments.append(
                {
                    "fragment_id": results["ids"][i],
                    "content": doc,
                    "memory_type": meta.get("memory_type", "fact"),
                    "source_session_id": meta.get("source_session_id", ""),
                    "source_query": meta.get("source_query", ""),
                    "importance": meta.get("importance", 0.5),
                    "tags": meta.get("tags", "[]"),
                    "created_at": meta.get("created_at", 0),
                    "invalid": bool(meta.get("invalid_at")),
                    "invalid_at": meta.get("invalid_at", 0),
                    "superseded_by": meta.get("superseded_by", ""),
                    "archived": bool(meta.get("archived", False)),
                }
            )
        return fragments

    def get_all_with_embeddings(self, limit=500):
        """Return all live stored memories (soft-invalidated and procedural fragments
        excluded — consolidation must never resurrect superseded facts) with embeddings."""
        if self.count == 0:
            return [], []
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
        )
        fragments = []
        embeddings = []
        for i, (doc, meta, emb) in enumerate(zip(results["documents"], results["metadatas"], results["embeddings"])):
            if results["ids"][i].startswith("proc_") or meta.get("invalid_at"):
                continue
            fragments.append(
                {
                    "fragment_id": results["ids"][i],
                    "content": doc,
                    "memory_type": meta.get("memory_type", "fact"),
                    "source_session_id": meta.get("source_session_id", ""),
                    "source_query": meta.get("source_query", ""),
                    "importance": meta.get("importance", 0.5),
                    "tags": meta.get("tags", "[]"),
                    "created_at": meta.get("created_at", 0),
                }
            )
            embeddings.append(emb)
        return fragments, embeddings

    def delete_fragment(self, fragment_id):
        try:
            self.collection.delete(ids=[fragment_id])
            return True
        except Exception as e:
            logger.error(f"Delete memory {fragment_id} failed: {e}")
            return False

    def clear(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._turn_counter = 0
        logger.info(f"Cleared all memories for '{self.user_id}'")


# -- Memory Extraction (LLM-based, Token-Optimized) --

from core import llm_client as _llm_client  # noqa: E402 (deliberately deferred, keeps module import light)

# Compact system prompt (saves ~100 tokens vs verbose version)
EXTRACT_SYSTEM = """Extract memorable facts from this Q&A exchange.
Return a JSON array. Each item: {"c":"content","t":"fact|pref|decision|insight","i":0.5,"g":["tag"]}
- c: concise standalone statement (1 sentence)
- t: type
- i: importance 0-1
- g: 1-2 tags
Only extract genuinely useful info for future chats. Return [] if nothing worth remembering.
Return ONLY the JSON array."""

# Ollama structured-output schema mirroring the JSON EXTRACT_SYSTEM requests;
# ignored on Anthropic, and the fail-open parsing below stays as the safety net.
_EXTRACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "c": {"type": "string"},
            "t": {"type": "string", "enum": ["fact", "pref", "decision", "insight"]},
            "i": {"type": "number"},
            "g": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["c", "t", "i", "g"],
    },
}

SUMMARIZE_SYSTEM = """Summarize this conversation in 2-3 sentences. Return JSON: {"s":"summary","g":["topic1","topic2"],"i":0.7}
Return ONLY the JSON object."""

NOVELTY_SYSTEM = """Does this Q&A contain new facts, preferences, decisions, or technical details worth remembering?
Answer YES if it has: specific facts, user preferences, technical choices, decisions, concrete details.
Answer NO if it's only: a clarification, rephrasing, follow-up with no new info, or small talk.
Reply with ONLY "YES" or "NO"."""

MERGE_SYSTEM = """Merge these related memories into one concise, complete statement that captures all key details.
Return JSON: {"c":"merged content (1-2 sentences)","t":"fact|pref|decision|insight","i":0.8,"g":["tag1","tag2"]}
Return ONLY the JSON object."""

RECONCILE_SYSTEM = """You reconcile a NEW memory against the closest EXISTING stored memory. Pick exactly one op:
- ADD: different facts, both worth keeping
- UPDATE: NEW supersedes or corrects EXISTING — put the single up-to-date statement in "content"
- DELETE: NEW invalidates EXISTING and nothing replaces it
- NOOP: NEW adds nothing beyond EXISTING
Return JSON: {"op":"ADD|UPDATE|DELETE|NOOP","content":"updated statement (op=UPDATE only, else empty)"}
Return ONLY the JSON object."""

_RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
        "content": {"type": "string"},
    },
    "required": ["op", "content"],
}

STYLE_RULES_SYSTEM = """Distill these user preferences into at most 5 short imperative rules for how an assistant
should respond to this user. Merge overlapping preferences; drop one-off or stale items.
Return JSON: {"rules":["rule 1","rule 2"]}
Return ONLY the JSON object."""

_STYLE_RULES_SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
    "required": ["rules"],
}


def _strip_json_response(text):
    """Strip markdown code fences from an LLM JSON reply (fail-open parsing safety net)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _reconcile_op(new_content, existing_content):
    """Mem0-style write reconciliation: one memory-model call deciding how a new memory
    relates to its closest existing match. Fail-open: any LLM/JSON error degrades to a
    plain ADD — strictly no worse than the old dedup-only behavior."""
    try:
        text = _llm_client.chat(
            messages=[{"role": "user", "content": f"EXISTING: {existing_content[:300]}\nNEW: {new_content[:300]}"}],
            system=RECONCILE_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=200,
            temperature=0.0,
            stream=False,
            json_schema=_RECONCILE_SCHEMA,
        )
        data = json.loads(_strip_json_response(text))
        op = str(data.get("op", "ADD")).strip().upper()
        if op not in ("ADD", "UPDATE", "DELETE", "NOOP"):
            op = "ADD"
        return op, str(data.get("content") or "").strip()
    except Exception as e:
        logger.warning(f"Memory reconciliation failed, defaulting to ADD: {e}")
        return "ADD", ""


def _has_new_information(query, answer):
    """Cheap novelty gate: a tiny YES/NO LLM call before running full extraction.
    Uses max_tokens=5 so this costs almost nothing vs the full extraction call."""
    try:
        result = _llm_client.chat(
            messages=[{"role": "user", "content": f"Q: {query[:200]}\nA: {answer[:350]}"}],
            system=NOVELTY_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )
        return result.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(f"Novelty check failed, defaulting to extract: {e}")
        return True  # fail open: extract anyway


def _should_extract(query, answer):
    """Quick heuristic: skip trivial Q&A to save tokens."""
    q = query.lower().strip()
    # Skip greetings
    if len(q) < 8 or q in ("hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye"):
        return False
    # Skip very short answers (probably errors or "I don't know")
    if len(answer) < settings.memory_min_answer_length:
        return False
    # Skip meta-questions about the system
    return not any(p in q for p in ["how do you work", "what can you do", "help me", "what is this"])


def _truncate_for_extraction(text, max_chars=800):
    """Truncate text to save input tokens for extraction."""
    if len(text) <= max_chars:
        return text
    # Keep start and end (most important parts)
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def extract_memories_from_turn(query, answer, session_id=""):
    """Extract memory fragments from a single Q&A turn using cheapest model."""
    if not settings.memory_enabled:
        return []
    # Anthropic-only guard: skip if API key missing and we're using Anthropic
    if _llm_client.get_backend() == "anthropic" and not settings.anthropic_api_key:
        return []
    if not _should_extract(query, answer):
        return []

    try:
        # Truncate inputs to minimize tokens
        q_trunc = _truncate_for_extraction(query, 400)
        a_trunc = _truncate_for_extraction(answer, 800)

        text = _llm_client.chat(
            messages=[{"role": "user", "content": f"Q: {q_trunc}\nA: {a_trunc}"}],
            system=EXTRACT_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=512,
            temperature=0.0,
            stream=False,
            json_schema=_EXTRACT_SCHEMA,
        )
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        fragments_data = json.loads(text)
        if not isinstance(fragments_data, list):
            return []

        fragments = []
        for fd in fragments_data[:3]:  # Max 3 per turn
            content = fd.get("c") or fd.get("content", "")
            if not content or len(content) < 10:
                continue
            mem_type = fd.get("t") or fd.get("memory_type", "fact")
            mem_type = _TYPE_ALIASES.get(mem_type, mem_type)
            fragments.append(
                MemoryFragment(
                    content=content,
                    memory_type=mem_type,
                    importance=min(1.0, max(0.0, fd.get("i", fd.get("importance", 0.5)))),
                    tags=(fd.get("g") or fd.get("tags", []))[:2],
                    source_session_id=session_id,
                    source_query=query[:100],
                )
            )
        logger.info(f"Extracted {len(fragments)} memories")
        return fragments

    except json.JSONDecodeError as e:
        logger.warning(f"Memory extraction JSON error: {e}")
        return []
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        return []


def summarize_conversation(messages, session_id=""):
    """Summarize an entire conversation into a single compact memory."""
    if not settings.memory_enabled:
        return None
    if _llm_client.get_backend() == "anthropic" and not settings.anthropic_api_key:
        return None
    if len(messages) < 4:
        return None

    try:
        parts = []
        for m in messages[:20]:
            role = m.get("role", "user")[0].upper()
            content = m.get("content", "")[:200]
            parts.append(f"{role}: {content}")
        conv_text = "\n".join(parts)

        text = _llm_client.chat(
            messages=[{"role": "user", "content": conv_text}],
            system=SUMMARIZE_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=256,
            temperature=0.0,
            stream=False,
        )
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        return MemoryFragment(
            content=data.get("s") or data.get("summary", ""),
            memory_type="summary",
            importance=min(1.0, max(0.0, data.get("i", data.get("importance", 0.6)))),
            tags=(data.get("g") or data.get("topics", []))[:3],
            source_session_id=session_id,
            source_query="session summary",
        )
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return None


# -- Conversation History Optimization --


def compact_history(messages, max_turns=None):
    """
    Optimize conversation history to minimize tokens.
    - Keeps last N turns
    - Summarizes older messages into a compact preamble
    - Truncates individual messages
    """
    max_turns = max_turns or settings.max_history_turns
    if not messages or len(messages) <= max_turns * 2:
        # Already short enough — just truncate individual messages
        return [{"role": m["role"], "content": m["content"][:600]} for m in messages[-max_turns * 2 :]]

    # Split into old and recent
    recent = messages[-(max_turns * 2) :]
    old = messages[: -(max_turns * 2)]

    # Build compact summary of old messages
    old_topics = set()
    for m in old:
        if m["role"] == "user":
            # Extract key words (cheap, no LLM)
            words = m["content"].lower().split()
            old_topics.update(w for w in words if len(w) > 4 and w.isalpha())

    preamble = "[Earlier in this conversation, we discussed: {}]".format(", ".join(sorted(old_topics)[:15]))

    compacted = [
        {"role": "user", "content": preamble},
        {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
    ]
    for m in recent:
        compacted.append({"role": m["role"], "content": m["content"][:600]})

    return compacted


# -- Memory Retrieval & Formatting --

_DAYS_AGO_RE = re.compile(r"\b(\d{1,3})\s+days?\s+ago\b")


def _parse_temporal_range(query):
    """Map temporal phrases in the query to a (lo, hi) epoch-seconds range, or None when nothing matches."""
    q = query.lower()
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "yesterday" in q:
        return (today - timedelta(days=1)).timestamp(), today.timestamp()
    if "this week" in q:
        week_start = today - timedelta(days=today.weekday())
        return week_start.timestamp(), now.timestamp()
    if "last week" in q:
        week_start = today - timedelta(days=today.weekday())
        return (week_start - timedelta(days=7)).timestamp(), week_start.timestamp()
    if "last month" in q:
        month_start = today.replace(day=1)
        prev_month_start = (month_start - timedelta(days=1)).replace(day=1)
        return prev_month_start.timestamp(), month_start.timestamp()
    m = _DAYS_AGO_RE.search(q)
    if m:
        day = today - timedelta(days=int(m.group(1)))
        return day.timestamp(), (day + timedelta(days=1)).timestamp()
    # Deliberately no month-name matching ("in June"): that phrasing usually
    # refers to when the EVENT happened, not when the memory was stored, and a
    # created_at filter would hide exactly the fragment that answers it.
    return None


def retrieve_memories(mem_store, query, top_k=None):
    """Retrieve relevant memories — composite-ranked (relevance x recency x importance), compact format."""
    top_k = top_k or settings.memory_top_k
    start = time.perf_counter()

    created_range = _parse_temporal_range(query)
    candidates = mem_store.search(query, top_k=max(20, top_k * 4), created_range=created_range)
    if not candidates and created_range:
        # Temporal phrase may refer to event time rather than storage time —
        # an empty filtered result should not hide otherwise-relevant memories.
        candidates = mem_store.search(query, top_k=max(20, top_k * 4))

    if not candidates:
        return MemoryContext(fragments=[], formatted="", count=0, retrieval_ms=0)

    now = time.time()
    for frag in candidates:
        last_accessed = frag.get("last_accessed") or frag.get("created_at", 0)
        hours_since = max(0.0, (now - last_accessed) / 3600.0)
        recency = math.exp(-hours_since / RECENCY_DECAY_HOURS)
        # search()'s similarity is (1+cos)/2, compressing real candidates into
        # [0.5, 1] — rescale to true cosine so relevance keeps its full weight
        # and an off-topic-but-recent memory can't outrank an on-topic one.
        relevance = max(0.0, 2.0 * frag["similarity"] - 1.0)
        frag["composite_score"] = 0.4 * relevance + 0.3 * recency + 0.3 * frag.get("importance", 0.5)

    candidates.sort(key=lambda f: f["composite_score"], reverse=True)
    fragments = candidates[:top_k]

    # Reinforcement touch: injected winners become "recently accessed" — never fail retrieval over this
    try:
        winners = [f for f in fragments if f.get("fragment_id")]
        if winners:
            mem_store.collection.update(
                ids=[f["fragment_id"] for f in winners],
                metadatas=[{"last_accessed": now, "access_count": int(f.get("access_count", 0)) + 1} for f in winners],
            )
    except Exception as e:
        logger.warning(f"Memory access touch failed (non-fatal): {e}")

    # Compact format (saves ~40% tokens vs verbose)
    lines = ["[Memory] Relevant facts from past chats:"]
    for frag in fragments:
        sim_pct = int(frag["similarity"] * 100)
        created_at = frag.get("created_at", 0)
        if created_at:
            date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
            lines.append("- {} [{}] ({}%)".format(frag["content"], date_str, sim_pct))
        else:
            lines.append("- {} ({}%)".format(frag["content"], sim_pct))
    formatted = "\n".join(lines)

    ms = (time.perf_counter() - start) * 1000
    return MemoryContext(
        fragments=fragments,
        formatted=formatted,
        count=len(fragments),
        retrieval_ms=round(ms, 1),
    )


# -- Optimized Context Builder --


def optimize_context_chunks(hits):
    """Reduce token cost of retrieved chunks sent to LLM.
    - Limit number of chunks
    - Truncate each chunk
    """
    max_chunks = settings.max_context_chunks
    max_tokens = settings.max_chunk_preview_tokens

    optimized = []
    for hit in hits[:max_chunks]:
        content = hit["content"]
        # Rough truncation by character (4 chars ≈ 1 token)
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...truncated]"
        optimized.append({**hit, "content": content})
    return optimized


# -- Memory Consolidation --


def _cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix using numpy (already a dep via sentence-transformers)."""
    import numpy as np

    A = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    A_norm = A / norms
    return (A_norm @ A_norm.T).tolist()


def _merge_cluster(cluster_mems):
    """Use LLM to merge a cluster of related memories into one richer memory."""
    memories_text = "\n".join("- " + m["content"] for m in cluster_mems)
    max_importance = max(m.get("importance", 0.5) for m in cluster_mems)
    try:
        text = _llm_client.chat(
            messages=[{"role": "user", "content": "Merge these related memories:\n" + memories_text}],
            system=MERGE_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=150,
            temperature=0.0,
            stream=False,
        )
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        data = json.loads(text)
        mem_type = _TYPE_ALIASES.get(data.get("t", "fact"), data.get("t", "fact"))
        return MemoryFragment(
            content=data.get("c", ""),
            memory_type=mem_type,
            importance=min(1.0, max(max_importance, float(data.get("i", 0.7)))),
            tags=data.get("g", [])[:2],
            source_query="consolidation",
        )
    except Exception as e:
        logger.error(f"Cluster merge failed: {e}")
        return None


def consolidate_memories(mem_store, merge_threshold=0.72):
    """
    Scan all memories for semantically related clusters and merge them.

    Memories that are similar enough to be related (above merge_threshold)
    but not exact duplicates (below the 0.92 dedup threshold) are candidates
    for consolidation. Each cluster is merged into one richer memory via LLM.

    Args:
        merge_threshold: Cosine similarity floor for clustering (default 0.72).
                         Higher = only very close memories merged.
                         Lower = more aggressive merging.

    Returns:
        dict with keys: merged (clusters merged), deleted (originals removed), skipped.
    """
    # Per-user lock (reentrant, so the run_maintenance path nests safely) — prevents a
    # concurrent manual consolidation or write from double-merging or double-deleting.
    with _get_user_lock(mem_store.user_id):
        if mem_store.count < 3:
            return {"merged": 0, "deleted": 0, "skipped": 0, "message": "Not enough memories to consolidate"}

        fragments, embeddings = mem_store.get_all_with_embeddings(limit=500)
        n = len(fragments)
        if n < 3:
            return {"merged": 0, "deleted": 0, "skipped": 0, "message": "Not enough memories to consolidate"}

        logger.info(f"Consolidation: computing similarity matrix for {n} memories")
        sim_matrix = _cosine_similarity_matrix(embeddings)

        # Greedy clustering: each fragment joins the first cluster it's similar to
        visited = set()
        clusters = []
        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= merge_threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        if not clusters:
            return {"merged": 0, "deleted": 0, "skipped": 0, "message": "No related memory clusters found"}

        logger.info(f"Consolidation: found {len(clusters)} clusters to merge")
        merged_count = 0
        deleted_count = 0
        skipped = 0

        for cluster_indices in clusters:
            cluster_mems = [fragments[i] for i in cluster_indices]
            merged = _merge_cluster(cluster_mems)
            if merged and merged.content:
                for m in cluster_mems:
                    mem_store.delete_fragment(m["fragment_id"])
                    deleted_count += 1
                mem_store.add_fragment(merged)
                merged_count += 1
                logger.info(f"Merged {len(cluster_mems)} memories → 1: {merged.content[:60]}")
            else:
                skipped += len(cluster_mems)

        return {
            "merged": merged_count,
            "deleted": deleted_count,
            "skipped": skipped,
            "message": f"Merged {merged_count} clusters ({deleted_count} memories → {merged_count} consolidated)",
        }


# -- Global Store Cache --

_memory_stores = {}


def get_memory_store(user_id="default"):
    if user_id not in _memory_stores:
        _memory_stores[user_id] = MemoryStore(user_id=user_id)
    return _memory_stores[user_id]


def process_turn_memories(user_id, query, answer, session_id=""):
    """
    Full pipeline: extract memories from a turn and store them.
    - Respects interval setting (only runs every N turns)
    - Within each interval, a cheap novelty check gates the full extraction
      so follow-up / clarification turns never trigger costly extraction calls
    """
    if not settings.memory_enabled or not settings.memory_auto_extract:
        return []

    mem_store = get_memory_store(user_id)
    should_run = mem_store.increment_turn()

    if not should_run:
        logger.debug(
            f"Skipping extraction (turn {mem_store._turn_counter}, interval {settings.memory_extract_interval})"
        )
        return []

    # Cheap novelty gate — saves full extraction cost on clarification turns
    if not _has_new_information(query, answer):
        logger.debug("Novelty check: no new info detected, skipping extraction")
        return []

    fragments = extract_memories_from_turn(query, answer, session_id)
    if fragments:
        mem_store.add_fragments(fragments)
    return fragments


def process_session_summary(user_id, session_id, messages):
    """Summarize a completed session and store as memory."""
    if not settings.memory_enabled or not settings.memory_auto_summarize:
        return None
    summary = summarize_conversation(messages, session_id)
    if summary:
        store = get_memory_store(user_id)
        store.add_fragment(summary)
    return summary


# -- Idle-Time Maintenance (consolidation + pruning + procedural distillation) --


def _archive_overflow(mem_store):
    """Soft-archive the lowest-value fragments when the live count exceeds
    settings.memory_max_fragments. Value = importance x recency x (1 + log1p(access_count))."""
    results = mem_store.collection.get(include=["metadatas"])
    now = time.time()
    live = []
    for fid, meta in zip(results["ids"], results["metadatas"]):
        if fid.startswith("proc_") or meta.get("invalid_at"):
            continue
        last_accessed = meta.get("last_accessed", meta.get("created_at", 0)) or 0
        hours_since = max(0.0, (now - last_accessed) / 3600.0)
        recency = math.exp(-hours_since / RECENCY_DECAY_HOURS)
        value = float(meta.get("importance", 0.5)) * recency * (1.0 + math.log1p(int(meta.get("access_count", 0))))
        live.append((value, fid))
    overflow = len(live) - settings.memory_max_fragments
    if overflow <= 0:
        return 0
    live.sort(key=lambda pair: pair[0])
    victims = [fid for _, fid in live[:overflow]]
    mem_store.collection.update(ids=victims, metadatas=[{"invalid_at": now, "archived": True} for _ in victims])
    logger.info(f"Soft-archived {len(victims)} low-value memories for '{mem_store.user_id}'")
    return len(victims)


def _purge_invalidated(mem_store):
    """Hard-delete fragments soft-invalidated more than INVALID_PURGE_DAYS ago, so
    superseded/archived rows stop crowding live memories out of search()'s over-fetch
    window. Returns the number of fragments purged."""
    cutoff = time.time() - INVALID_PURGE_DAYS * 86400
    results = mem_store.collection.get(where={"invalid_at": {"$lt": cutoff}})
    victims = results["ids"]
    if victims:
        mem_store.collection.delete(ids=victims)
        logger.info(f"Purged {len(victims)} long-invalidated memories for '{mem_store.user_id}'")
    return len(victims)


def _distill_style_rules(mem_store):
    """Distill all live 'pref' fragments into at most 5 bullet rules stored as one
    procedural fragment (id 'proc_<user_id>'), upserted over the previous version.
    Returns the number of rules stored (0 when no prefs exist or the LLM call fails)."""
    results = mem_store.collection.get(where={"memory_type": "pref"}, include=["documents", "metadatas"])
    prefs = [doc for doc, meta in zip(results["documents"], results["metadatas"]) if not meta.get("invalid_at")]
    if not prefs:
        return 0
    prefs_text = "\n".join("- " + p[:200] for p in prefs[:40])
    try:
        text = _llm_client.chat(
            messages=[{"role": "user", "content": prefs_text}],
            system=STYLE_RULES_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=300,
            temperature=0.0,
            stream=False,
            json_schema=_STYLE_RULES_SCHEMA,
        )
        data = json.loads(_strip_json_response(text))
        rules = [str(r).strip() for r in data.get("rules", []) if str(r).strip()][:5]
    except Exception as e:
        logger.warning(f"Style-rule distillation failed (LLM unreachable?): {e}")
        return 0
    if not rules:
        return 0
    content = "\n".join("- " + r for r in rules)
    now = time.time()
    embedding = _embed_texts([content])[0]
    mem_store.collection.upsert(
        ids=["proc_" + mem_store.user_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[
            {
                "memory_type": "summary",
                "source_session_id": "",
                "source_query": "style-rules distillation",
                "importance": 1.0,
                "tags": json.dumps(["style-rules"]),
                "created_at": now,
                "last_accessed": now,
                "access_count": 0,
                "user_id": mem_store.user_id,
            }
        ],
    )
    logger.info(f"Distilled {len(prefs)} prefs into {len(rules)} style rules for '{mem_store.user_id}'")
    return len(rules)


def run_maintenance(user_id):
    """Idle-time memory upkeep: (a) consolidation of related clusters, (b) bounded
    soft-archiving above memory_max_fragments, (c) distillation of 'pref' fragments
    into a procedural style-rules block, (d) hard purge of fragments soft-invalidated
    more than INVALID_PURGE_DAYS ago. Serialized against memory writes via the
    per-user lock. Each step tolerates the LLM backend being unreachable — the
    failure is logged and that step contributes zero to the returned counts.

    Returns {"merged": int, "archived": int, "distilled": int, "purged": int}.
    """
    counts = {"merged": 0, "archived": 0, "distilled": 0, "purged": 0}
    mem_store = get_memory_store(user_id)
    with _get_user_lock(user_id):
        try:
            counts["merged"] = consolidate_memories(mem_store).get("merged", 0)
        except Exception as e:
            logger.warning(f"Maintenance consolidation failed for '{user_id}': {e}")
        try:
            counts["archived"] = _archive_overflow(mem_store)
        except Exception as e:
            logger.warning(f"Maintenance archiving failed for '{user_id}': {e}")
        try:
            counts["distilled"] = _distill_style_rules(mem_store)
        except Exception as e:
            logger.warning(f"Maintenance distillation failed for '{user_id}': {e}")
        try:
            counts["purged"] = _purge_invalidated(mem_store)
        except Exception as e:
            logger.warning(f"Maintenance purge failed for '{user_id}': {e}")
    logger.info(f"Memory maintenance for '{user_id}': {counts}")
    return counts


def get_style_rules(user_id):
    """Return the distilled style-rules bullet text for a user, or None when absent."""
    try:
        mem_store = get_memory_store(user_id)
        result = mem_store.collection.get(ids=["proc_" + user_id], include=["documents"])
        if result["ids"] and result["documents"] and result["documents"][0]:
            return result["documents"][0]
    except Exception as e:
        logger.warning(f"get_style_rules failed for '{user_id}': {e}")
    return None
