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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

MEMORY_TYPES = ["fact", "pref", "decision", "insight", "summary"]

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
        logger.info("MemoryStore ready for '{}': {} fragments".format(
            user_id, self.collection.count()))

    @property
    def count(self):
        return self.collection.count()

    def increment_turn(self):
        """Increment turn counter. Returns True if extraction should run."""
        self._turn_counter += 1
        return self._turn_counter % settings.memory_extract_interval == 0

    def add_fragment(self, fragment):
        """Store a single memory fragment."""
        # Deduplicate: check if very similar memory already exists
        if self.count > 0:
            existing = self.search(fragment.content, top_k=1)
            if existing and existing[0].get("similarity", 0) > 0.92:
                logger.info("Skipping duplicate memory: {}".format(fragment.content[:50]))
                return None

        embedding = _embed_texts([fragment.content])[0]
        self.collection.upsert(
            ids=[fragment.fragment_id],
            embeddings=[embedding],
            documents=[fragment.content],
            metadatas=[{
                "memory_type": fragment.memory_type,
                "source_session_id": fragment.source_session_id,
                "source_query": fragment.source_query[:150],
                "importance": fragment.importance,
                "tags": json.dumps(fragment.tags),
                "created_at": fragment.created_at,
                "user_id": self.user_id,
            }],
        )
        logger.info("Stored memory: {} [{}]".format(
            fragment.fragment_id, fragment.memory_type))
        return fragment.fragment_id

    def add_fragments(self, fragments):
        """Store multiple fragments with deduplication."""
        if not fragments:
            return []
        stored = []
        for f in fragments:
            fid = self.add_fragment(f)
            if fid:
                stored.append(fid)
        logger.info("Stored {}/{} memory fragments (deduped)".format(
            len(stored), len(fragments)))
        return stored

    def search(self, query, top_k=5, memory_type=None, min_importance=0.0):
        """Retrieve relevant memory fragments."""
        if self.count == 0:
            return []
        query_embedding = _embed_texts([query])[0]
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        fragments = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - (dist / 2)
                if similarity < 0.2:
                    continue
                importance = meta.get("importance", 0.5)
                if importance < min_importance:
                    continue
                fragments.append({
                    "content": doc,
                    "memory_type": meta.get("memory_type", "fact"),
                    "source_session_id": meta.get("source_session_id", ""),
                    "source_query": meta.get("source_query", ""),
                    "importance": importance,
                    "tags": json.loads(meta.get("tags", "[]")),
                    "created_at": meta.get("created_at", 0),
                    "similarity": round(similarity, 4),
                })
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
            fragments.append({
                "fragment_id": results["ids"][i],
                "content": doc,
                "memory_type": meta.get("memory_type", "fact"),
                "source_session_id": meta.get("source_session_id", ""),
                "source_query": meta.get("source_query", ""),
                "importance": meta.get("importance", 0.5),
                "tags": meta.get("tags", "[]"),
                "created_at": meta.get("created_at", 0),
            })
        return fragments

    def get_all_with_embeddings(self, limit=500):
        """Return all stored memories along with their stored embeddings."""
        if self.count == 0:
            return [], []
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
        )
        fragments = []
        embeddings = []
        for i, (doc, meta, emb) in enumerate(zip(
            results["documents"], results["metadatas"], results["embeddings"]
        )):
            fragments.append({
                "fragment_id": results["ids"][i],
                "content": doc,
                "memory_type": meta.get("memory_type", "fact"),
                "source_session_id": meta.get("source_session_id", ""),
                "source_query": meta.get("source_query", ""),
                "importance": meta.get("importance", 0.5),
                "tags": meta.get("tags", "[]"),
                "created_at": meta.get("created_at", 0),
            })
            embeddings.append(emb)
        return fragments, embeddings

    def delete_fragment(self, fragment_id):
        try:
            self.collection.delete(ids=[fragment_id])
            return True
        except Exception as e:
            logger.error("Delete memory {} failed: {}".format(fragment_id, e))
            return False

    def clear(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"},
        )
        self._turn_counter = 0
        logger.info("Cleared all memories for '{}'".format(self.user_id))


# -- Memory Extraction (LLM-based, Token-Optimized) --

from core import llm_client as _llm_client


# Compact system prompt (saves ~100 tokens vs verbose version)
EXTRACT_SYSTEM = """Extract memorable facts from this Q&A exchange.
Return a JSON array. Each item: {"c":"content","t":"fact|pref|decision|insight","i":0.5,"g":["tag"]}
- c: concise standalone statement (1 sentence)
- t: type
- i: importance 0-1
- g: 1-2 tags
Only extract genuinely useful info for future chats. Return [] if nothing worth remembering.
Return ONLY the JSON array."""

SUMMARIZE_SYSTEM = """Summarize this conversation in 2-3 sentences. Return JSON: {"s":"summary","g":["topic1","topic2"],"i":0.7}
Return ONLY the JSON object."""

NOVELTY_SYSTEM = """Does this Q&A contain new facts, preferences, decisions, or technical details worth remembering?
Answer YES if it has: specific facts, user preferences, technical choices, decisions, concrete details.
Answer NO if it's only: a clarification, rephrasing, follow-up with no new info, or small talk.
Reply with ONLY "YES" or "NO"."""

MERGE_SYSTEM = """Merge these related memories into one concise, complete statement that captures all key details.
Return JSON: {"c":"merged content (1-2 sentences)","t":"fact|pref|decision|insight","i":0.8,"g":["tag1","tag2"]}
Return ONLY the JSON object."""


def _has_new_information(query, answer):
    """Cheap novelty gate: a tiny YES/NO LLM call before running full extraction.
    Uses max_tokens=5 so this costs almost nothing vs the full extraction call."""
    try:
        result = _llm_client.chat(
            messages=[{"role": "user", "content": "Q: {}\nA: {}".format(
                query[:200], answer[:350])}],
            system=NOVELTY_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )
        return result.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning("Novelty check failed, defaulting to extract: {}".format(e))
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
    if any(p in q for p in ["how do you work", "what can you do", "help me", "what is this"]):
        return False
    return True


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
            messages=[{"role": "user", "content": "Q: {}\nA: {}".format(q_trunc, a_trunc)}],
            system=EXTRACT_SYSTEM,
            model=_llm_client.get_memory_model(),
            max_tokens=512,
            temperature=0.0,
            stream=False,
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
            fragments.append(MemoryFragment(
                content=content,
                memory_type=mem_type,
                importance=min(1.0, max(0.0, fd.get("i", fd.get("importance", 0.5)))),
                tags=(fd.get("g") or fd.get("tags", []))[:2],
                source_session_id=session_id,
                source_query=query[:100],
            ))
        logger.info("Extracted {} memories".format(len(fragments)))
        return fragments

    except json.JSONDecodeError as e:
        logger.warning("Memory extraction JSON error: {}".format(e))
        return []
    except Exception as e:
        logger.error("Memory extraction failed: {}".format(e))
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
            parts.append("{}: {}".format(role, content))
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
        logger.error("Summarization failed: {}".format(e))
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
        return [{"role": m["role"], "content": m["content"][:600]}
                for m in messages[-max_turns * 2:]]

    # Split into old and recent
    recent = messages[-(max_turns * 2):]
    old = messages[:-(max_turns * 2)]

    # Build compact summary of old messages
    old_topics = set()
    for m in old:
        if m["role"] == "user":
            # Extract key words (cheap, no LLM)
            words = m["content"].lower().split()
            old_topics.update(w for w in words if len(w) > 4 and w.isalpha())

    preamble = "[Earlier in this conversation, we discussed: {}]".format(
        ", ".join(sorted(old_topics)[:15])
    )

    compacted = [{"role": "user", "content": preamble},
                 {"role": "assistant", "content": "Understood, I have context from our earlier discussion."}]
    for m in recent:
        compacted.append({"role": m["role"], "content": m["content"][:600]})

    return compacted


# -- Memory Retrieval & Formatting --

def retrieve_memories(mem_store, query, top_k=None):
    """Retrieve relevant memories — compact format to minimize prompt tokens."""
    top_k = top_k or settings.memory_top_k
    start = time.perf_counter()

    fragments = mem_store.search(query, top_k=top_k)

    if not fragments:
        return MemoryContext(fragments=[], formatted="", count=0, retrieval_ms=0)

    # Compact format (saves ~40% tokens vs verbose)
    lines = ["[Memory] Relevant facts from past chats:"]
    for frag in fragments:
        lines.append("- {} ({}%)".format(frag["content"], int(frag["similarity"] * 100)))
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
        logger.error("Cluster merge failed: {}".format(e))
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
    if mem_store.count < 3:
        return {"merged": 0, "deleted": 0, "skipped": 0, "message": "Not enough memories to consolidate"}

    fragments, embeddings = mem_store.get_all_with_embeddings(limit=500)
    n = len(fragments)
    if n < 3:
        return {"merged": 0, "deleted": 0, "skipped": 0, "message": "Not enough memories to consolidate"}

    logger.info("Consolidation: computing similarity matrix for {} memories".format(n))
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

    logger.info("Consolidation: found {} clusters to merge".format(len(clusters)))
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
            logger.info("Merged {} memories → 1: {}".format(
                len(cluster_mems), merged.content[:60]))
        else:
            skipped += len(cluster_mems)

    return {
        "merged": merged_count,
        "deleted": deleted_count,
        "skipped": skipped,
        "message": "Merged {} clusters ({} memories → {} consolidated)".format(
            merged_count, deleted_count, merged_count),
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
        logger.debug("Skipping extraction (turn {}, interval {})".format(
            mem_store._turn_counter, settings.memory_extract_interval))
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
