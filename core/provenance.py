"""
core/provenance.py — Ancestry Trace / Thought Provenance System

Post-hoc attribution: split the LLM response into sentences, then score
each sentence against retrieved chunks, long-term memories, the user query,
and recent conversation history using local embeddings.

No LLM calls — pure embedding cosine similarity, so it runs in < 100 ms
even for long responses with many chunks.

Risk levels per sentence:
  sourced  — novel_score < 0.35  (clearly grounded in retrieval)
  inferred — novel_score 0.35–0.65 (partially supported)
  orphan   — novel_score > 0.65  (structurally unverifiable)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────
SOURCED_THRESHOLD = 0.35   # novel_score below this → sourced
ORPHAN_THRESHOLD  = 0.65   # novel_score above this → orphan

# Maximum chars we pass per chunk / memory to the embedder
MAX_CHUNK_CHARS = 1000
MAX_MEM_CHARS   = 400
MAX_HIST_CHARS  = 300


# ── lazy embedder (same model the rest of the project uses) ──────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── sentence splitting ────────────────────────────────────────────────────────

# Simple regex-based sentence splitter. Handles abbreviations poorly but is
# fast and dependency-free. NLTK / spaCy are overkill here.
_SENT_RE = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
)

def split_sentences(text: str) -> List[str]:
    """Split *text* into individual sentences, filtering empty strings."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) > 10:          # skip fragments shorter than a typical clause
            out.append(p)
    return out or [text]          # fallback: return whole text as one sentence


# ── cosine helpers ────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_norm(a), _norm(b)))

def _max_cosine(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> float:
    """Return the maximum cosine similarity between *query_vec* and any row of *corpus_vecs*."""
    if corpus_vecs.shape[0] == 0:
        return 0.0
    dots = corpus_vecs @ _norm(query_vec)
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    normed = corpus_vecs / norms
    sims = normed @ _norm(query_vec)
    return float(np.max(sims))


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class SourceAttribution:
    """Best-matching source for a sentence."""
    source_type: str          # "chunk" | "memory" | "query" | "history"
    source_id: str            # chunk index / memory fragment id / "query" / "history_N"
    source_preview: str       # first 120 chars of the source text
    similarity: float         # cosine similarity (0–1)


@dataclass
class SentenceProvenance:
    text: str
    attributions: List[SourceAttribution]   # top-3 sources, sorted by similarity desc
    query_echo: float                        # similarity to user query
    history_echo: float                     # max similarity to any history turn
    novel_score: float                      # 1 – max(all source sims)
    risk: str                               # "sourced" | "inferred" | "orphan"

    def to_dict(self):
        d = asdict(self)
        d["attributions"] = [asdict(a) for a in self.attributions]
        return d


@dataclass
class ProvenanceMap:
    sentences: List[SentenceProvenance]
    doc_coverage: float         # fraction of sentences that are "sourced"
    memory_coverage: float      # fraction of sentences whose best source is a memory
    novel_fraction: float       # fraction of sentences that are "orphan"
    orphan_count: int
    sourced_count: int
    inferred_count: int

    def to_dict(self):
        return {
            "sentences": [s.to_dict() for s in self.sentences],
            "doc_coverage": round(self.doc_coverage, 3),
            "memory_coverage": round(self.memory_coverage, 3),
            "novel_fraction": round(self.novel_fraction, 3),
            "orphan_count": self.orphan_count,
            "sourced_count": self.sourced_count,
            "inferred_count": self.inferred_count,
        }


# ── main entry point ─────────────────────────────────────────────────────────

def compute_provenance(
    response: str,
    chunks: list,              # list of hit dicts with "content" key
    memories: list,            # list of MemoryFragment objects (have .content attr) or plain strings
    query: str,
    history: Optional[list] = None,  # list of {"role": ..., "content": ...} dicts
) -> Optional[ProvenanceMap]:
    """
    Compute sentence-level provenance for *response* against *chunks*, *memories*,
    *query*, and *history*.

    Returns None if response is empty or there are no sources to compare against.
    """
    if not response or not response.strip():
        return None

    sentences = split_sentences(response)
    if not sentences:
        return None

    # ── build source corpus ────────────────────────────────────────────────
    chunk_texts: List[str] = []
    chunk_ids:   List[str] = []
    for i, h in enumerate(chunks or []):
        text = (h.get("content") or "")[:MAX_CHUNK_CHARS]
        if text.strip():
            chunk_texts.append(text)
            fname = h.get("metadata", {}).get("document_path", f"chunk_{i}")
            chunk_ids.append(str(fname))

    mem_texts: List[str] = []
    mem_ids:   List[str] = []
    for i, m in enumerate(memories or []):
        if hasattr(m, "content"):
            text = str(m.content)[:MAX_MEM_CHARS]
            mid  = getattr(m, "id", f"mem_{i}")
        else:
            text = str(m)[:MAX_MEM_CHARS]
            mid  = f"mem_{i}"
        if text.strip():
            mem_texts.append(text)
            mem_ids.append(str(mid))

    hist_texts: List[str] = []
    hist_ids:   List[str] = []
    for i, turn in enumerate(history or []):
        content = (turn.get("content") or "")[:MAX_HIST_CHARS]
        if content.strip():
            hist_texts.append(content)
            hist_ids.append(f"history_{i}")

    query_text = (query or "")[:500]

    # Nothing to compare against → skip provenance
    if not chunk_texts and not mem_texts and not query_text:
        return None

    # ── batch embed everything in a single call ───────────────────────────
    embedder = _get_embedder()

    all_texts = sentences + chunk_texts + mem_texts + hist_texts + ([query_text] if query_text else [])
    try:
        all_vecs = embedder.encode(all_texts, batch_size=64, show_progress_bar=False)
    except Exception as e:
        logger.warning("Provenance embedding failed: {}".format(e))
        return None

    n_sent  = len(sentences)
    n_chunk = len(chunk_texts)
    n_mem   = len(mem_texts)
    n_hist  = len(hist_texts)

    sent_vecs  = all_vecs[:n_sent]
    chunk_vecs = all_vecs[n_sent            : n_sent + n_chunk]
    mem_vecs   = all_vecs[n_sent + n_chunk  : n_sent + n_chunk + n_mem]
    hist_vecs  = all_vecs[n_sent + n_chunk + n_mem : n_sent + n_chunk + n_mem + n_hist]
    query_vec  = all_vecs[-1] if query_text else None

    # Pre-normalise corpus vecs (used for matrix multiply)
    def _prenorm(mat):
        if mat.shape[0] == 0:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return mat / norms

    chunk_n = _prenorm(chunk_vecs)
    mem_n   = _prenorm(mem_vecs)
    hist_n  = _prenorm(hist_vecs)
    query_n = _norm(query_vec) if query_vec is not None else None

    # ── per-sentence attribution ──────────────────────────────────────────
    sent_provenances: List[SentenceProvenance] = []

    for idx, sentence in enumerate(sentences):
        sv = _norm(sent_vecs[idx])

        # similarity to each corpus group
        chunk_sims = (chunk_n @ sv).tolist() if chunk_n.shape[0] > 0 else []
        mem_sims   = (mem_n   @ sv).tolist() if mem_n.shape[0] > 0 else []
        hist_sims  = (hist_n  @ sv).tolist() if hist_n.shape[0] > 0 else []
        query_sim  = float(np.dot(query_n, sv)) if query_n is not None else 0.0

        # collect all (sim, type, id, preview) candidates
        candidates = []
        for j, sim in enumerate(chunk_sims):
            candidates.append((sim, "chunk", chunk_ids[j], chunk_texts[j][:120]))
        for j, sim in enumerate(mem_sims):
            candidates.append((sim, "memory", mem_ids[j], mem_texts[j][:120]))
        if query_sim > 0:
            candidates.append((query_sim, "query", "query", query_text[:120]))
        for j, sim in enumerate(hist_sims):
            candidates.append((sim, "history", hist_ids[j], hist_texts[j][:120]))

        # top-3 by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)
        top3 = candidates[:3]

        attributions = [
            SourceAttribution(
                source_type=t, source_id=sid, source_preview=preview,
                similarity=round(sim, 4)
            )
            for sim, t, sid, preview in top3
        ]

        max_source_sim = top3[0][0] if top3 else 0.0
        novel_score    = 1.0 - max_source_sim
        history_echo   = max(hist_sims) if hist_sims else 0.0

        if novel_score < SOURCED_THRESHOLD:
            risk = "sourced"
        elif novel_score > ORPHAN_THRESHOLD:
            risk = "orphan"
        else:
            risk = "inferred"

        sent_provenances.append(SentenceProvenance(
            text=sentence,
            attributions=attributions,
            query_echo=round(query_sim, 4),
            history_echo=round(history_echo, 4),
            novel_score=round(novel_score, 4),
            risk=risk,
        ))

    # ── aggregate stats ───────────────────────────────────────────────────
    total = len(sent_provenances)
    sourced_count  = sum(1 for s in sent_provenances if s.risk == "sourced")
    inferred_count = sum(1 for s in sent_provenances if s.risk == "inferred")
    orphan_count   = sum(1 for s in sent_provenances if s.risk == "orphan")

    # memory_coverage: sentences whose best attribution is a memory
    mem_primary = sum(
        1 for sp in sent_provenances
        if sp.attributions and sp.attributions[0].source_type == "memory"
    )

    doc_coverage    = sourced_count / total if total else 0.0
    memory_coverage = mem_primary   / total if total else 0.0
    novel_fraction  = orphan_count  / total if total else 0.0

    return ProvenanceMap(
        sentences=sent_provenances,
        doc_coverage=doc_coverage,
        memory_coverage=memory_coverage,
        novel_fraction=novel_fraction,
        orphan_count=orphan_count,
        sourced_count=sourced_count,
        inferred_count=inferred_count,
    )
