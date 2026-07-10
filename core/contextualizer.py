"""
core/contextualizer.py — Contextual retrieval (Anthropic-style chunk situating)

Generates a short situating sentence for a chunk within its source document at
ingest time. Storing the enriched text as the Chroma document gives contextual
embeddings + contextual BM25 + contextual SPLADE automatically, since both
sparse indexes rebuild from collection documents.

Opt-in via RAG_CONTEXTUAL_ENRICH (2-6s/chunk on CPU Ollama). Any failure
returns "" so ingest is never blocked or degraded by this step.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SITUATE_SYSTEM = (
    "You situate document chunks for search retrieval. "
    "Given a document and one chunk from it, reply with 1-2 short sentences that place the chunk "
    "within the overall document: what it is part of and what it covers. "
    "Reply with the situating context only — no preamble, no quotes, no restating the chunk."
)

# Keep the document excerpt small enough for a 3B model's context window.
_DOC_WINDOW_CHARS = 6000
_MAX_CONTEXT_CHARS = 500


def _window_around_chunk(doc_text: str, chunk_text: str) -> str:
    """Return up to _DOC_WINDOW_CHARS of doc_text centered on the chunk's location."""
    if len(doc_text) <= _DOC_WINDOW_CHARS:
        return doc_text
    # Skip the breadcrumb/location header line — it does not appear in the source document.
    probe = chunk_text.split("\n", 1)[-1].strip()[:200]
    idx = doc_text.find(probe) if probe else -1
    if idx < 0:
        return doc_text[:_DOC_WINDOW_CHARS]
    start = max(0, idx - _DOC_WINDOW_CHARS // 2)
    return doc_text[start : start + _DOC_WINDOW_CHARS]


def situate(doc_text: str, chunk_text: str) -> str:
    """
    Generate a 1-2 sentence situating context for chunk_text within doc_text
    using the lightweight memory model.

    Returns the situating sentence on success, or "" on any failure so the
    caller can keep the chunk unmodified.
    """
    from core import llm_client

    if not doc_text.strip() or not chunk_text.strip():
        return ""

    try:
        excerpt = _window_around_chunk(doc_text, chunk_text)
        prompt = (
            f"<document>\n{excerpt}\n</document>\n\n"
            f"<chunk>\n{chunk_text}\n</chunk>\n\n"
            "Situate this chunk within the document. Answer with 1-2 sentences only."
        )
        out = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_SITUATE_SYSTEM,
            model=llm_client.get_memory_model(),
            max_tokens=80,
            temperature=0.2,
            stream=False,
            keep_alive="5m",
        )
        out = " ".join((out or "").split())
        if 0 < len(out) <= _MAX_CONTEXT_CHARS:
            return out
    except Exception as e:
        logger.debug("situate failed: %s", e)

    return ""
