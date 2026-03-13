"""
core/hyde.py — Hypothetical Document Embeddings (HyDE)

Generates a short hypothetical answer paragraph for a query, then embeds it.
The hypothetical doc embedding is used for vector search instead of the raw
query embedding — placing the search vector closer to real answer documents.

BM25 still uses the original query text (lexical matching is not improved by
paraphrasing). Only the vector search path benefits from HyDE.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels", arXiv 2212.10496.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are a technical documentation assistant. "
    "Write a precise, factual passage of 80-120 words that directly answers the question. "
    "Use domain-specific vocabulary. Do not say 'I' or reference the question. "
    "Write as if you are the relevant section of a technical document."
)


def generate_hypothetical_doc(query: str) -> Optional[str]:
    """
    Use the active LLM (Anthropic or Ollama) to generate a hypothetical answer
    document for the given query.

    Returns the generated passage on success, or None on any failure so the
    caller can transparently fall back to the raw query embedding.
    """
    from core import llm_client

    try:
        doc = llm_client.chat(
            messages=[{"role": "user", "content": query.strip()}],
            system=_HYDE_SYSTEM,
            max_tokens=180,
            temperature=0.3,
            stream=False,
        )
        if doc and len(doc.strip()) > 20:
            logger.info(
                "HyDE: generated hypothetical doc ({} chars) for: {}".format(
                    len(doc), query[:60]
                )
            )
            return doc.strip()
    except Exception as e:
        logger.warning(
            "HyDE generation failed, falling back to raw query: {}".format(e)
        )

    return None
