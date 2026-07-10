"""
core/multi_query.py — Multi-Query Expansion (RAG-Fusion)

Generates 2 paraphrases + 1 step-back generalization of a query. Each variant
is retrieved independently (vector + sparse) and the ranked lists are fused
with reciprocal rank fusion in `core.retriever.retrieve()`, then reranked
against the original query.

Targets vocabulary mismatch: a small 384-dim embedder often misses relevant
chunks when the query uses different terminology than the documents — the
paraphrases cover synonyms while the step-back query pulls broader context.

Reference: Rackauckas, "RAG-Fusion: A New Take on Retrieval-Augmented
Generation", arXiv 2402.03367.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Ollama structured-output schema (maps to `format`); ignored on Anthropic.
_VARIANTS_SCHEMA = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}

_MULTI_QUERY_SYSTEM = """You are a search query expander. Given a user's search query, generate exactly 3 alternative queries:
- Two paraphrases that reword the query using different but equivalent vocabulary.
- One step-back query that generalizes to the broader topic or underlying concept.

Rules:
- Return ONLY a JSON array of 3 strings. No explanation, no markdown.
- Keep each variant short and self-contained.
- Preserve exact identifiers (function names, file paths, error codes) verbatim.
- Do not repeat the original query verbatim.

Example input:  "how do I revoke a JWT before it expires?"
Example output: ["how to invalidate a JSON Web Token early?", "can an issued auth token be forcibly expired?", "how does JWT token lifecycle management work?"]"""


def generate_query_variants(query: str, max_variants: int = 3) -> list[str] | None:
    """
    Generate 2 paraphrases + 1 step-back generalization for *query* via LLM.

    Returns the variant queries (original NOT included) on success, or None on
    any failure so the caller can transparently fall back to single-query
    retrieval.

    Model selection mirrors core.decomposer.decompose():
      - Tries the lightweight memory model first (cheaper/faster).
      - If that model is unavailable (404 / not found), retries once with the
        active chat model before giving up.
    """
    from core import llm_client

    def _call(model: str | None) -> str:
        return llm_client.chat(
            messages=[{"role": "user", "content": query.strip()}],
            system=_MULTI_QUERY_SYSTEM,
            model=model,
            max_tokens=200,
            temperature=0.7,
            stream=False,
            json_schema=_VARIANTS_SCHEMA,
        )

    memory_model = llm_client.get_memory_model()
    chat_model = llm_client.get_model()

    try:
        raw = _call(memory_model)
    except Exception as e:
        err_str = str(e).lower()
        unavailable = "not found" in err_str or "404" in err_str or "no such" in err_str
        if unavailable and memory_model != chat_model:
            logger.info(f"Memory model '{memory_model}' unavailable for multi-query, retrying with '{chat_model}'")
            try:
                raw = _call(chat_model)
            except Exception as e2:
                logger.warning(f"Multi-query fallback also failed: {e2}")
                return None
        else:
            logger.warning(f"Multi-query LLM call failed: {e}")
            return None

    variants = _parse_variants(raw)
    if variants is None:
        logger.warning(f"Multi-query parse failed — no JSON array in response: {raw[:120]!r}")
        return None

    # Drop empties, the original query, and duplicates (order-preserving)
    q_norm = query.strip().lower()
    seen: set[str] = set()
    cleaned: list[str] = []
    for v in variants:
        v = v.strip()
        key = v.lower()
        if not v or key == q_norm or key in seen:
            continue
        seen.add(key)
        cleaned.append(v)
    cleaned = cleaned[:max_variants]

    if not cleaned:
        return None

    logger.info(f"Multi-query: {len(cleaned)} variants for: {query[:60]}")
    return cleaned


def _parse_variants(text: str) -> list[str] | None:
    """Strict JSON-array parse. Unlike the decomposer, no line-split last resort:
    a prose reply ("Sure! Here are 3 queries: ...") must fail to None, not become
    junk variants that pollute the fusion."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    for candidate in (text, m.group() if (m := re.search(r"\[.*\]", text, re.DOTALL)) else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [str(s) for s in obj if isinstance(s, str | int | float)]
    return None
