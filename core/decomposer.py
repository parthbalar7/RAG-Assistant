"""
core/decomposer.py — Query Decomposition

Detects multi-part queries and splits them into atomic sub-queries using a
lightweight LLM call (uses the memory/extraction model — cheap and fast).

Detection heuristics (rule-based, zero cost):
  - Query contains " and " or " also " joining two question fragments
  - More than one question mark
  - Length > 120 chars AND contains a second interrogative word

If detected → one LLM call (max 150 tokens) → returns a list of atomic queries.
Each sub-query is then retrieved independently; results are merged + deduplicated
before being passed to the main LLM as combined context.

Example
-------
Input:  "how does auth work and where are tokens stored and how long do they last?"
Output: [
    "how does authentication work?",
    "where are JWT tokens stored?",
    "what is the token expiry duration?",
]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── detection ─────────────────────────────────────────────────────────────────

# Second interrogative words that signal a new sub-question
_INTERROGATIVES = {"how", "where", "what", "when", "why", "which", "who", "is", "are", "does", "do", "can"}

_AND_RE = re.compile(r'\b(and|also)\b', re.IGNORECASE)


def is_multi_part(query: str) -> bool:
    """
    Fast rule-based check — no LLM call.
    Returns True if the query likely contains more than one distinct question.
    """
    q = query.strip()

    # Explicit multiple question marks
    if q.count("?") > 1:
        return True

    # "and" or "also" linking two fragments that each start with an interrogative
    if _AND_RE.search(q):
        parts = _AND_RE.split(q)
        # parts alternates: [text, connector, text, connector, text, ...]
        # filter to just the text parts (skip connector tokens)
        text_parts = [p.strip() for i, p in enumerate(parts) if i % 2 == 0]
        interrogative_parts = 0
        for part in text_parts:
            first_word = part.split()[0].lower().rstrip("?,:") if part.split() else ""
            if first_word in _INTERROGATIVES:
                interrogative_parts += 1
        if interrogative_parts >= 2:
            return True

    # Long query with a second interrogative appearing mid-sentence
    if len(q) > 120:
        words = q.lower().split()
        # Count interrogatives after the first 5 words
        mid_interrogatives = sum(1 for w in words[5:] if w.rstrip("?,") in _INTERROGATIVES)
        if mid_interrogatives >= 2:
            return True

    return False


# ── LLM decomposition ─────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """You are a query decomposer. Split the user's multi-part question into a list of short, self-contained, atomic sub-questions.

Rules:
- Return ONLY a JSON array of strings. No explanation, no markdown.
- Each sub-question must be answerable independently.
- Keep the original intent and terminology.
- Max 5 sub-questions. If the query is actually a single question, return a 1-element array.
- Each sub-question should end with a question mark.

Example input:  "how does auth work and where are tokens stored and how long do they last?"
Example output: ["How does authentication work?", "Where are JWT tokens stored?", "What is the JWT token expiry duration?"]"""


def decompose(query: str, max_sub_queries: int = 5) -> list[str]:
    """
    Split *query* into atomic sub-queries via LLM.
    Falls back to [query] on any error so the caller always gets a usable list.

    Model selection:
      - Tries the lightweight memory model first (cheaper/faster).
      - If that model is not available (404 / connection error), retries once
        with the active chat model before giving up.
    """
    from core import llm_client

    def _call(model: Optional[str]) -> str:
        return llm_client.chat(
            messages=[{"role": "user", "content": query.strip()}],
            system=_DECOMPOSE_SYSTEM,
            model=model,
            max_tokens=200,
            temperature=0.0,
            stream=False,
        )

    # Try lightweight memory model first, fall back to the active chat model
    memory_model = llm_client.get_memory_model()
    chat_model   = llm_client.get_model()

    try:
        raw = _call(memory_model)
    except Exception as e:
        err_str = str(e).lower()
        # Model not found or unavailable — retry with the main chat model
        if "not found" in err_str or "404" in err_str or "no such" in err_str:
            if memory_model != chat_model:
                logger.info("Memory model '{}' unavailable for decomposition, retrying with '{}'".format(
                    memory_model, chat_model))
                try:
                    raw = _call(chat_model)
                except Exception as e2:
                    logger.warning("Decomposition fallback also failed: {}".format(e2))
                    return [query]
            else:
                logger.warning("Decomposition LLM call failed: {}".format(e))
                return [query]
        else:
            logger.warning("Decomposition LLM call failed: {}".format(e))
            return [query]

    sub_queries = _parse_sub_queries(raw, query)
    # Cap and filter empty strings
    sub_queries = [s.strip() for s in sub_queries if s.strip()][:max_sub_queries]
    if not sub_queries:
        return [query]

    logger.info("Decomposed into {} sub-queries: {}".format(len(sub_queries), sub_queries))
    return sub_queries


def _parse_sub_queries(text: str, fallback: str) -> list[str]:
    """Parse LLM response into a list of strings. Robust to markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(s) for s in obj]
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON array from the text
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return [str(s) for s in obj]
        except json.JSONDecodeError:
            pass

    # Last resort: split on numbered list lines  "1. ..." / "- ..."
    lines = [re.sub(r'^[\d\.\-\*\s]+', '', l).strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 8]
    if lines:
        return lines

    return [fallback]


# ── retrieval merger ──────────────────────────────────────────────────────────

def merge_hits(hits_per_query: list[list[dict]], max_total: int = 12) -> list[dict]:
    """
    Merge and deduplicate hit lists from multiple sub-queries.

    Strategy:
    - Round-robin interleave (take the top hit from each sub-query in turn)
      so every sub-query gets at least one chunk represented.
    - Deduplicate by chunk content prefix (first 80 chars).
    - Cap at max_total.
    """
    seen: set[str] = set()
    merged: list[dict] = []

    # Round-robin across sub-query result lists
    max_len = max((len(h) for h in hits_per_query), default=0)
    for i in range(max_len):
        for hits in hits_per_query:
            if i < len(hits):
                hit = hits[i]
                key = hit.get("content", "")[:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(hit)
                    if len(merged) >= max_total:
                        return merged

    return merged
