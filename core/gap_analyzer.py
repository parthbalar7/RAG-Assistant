"""
core/gap_analyzer.py — Knowledge Gap Detector

Runs after every retrieval. Checks whether the retrieved chunks actually cover
the query well enough. If not, emits a gap signal so the frontend can ask the
user whether to search the web.

Three independent signals (any one is enough to declare a gap):
  1. Score-based  — top-hit cosine score below GAP_SCORE_THRESHOLD
  2. Sparse       — fewer than MIN_HITS chunks returned
  3. Answer-based — LLM's own answer contains "I don't have / cannot find"
                    phrases, catching cases where hits exist but are irrelevant

Short/trivial queries (greetings, single words) are never flagged.

Returns a GapResult with:
  - is_gap (bool)
  - topic   (str)  — the query rephrased as a compact topic label
  - reason  (str)  — human-readable explanation shown in the chat card
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

GAP_SCORE_THRESHOLD = 0.45   # top-hit cosine below this = poor retrieval
MIN_HITS            = 2      # fewer hits than this = sparse coverage
MIN_QUERY_WORDS     = 3      # ignore single-word / greeting queries

# Patterns that are never knowledge gaps (greetings, meta commands, etc.)
_SKIP_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye|quit|exit|clear|help|\?+)$",
    re.IGNORECASE,
)

# Phrases the LLM uses when it genuinely can't find the answer in context.
# Matching any of these in the answer overrides score-based passing.
_NO_ANSWER_PATTERNS = re.compile(
    r"(does not contain|do not contain|no information|not mention|not covered|"
    r"cannot (provide|find|answer|give)|not (found|available) in|"
    r"the context (does not|doesn't)|no (relevant|specific) (information|content|data)|"
    r"i (don't|do not) (have|see|find) (any )?(information|details|data)|"
    r"not (present|included|discussed) in|outside (the scope|my knowledge|the context))",
    re.IGNORECASE,
)


@dataclass
class GapResult:
    is_gap: bool
    topic: str       = ""
    reason: str      = ""
    top_score: float = 0.0


def analyze(query: str, hits: list[dict], answer: str = "") -> GapResult:
    """
    Inspect retrieval hits (and optionally the LLM's answer) to detect gaps.
    No LLM calls — pure heuristics, < 1 ms.

    Args:
        query:  The user's original question.
        hits:   Retrieved chunks from the vector/hybrid search.
        answer: The LLM's generated answer (optional but improves detection).
    """
    q = query.strip()

    # Never flag trivial / greeting queries
    if len(q.split()) < MIN_QUERY_WORDS:
        return GapResult(is_gap=False)
    if _SKIP_PATTERNS.match(q):
        return GapResult(is_gap=False)

    # ── Signal 1: no chunks returned at all ───────────────────────────────────
    if not hits:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason="No relevant chunks were found in your indexed documents.",
            top_score=0.0,
        )

    top_score = float(hits[0].get("rerank_score", hits[0].get("score", 0.0)))

    # ── Signal 2: weak retrieval score ────────────────────────────────────────
    if top_score < GAP_SCORE_THRESHOLD:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason="Best match score was {:.0%} — the indexed documents have limited coverage of this topic.".format(top_score),
            top_score=top_score,
        )

    # ── Signal 3: sparse hits ─────────────────────────────────────────────────
    if len(hits) < MIN_HITS:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason="Only {} chunk(s) matched — coverage looks thin for this topic.".format(len(hits)),
            top_score=top_score,
        )

    # ── Signal 4: LLM explicitly said it couldn't find the answer ────────────
    # Catches the case where chunks exist but are irrelevant (false-positive
    # retrieval). The LLM's own admission is the most reliable signal.
    if answer and _NO_ANSWER_PATTERNS.search(answer):
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason="The assistant couldn't find this topic in the indexed documents.",
            top_score=top_score,
        )

    return GapResult(is_gap=False, top_score=top_score)


def _topic(query: str) -> str:
    """
    Derive a compact topic label from the query.
    Strips interrogative openers so the topic is noun-phrase-like.
    """
    q = query.strip().rstrip("?.,!")
    # Remove common leading interrogative phrases
    q = re.sub(
        r"^(how does|how do|how is|where is|where are|what is|what are|"
        r"why does|why is|explain|show me|find|tell me about|describe)\s+",
        "", q, flags=re.IGNORECASE,
    )
    # Capitalise first letter, limit length
    q = q[:80].strip()
    return q[:1].upper() + q[1:] if q else query[:60]
