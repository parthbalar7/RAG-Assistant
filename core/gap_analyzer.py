"""
core/gap_analyzer.py — Knowledge Gap Detector

Runs around every retrieval. Checks whether the retrieved chunks actually
cover the query well enough. If not, emits a gap signal so the frontend can
ask the user whether to search the web.

Five independent signals (any one is enough to declare a gap):
  1. No hits      — nothing returned from retrieval
  2. Score-based  — top-hit rerank/cosine score below GAP_SCORE_THRESHOLD
  3. Sparse       — fewer than MIN_HITS chunks returned
  4. Answer-based — LLM's own answer contains "I don't have / cannot find"
                    phrases, catching cases where hits exist but are irrelevant
  5. Groundedness — provenance map shows the answer is mostly novel text with
                    little document coverage (confident hallucinations that
                    never admit "I don't know")

Signals 1-3 are pure retrieval heuristics available before generation —
`analyze_pre()` exposes them for a CRAG-style pre-generation gate. `analyze()`
runs all five post-generation.

Short/trivial queries (greetings, single words) are never flagged.

Returns a GapResult with:
  - is_gap (bool)
  - topic   (str)  — the query rephrased as a compact topic label
  - reason  (str)  — human-readable explanation shown in the chat card
  - stage   (str)  — "pre" (before generation) or "post" (after)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

# rerank() now sigmoid-normalises CrossEncoder logits into (0,1), so this is a
# probability-like relevance score, not a raw cosine — recalibrated from 0.45.
GAP_SCORE_THRESHOLD = 0.55  # top-hit score below this = poor retrieval
MIN_HITS = 2  # fewer hits than this = sparse coverage
MIN_QUERY_WORDS = 3  # ignore single-word / greeting queries

# Signal 5 (groundedness) thresholds over the provenance map
NOVEL_FRACTION_THRESHOLD = 0.5  # more than half the sentences are orphans
DOC_COVERAGE_THRESHOLD = 0.3  # fewer than 30% of sentences are sourced

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
    topic: str = ""
    reason: str = ""
    top_score: float = 0.0
    stage: str = "post"  # "pre" | "post" — which analysis phase produced this


def _is_trivial(q: str) -> bool:
    """Trivial / greeting queries are never flagged as gaps."""
    return len(q.split()) < MIN_QUERY_WORDS or bool(_SKIP_PATTERNS.match(q))


def analyze_pre(query: str, hits: list[dict]) -> GapResult:
    """
    Pre-generation gate: Signals 1-3 only (no hits / low score / sparse).
    Pure heuristics over the retrieval hits, < 1 ms — safe to call before the
    generation stream starts.
    """
    q = query.strip()

    if _is_trivial(q):
        return GapResult(is_gap=False, stage="pre")

    # ── Signal 1: no chunks returned at all ───────────────────────────────────
    if not hits:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason="No relevant chunks were found in your indexed documents.",
            top_score=0.0,
            stage="pre",
        )

    _rs = hits[0].get("rerank_score") or hits[0].get("score") or 0.0
    top_score = float(_rs)

    # ── Signal 2: weak retrieval score ────────────────────────────────────────
    # Only meaningful when a sigmoid-normalised rerank_score exists: without
    # reranking, hits carry RRF rank-sum scores (~0.016 max) that would trip
    # the probability-scale threshold on every query.
    if hits[0].get("rerank_score") is not None and top_score < GAP_SCORE_THRESHOLD:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason=f"Best match score was {top_score:.0%} — the indexed documents have limited coverage of this topic.",
            top_score=top_score,
            stage="pre",
        )

    # ── Signal 3: sparse hits ─────────────────────────────────────────────────
    if len(hits) < MIN_HITS:
        return GapResult(
            is_gap=True,
            topic=_topic(q),
            reason=f"Only {len(hits)} chunk(s) matched — coverage looks thin for this topic.",
            top_score=top_score,
            stage="pre",
        )

    return GapResult(is_gap=False, top_score=top_score, stage="pre")


def analyze(query: str, hits: list[dict], answer: str = "", provenance=None) -> GapResult:
    """
    Full post-generation analysis: Signals 1-3 (via analyze_pre) plus the
    answer-phrase check (Signal 4) and provenance groundedness (Signal 5).
    No LLM calls — pure heuristics, < 1 ms.

    Args:
        query:      The user's original question.
        hits:       Retrieved chunks from the vector/hybrid search.
        answer:     The LLM's generated answer (optional but improves detection).
        provenance: Optional ProvenanceMap (or its to_dict()) from
                    core.provenance.compute_provenance.
    """
    result = analyze_pre(query, hits)
    result.stage = "post"
    if result.is_gap:
        return result

    q = query.strip()
    if _is_trivial(q):
        return result
    top_score = result.top_score

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

    # ── Signal 5: answer mostly unsupported by the retrieved context ─────────
    # Provenance sentence-splitting handles code poorly, so skip code-heavy answers.
    if provenance is not None and answer and "```" not in answer:
        try:
            novel_fraction = float(_prov_field(provenance, "novel_fraction", 0.0))
            doc_coverage = float(_prov_field(provenance, "doc_coverage", 1.0))
            if novel_fraction > NOVEL_FRACTION_THRESHOLD and doc_coverage < DOC_COVERAGE_THRESHOLD:
                return GapResult(
                    is_gap=True,
                    topic=_topic(q),
                    reason="answer mostly unsupported by retrieved context",
                    top_score=top_score,
                )
        except (TypeError, ValueError) as e:
            logger.warning(f"Gap Signal 5 skipped — malformed provenance: {e}")

    return GapResult(is_gap=False, top_score=top_score)


def _prov_field(provenance, name: str, default: float) -> float:
    """Read a field from a ProvenanceMap instance or its to_dict() form."""
    if isinstance(provenance, dict):
        value = provenance.get(name, default)
    else:
        value = getattr(provenance, name, default)
    return default if value is None else value


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
        "",
        q,
        flags=re.IGNORECASE,
    )
    # Capitalise first letter, limit length
    q = q[:80].strip()
    return q[:1].upper() + q[1:] if q else query[:60]
