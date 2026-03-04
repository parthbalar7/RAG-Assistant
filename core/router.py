"""
Query Router: classifies incoming queries and routes to optimal retrieval strategy.
Uses fast rule-based matching (no LLM call needed).
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryRoute:
    category: str
    sub_queries: list
    language_hint: str
    retrieval_strategy: str
    suggested_top_k: int
    confidence: float


def route_query_fast(query):
    """Fast rule-based routing (no LLM call)."""
    q = query.lower().strip()

    # Code lookup patterns
    if any(p in q for p in ["show me", "find the", "where is", "what is the function", "get the class"]):
        return QueryRoute("code_lookup", [query], None, "focused", 4, 0.7)

    # Debugging patterns
    if any(p in q for p in ["error", "bug", "fail", "crash", "exception", "not working", "broken"]):
        return QueryRoute("debugging", [query], None, "broad", 8, 0.7)

    # Architecture patterns
    if any(p in q for p in ["structure", "architecture", "overview", "organized", "layout", "project"]):
        return QueryRoute("architecture", [query], None, "broad", 8, 0.7)

    # Configuration patterns
    if any(p in q for p in ["config", "setting", "environment", "env", ".env", "option"]):
        return QueryRoute("configuration", [query], None, "focused", 4, 0.7)

    # Comparison patterns
    if any(p in q for p in ["difference", "compare", "vs ", "versus", "between"]):
        return QueryRoute("comparison", [query], None, "multi", 6, 0.6)

    # Explanation patterns
    if any(p in q for p in ["how does", "explain", "what does", "why does", "how to"]):
        return QueryRoute("explanation", [query], None, "broad", 6, 0.6)

    return QueryRoute("general", [query], None, "broad", 5, 0.3)
