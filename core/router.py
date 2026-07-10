"""
Query Router: classifies incoming queries and routes to optimal retrieval strategy.

Two tiers:
  - route_query_learned(): trained 3-way complexity classifier (direct / single / multi),
    TF-IDF + LinearSVC loaded lazily from settings.router_model_path
    (train with scripts/train_router.py).
  - route_query_fast(): rule-based keyword matching, zero dependencies.

route_query() is the single entry point: learned when a model is available,
heuristic fallback otherwise.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

try:  # graceful degradation: without joblib/sklearn the heuristic router still works
    import joblib
except ImportError:
    joblib = None


@dataclass
class QueryRoute:
    category: str
    sub_queries: list
    language_hint: str
    retrieval_strategy: str
    suggested_top_k: int
    confidence: float
    # False only for learned "direct" routes: answer from memory/history, skip retrieval.
    needs_retrieval: bool = True


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


# ── learned 3-way complexity router ───────────────────────────────────────────

# Learned label -> (retrieval_strategy, suggested_top_k, needs_retrieval).
# "direct" keeps a small nonzero top_k so consumers that ignore needs_retrieval
# still retrieve sanely instead of asking the store for 0 results.
_LEARNED_ROUTES = {
    "direct": ("direct", 3, False),
    "single": ("focused", 5, True),
    "multi": ("multi", 8, True),
}

_learned_model = None
_learned_load_attempted = False


def _get_learned_model():
    """Lazy-load {"vectorizer", "clf"} from settings.router_model_path. Cached; None on any failure."""
    global _learned_model, _learned_load_attempted
    if _learned_load_attempted:
        return _learned_model
    _learned_load_attempted = True

    if joblib is None:
        logger.info("joblib/scikit-learn unavailable — learned router disabled, using heuristics")
        return None
    path = Path(settings.router_model_path)
    if not path.exists():
        logger.info("No trained router at %s — using heuristic routing (train via scripts/train_router.py)", path)
        return None
    try:
        bundle = joblib.load(path)
        if not (isinstance(bundle, dict) and "vectorizer" in bundle and "clf" in bundle):
            raise ValueError("expected dict with 'vectorizer' and 'clf' keys")
        _learned_model = bundle
        logger.info("Loaded learned query router from %s", path)
    except Exception as e:
        logger.warning("Failed to load learned router from %s: %s", path, e)
        _learned_model = None
    return _learned_model


def _margin_confidence(clf, features):
    """Squash the LinearSVC decision margin into (0.5, 1.0) as a pseudo-confidence."""
    try:
        row = clf.decision_function(features)[0]
        try:
            margins = sorted((float(v) for v in row), reverse=True)
        except TypeError:  # binary classifier: row is a scalar margin
            margins = [abs(float(row))]
        margin = margins[0] - margins[1] if len(margins) > 1 else margins[0]
        return round(1.0 / (1.0 + math.exp(-margin)), 3)
    except Exception:
        return 0.6


def route_query_learned(query):
    """
    Classify *query* with the trained 3-way complexity router.

    Label -> QueryRoute mapping (via _LEARNED_ROUTES):
      direct -> category "direct", strategy "direct",  top_k 3, needs_retrieval=False
      single -> category "single", strategy "focused", top_k 5, needs_retrieval=True
      multi  -> category "multi",  strategy "multi",   top_k 8, needs_retrieval=True

    Returns None when no model is available, the label is unknown, or prediction
    fails — callers fall back to route_query_fast().
    """
    model = _get_learned_model()
    if model is None:
        return None
    try:
        features = model["vectorizer"].transform([query])
        label = str(model["clf"].predict(features)[0])
        mapped = _LEARNED_ROUTES.get(label)
        if mapped is None:
            logger.warning("Learned router produced unknown label %r — falling back to heuristics", label)
            return None
        strategy, top_k, needs_retrieval = mapped
        confidence = _margin_confidence(model["clf"], features)
        return QueryRoute(label, [query], None, strategy, top_k, confidence, needs_retrieval)
    except Exception as e:
        logger.warning("Learned router prediction failed: %s", e)
        return None


def route_query(query):
    """Single routing entry point: learned classifier when trained, keyword heuristics otherwise."""
    return route_query_learned(query) or route_query_fast(query)
