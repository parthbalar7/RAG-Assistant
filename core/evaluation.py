"""Lightweight RAG evaluation framework."""

import json
import logging
from dataclasses import dataclass

import anthropic
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    query: str
    retrieval_hit: bool
    mrr: float
    faithfulness_score: float
    relevance_score: float
    details: str = ""


JUDGE_PROMPT = """Score this RAG response (0.0-1.0).
Context: {context}
Question: {query}
Answer: {answer}
Respond ONLY with JSON: {{"faithfulness": <float>, "relevance": <float>, "reasoning": "<brief>"}}"""


def evaluate_response(query, answer, context_chunks, expected_sources=None):
    retrieval_hit = True
    mrr = 1.0

    if expected_sources:
        sources = [c["metadata"].get("document_path", "") for c in context_chunks]
        retrieval_hit = any(e in sources for e in expected_sources)
        mrr = next((1.0 / (i + 1) for i, s in enumerate(sources) if s in expected_sources), 0.0)

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        ctx = "\n---\n".join(c["content"] for c in context_chunks[:4])
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                context=ctx[:3000], query=query, answer=answer[:2000])}]
        )
        text = resp.content[0].text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        return EvalResult(
            query, retrieval_hit, mrr,
            float(scores.get("faithfulness", 0)),
            float(scores.get("relevance", 0)),
            scores.get("reasoning", "")
        )
    except Exception as e:
        return EvalResult(query, retrieval_hit, mrr, -1.0, -1.0, str(e))


def run_eval_suite(test_cases, query_fn):
    results = []
    for tc in test_cases:
        answer, hits = query_fn(tc["query"])
        result = evaluate_response(tc["query"], answer, hits, tc.get("expected_sources"))
        results.append(result)

    n = len(results)
    if n == 0:
        return {"error": "No cases"}

    vf = [r.faithfulness_score for r in results if r.faithfulness_score >= 0]
    vr = [r.relevance_score for r in results if r.relevance_score >= 0]

    return {
        "total_cases": n,
        "retrieval_hit_rate": sum(r.retrieval_hit for r in results) / n,
        "avg_mrr": sum(r.mrr for r in results) / n,
        "avg_faithfulness": sum(vf) / len(vf) if vf else -1,
        "avg_relevance": sum(vr) / len(vr) if vr else -1,
    }
