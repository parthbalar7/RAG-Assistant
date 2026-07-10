"""
scripts/eval_answers.py — DeepEval answer-quality eval: Faithfulness + ContextualRelevancy.

Answers each golden question through the standard retrieve() -> generate() path, then judges
the answers with DeepEval using the currently configured LLM backend (core.llm_client.chat)
as the judge model. Judge noise is about +/-0.1 — scores are comparative only (before/after
deltas on the same golden set and judge), never absolute quality claims or CI gates.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_answers.py [--golden tests/eval/golden.jsonl]
        [--user default] [--top-k 5] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INSTALL_HINT = (
    "DeepEval is not installed in this environment. Install it with: .venv\\Scripts\\pip.exe install deepeval"
)


def _extract_json(text: str) -> str:
    """Strip code fences and grab the outermost JSON object — safety net for sloppy judges."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _make_judge(deepeval_base_llm_cls):
    from core import llm_client

    class LocalJudge(deepeval_base_llm_cls):
        """DeepEval judge wrapping core.llm_client.chat (Anthropic or Ollama, whichever is active)."""

        def load_model(self):
            return None

        def get_model_name(self) -> str:
            return f"{llm_client.get_backend()}:{llm_client.get_model()}"

        def generate(self, prompt: str, schema=None):
            json_schema = None
            if schema is not None:
                try:
                    json_schema = schema.model_json_schema()
                except Exception:
                    schema = None
            text = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                json_schema=json_schema,
            )
            if schema is None:
                return text
            try:
                return schema.model_validate_json(_extract_json(text))
            except Exception as e:
                # DeepEval catches TypeError from schema-aware generate() and retries via plain
                # generate(prompt) + its own JSON parsing — fail open into that path.
                raise TypeError(f"schema-constrained judge output failed validation: {e}") from e

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema=schema)

    return LocalJudge()


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"error: golden set not found at {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: {path}:{line_no} is not valid JSON: {e}") from e
        if not row.get("question"):
            raise SystemExit(f"error: {path}:{line_no} needs a non-empty 'question'")
        rows.append(row)
    if not rows:
        raise SystemExit(f"error: golden set {path} is empty")
    if any("PLACEHOLDER" in str(row.get("note", "")) or "PLACEHOLDER" in row["question"] for row in rows):
        print(
            "warning: golden set still contains PLACEHOLDER template rows — replace them with real "
            "questions before trusting any scores (see tests/eval/README.md)",
            file=sys.stderr,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepEval answer-quality eval (Faithfulness + ContextualRelevancy).")
    parser.add_argument("--golden", default="tests/eval/golden.jsonl", help="Path to the golden JSONL set")
    parser.add_argument("--user", default="default", help="User id — evaluates against collection docs_<user>")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per question (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions (0 = all)")
    args = parser.parse_args()

    try:
        from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric
        from deepeval.models import DeepEvalBaseLLM
        from deepeval.test_case import LLMTestCase
    except ImportError:
        raise SystemExit(f"error: {INSTALL_HINT}") from None

    rows = load_golden(Path(args.golden))
    if args.limit > 0:
        rows = rows[: args.limit]

    from core.generator import generate
    from core.retriever import VectorStore, retrieve

    collection_name = f"docs_{args.user}"
    store = VectorStore(collection_name=collection_name)
    if store.count == 0:
        raise SystemExit(f"error: collection '{collection_name}' is empty — ingest documents first.")

    judge = _make_judge(DeepEvalBaseLLM)
    # threshold only controls DeepEval's pass/fail flag, which we ignore — scores are what matter
    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=0.0, model=judge, include_reason=False, async_mode=False),
        "ctx_relevancy": ContextualRelevancyMetric(threshold=0.0, model=judge, include_reason=False, async_mode=False),
    }

    print(
        "warning: judge noise is about +/-0.1 — treat these scores as comparative between runs on "
        "the same golden set and judge model, never as absolute quality or a CI gate",
        file=sys.stderr,
    )
    print(f"Golden set: {args.golden} ({len(rows)} questions)")
    print(f"Collection: {collection_name} ({store.count} chunks) | judge: {judge.get_model_name()}\n")

    scores: dict[str, list[float]] = {name: [] for name in metrics}
    header = f"{'#':<4} {'faithfulness':>13} {'ctx_relevancy':>14}  question"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, 1):
        question = row["question"]
        hits = retrieve(store, question, rerank_top_k=args.top_k)
        if not hits:
            print(f"{i:<4} {'—':>13} {'—':>14}  {question[:60]} (no chunks retrieved — skipped)")
            continue
        answer = generate(question, hits[: args.top_k]).answer
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[hit["content"] for hit in hits[: args.top_k]],
        )
        cells = []
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                score = float(metric.score)
                scores[name].append(score)
                cells.append(f"{score:.3f}")
            except Exception as e:
                print(f"warning: {name} failed on question {i}: {e}", file=sys.stderr)
                cells.append("err")
        print(f"{i:<4} {cells[0]:>13} {cells[1]:>14}  {question[:60]}")

    print()
    for name, values in scores.items():
        if values:
            print(f"mean {name}: {sum(values) / len(values):.3f} (n={len(values)})")
        else:
            print(f"mean {name}: no successful measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
