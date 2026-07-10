"""
scripts/train_router.py — Train the 3-way query-complexity router (direct / single / multi).

Training data sources (combinable; hand labels win on duplicate queries):
  --csv PATH   CSV with query,label rows (label in {direct, single, multi})
  --from-db    user queries from the SQLite `messages` table, weak-labeled with
               the current rule-based heuristics so a zero-hand-label bootstrap exists:
                 is_multi_part(query)                          -> multi
                 < 4 words and no interrogative word or "?"    -> direct
                 otherwise                                     -> single

A small built-in seed set (disable with --no-seed) guarantees every class is
represented even on an empty database. Trains TfidfVectorizer(ngram_range=(1,2))
+ LinearSVC, prints a cross-validated classification report, and saves
{"vectorizer", "clf"} to settings.router_model_path via joblib.dump — loaded
lazily by core.router.route_query_learned().

Usage:
    .venv\\Scripts\\python.exe scripts/train_router.py --from-db
    .venv\\Scripts\\python.exe scripts/train_router.py --csv data/router_labels.csv --from-db
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = ("direct", "single", "multi")

# Hand-written seed examples so all three classes exist before any real usage data.
SEED_EXAMPLES = [
    # direct: greetings / acknowledgements / conversation-only follow-ups — no retrieval needed
    ("hi", "direct"),
    ("hello there", "direct"),
    ("thanks", "direct"),
    ("thank you!", "direct"),
    ("ok sounds good", "direct"),
    ("yes please", "direct"),
    ("no thanks", "direct"),
    ("got it", "direct"),
    ("never mind", "direct"),
    ("good morning", "direct"),
    ("bye", "direct"),
    ("continue", "direct"),
    ("try again", "direct"),
    ("summarize that", "direct"),
    ("make it shorter", "direct"),
    # single: one self-contained question -> one retrieval pass
    ("how does authentication work?", "single"),
    ("where are JWT tokens stored?", "single"),
    ("what does the retriever module do?", "single"),
    ("explain the memory extraction pipeline", "single"),
    ("show me the websocket handler", "single"),
    ("what is the default chunk size?", "single"),
    ("how do I run the backend locally?", "single"),
    ("which model is used for reranking?", "single"),
    ("what does RRF merge do?", "single"),
    ("how is the query cache invalidated?", "single"),
    ("where is the SPLADE configuration?", "single"),
    ("what port does the API listen on?", "single"),
    ("how do I enable HyDE?", "single"),
    ("describe the document ingestion flow", "single"),
    ("what happens when a knowledge gap is detected?", "single"),
    # multi: multi-part questions -> decompose + retrieve per sub-query
    ("how does auth work and where are tokens stored?", "multi"),
    ("compare BM25 and SPLADE and explain when to use each", "multi"),
    ("what is HyDE and how does it interact with the semantic cache?", "multi"),
    ("explain ingestion and how chunking and embedding fit together", "multi"),
    ("where is memory stored and how is it retrieved and when is it summarized?", "multi"),
    ("what's the difference between the agent path and the standard RAG path, and which is faster?", "multi"),
    ("how do I configure Ollama and what models are recommended?", "multi"),
    ("how does gap detection work and what happens after the user approves web search?", "multi"),
    ("what does the tree indexer do and how does tree search use it?", "multi"),
    ("how are sessions stored and how are messages persisted and what triggers renaming?", "multi"),
    ("explain RRF merging and cross-encoder reranking and how they interact", "multi"),
    ("what is SPLADE and how does it differ from BM25 and when does it fall back?", "multi"),
    ("how does the router work and when does query decomposition trigger?", "multi"),
    ("where are embeddings cached and how is the cache invalidated and what keys are used?", "multi"),
    ("list the websocket payload flags and what each one does and which are on by default", "multi"),
]


def weak_label(query: str) -> str:
    """Bootstrap label from the current heuristics: multi > direct > single."""
    from core.decomposer import _INTERROGATIVES, is_multi_part

    if is_multi_part(query):
        return "multi"
    words = query.split()
    interrogative = "?" in query or any(w.lower().strip("?,.:;!") in _INTERROGATIVES for w in words)
    if len(words) < 4 and not interrogative:
        return "direct"
    return "single"


def load_csv(path: str) -> list[tuple[str, str]]:
    pairs = []
    skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            query, label = row[0].strip(), row[1].strip().lower()
            if query and label in LABELS:
                pairs.append((query, label))
            else:
                skipped += 1  # header row or unknown label
    if skipped:
        print(f"  {path}: skipped {skipped} row(s) without a valid direct/single/multi label")
    return pairs


def load_db_queries(db_path: str) -> list[tuple[str, str]]:
    if not Path(db_path).exists():
        print(f"  no database at {db_path} — skipping --from-db")
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT content FROM messages WHERE role = 'user'").fetchall()
    finally:
        conn.close()
    pairs = []
    for (content,) in rows:
        query = " ".join(str(content or "").split())
        if 1 <= len(query) <= 500:
            pairs.append((query, weak_label(query)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the 3-way query-complexity router (direct/single/multi).")
    parser.add_argument("--csv", default=None, help="CSV file with query,label rows (label: direct|single|multi)")
    parser.add_argument("--from-db", action="store_true", help="Weak-label user queries from the SQLite messages table")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (default: settings.database_path)")
    parser.add_argument("--out", default=None, help="Output model path (default: settings.router_model_path)")
    parser.add_argument("--no-seed", action="store_true", help="Exclude the built-in seed examples")
    args = parser.parse_args()

    if not args.csv and not args.from_db:
        parser.error("provide --csv and/or --from-db")

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report
        from sklearn.model_selection import cross_val_predict
        from sklearn.svm import LinearSVC
    except ImportError as e:
        print(f"scikit-learn and joblib are required: pip install scikit-learn joblib ({e})")
        return 1

    from config import settings

    print("Collecting training data...")
    # Later sources override earlier ones on duplicate queries: weak DB labels
    # < seed labels < hand labels from --csv.
    by_query: dict[str, tuple[str, str]] = {}

    def _add(pairs: list[tuple[str, str]]) -> None:
        for query, label in pairs:
            by_query[query.lower()] = (query, label)

    if args.from_db:
        db_pairs = load_db_queries(args.db_path or settings.database_path)
        print(f"  {len(db_pairs)} weak-labeled queries from the messages table")
        _add(db_pairs)
    if not args.no_seed:
        _add(SEED_EXAMPLES)
        print(f"  {len(SEED_EXAMPLES)} built-in seed examples")
    if args.csv:
        csv_pairs = load_csv(args.csv)
        print(f"  {len(csv_pairs)} hand-labeled queries from {args.csv}")
        _add(csv_pairs)

    queries = [q for q, _ in by_query.values()]
    labels = [label for _, label in by_query.values()]
    class_counts = Counter(labels)
    print(
        f"\n{len(queries)} unique examples: " + ", ".join(f"{k}={class_counts[k]}" for k in LABELS if k in class_counts)
    )

    if len(class_counts) < 2:
        print("Need at least 2 classes to train — add labeled examples or drop --no-seed.")
        return 1

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform(queries)
    # balanced: weak-labeled real traffic is dominated by "single"
    clf = LinearSVC(class_weight="balanced")

    folds = min(5, min(class_counts.values()))
    if folds >= 2:
        print(f"\nCross-validated classification report ({folds}-fold):")
        predicted = cross_val_predict(LinearSVC(class_weight="balanced"), features, labels, cv=folds)
        print(classification_report(labels, predicted, zero_division=0))
    else:
        print("\nToo few samples in the smallest class for cross-validation — skipping report.")

    clf.fit(features, labels)

    out_path = Path(args.out or settings.router_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "clf": clf,
            "labels": sorted(class_counts),
            "n_samples": len(queries),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        out_path,
    )
    print(f"Saved router model to {out_path} — core.router.route_query() picks it up on next server start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
