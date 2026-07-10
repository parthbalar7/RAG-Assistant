"""
scripts/eval_retrieval.py — Deterministic retrieval eval: hit-rate@k and MRR per toggle combo.

Runs core.retriever.retrieve() directly against a user's vector store for each retrieval
config in the matrix and scores the results against tests/eval/golden.jsonl. A hit is any
returned chunk whose metadata document_path matches any of the row's expected_paths.
No judge LLM — metrics are pure rank math (hyde/multiquery configs do call the configured
LLM for query *expansion*, which makes those two rows slower and not perfectly repeatable).

Usage:
    .venv\\Scripts\\python.exe scripts/eval_retrieval.py [--golden tests/eval/golden.jsonl]
        [--user default] [--top-k 5] [--configs baseline,rerank,splade | all]
        [--collection eval_ab12cd34] [--ingest-dir docs --ingest-dir core] [--json]

--collection overrides the docs_<user> naming (used by scripts/eval_pipeline.py for
per-embedder eval collections). --ingest-dir (repeatable) populates an empty target
collection from local directories first; document_path values are prefixed with each
directory's basename ("core/retriever.py") so files from different roots never collide.
--json appends one machine-readable JSON line with the metric table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Sidecar recording the chunk count of each successful eval-collection ingest.
# A timeout-killed, half-embedded collection is non-empty and would otherwise be
# silently reused, poisoning every later profile comparison. Applies only to
# explicit --collection targets — user docs_* collections are never touched.
SIDECAR_PATH = REPO_ROOT / "data" / "eval_collections.json"

# Matrix of flag combos actually supported by core.retriever.retrieve()
CONFIGS: dict[str, dict] = {
    "vector": {"use_hybrid": False, "use_reranking": False},
    "baseline": {"use_hybrid": True, "use_reranking": False},
    "rerank": {"use_hybrid": True, "use_reranking": True},
    "hyde": {"use_hybrid": True, "use_reranking": True, "use_hyde": True},
    "splade": {"use_hybrid": True, "use_reranking": True, "use_splade": True},
    "multiquery": {"use_hybrid": True, "use_reranking": True, "use_multiquery": True},
}
LLM_EXPANSION_CONFIGS = {"hyde", "multiquery"}


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
        if not row.get("question") or not row.get("expected_paths"):
            raise SystemExit(f"error: {path}:{line_no} needs non-empty 'question' and 'expected_paths'")
        rows.append(row)
    if not rows:
        raise SystemExit(f"error: golden set {path} is empty")
    if any("PLACEHOLDER" in str(row.get("note", "")) or "PLACEHOLDER" in row["question"] for row in rows):
        print(
            "warning: golden set still contains PLACEHOLDER template rows — numbers below are "
            "meaningless until you replace them with real questions (see tests/eval/README.md)",
            file=sys.stderr,
        )
    return rows


def ingest_dirs(store, dirs: list[str]) -> int:
    """Ingest each directory into *store* (embedding with the configured model).

    Filepaths are prefixed with the directory's basename so chunks from different
    roots keep unambiguous, collision-free document_path values. Returns the number
    of chunks added.
    """
    from config import settings
    from core.ingestion import chunk_document, load_documents

    all_chunks = []
    for directory in dirs:
        d = Path(directory)
        prefix = d.name or d.resolve().name
        docs = load_documents(d)
        n_before = len(all_chunks)
        for doc in docs:
            doc.filepath = f"{prefix}/{doc.filepath}"
            all_chunks.extend(chunk_document(doc))
        print(f"Ingesting {directory}: {len(docs)} files -> {len(all_chunks) - n_before} chunks")
    if not all_chunks:
        raise SystemExit(f"error: no ingestable documents found under {', '.join(dirs)}")
    print(f"Embedding {len(all_chunks)} chunks with '{settings.embedding_model}' ...")
    added = store.add_chunks(all_chunks)
    # Sparse (BM25/SPLADE) rebuilds are debounced into a background thread; block
    # until they finish so the first eval queries don't hit an empty sparse index.
    if hasattr(store, "flush_index_rebuild"):
        store.flush_index_rebuild()
    print(f"Ingested {added} chunks into '{store.collection.name}'")
    return added


def _sidecar_load() -> dict:
    try:
        data = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _sidecar_record(collection: str, count: int) -> None:
    data = _sidecar_load()
    data[collection] = count
    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").strip().strip("/").lower()


def path_matches(document_path: str, expected_paths: list[str]) -> bool:
    """Slash-normalized, case-insensitive match tolerating prefix differences on either side."""
    doc = _norm_path(document_path or "")
    if not doc:
        return False
    for expected in expected_paths:
        exp = _norm_path(str(expected))
        if not exp:
            continue
        if doc == exp or doc.endswith("/" + exp) or exp.endswith("/" + doc):
            return True
    return False


def evaluate_config(store, rows: list[dict], flags: dict, top_k: int) -> dict:
    from config import settings
    from core.retriever import retrieve

    # Candidate pool sizing: core.retriever.rerank() returns hits UNRERANKED when
    # len(hits) <= rerank_top_k, so a pool of max(settings.top_k, top_k) silently
    # skips reranking for any --top-k >= settings.top_k — and even below that the
    # reranker gets almost no headroom to demonstrate recall gains. top_k * 4
    # guarantees a real candidate set and no early-out for any sane k.
    pool_k = max(settings.top_k, top_k * 4)

    hits_at_k = 0
    rr_sum = 0.0
    elapsed = 0.0
    for row in rows:
        start = time.perf_counter()
        hits = retrieve(store, row["question"], top_k=pool_k, rerank_top_k=top_k, **flags)
        elapsed += time.perf_counter() - start
        rank = None
        for i, hit in enumerate(hits[:top_k], 1):
            meta = hit.get("metadata") or {}
            if path_matches(meta.get("document_path", ""), row["expected_paths"]):
                rank = i
                break
        if rank is not None:
            hits_at_k += 1
            rr_sum += 1.0 / rank
    n = len(rows)
    return {"hit_rate": hits_at_k / n, "mrr": rr_sum / n, "avg_ms": elapsed / n * 1000}


def measure_cold_embed_ms() -> float:
    """Time one guaranteed-cache-miss embed_texts() call (model already warm).

    With a warm per-model embed cache, every eval query's embedding reduces to a
    SHA-256 lookup — avg_ms above therefore excludes embedder cost. This isolates
    the true per-query embedding cost by embedding a string that cannot be cached.
    """
    import uuid

    from core.retriever import embed_texts

    probe = f"cold embed timing probe {uuid.uuid4()}"
    start = time.perf_counter()
    embed_texts([probe], persist=False)
    return (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic retrieval eval: hit-rate@k and MRR per toggle combo.")
    parser.add_argument("--golden", default="tests/eval/golden.jsonl", help="Path to the golden JSONL set")
    parser.add_argument("--user", default="default", help="User id — evaluates against collection docs_<user>")
    parser.add_argument("--top-k", type=int, default=5, help="k for hit-rate@k / MRR@k (default: 5)")
    parser.add_argument(
        "--configs",
        default="all",
        help=f"Comma-separated subset of: {', '.join(CONFIGS)} (default: all)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Evaluate this exact collection instead of docs_<user> (disables the single-collection fallback)",
    )
    parser.add_argument(
        "--ingest-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="When the target collection is empty, ingest this directory into it first (repeatable)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print one machine-readable JSON line with the results (parsed by scripts/eval_pipeline.py)",
    )
    args = parser.parse_args()

    if args.configs.strip().lower() == "all":
        selected = list(CONFIGS)
    else:
        selected = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in selected if c not in CONFIGS]
        if unknown:
            raise SystemExit(f"error: unknown config(s) {unknown}; choose from: {', '.join(CONFIGS)}")

    rows = load_golden(Path(args.golden))

    from config import settings
    from core.retriever import VectorStore

    collection_name = args.collection or f"docs_{args.user}"
    store = VectorStore(collection_name=collection_name)
    if store.count == 0 and args.ingest_dir:
        ingest_dirs(store, args.ingest_dir)
        if args.collection:
            _sidecar_record(collection_name, store.count)
    elif args.collection and store.count > 0:
        # Completeness check for explicit eval collections: any non-empty collection
        # would otherwise skip ingest, so a half-embedded one (e.g. a timed-out
        # earlier run) silently poisons every later comparison. docs_* user
        # collections never go through this branch.
        expected = _sidecar_load().get(collection_name)
        if expected == store.count:
            if args.ingest_dir:
                print(f"note: collection '{collection_name}' already has {store.count} chunks — skipping ingest")
        else:
            reason = (
                "no recorded ingest count in sidecar"
                if expected is None
                else f"sidecar records {expected} chunks, store has {store.count}"
            )
            if args.ingest_dir:
                print(
                    f"warning: collection '{collection_name}' may be partially ingested ({reason}) — "
                    "deleting and re-ingesting",
                    file=sys.stderr,
                )
                store.clear()
                ingest_dirs(store, args.ingest_dir)
                _sidecar_record(collection_name, store.count)
            else:
                print(
                    f"warning: collection '{collection_name}' has {store.count} chunks but {reason}; "
                    "results may reflect a partial ingest — pass --ingest-dir to force a re-ingest",
                    file=sys.stderr,
                )
    elif args.ingest_dir:
        print(f"note: collection '{collection_name}' already has {store.count} chunks — skipping ingest")
    if store.count == 0:
        if args.collection:
            raise SystemExit(f"error: collection '{collection_name}' is empty — pass --ingest-dir to populate it.")
        names = sorted(c if isinstance(c, str) else c.name for c in store.client.list_collections())
        docs_collections = [n for n in names if n.startswith("docs_") and n != collection_name]
        # User ids are uuid4()[:8], so the 'default' default rarely exists — when
        # exactly one real user collection does, use it instead of erroring.
        if len(docs_collections) == 1:
            collection_name = docs_collections[0]
            print(f"note: 'docs_{args.user}' is empty — using the only populated collection '{collection_name}'")
            store = VectorStore(collection_name=collection_name)
        if store.count == 0:
            hint = f" Existing user collections: {', '.join(docs_collections)}" if docs_collections else ""
            raise SystemExit(f"error: collection '{collection_name}' is empty — ingest documents first.{hint}")

    if "splade" in selected and getattr(store, "splade_index", None) is None:
        print(
            "warning: SPLADE index unavailable (RAG_SPLADE_ENABLED=false or model load failed) — "
            "the 'splade' config degrades to BM25 and will match 'rerank'",
            file=sys.stderr,
        )
    llm_selected = sorted(LLM_EXPANSION_CONFIGS & set(selected))
    if llm_selected:
        print(
            f"note: config(s) {', '.join(llm_selected)} call the configured LLM for query expansion — "
            "slower and not perfectly repeatable run-to-run",
            file=sys.stderr,
        )

    print(f"Golden set: {args.golden} ({len(rows)} questions)")
    print(
        f"Collection: {collection_name} ({store.count} chunks) | embedder: {settings.embedding_model} | k={args.top_k}\n"
    )

    results = {}
    for name in selected:
        print(f"Running config '{name}' ...", file=sys.stderr)
        results[name] = evaluate_config(store, rows, CONFIGS[name], args.top_k)

    # Measured after the configs so the embedding model is loaded: this times the
    # embedder alone, not its one-off weight load.
    embed_cold_ms = measure_cold_embed_ms()

    header = f"{'config':<12} {'hit@' + str(args.top_k):>8} {'MRR@' + str(args.top_k):>8} {'avg ms':>8}"
    print(header)
    print("-" * len(header))
    for name in selected:
        r = results[name]
        print(f"{name:<12} {r['hit_rate']:>8.3f} {r['mrr']:>8.3f} {r['avg_ms']:>8.0f}")
    print(f"\ncold query embed (cache miss): {embed_cold_ms:.0f} ms — warm-cache queries above skip this cost")

    if args.json:
        print(
            json.dumps(
                {
                    "collection": collection_name,
                    "chunks": store.count,
                    "files": len(store.get_all_files()),
                    "embedder": settings.embedding_model,
                    "reranker": settings.reranker_model,
                    "golden": args.golden,
                    "n_questions": len(rows),
                    "top_k": args.top_k,
                    "embed_cold_ms": embed_cold_ms,
                    "results": results,
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
