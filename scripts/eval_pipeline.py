"""
scripts/eval_pipeline.py — Tier 3.1 driver: measure embedder/reranker upgrade profiles end-to-end.

For each profile (baseline MiniLM, gte-modernbert embedder, gte embedder + gte reranker,
optionally Qwen3-Embedding) this script:
  1. computes a per-embedder eval collection name eval_<sha256(embedder)[:8]>,
  2. runs scripts/eval_retrieval.py as a subprocess with the profile's env vars,
     ingesting the corpus dirs into the collection on first use (profiles that share
     an embedder share the collection — reranker-only profiles never re-embed),
  3. parses the machine-readable --json line for hit-rate@k / MRR / avg-ms per config.

Results are written to docs/EVAL_RESULTS.md (and printed). A profile whose model
download or run fails is recorded as failed and the remaining profiles still run.

First use of the gte profiles downloads ~600MB of model weights each and CPU-embeds
the corpus — expect minutes per profile; timeouts are generous by default.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_pipeline.py
        [--golden tests/eval/golden_repo.jsonl] [--ingest-dir docs --ingest-dir core --ingest-dir api]
        [--configs vector,baseline,rerank] [--top-k 5] [--with-qwen3] [--timeout 3600]
        [--out docs/EVAL_RESULTS.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_retrieval.py"

BASELINE_EMBEDDER = "all-MiniLM-L6-v2"
BASELINE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GTE_EMBEDDER = "Alibaba-NLP/gte-modernbert-base"
GTE_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
QWEN3_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"

PROFILES: list[dict] = [
    {
        "name": "baseline",
        "env": {"RAG_EMBEDDING_MODEL": BASELINE_EMBEDDER, "RAG_RERANKER_MODEL": BASELINE_RERANKER},
    },
    {
        "name": "gte-embed",
        "env": {"RAG_EMBEDDING_MODEL": GTE_EMBEDDER, "RAG_RERANKER_MODEL": BASELINE_RERANKER},
    },
    {
        "name": "gte-embed+gte-rerank",
        "env": {"RAG_EMBEDDING_MODEL": GTE_EMBEDDER, "RAG_RERANKER_MODEL": GTE_RERANKER},
    },
]

QWEN3_PROFILE: dict = {
    "name": "qwen3-embed",
    "env": {"RAG_EMBEDDING_MODEL": QWEN3_EMBEDDER, "RAG_RERANKER_MODEL": BASELINE_RERANKER},
}


def collection_for(embedder: str) -> str:
    return f"eval_{hashlib.sha256(embedder.encode('utf-8')).hexdigest()[:8]}"


def run_profile(profile: dict, args) -> dict:
    """Run eval_retrieval.py for one profile in a subprocess; returns a result record.

    A subprocess (not an in-process settings mutation) is mandatory: core.retriever
    caches the embedding model, reranker, and per-model embed cache in module-level
    singletons, so switching RAG_EMBEDDING_MODEL/RAG_RERANKER_MODEL inside one process
    would keep serving the previously loaded models.
    """
    embedder = profile["env"]["RAG_EMBEDDING_MODEL"]
    collection = collection_for(embedder)
    env = dict(os.environ)
    env.update(profile["env"])
    # Keep the eval deterministic and self-contained: no SPLADE build attempts
    # (the configured naver/splade-v3 is gated and would fail-load on every start),
    # cross-encoder reranker type, UTF-8 child stdout for reliable parsing on Windows.
    env.update({"RAG_RERANKER_TYPE": "cross-encoder", "RAG_SPLADE_ENABLED": "false", "PYTHONIOENCODING": "utf-8"})

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--golden",
        args.golden,
        "--collection",
        collection,
        "--top-k",
        str(args.top_k),
        "--configs",
        args.configs,
        "--json",
    ]
    for d in args.ingest_dir:
        cmd += ["--ingest-dir", d]

    record = {"profile": profile["name"], "env": profile["env"], "collection": collection}
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        record["error"] = f"timed out after {args.timeout}s"
        return record
    record["elapsed_s"] = time.perf_counter() - start

    payload = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "results" in parsed:
                payload = parsed
                break
    if proc.returncode != 0 or payload is None:
        tail = "\n".join(((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()[-8:])
        record["error"] = f"exit={proc.returncode}, no parseable results. Output tail:\n{tail}"
        return record
    record["data"] = payload
    return record


def _fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_report(records: list[dict], args) -> str:
    ok = [r for r in records if "data" in r]
    failed = [r for r in records if "error" in r]
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Retrieval Eval Results (Tier 3.1)",
        "",
        f"Generated by `scripts/eval_pipeline.py` on {generated}. CPU-only run; each profile",
        "executes `scripts/eval_retrieval.py` in its own subprocess with the profile's",
        "`RAG_EMBEDDING_MODEL` / `RAG_RERANKER_MODEL` env vars (module-level model caches make",
        "in-process switching unsafe). Profiles sharing an embedder share one eval collection,",
        "so reranker-only variants reuse the same embeddings.",
        "",
        "## Corpus & golden set",
        "",
    ]
    if ok:
        d = ok[0]["data"]
        lines += [
            f"- Corpus: {', '.join(f'`{x}/`' for x in args.ingest_dir)} of this repository — "
            f"{d['files']} files, {d['chunks']} chunks (chunking per `config.py` settings).",
            f"- Golden set: `{d['golden']}` — {d['n_questions']} generated questions "
            "(`scripts/build_golden.py`; expected_paths = source chunk's document_path).",
            f"- Metrics: hit-rate@{d['top_k']} and MRR@{d['top_k']} per retrieval config; "
            "avg ms is per-query wall time (includes first-query model warm-up).",
            "",
            "**Latency caveat**: `avg ms` excludes query-embedding cost whenever the per-model",
            "embed cache is warm — repeat questions reduce embedding to a SHA-256 disk-cache",
            "lookup, so cross-profile `avg ms` compares the retrieval+rerank stages, not embedder",
            "cost. The per-profile cold-embed figures below time one deliberately cache-missing",
            "`embed_texts()` call (model already loaded) and are the true per-query embedder cost.",
        ]
    else:
        lines += [f"- Corpus dirs: {', '.join(args.ingest_dir)} (no profile completed — see failures)."]
    lines += ["", "## Profile comparison", ""]

    header = ["profile", "embedder", "reranker", "config", f"hit@{args.top_k}", f"MRR@{args.top_k}", "avg ms"]
    lines.append(_fmt_row(header))
    lines.append(_fmt_row(["---"] * len(header)))
    for r in ok:
        embedder = r["env"]["RAG_EMBEDDING_MODEL"]
        reranker = r["env"]["RAG_RERANKER_MODEL"]
        for cfg, m in r["data"]["results"].items():
            lines.append(
                _fmt_row(
                    [
                        r["profile"],
                        f"`{embedder}`",
                        f"`{reranker}`" if cfg not in ("vector", "baseline") else "—",
                        cfg,
                        f"{m['hit_rate']:.3f}",
                        f"{m['mrr']:.3f}",
                        f"{m['avg_ms']:.0f}",
                    ]
                )
            )
    lines.append("")
    lines.append(
        "Configs: `vector` = dense only; `baseline` = hybrid dense+BM25 RRF; `rerank` = hybrid + cross-encoder."
    )
    lines.append("The reranker column applies to the `rerank` config only — `vector`/`baseline` never call it.")
    lines.append("")

    cold_rows = [r for r in ok if r["data"].get("embed_cold_ms") is not None]
    if cold_rows:
        lines += ["Cold query-embed cost per profile (one cache-missing embed, model already loaded):", ""]
        for r in cold_rows:
            lines.append(
                f"- **{r['profile']}** (`{r['env']['RAG_EMBEDDING_MODEL']}`): {r['data']['embed_cold_ms']:.0f} ms"
            )
        lines.append("")

    if failed:
        lines += ["## Failed profiles", ""]
        for r in failed:
            lines += [f"- **{r['profile']}** (`{r['env']['RAG_EMBEDDING_MODEL']}`): {r['error']}", ""]

    lines += ["## Recommendation", "", _recommendation(ok), ""]
    if not any(r["profile"] == "qwen3-embed" for r in records):
        lines += [
            "`Qwen/Qwen3-Embedding-0.6B` was skipped (run with `--with-qwen3` to include): ~1.2GB download,",
            "5-15x MiniLM per-chunk CPU cost, and it wants a query/document role prefix that",
            "`embed_texts()` does not thread through yet — un-prefixed numbers would understate it.",
            "",
        ]
    return "\n".join(lines)


def _best_config(record: dict) -> tuple[str, dict]:
    return max(record["data"]["results"].items(), key=lambda kv: (kv[1]["hit_rate"], kv[1]["mrr"]))


def _recommendation(ok: list[dict]) -> str:
    if not ok:
        return "No profile completed — fix the failures above and re-run before drawing conclusions."
    baseline = next((r for r in ok if r["profile"] == "baseline"), ok[0])
    winner = max(ok, key=lambda r: _best_config(r)[1]["hit_rate"] + _best_config(r)[1]["mrr"])
    w_cfg, w = _best_config(winner)
    b_cfg, b = _best_config(baseline)
    if winner is baseline:
        return (
            f"On this corpus the baseline profile already wins: `{baseline['env']['RAG_EMBEDDING_MODEL']}` at "
            f"config `{b_cfg}` scores hit@k {b['hit_rate']:.3f} / MRR {b['mrr']:.3f} ({b['avg_ms']:.0f} ms/query). "
            "None of the candidate upgrades earned their extra latency/download — keep the current models and "
            "re-run after the corpus or golden set changes."
        )
    return (
        f"**{winner['profile']}** (`{winner['env']['RAG_EMBEDDING_MODEL']}` + "
        f"`{winner['env']['RAG_RERANKER_MODEL']}`) is the best measured profile: config `{w_cfg}` reaches "
        f"hit@k {w['hit_rate']:.3f} / MRR {w['mrr']:.3f} vs baseline's {b['hit_rate']:.3f} / {b['mrr']:.3f} "
        f"(Δhit {w['hit_rate'] - b['hit_rate']:+.3f}, ΔMRR {w['mrr'] - b['mrr']:+.3f}) at "
        f"{w['avg_ms']:.0f} ms/query vs {b['avg_ms']:.0f} ms. To adopt it, set `RAG_EMBEDDING_MODEL` / "
        "`RAG_RERANKER_MODEL` in `.env` and run `scripts/migrate_embeddings.py` to re-embed existing collections."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measured embedder/reranker upgrade program (Tier 3.1). Runs eval_retrieval.py per profile."
    )
    parser.add_argument("--golden", default="tests/eval/golden_repo.jsonl", help="Golden set JSONL path")
    parser.add_argument(
        "--ingest-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="Corpus directory (repeatable; default: docs, core, api)",
    )
    parser.add_argument(
        "--configs",
        default="vector,baseline,rerank",
        help="eval_retrieval.py configs to run per profile (default: the deterministic, LLM-free trio)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="k for hit-rate@k / MRR@k (default: 5)")
    parser.add_argument(
        "--with-qwen3",
        action="store_true",
        help=f"Also evaluate {QWEN3_EMBEDDER}. Skipped by default: ~1.2GB download, 5-15x MiniLM "
        "per-chunk CPU cost, and best quality needs a query/document role flag that embed_texts() "
        "does not support yet, so its numbers here would be a lower bound.",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Per-profile subprocess timeout in seconds (default: 3600)"
    )
    parser.add_argument("--out", default="docs/EVAL_RESULTS.md", help="Report output path")
    args = parser.parse_args()
    args.ingest_dir = args.ingest_dir or ["docs", "core", "api"]

    # Windows pipes default to cp1252; the report uses em dashes / Δ, so never crash on print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    golden_path = REPO_ROOT / args.golden
    if not golden_path.exists():
        raise SystemExit(
            f"error: golden set not found at {args.golden} — generate it first, e.g.\n"
            f"  .venv\\Scripts\\python.exe scripts/build_golden.py --collection "
            f"{collection_for(BASELINE_EMBEDDER)} " + " ".join(f"--ingest-dir {d}" for d in args.ingest_dir)
        )

    profiles = list(PROFILES)
    if args.with_qwen3:
        profiles.append(QWEN3_PROFILE)

    records = []
    for profile in profiles:
        embedder = profile["env"]["RAG_EMBEDDING_MODEL"]
        print(f"\n=== Profile '{profile['name']}' (embedder={embedder}) -> {collection_for(embedder)} ===")
        print("(first use downloads model weights and CPU-embeds the corpus — this can take minutes)")
        record = run_profile(profile, args)
        records.append(record)
        if "error" in record:
            print(f"FAILED: {record['error']}")
        else:
            print(f"done in {record['elapsed_s']:.0f}s")
            for cfg, m in record["data"]["results"].items():
                print(f"  {cfg:<10} hit@{args.top_k}={m['hit_rate']:.3f} MRR={m['mrr']:.3f} avg_ms={m['avg_ms']:.0f}")
            cold = record["data"].get("embed_cold_ms")
            if cold is not None:
                print(f"  embed_cold_ms={cold:.0f} (cache-missing query embed; warm-cache queries skip this)")

    report = build_report(records, args)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {args.out}\n")
    print(report)
    return 0 if any("data" in r for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
