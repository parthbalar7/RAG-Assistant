"""
scripts/build_golden.py — Generate a golden retrieval-eval set from an ingested collection.

Samples N chunks spread across distinct document_path values (preferring prose/markdown
chunks, mixing in some code) and asks the configured LLM for one specific, self-contained
question per chunk that the chunk answers. expected_paths is the sampled chunk's own
document_path, so the resulting JSONL plugs straight into scripts/eval_retrieval.py.

When the LLM is unreachable (or a single call fails) the script degrades to deterministic
template questions built from heading_path / file names — it never hard-fails, but template
rows are far weaker signal than LLM rows, so re-run with Ollama up when possible.

Usage:
    .venv\\Scripts\\python.exe scripts/build_golden.py --collection eval_ab12cd34
        [--out tests/eval/golden_repo.jsonl] [--n 20] [--model qwen3.5:9b]
        [--ingest-dir docs --ingest-dir core]  # populate the collection first if empty
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import: eval_retrieval

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

_GEN_SYSTEM = (
    "You write retrieval-benchmark questions for a project's documentation and source code. "
    "Given one excerpt, produce ONE specific question a developer using this project would ask "
    "that THIS excerpt answers. The question must stand alone: never mention 'the chunk', 'the "
    "context', 'this excerpt', 'the snippet', or 'the file above', and never mention file names, "
    "file paths, or line numbers. Ask about the CONCEPT or BEHAVIOR the excerpt covers — never "
    "use the module's own name or the words it is built from (listed as banned words in the "
    "prompt); paraphrase what the code or doc does instead. Prefer questions about concrete "
    "behavior, configuration values, function purposes, or design decisions unique to the excerpt. "
    'Respond with ONLY a JSON object of the form {"question": "..."} — no markdown, no commentary.'
)

# Chunk-body sizes worth asking about: too short has no content, too long is unfocused.
_MIN_CHUNK_CHARS = 200
_MAX_PROMPT_CHARS = 2400
# Sample extra candidates so dedup/filter losses still leave N questions.
_OVERSAMPLE = 2
# Normalized-token Jaccard above this counts as a duplicate question.
_DEDUP_JACCARD = 0.8

_WORD_RE = re.compile(r"[a-z0-9]+")

# Stem tokens too generic to count as answer-key leaks (a question naturally says
# "server" or "config" without pointing at any one file).
_GENERIC_STEM_TOKENS = {
    "core",
    "api",
    "apis",
    "app",
    "base",
    "client",
    "code",
    "common",
    "config",
    "data",
    "db",
    "doc",
    "docs",
    "file",
    "files",
    "helper",
    "helpers",
    "index",
    "init",
    "lib",
    "main",
    "misc",
    "model",
    "models",
    "server",
    "settings",
    "src",
    "test",
    "tests",
    "types",
    "util",
    "utils",
    "utilities",
}


def _norm_tokens(question: str) -> set[str]:
    return set(_WORD_RE.findall(question.lower()))


def _stem_tokens(path: str) -> list[str]:
    """Path stem -> lowercase word tokens ('gap_analyzer' / 'GapAnalyzer' -> ['gap', 'analyzer']).

    Package __init__ files take the parent directory's name — that IS the module name
    a question could leak ('core/integrity/__init__.py' -> ['integrity']).
    """
    p = Path(path)
    stem = p.stem
    if stem.strip("_") == "init" and p.parent.name:
        stem = p.parent.name
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)
    return _WORD_RE.findall(stem.lower())


def _leaks_target(question: str, path: str) -> bool:
    """True when the question leaks its target file's name into the query text.

    Signal 1: the stem's token sequence appears as a contiguous phrase — stem
    'gap_analyzer' leaks as 'gap analyzer', 'gap-analyzer', or joined
    'gapanalyzer' (single-token stems leak when the token itself appears and is
    informative). Signal 2: >=2 distinct informative stem tokens (len>=4,
    non-generic) appear anywhere in the question. Either signal hands sparse
    retrieval the answer key, so both reject.
    """
    stem_tokens = _stem_tokens(path)
    if not stem_tokens:
        return False
    q_tokens = _WORD_RE.findall(question.lower())
    informative = {t for t in stem_tokens if len(t) >= 4 and t not in _GENERIC_STEM_TOKENS}

    if len(stem_tokens) >= 2:
        n = len(stem_tokens)
        if any(q_tokens[i : i + n] == stem_tokens for i in range(len(q_tokens) - n + 1)):
            return True
        if "".join(stem_tokens) in q_tokens:
            return True
    elif informative and stem_tokens[0] in q_tokens:
        return True

    return len(informative & set(q_tokens)) >= 2


def _is_duplicate(question: str, accepted: list[str]) -> bool:
    tokens = _norm_tokens(question)
    if not tokens:
        return True
    for prev in accepted:
        prev_tokens = _norm_tokens(prev)
        union = tokens | prev_tokens
        if union and len(tokens & prev_tokens) / len(union) >= _DEDUP_JACCARD:
            return True
    return False


def _template_question(path: str, meta: dict) -> str:
    """Deterministic fallback when the LLM is unreachable or a call fails."""
    heading = (meta.get("heading_path") or "").split(" > ")[-1].strip()
    name = Path(path).name
    if heading:
        return f"What does the '{heading}' section in {name} cover?"
    start = meta.get("start_line")
    if meta.get("chunk_type") == "code" and start is not None:
        return f"What does the code in {path} starting at line {start} do?"
    return f"What does {path} cover?"


# Set once llm_client.chat returns empty content on Ollama: thinking models (e.g.
# qwen3.5) spend the whole num_predict budget on `thinking` and emit no `content`,
# and llm_client.chat has no think kwarg — subsequent calls go straight to Ollama
# with think=False instead of burning minutes of CPU thinking per question.
_thinking_model_detected = False


def _chat_question(prompt: str, model: str) -> str:
    """Primary path: schema-constrained question via core.llm_client.chat."""
    from core.llm_client import chat

    return (
        chat(
            [{"role": "user", "content": prompt}],
            system=_GEN_SYSTEM,
            model=model,
            max_tokens=400,
            temperature=0.4,
            keep_alive="5m",
            json_schema=QUESTION_SCHEMA,
        )
        or ""
    )


def _chat_question_nothink(prompt: str, model: str) -> str:
    """Fallback for Ollama thinking models: direct client call with think=False."""
    import ollama

    from config import settings

    resp = ollama.Client(host=settings.ollama_base_url).chat(
        model=model,
        messages=[{"role": "system", "content": _GEN_SYSTEM}, {"role": "user", "content": prompt}],
        format=QUESTION_SCHEMA,
        options={"temperature": 0.4, "num_predict": 400, "num_ctx": settings.ollama_num_ctx},
        keep_alive="5m",
        think=False,
    )
    return resp.message.content or ""


def _extract_question(raw: str) -> str | None:
    """Lenient parse: strict JSON first, else the first {"question": ...} object in the text.

    Some Ollama model renderers (e.g. qwen3.5's thinking renderer on Ollama 0.20) ignore the
    `format` schema entirely, so the model may wrap the JSON in prose despite the constraint.
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\"question\"[^{}]*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    question = (parsed.get("question") or "").strip()
    return question or None


def _llm_question(content: str, path: str, model: str) -> str | None:
    """One schema-constrained chat call -> question string, or None on any failure."""
    global _thinking_model_detected

    stem_tokens = _stem_tokens(path)
    banned = (
        f"Banned words (the module's own name — ask about the concept/behavior instead): "
        f"{', '.join(dict.fromkeys(stem_tokens))}\n\n"
        if stem_tokens
        else ""
    )
    prompt = (
        f"File: {path}\n\n{banned}Excerpt:\n---\n{content[:_MAX_PROMPT_CHARS]}\n---\n\n"
        "Write the question. Output only the JSON object."
    )
    try:
        if _thinking_model_detected:
            raw = _chat_question_nothink(prompt, model)
        else:
            raw = _chat_question(prompt, model)
            # Empty content (thinking budget exhausted) or unparseable prose (format
            # schema ignored) both mean a thinking-model renderer — retry with think=False.
            if not raw.strip() or _extract_question(raw) is None:
                from core.llm_client import get_backend

                if get_backend() != "ollama":
                    return None
                _thinking_model_detected = True
                print(
                    "note: unusable content from chat() — thinking model; retrying with think=False",
                    file=sys.stderr,
                )
                raw = _chat_question_nothink(prompt, model)
        question = _extract_question(raw) or ""
    except Exception as e:
        print(f"warning: LLM question generation failed for {path}: {e}", file=sys.stderr)
        return None
    if not question or len(question) < 12:
        return None
    # Self-containment guard: the eval question must not reference the excerpt itself,
    # and must not leak the answer key (file name / path / line number) into the query.
    if re.search(r"\b(chunk|excerpt|snippet|context|passage|above)\b", question, re.IGNORECASE):
        return None
    if Path(path).name.lower() in question.lower() or re.search(r"\bline\s+\d+", question, re.IGNORECASE):
        return None
    # Leak guard: the module's own name (as a phrase, or as several distinct stem
    # words) lets sparse retrieval trivially win — the question must earn the hit.
    if _leaks_target(question, path):
        return None
    return question


def _pick_chunk(chunks: list[dict]) -> dict:
    """Prefer a mid-sized chunk (most likely a coherent, answerable unit), else the longest."""
    sized = [c for c in chunks if _MIN_CHUNK_CHARS <= len(c["content"]) <= 1600]
    pool = sized or chunks
    return max(pool, key=lambda c: len(c["content"]))


def sample_chunks(collection, n: int, rng: random.Random) -> list[dict]:
    """Pick ~n*_OVERSAMPLE candidate chunks spread across distinct document_path values.

    Prose paths (markdown/docs) are preferred; roughly a quarter of the target comes
    from code paths so the set exercises identifier-style queries too.
    """
    raw = collection.get(include=["documents", "metadatas"])
    by_path: dict[str, list[dict]] = {}
    for doc, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        meta = meta or {}
        path = meta.get("document_path", "")
        if not path or not (doc or "").strip():
            continue
        by_path.setdefault(path, []).append({"content": doc, "meta": meta, "path": path})

    if not by_path:
        raise SystemExit("error: collection has no chunks with document_path metadata")

    prose_paths = [p for p, cs in by_path.items() if any(c["meta"].get("chunk_type") == "prose" for c in cs)]
    code_paths = [p for p in by_path if p not in set(prose_paths)]
    rng.shuffle(prose_paths)
    rng.shuffle(code_paths)

    target = n * _OVERSAMPLE
    n_code = min(len(code_paths), max(1, target // 4))
    ordered_paths = prose_paths[: target - n_code] + code_paths[:n_code]
    # Backfill without replacement from ALL remaining unused paths (including code
    # paths beyond the initial quarter) so every distinct file becomes a candidate
    # before any file is sampled twice.
    if len(ordered_paths) < target:
        unused = [p for p in prose_paths + code_paths if p not in set(ordered_paths)]
        rng.shuffle(unused)
        ordered_paths += unused[: target - len(ordered_paths)]
    # Only once every path is used may a path contribute a second (third, ...) chunk.
    while len(ordered_paths) < target and (prose_paths or code_paths):
        ordered_paths += (prose_paths + code_paths)[: target - len(ordered_paths)]

    candidates = []
    seen_ids: set[int] = set()
    for path in ordered_paths[:target]:
        remaining = [c for c in by_path[path] if id(c) not in seen_ids]
        if not remaining:
            continue
        chunk = _pick_chunk(remaining)
        seen_ids.add(id(chunk))
        candidates.append(chunk)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a golden eval set from an ingested collection.")
    parser.add_argument("--collection", required=True, help="ChromaDB collection to sample chunks from")
    parser.add_argument("--out", default="tests/eval/golden_repo.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=20, help="Number of golden questions to generate (default: 20)")
    parser.add_argument("--model", default="qwen3.5:9b", help="LLM for question generation (default: qwen3.5:9b)")
    parser.add_argument(
        "--ingest-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="When the collection is empty, ingest this directory into it first (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    args = parser.parse_args()

    from eval_retrieval import ingest_dirs

    from config import settings
    from core.llm_client import get_backend, ollama_reachable, set_backend
    from core.retriever import VectorStore

    store = VectorStore(collection_name=args.collection)
    if store.count == 0 and args.ingest_dir:
        ingest_dirs(store, args.ingest_dir)
    if store.count == 0:
        raise SystemExit(f"error: collection '{args.collection}' is empty — pass --ingest-dir to populate it.")

    # The default backend may be Anthropic with no key configured — prefer a live Ollama.
    llm_ok = True
    if get_backend() != "ollama" and not settings.anthropic_api_key:
        if ollama_reachable():
            set_backend("ollama")
        else:
            llm_ok = False
    elif get_backend() == "ollama" and not ollama_reachable():
        llm_ok = False
    if not llm_ok:
        print("warning: no reachable LLM — falling back to deterministic template questions", file=sys.stderr)

    rng = random.Random(args.seed)
    candidates = sample_chunks(store.collection, args.n, rng)
    print(f"Collection '{args.collection}': {store.count} chunks; sampled {len(candidates)} candidates")

    rows: list[dict] = []
    accepted_questions: list[str] = []
    llm_count = template_count = 0
    for chunk in candidates:
        if len(rows) >= args.n:
            break
        path, meta = chunk["path"], chunk["meta"]
        question = _llm_question(chunk["content"], path, args.model) if llm_ok else None
        from_llm = question is not None
        if question is None:
            if llm_ok:
                continue  # skip failed candidates: template rows echo file names and weaken the set
            question = _template_question(path, meta)
        if _is_duplicate(question, accepted_questions):
            continue
        if from_llm:
            llm_count += 1
        else:
            template_count += 1
        accepted_questions.append(question)
        rows.append({"question": question, "expected_paths": [path], "note": "generated"})
        print(f"  [{len(rows)}/{args.n}] {path}: {question}")

    if not rows:
        raise SystemExit("error: no questions generated")
    if len(rows) < args.n:
        print(f"warning: only {len(rows)}/{args.n} questions survived dedup", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"\nWrote {len(rows)} questions to {out_path} ({llm_count} LLM-generated, {template_count} template)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
