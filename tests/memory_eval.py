"""
tests/memory_eval.py — Tier 3.2: LongMemEval-S memory regression harness (standalone script, NOT pytest).

Replays LongMemEval haystack sessions through the real extraction pipeline
(core.memory.process_turn_memories), answers each benchmark question from
retrieve_memories() + llm_client.chat, and judges answers against the gold
answer with a schema-constrained lenient-equivalence LLM judge (same fact =
correct regardless of wording — the paper's evaluation framing, arXiv 2410.10813).

Dataset: the SMALL variant (longmemeval_s, ~277 MB, ~40-60 sessions/question).
Auto-downloaded on first run from the maintainer's cleaned HuggingFace copy
(xiaowu0162/longmemeval-cleaned :: longmemeval_s_cleaned.json — the original
xiaowu0162/longmemeval repo is deprecated in favor of it) and saved to --dataset.

Sampling is deterministic (seeded) and stratified across question_type, weighted
3x toward temporal-reasoning and knowledge-update — the categories the roadmap
predicts our memory design changes (composite ranking, time-aware retrieval,
Mem0-style reconciliation) move most. Same --questions/--seed => same questions,
so before/after runs are comparable.

Caching/resume: each question replays into its own memory collection
(user "lme_<question_id>"). A collection is reused ONLY when a completion marker in
the replay_state.json sidecar (next to --results) records that the question finished
replaying with the same --sessions-cap, --model and timeshift setting; a non-empty
collection without a matching marker (interrupted replay, changed cap/model, or a
store from a pre-marker run) is cleared and fully re-replayed. Judged rows already
in --results are not re-run unless --fresh; rows whose judge verdict is null are NOT
treated as done — they are re-judged on the next run (replay stays cached). Replay
through a local Ollama model is the slow part on CPU — progress is printed per session.

Time-shift (--timeshift, default ON): dataset sessions carry ~2023 timestamps, so the
recency decay under test (exp(-hours/168)) would score every memory as ancient and the
now()-anchored _parse_temporal_range windows would never intersect them — the time-aware
components would be invisible to the eval. Each question's timestamps are therefore
shifted by one constant offset placing its newest session at now-minus-1-day; the
question_date is shifted by the same offset, so relative deltas ("how many days between
X and Y") are preserved exactly and the shifted question date is what the answer model
sees. Tradeoff: absolute calendar dates in model answers land in the shifted frame while
gold answers keep the original ~2023 frame — the judge prompt says dates were uniformly
shifted and instructs grading relative-time reasoning on deltas, treating absolute-date
frame differences as non-errors, which makes absolute-date grading slightly more lenient.
Cached replays also keep the stamps from when they were replayed, so the "newest session
= yesterday" anchor drifts as the cache ages; use --fresh for a clean frame.
--no-timeshift restores raw dataset dates.

Usage:
    .venv\\Scripts\\python.exe tests/memory_eval.py [--questions 20] [--sessions-cap 10]
        [--model qwen3.5:9b] [--judge-model qwen3.5:9b] [--dataset data/longmemeval/longmemeval_s.json]
        [--results data/longmemeval/results.jsonl] [--seed 42] [--fresh] [--no-timeshift]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"
HF_FILENAME = "longmemeval_s_cleaned.json"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"
DATASET_SIZE_MB = 277  # verified against the HF repo listing on 2026-07-09

REQUIRED_KEYS = (
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_sessions",
    "haystack_dates",
    "haystack_session_ids",
    "answer_session_ids",
)

# Stratified-sampling weights: the roadmap (item 3.2) predicts the memory redesign moves
# temporal-reasoning and knowledge-update most, so they get 3x the share of other categories.
CATEGORY_WEIGHTS = {"temporal-reasoning": 3.0, "knowledge-update": 3.0}
DEFAULT_CATEGORY_WEIGHT = 1.0

ANSWER_SYSTEM = """You are a personal assistant answering a question about the user, grounded ONLY in the memory
notes provided. The notes are facts extracted from the user's past chat sessions; each ends with the date it was
learned in [YYYY-MM-DD] form. Reason carefully about dates when the question involves time. Answer directly and
concisely in one short sentence. If the notes do not contain the information needed, reply exactly:
"I don't have that information." """

JUDGE_SYSTEM = """You grade a memory-augmented assistant's answer against the gold answer for a question about a
user's chat history. Be lenient about form: the answer is CORRECT if it conveys the same fact(s) as the gold
answer, regardless of wording, format, or extra supporting detail. Equivalent date/duration expressions count
(e.g. "May 5, 2023" matches "2023/05/05"). The answer is INCORRECT if it contradicts the gold answer, omits the
asked fact, or claims the information is unavailable when the gold answer states it.
Return JSON: {"correct": true|false}."""

ABSTENTION_NOTE = """Note: this is an abstention test — the gold behavior is recognizing the chat history never
contained this information. Judge correct=true only if the model declined to answer or said it does not know;
judge correct=false if it fabricated a concrete answer."""

_JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}

_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


# ---------------------------------------------------------------------------
# Dataset acquisition & loading
# ---------------------------------------------------------------------------


def ensure_dataset(path: Path) -> None:
    """Download longmemeval_s to `path` if missing (HF hub, ~277 MB); fail with manual instructions."""
    if path.exists():
        return
    manual_help = f"Manual download: {HF_URL}\nSave it as {path} (or pass --dataset <path> pointing at your copy)."
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise SystemExit(
            f"error: dataset missing at {path} and huggingface_hub is not installed.\n{manual_help}"
        ) from e

    size_note = f"~{DATASET_SIZE_MB} MB"
    try:  # verify the remote size before committing to the download
        info = HfApi().get_paths_info(HF_REPO_ID, [HF_FILENAME], repo_type="dataset")
        if info and getattr(info[0], "size", None):
            size_note = f"{info[0].size / 1e6:.0f} MB"
    except Exception:
        pass
    print(f"Dataset not found at {path} — downloading {HF_REPO_ID}/{HF_FILENAME} ({size_note}) ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(HF_REPO_ID, HF_FILENAME, repo_type="dataset", local_dir=str(path.parent))
    except Exception as e:
        raise SystemExit(f"error: dataset download failed: {e}\n{manual_help}") from e
    if Path(downloaded).resolve() != path.resolve():
        os.replace(downloaded, path)
    print(f"Saved dataset to {path}")


def load_dataset(path: Path) -> list[dict]:
    print(f"Loading {path} ({path.stat().st_size / 1e6:.0f} MB) ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"error: {path} is not a non-empty JSON array of LongMemEval instances")
    missing = [k for k in REQUIRED_KEYS if k not in data[0]]
    if missing:
        raise SystemExit(f"error: {path} instances lack expected LongMemEval fields: {missing}")
    return data


def disable_ollama_thinking() -> None:
    """Default think=False on every Ollama chat call in this process.

    qwen3-family models think by default, and Ollama routes those tokens to
    message.thinking — with the small num_predict budgets used by the pipeline's
    novelty gate (5 tokens) and this harness's judge, the whole budget is spent
    thinking and message.content comes back EMPTY, silently zeroing extraction
    and judging. Models that ignore or reject the parameter fall back cleanly.
    """
    try:
        import ollama
    except ImportError:
        return
    original = ollama.Client.chat
    if getattr(original, "_memory_eval_no_think", False):
        return

    def no_think_chat(self, *args, **kwargs):
        if "think" in kwargs:
            return original(self, *args, **kwargs)
        try:
            return original(self, *args, think=False, **kwargs)
        except ollama.ResponseError:  # server/model rejects the parameter — use default behavior
            return original(self, *args, **kwargs)

    no_think_chat._memory_eval_no_think = True
    ollama.Client.chat = no_think_chat


def _close_open_brackets(text: str) -> str:
    """Append the closers ('"', ']', '}') a truncated JSON value still needs."""
    stack = []
    in_str = escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack:
            stack.pop()
    if in_str:
        text += '"'
    return text + "".join("]" if opener == "[" else "}" for opener in reversed(stack))


def _repair_schema_json(text: str) -> str:
    """Salvage qwen3.5's malformed Ollama structured outputs (format=<schema>).

    Two artifacts observed with qwen3.5:9b at temperature 0:
    1. A spurious '"' directly after '}' or ']' (e.g. trailing '...}]"') — never
       legal JSON in that position, safe to drop.
    2. Premature EOS: the value stops mid-array/object (e.g. '[{...},{...}' with
       no closing ']') — repaired by appending the missing closers, and if the
       tail element itself was cut mid-value, by dropping it (cut at last '}')
       before closing.
    Also keeps only the first top-level JSON value (raw_decode) to shed any other
    trailing junk. Unrepairable text is returned unchanged so callers' own
    fail-open error handling still applies.
    """
    if not isinstance(text, str):
        return text
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r'(?<=[}\]])\s*"', "", text.strip())  # artifact 1
    for candidate in (
        stripped,
        # artifact 2 — drop-partial first: cutting at the last '}' sheds an element cut
        # mid-value instead of storing its truncated content (no-op for boundary cuts)
        _close_open_brackets(stripped[: stripped.rfind("}") + 1]) if "}" in stripped else "",
        _close_open_brackets(stripped),  # no complete element to cut back to (small objects)
    ):
        if not candidate:
            continue
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
            return candidate[:end]
        except json.JSONDecodeError:
            continue
    return text


def repair_ollama_schema_json() -> None:
    """Post-process every schema-constrained (format=...) non-streaming Ollama chat
    response in this process through _repair_schema_json. Without it, most qwen3.5
    extraction/reconcile calls fail JSON parsing on the stray-quote artifact and the
    pipeline silently stores zero fragments (the prior smoke's failure mode)."""
    try:
        import ollama
    except ImportError:
        return
    original = ollama.Client.chat
    if getattr(original, "_memory_eval_json_repair", False):
        return

    def repairing_chat(self, *args, **kwargs):
        resp = original(self, *args, **kwargs)
        if kwargs.get("format") is not None and not kwargs.get("stream"):
            content = getattr(getattr(resp, "message", None), "content", None)
            if content:
                resp.message.content = _repair_schema_json(content)
        return resp

    repairing_chat._memory_eval_json_repair = True
    ollama.Client.chat = repairing_chat


# ---------------------------------------------------------------------------
# Deterministic stratified sampling
# ---------------------------------------------------------------------------


def sample_questions(data: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded stratified sample across question_type, weighted by CATEGORY_WEIGHTS.

    Quotas follow weight shares (largest-deficit rounding), capped by category size;
    leftover slots spill to whichever weighted category still has questions.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for inst in data:
        by_cat[inst["question_type"]].append(inst)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda d: d["question_id"])
    cats = sorted(by_cat)
    weights = {c: CATEGORY_WEIGHTS.get(c, DEFAULT_CATEGORY_WEIGHT) for c in cats}
    total_weight = sum(weights.values())
    n = min(n, len(data))

    ideal = {c: n * weights[c] / total_weight for c in cats}
    quota = {c: min(int(ideal[c]), len(by_cat[c])) for c in cats}
    while sum(quota.values()) < n:
        open_cats = [c for c in cats if quota[c] < len(by_cat[c])]
        if not open_cats:
            break
        open_cats.sort(key=lambda c: (ideal[c] - quota[c], weights[c], c), reverse=True)
        quota[open_cats[0]] += 1

    rng = random.Random(seed)
    sampled: list[dict] = []
    for c in cats:  # fixed category order keeps rng consumption deterministic
        sampled.extend(rng.sample(by_cat[c], quota[c]))
    sampled.sort(key=lambda d: d["question_id"])
    return sampled


# ---------------------------------------------------------------------------
# Session replay through the real memory pipeline
# ---------------------------------------------------------------------------


def eval_user_id(question_id: str) -> str:
    """Stable per-question user id. MemoryStore truncates collection names to the first
    20 chars of user_id, so long question_ids get a hash suffix to avoid collisions."""
    base = "lme_" + re.sub(r"[^a-zA-Z0-9_]", "_", question_id)
    if len(base) <= 20:
        return base
    return base[:12] + hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:8]


def parse_lme_date(raw: str) -> float | None:
    """'2023/05/20 (Sat) 02:21' -> epoch seconds (regex-based: locale-free, tolerant)."""
    from datetime import datetime

    m = _DATE_RE.search(raw or "")
    if not m:
        return None
    hour = minute = 0
    t = _TIME_RE.search(raw[m.end() :])
    if t:
        hour, minute = int(t.group(1)), int(t.group(2))
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), hour, minute).timestamp()
    except ValueError:
        return None


def format_lme_date(ts: float) -> str:
    """Epoch seconds -> the dataset's '2023/05/20 (Sat) 02:21' format."""
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y/%m/%d (%a) %H:%M")


def compute_timeshift(inst: dict) -> float:
    """Constant offset (seconds) placing this question's NEWEST session at now-minus-1-day.

    Computed over all haystack dates (not just the selected subset) so the offset is
    stable across --sessions-cap values. Applied uniformly to every session timestamp
    and the question_date, so relative deltas between them are preserved exactly.
    """
    stamps = [parse_lme_date(d) for d in inst["haystack_dates"]]
    stamps = [s for s in stamps if s is not None]
    if not stamps:
        return 0.0
    return (time.time() - 86400.0) - max(stamps)


def shift_lme_date(raw: str, shift: float) -> str:
    """Re-render a dataset date string with the timeshift applied (unparseable -> unchanged)."""
    ts = parse_lme_date(raw)
    if ts is None or not shift:
        return raw
    return format_lme_date(ts + shift)


def select_sessions(inst: dict, cap: int, seed: int) -> list[int]:
    """Indices of sessions to replay: every evidence session (kept even past the cap —
    correctness first), plus seeded-random distractors up to `cap` total, in chronological order."""
    answer_ids = set(inst["answer_session_ids"])
    evidence = [i for i, sid in enumerate(inst["haystack_session_ids"]) if sid in answer_ids]
    distractors = [i for i in range(len(inst["haystack_sessions"])) if i not in set(evidence)]
    rng = random.Random(f"{seed}:{inst['question_id']}")  # per-question rng: independent of sample order
    n_fill = min(max(0, cap - len(evidence)), len(distractors))
    fill = rng.sample(distractors, n_fill) if n_fill else []
    return sorted(set(evidence) | set(fill))


def iter_turn_pairs(session: list[dict]):
    """Yield (user_content, assistant_content) pairs from a session's alternating turns."""
    pending_user = None
    for turn in session:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            yield pending_user, content
            pending_user = None


def load_replay_state(path: Path) -> dict:
    """question_id -> completion marker written after that question fully replayed."""
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return state if isinstance(state, dict) else {}


def save_replay_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def replay_question(
    inst: dict,
    user_id: str,
    cap: int,
    seed: int,
    fresh: bool,
    shift: float,
    timeshift: bool,
    model: str,
    state_path: Path,
) -> dict:
    """Replay selected haystack sessions through process_turn_memories into the question's
    own memory collection.

    Resume is marker-based: the store is reused only when `state_path` records this
    question as complete with the same sessions-cap, model and timeshift setting.
    A non-empty store without a matching marker — interrupted replay, changed
    --sessions-cap/--model/timeshift, or a store from a pre-marker run — is cleared
    and fully re-replayed, so a partial replay can never poison the verdict.
    """
    from core import memory

    qid = inst["question_id"]
    mem_store = memory.get_memory_store(user_id)
    selected = select_sessions(inst, cap, seed)
    evidence_ids = set(inst["answer_session_ids"])

    state = load_replay_state(state_path)
    marker = state.get(qid)
    marker_ok = (
        isinstance(marker, dict)
        and marker.get("complete") is True
        and marker.get("cap") == cap
        and marker.get("model") == model
        and bool(marker.get("timeshift", False)) == timeshift
    )
    if mem_store.count > 0:
        if marker_ok and not fresh:
            print(f"    replay cached: {mem_store.count} fragments already stored — skipping (use --fresh to redo)")
            return {"replayed_sessions": 0, "fragments": mem_store.count, "cached": True, "seconds": 0.0}
        if not fresh:
            why = "no completion marker (interrupted or pre-marker replay)" if marker is None else "marker mismatch"
            print(f"    store has {mem_store.count} fragments but {why} — clearing and re-replaying")
        mem_store.clear()
    if marker is not None:  # drop stale marker so an interrupted re-replay is never marked complete
        state.pop(qid, None)
        save_replay_state(state_path, state)

    total_fragments = 0
    start = time.perf_counter()
    for pos, idx in enumerate(selected, 1):
        session = inst["haystack_sessions"][idx]
        session_id = inst["haystack_session_ids"][idx]
        session_date = inst["haystack_dates"][idx]
        session_ts = parse_lme_date(session_date)
        if session_ts is not None:
            session_ts += shift
        tag = "evidence" if session_id in evidence_ids else "distractor"
        t0 = time.perf_counter()

        fragment_ids: list[str] = []
        n_pairs = 0
        for query, answer in iter_turn_pairs(session):
            n_pairs += 1
            fragments = memory.process_turn_memories(user_id, query, answer, session_id=session_id)
            fragment_ids.extend(f.fragment_id for f in fragments)

        # Re-stamp stored fragments with the session's real date: in production created_at IS
        # the session time, and the time-aware ranking/filtering under test reads it. Replay
        # would otherwise collapse months of history onto "now". get() filters to fragments
        # that survived dedup/reconciliation.
        stored = 0
        if fragment_ids and session_ts:
            existing_ids = mem_store.collection.get(ids=fragment_ids)["ids"]
            if existing_ids:
                stamps = [{"created_at": session_ts, "last_accessed": session_ts} for _ in existing_ids]
                mem_store.collection.update(ids=existing_ids, metadatas=stamps)
                stored = len(existing_ids)
        elif fragment_ids:
            stored = len(mem_store.collection.get(ids=fragment_ids)["ids"])
        total_fragments += stored

        print(
            f"    [{pos}/{len(selected)}] {tag} session {session_date[:10]} "
            f"({n_pairs} turns) -> {stored} fragments ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    # Marker written only after ALL selected sessions replayed — its absence on the
    # next run is what forces an interrupted replay to start over.
    state = load_replay_state(state_path)
    state[qid] = {"sessions": len(selected), "cap": cap, "model": model, "timeshift": timeshift, "complete": True}
    save_replay_state(state_path, state)

    return {
        "replayed_sessions": len(selected),
        "fragments": total_fragments,
        "cached": False,
        "seconds": round(time.perf_counter() - start, 1),
    }


# ---------------------------------------------------------------------------
# Answering & judging
# ---------------------------------------------------------------------------


def answer_question(inst: dict, user_id: str, model: str, question_date: str) -> tuple[str, int]:
    from core import llm_client, memory

    mem_store = memory.get_memory_store(user_id)
    mem_ctx = memory.retrieve_memories(mem_store, inst["question"])
    memories_block = mem_ctx.formatted or "[Memory] (no stored memories matched this question)"
    user_msg = f"Today's date: {question_date}\n\n{memories_block}\n\nQuestion: {inst['question']}"
    text = llm_client.chat(
        messages=[{"role": "user", "content": user_msg}],
        system=ANSWER_SYSTEM,
        model=model,
        max_tokens=300,
        temperature=0.0,
        stream=False,
        keep_alive="5m",
    )
    return (text or "").strip(), mem_ctx.count


_TRUTHY_WORDS = {"true", "yes", "correct", "1"}
_FALSY_WORDS = {"false", "no", "incorrect", "0"}


def _normalize_correct(value) -> bool | None:
    """Coerce a judge-emitted 'correct' value to bool. Guards the {"correct": "false"}
    trap — bool("false") is True — by mapping string/int spellings explicitly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().strip(".\"'").lower()
        if v in _TRUTHY_WORDS:
            return True
        if v in _FALSY_WORDS:
            return False
    return None


def parse_judge_verdict(text: str) -> tuple[bool | None, str]:
    """Extract the judge verdict leniently; returns (verdict, how).

    Ollama 0.20.0 ignores the format schema for some models (observed with qwen3.5),
    so the judge may reply in prose. Tries, in order:
      strict    — the whole reply is a JSON object with a "correct" key;
      embedded  — first JSON object inside the reply containing a "correct" key;
      prose     — keyword heuristic: an unambiguous yes/correct vs no/incorrect statement.
    All values are normalized via _normalize_correct. (None, "none") when nothing matches.
    """
    if not text or not text.strip():
        return None, "none"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            verdict = _normalize_correct(obj.get("correct"))
            if verdict is not None:
                return verdict, "strict"
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "correct" in obj:
            verdict = _normalize_correct(obj["correct"])
            if verdict is not None:
                return verdict, "embedded"
    lowered = text.lower()
    lead = re.match(r"[\s\"'`*]*(yes|no)\b", lowered)  # a leading bare yes/no is unambiguous
    if lead:
        return lead.group(1) == "yes", "prose"
    negative = bool(re.search(r"\b(incorrect|not\s+correct|wrong|false)\b", lowered))
    positive = bool(re.search(r"\b(correct|true)\b", re.sub(r"\bincorrect\b|\bnot\s+correct\b", " ", lowered)))
    if positive != negative:  # both or neither -> ambiguous
        return positive, "prose"
    return None, "none"


TIMESHIFT_JUDGE_NOTE = (
    "Session dates were uniformly time-shifted; judge relative-time reasoning on deltas, and treat "
    "absolute calendar dates in the gold answer as reference-frame differences, not errors."
)


def judge_answer(inst: dict, model_answer: str, judge_model: str, question_date: str, timeshifted: bool) -> bool | None:
    """Lenient-equivalence LLM judge. Schema-constrained JSON requested, but the verdict is
    parsed leniently (see parse_judge_verdict) because Ollama may ignore the schema.
    None = no parseable verdict after two attempts."""
    from core import llm_client

    prompt = (
        f"Question (asked on {question_date}): {inst['question']}\n"
        f"Gold answer: {inst['answer']}\n"
        f"Model answer: {model_answer or '(empty)'}"
    )
    if timeshifted:
        prompt += f"\n\n{TIMESHIFT_JUDGE_NOTE}"
    if str(inst["question_id"]).endswith("_abs"):
        prompt += f"\n\n{ABSTENTION_NOTE}"
    for attempt in (1, 2):
        try:
            text = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=JUDGE_SYSTEM,
                model=judge_model,
                max_tokens=50,
                temperature=0.0,
                stream=False,
                keep_alive="5m",
                json_schema=_JUDGE_SCHEMA,
            )
        except Exception as e:
            print(f"    judge attempt {attempt} failed, retrying: {e}", file=sys.stderr)
            continue
        verdict, how = parse_judge_verdict(text or "")
        if verdict is not None:
            if how != "strict":
                print(f"    judge verdict recovered via lenient path ({how}) from: {(text or '').strip()[:100]!r}")
            return verdict
        print(f"    judge attempt {attempt} gave no parseable verdict: {(text or '').strip()[:120]!r}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Results file & report
# ---------------------------------------------------------------------------


def load_done(results_path: Path) -> dict[str, dict]:
    """question_id -> latest properly-judged row from previous runs (summary lines skipped).

    Rows with correct=None (judge failed) are NOT counted as done, so a re-run
    re-judges them — the replay stays cached via its completion marker, so only
    the cheap answer+judge steps repeat.
    """
    done: dict[str, dict] = {}
    unjudged_qids: set[str] = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (isinstance(row, dict) and not row.get("summary") and row.get("question_id")):
            continue
        if row.get("correct") is None:
            unjudged_qids.add(row["question_id"])
        else:
            done[row["question_id"]] = row
    to_rejudge = unjudged_qids - done.keys()
    if to_rejudge:
        print(f"Resume: {len(to_rejudge)} question(s) with unjudged (correct=null) rows will be re-judged")
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(rows: list[dict]) -> tuple[str, dict]:
    per_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_cat[row["question_type"]].append(row)

    def acc(bucket: list[dict]) -> tuple[int, int, float | None]:
        judged = [r for r in bucket if r.get("correct") is not None]
        correct = sum(1 for r in judged if r["correct"])
        return len(judged), correct, (correct / len(judged) if judged else None)

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    header = f"{'category':<28} {'n':>4} {'correct':>8} {'accuracy':>9}"
    lines = [header, "-" * len(header)]
    summary_cats = {}
    for cat in sorted(per_cat):
        n, correct, accuracy = acc(per_cat[cat])
        summary_cats[cat] = {"n": n, "correct": correct, "accuracy": accuracy}
        lines.append(f"{cat:<28} {n:>4} {correct:>8} {fmt(accuracy):>9}")
    n_all, correct_all, acc_all = acc(rows)
    lines.append("-" * len(header))
    lines.append(f"{'overall':<28} {n_all:>4} {correct_all:>8} {fmt(acc_all):>9}")
    judge_errors = sum(1 for r in rows if r.get("correct") is None)
    if judge_errors:
        lines.append(f"(judge failed on {judge_errors} question(s) — excluded from accuracy)")
    summary = {
        "overall": {"n": n_all, "correct": correct_all, "accuracy": acc_all},
        "per_category": summary_cats,
        "judge_errors": judge_errors,
    }
    return "\n".join(lines), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # progress must reach redirected logs live, not on exit
        sys.stdout.reconfigure(line_buffering=True)
    # Surface extraction/reconciliation activity (and its silent failure paths) without httpx noise
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("core.memory").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="LongMemEval-S regression harness: replay sessions through core.memory, answer, judge."
    )
    parser.add_argument("--dataset", default="data/longmemeval/longmemeval_s.json", help="LongMemEval-S JSON path")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions to evaluate (default: 20)")
    parser.add_argument("--sessions-cap", type=int, default=10, help="Max sessions replayed per question (default: 10)")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model for extraction + answering")
    parser.add_argument("--judge-model", default="qwen3.5:9b", help="Ollama model for judging")
    parser.add_argument("--results", default="data/longmemeval/results.jsonl", help="Append-only results JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached memory stores and prior judged results")
    parser.add_argument(
        "--timeshift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift each question's timestamps so its newest session is now-minus-1-day, "
        "keeping relative deltas — makes the time-aware ranking under test visible "
        "(default: on; --no-timeshift restores raw ~2023 dates)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    results_path = Path(args.results)
    replay_state_path = results_path.parent / "replay_state.json"
    ensure_dataset(dataset_path)
    data = load_dataset(dataset_path)

    from config import settings
    from core import llm_client

    # Eval-mode pipeline overrides. Interval=1 and min_answer_length=1: production gates
    # extraction to every 3rd substantial turn, which would drop most evidence turns and
    # measure the gating instead of the memory design under test. The novelty gate and
    # Mem0-style reconciliation stay active — they ARE part of what is measured.
    llm_client.set_backend("ollama")
    disable_ollama_thinking()
    repair_ollama_schema_json()
    settings.memory_enabled = True
    settings.memory_auto_extract = True
    settings.memory_extract_interval = 1
    settings.memory_min_answer_length = 1
    settings.ollama_memory_model = args.model
    settings.ollama_keep_alive = "5m"

    if not llm_client.ollama_reachable():
        raise SystemExit(f"error: no Ollama node reachable (base: {settings.ollama_base_url}) — start Ollama first")

    sampled = sample_questions(data, args.questions, args.seed)
    cat_counts = defaultdict(int)
    for inst in sampled:
        cat_counts[inst["question_type"]] += 1
    print(f"\nDataset: {len(data)} instances | sampled {len(sampled)} (seed {args.seed}):")
    for cat in sorted(cat_counts):
        print(f"  {cat:<28} {cat_counts[cat]}")
    print(
        f"Models: extract/answer={args.model} judge={args.judge_model} | sessions cap: {args.sessions_cap} | "
        f"timeshift: {'on' if args.timeshift else 'off'}\n"
    )

    done = {} if args.fresh else load_done(results_path)
    rows: list[dict] = []
    timings = {"replay_s": 0.0, "answer_s": 0.0, "judge_s": 0.0}

    for i, inst in enumerate(sampled, 1):
        qid = inst["question_id"]
        category = inst["question_type"]
        if qid in done:
            rows.append(done[qid])
            print(f"[{i}/{len(sampled)}] {qid} ({category}) cached result: correct={done[qid].get('correct')}")
            continue

        print(f"[{i}/{len(sampled)}] {qid} ({category}): {inst['question'][:90]}")
        user_id = eval_user_id(qid)

        shift = compute_timeshift(inst) if args.timeshift else 0.0
        question_date = shift_lme_date(inst["question_date"], shift)
        if shift:
            print(
                f"    timeshift: +{shift / 86400:.1f} days (question date {inst['question_date']} -> {question_date})"
            )

        replay = replay_question(
            inst,
            user_id,
            args.sessions_cap,
            args.seed,
            args.fresh,
            shift,
            args.timeshift,
            args.model,
            replay_state_path,
        )
        timings["replay_s"] += replay["seconds"]

        t0 = time.perf_counter()
        model_answer, n_memories = answer_question(inst, user_id, args.model, question_date)
        answer_s = time.perf_counter() - t0
        timings["answer_s"] += answer_s

        t0 = time.perf_counter()
        correct = judge_answer(inst, model_answer, args.judge_model, question_date, args.timeshift)
        judge_s = time.perf_counter() - t0
        timings["judge_s"] += judge_s

        row = {
            "question_id": qid,
            "question_type": category,
            "question": inst["question"],
            "gold_answer": inst["answer"],
            "model_answer": model_answer,
            "correct": correct,
            "n_memories_injected": n_memories,
            "replay": replay,
            "answer_s": round(answer_s, 1),
            "judge_s": round(judge_s, 1),
            "model": args.model,
            "judge_model": args.judge_model,
            "sessions_cap": args.sessions_cap,
            "timeshift": args.timeshift,
            "timeshift_days": round(shift / 86400, 2),
            "ts": time.time(),
        }
        rows.append(row)
        append_jsonl(results_path, row)
        print(
            f"    answer ({answer_s:.1f}s, {n_memories} memories): {model_answer[:100]}\n"
            f"    gold: {str(inst['answer'])[:100]}\n"
            f"    judged ({judge_s:.1f}s): correct={correct}",
            flush=True,
        )

    report, summary = build_report(rows)
    print(f"\n{report}\n")
    print(
        f"Timing (new work this run): replay {timings['replay_s']:.0f}s | "
        f"answer {timings['answer_s']:.0f}s | judge {timings['judge_s']:.0f}s"
    )
    append_jsonl(
        results_path,
        {
            "summary": True,
            "ts": time.time(),
            "questions": len(sampled),
            "seed": args.seed,
            "sessions_cap": args.sessions_cap,
            "model": args.model,
            "judge_model": args.judge_model,
            "timeshift": args.timeshift,
            **summary,
        },
    )
    print(f"Results appended to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
