# Offline eval harness

Two scripts measure the RAG pipeline against a golden set of question → expected-document pairs:

| Script | What it measures | Cost |
| --- | --- | --- |
| `scripts/eval_retrieval.py` | hit-rate@k and MRR per retrieval-toggle combo | Seconds, deterministic, no judge |
| `scripts/eval_answers.py` | DeepEval Faithfulness + ContextualRelevancy on generated answers | Minutes, one judge LLM call chain per question |

Run them via the Makefile:

```
make eval          # retrieval metrics (fast, run this often)
make eval-answers  # answer-quality metrics (slow, judge-based)
```

or directly:

```
.venv\Scripts\python.exe scripts/eval_retrieval.py --golden tests/eval/golden.jsonl --user default --top-k 5 --configs all
.venv\Scripts\python.exe scripts/eval_answers.py --golden tests/eval/golden.jsonl --user default --limit 10
```

## golden.jsonl format

One JSON object per line:

```json
{"question": "How do I rotate signing keys?", "expected_paths": ["docs/operations/key_rotation.md"], "note": "asked 2026-07-02, answer lives in the ops runbook"}
```

- `question` — a real question, phrased the way you actually ask it.
- `expected_paths` — document paths whose chunks answer the question. A retrieved chunk counts as a
  hit when its `document_path` metadata matches ANY listed path (comparison is case-insensitive,
  slash-normalized, and tolerates prefix differences, so `docs/guide.md` matches `project/docs/guide.md`).
- `note` — free text for humans: why this row exists, where the answer lives, when it was added.

**The shipped file is a placeholder template.** Every row is marked `PLACEHOLDER` and the retrieval
script warns loudly until you replace them. Numbers computed on the template are meaningless.

## Filling it from real usage

1. **Ask real questions.** Use the app normally. Good golden rows are questions you actually needed
   answered, not synthetic trivia.
2. **Find the correct document.** The chat Sources footer and the Files panel show `document_path`
   values exactly as stored (ingest-relative, forward slashes — e.g. `docs/guide.md`,
   `src/auth/service.py`). Copy them verbatim. `GET /api/files` lists every indexed path.
3. **Mine past sessions.** Questions and cited sources are persisted in SQLite:

   ```
   .venv\Scripts\python.exe -c "import sqlite3, json; rows = sqlite3.connect('data/rag_assistant.db').execute(\"SELECT content, sources FROM messages WHERE role='assistant' AND sources IS NOT NULL ORDER BY created_at DESC LIMIT 20\").fetchall(); [print(json.loads(s or '[]'), '\n---') for _, s in rows]"
   ```

   Pair each assistant message's cited sources with the preceding user question, keep the pairs
   where the citation was actually correct, and write them as rows.
4. **Aim for 20-50 rows.** Below ~20 the metrics are too noisy to compare configs; the template
   ships 10 slots to show the shape.
5. **Keep it stable.** The value of the golden set is trend-over-time — append new rows, avoid
   rewriting old ones, and re-run `make eval` before/after every retrieval change (embedder swap,
   reranker change, chunking change, ...).

## Retrieval configs measured

`eval_retrieval.py` builds its matrix from the flags `core.retriever.retrieve()` actually supports:

| Config | Flags |
| --- | --- |
| `vector` | dense only (`use_hybrid=False`, no rerank) |
| `baseline` | hybrid dense+BM25 RRF, no rerank |
| `rerank` | baseline + cross-encoder/ColBERT rerank |
| `hyde` | rerank + HyDE query expansion |
| `splade` | rerank + SPLADE sparse (needs `RAG_SPLADE_ENABLED=true`) |
| `multiquery` | rerank + RAG-Fusion multi-query expansion |

Caveats:

- `hyde` and `multiquery` call the configured LLM to expand the query, so those two rows are slower
  and not perfectly deterministic run-to-run. The metric computation itself never uses a judge.
- If SPLADE is disabled or fails to load, the `splade` row silently degrades to BM25 and will match
  `rerank` — the script warns when that happens.
- Metrics are per-document, not per-chunk: retrieving the right document at the wrong chunk still
  counts as a hit. That is deliberate — chunk boundaries shift with every re-ingest.

## Answer-quality metrics

`eval_answers.py` answers each golden question through the standard `retrieve()` → `generate()`
path, then scores with DeepEval's Faithfulness (is the answer grounded in the retrieved chunks?)
and ContextualRelevancy (were the retrieved chunks relevant to the question?) using the currently
configured LLM backend as judge.

**Judge noise is about ±0.1.** Scores are comparative only — useful for before/after deltas on the
same golden set and judge model, never as absolute quality claims or CI gates. Small local judge
models (3B-14B) are noisier than frontier APIs; keep the judge fixed when comparing runs.
