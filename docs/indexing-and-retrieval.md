# RAGv2 — Indexing & Retrieval Explained

> How a directory of 20-25 files gets indexed, stored, and queried.
> Walk-through example query: **"where and how is eslint used?"**

---

## Part 1 — Indexing (what happens when you click "Ingest Directory")

### Step 1: File Discovery (`core/ingestion.py` → `load_documents()`)

The system walks your project folder recursively. For every file it checks:

- Is the extension supported? (`.js`, `.ts`, `.json`, `.md`, `.css`, `.yaml`, etc. — 50+ types)
- Is the parent folder a junk folder? (`node_modules`, `.git`, `dist`, `build`, `__pycache__` are all **skipped**)

Each file becomes a `Document` object:

```python
Document(
    content   = "<full file text>",
    filepath  = "src/App.js",       # relative path, forward slashes
    language  = "javascript",
    metadata  = {}
)
```

---

### Step 2: Chunking (`chunk_document()`)

A large file can't be stored as one blob — so each document is split into **chunks** (~300–500 tokens each). Two strategies are used depending on file type:

#### Code files (`.js`, `.ts`, `.py`, `.go`, etc.) → `_chunk_code()`

Splits at structural boundaries using regex:

```python
# For JavaScript:
pattern = re.compile(r"^(function |class |const |let |var |export |import )")
```

Each logical block (a component, a function, an import group) becomes its own chunk. This means when you ask about `eslint`, the exact function or config block containing it comes back — not an entire 1700-line file.

#### Prose / config files (`.md`, `.json`, `.yaml`, `.css`) → `_chunk_prose()`

Splits on heading boundaries (`#`, `##`, `---`) and uses a sliding window with overlap so context is never cut mid-sentence.

Each chunk becomes a `Chunk` object:

```python
Chunk(
    content       = 'module.exports = { extends: ["react-app"], rules: {...} }',
    chunk_id      = "a3f8b2c1...",   # SHA-256 hash of filepath + line range
    document_path = ".eslintrc.json",
    language      = "json",
    start_line    = 1,
    end_line      = 12,
    chunk_type    = "prose"
)
```

---

### Step 3: Embedding + Storing in ChromaDB (`VectorStore.add_chunks()`)

For every batch of 64 chunks, the system embeds the text:

```python
embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(chunk_texts)
# Each chunk → a 384-dimensional float32 vector
```

Then calls ChromaDB's `upsert()` with:
- `ids` → SHA-256 chunk IDs
- `documents` → raw text content
- `embeddings` → 384-dim vectors
- `metadatas` → `{document_path, language, start_line, end_line, chunk_type}`

**Where is this stored on disk?**

```
data/chroma/
└── docs_{user_id}/           # one ChromaDB collection per user
    ├── chroma.sqlite3         # metadata, IDs, relationships
    └── index/
        └── *.bin              # HNSW vector index (approximate nearest neighbour)
```

ChromaDB's HNSW index is a web of connected vectors — each vector links to its nearest neighbours so retrieval is O(log N), not a full scan.

---

### Step 4: BM25 Index built in RAM (`BM25Index.build_from_collection()`)

After ChromaDB is updated, all stored chunks are re-read and a **BM25 keyword index is built in memory**:

```python
tokenized = [re.findall(r'\b\w+\b', doc.lower()) for doc in all_docs]
self.bm25 = BM25Okapi(tokenized)
```

This index lives **only in RAM** — it is rebuilt from ChromaDB every time the server starts or new files are added. Nothing extra is written to disk.

**After indexing your project you have:**
- `data/chroma/docs_{uid}/` — persistent vector store on disk
- `bm25_index` object — fast keyword index in RAM

---

## Part 2 — Querying: "where and how is eslint used?"

### Step 1: Query Router (`core/router.py` → `route_query_fast()`)

Before any retrieval, the query is classified in **< 1 ms** using keyword matching:

```python
# "where and how is eslint used?" contains "how"
if any(p in q for p in ["how does", "explain", "what does", "why does", "how to"]):
    return QueryRoute("explanation", strategy="broad", suggested_top_k=6)
```

Result: category `explanation`, top_k **6**, strategy `broad`. This tells retrieval to cast a wider net.

---

### Step 2: Hybrid Retrieval — two searches fire simultaneously (`VectorStore.hybrid_search()`)

#### Search A: Vector Search (semantic)

The query is embedded into a 384-dim vector:

```python
q_vec = embedder.encode(["where and how is eslint used?"])[0]  # shape: (384,)
```

ChromaDB's HNSW index finds the **top-6 nearest chunks** by cosine similarity — chunks whose *meaning* is most similar to the query:

| Chunk | Similarity |
|---|---|
| `.eslintrc.json` content | 0.82 |
| `package.json` with `"eslint": "^8.0.0"` | 0.76 |
| `src/App.js` with `eslint-disable-next-line` | 0.71 |
| `README.md` section on "Linting" | 0.68 |

Chunks below the minimum similarity threshold are discarded.

#### Search B: BM25 Keyword Search (exact token matching)

The query is tokenized → `["where", "and", "how", "is", "eslint", "used"]`

BM25 (Best Match 25) scores every chunk based on:
- **Term frequency** — how often "eslint" appears in the chunk
- **Inverse document frequency** — "eslint" is rare across all chunks, so matches on it are highly rewarded
- **Document length normalisation** — shorter chunks with "eslint" rank higher than long files that mention it in passing

This catches things like `"devDependencies": { "eslint": "^8.0" }` that might not score well semantically but literally contains the keyword.

---

### Step 3: Reciprocal Rank Fusion — merging both result lists

The two ranked lists are combined using **RRF**:

```python
# Score for each hit:
rrf_score = weight / (60 + rank)

# A chunk ranked #1 in both vector AND BM25:
score = (vector_weight / 61) + (bm25_weight / 61)   # highest possible fused score

# A chunk only in BM25 at rank #3:
score = bm25_weight / 63
```

Chunks that both methods agreed on get the highest combined score. Each result records its `search_type`: `"vector+bm25"` (both found it), `"vector"`, or `"bm25"` (only one found it).

---

### Step 4: Cross-Encoder Reranking (`rerank()`)

The fused top results are passed to a cross-encoder model:

```
model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike the embedding model (which embeds query and chunk *separately*), the cross-encoder reads the query **and** chunk **together**:

```
Input:  ["where and how is eslint used?",  ".eslintrc.json content..."]
Output: relevance score (a single float, e.g. 8.2)
```

This is much more accurate than cosine similarity but slower — so it only runs on the top ~6-10 fused hits to produce the final ordering.

---

### Step 5: Knowledge Graph augmentation (if graph has been built)

If you previously ran "Build Knowledge Graph", the KG kicks in alongside vector results (`api/server.py:426-438`):

1. Query is embedded → compared against the pre-cached entity name matrix
2. Seed nodes are selected: e.g. **`eslint`**, **`ESLintConfig`**, **`react-app`**
3. A BFS walk 2 hops out collects related entities: `webpack`, `babel`, `package.json`, `CRA`, etc.
4. Their associated chunk IDs are fetched from ChromaDB and **merged** into the hit list (deduplicating any already found by vector/BM25)

---

### Step 6: Memory retrieval (`retrieve_memories()`)

Your top-5 long-term memories most relevant to this query are pulled from ChromaDB's `memories_{uid}` collection. If you've asked about eslint before, fragments like `"User's MyApp uses eslint with react-app preset"` get injected into the prompt.

---

### Step 7: LLM prompt assembly and streaming (`core/generator.py`)

The final prompt sent to the LLM:

```
[SYSTEM]: You are a precise technical assistant. Cite sources as [file:lines]...

[USER]:
## Recalled memories
- User's MyApp uses eslint with react-app preset (from previous session)

## Retrieved context
--- 1 [.eslintrc.json:1-12] [json] [9.21] ---
{ "extends": ["react-app"], "rules": { "no-console": "warn" } }

--- 2 [package.json:15-22] [json] [8.74] ---
"devDependencies": { "eslint": "^8.0.0", "eslint-plugin-react": "^7.33.0" }

--- 3 [src/App.js:45-52] [javascript] [7.93] ---
// eslint-disable-next-line react-hooks/exhaustive-deps
useEffect(() => { ... }, [])

--- 4 [README.md:80-95] [markdown] [6.88] ---
## Linting — Run `npm run lint` to check your code...

## Question
where and how is eslint used?
```

The LLM streams its answer back token by token over WebSocket. You see it appear word by word in the UI.

---

### Step 8: Provenance trace (`core/provenance.py`)

Once streaming is done, the answer is split into sentences. Each sentence is scored against all retrieved chunks using embeddings:

```
novel_score = 1 - max_cosine_similarity(sentence, all_sources)
```

| Score range | Label | Meaning |
|---|---|---|
| < 0.35 | `sourced` (green) | Clearly backed by retrieved chunks |
| 0.35 – 0.65 | `inferred` (yellow) | Partially supported |
| > 0.65 | `orphan` (red) | No strong source found — potential hallucination |

---

## Complete Flow Diagram

```
YOU TYPE: "where and how is eslint used?"
          │
          ▼
  [Router] → category: "explanation", top_k: 6
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
  [Vector Search]                            [BM25 Search]
  embed query → HNSW cosine search           tokenize → BM25Okapi score
  ChromaDB on disk                           in-memory index
  top-6 semantic matches                     top-6 keyword matches
          │                                          │
          └──────────────┬───────────────────────────┘
                         ▼
              [RRF Fusion — combine & re-rank]
                         │
                         ▼
              [Cross-Encoder Reranker]
              reads (query + chunk) together
              produces final relevance scores
                         │
              (optional) ▼
              [Knowledge Graph BFS]
              seed nodes → 2-hop walk
              fetch chunk IDs from ChromaDB
              merge into hit list
                         │
                         ▼
              [Memory Retrieval]
              top-5 long-term memories
              from ChromaDB memories_{uid}
                         │
                         ▼
              [Build LLM Prompt]
              memories + context chunks + query
                         │
                         ▼
              [LLM streams answer] → WebSocket → UI
                         │
                         ▼
              [Provenance trace]
              sentence-level source attribution
              sourced / inferred / orphan labels
                         │
                         ▼
              [Save to SQLite]
              user message + assistant answer
              (short-term history for next turn)
```

---

## Summary: All retrieval methods and when each helps

| Method | How it works | Best at finding |
|---|---|---|
| **Vector search** | Cosine similarity on 384-dim embeddings via HNSW | Semantically related code, even if exact words differ |
| **BM25** | Term frequency × inverse document frequency | Exact keyword matches like `"eslint"` |
| **RRF fusion** | Combines both ranked lists with a rank-discount formula | Chunks agreed on by both methods rank highest |
| **Cross-encoder rerank** | Reads query + chunk jointly for a relevance score | Most accurate final ordering of top candidates |
| **Knowledge graph BFS** | Graph traversal from cosine-ranked seed entities | Related concepts 1-2 hops away from the query |
| **Long-term memory** | Per-user ChromaDB semantic search on past facts | Context and preferences from previous sessions |
| **Short-term history** | Last 6 turns fetched from SQLite | In-session follow-up and clarification questions |
