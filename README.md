# RAG Documentation Assistant v2

Production-grade RAG with hybrid search, agentic mode, query routing, multi-modal support, and a React UI with auth, analytics, and theme switching.

---

## Complete File List

```
RAGv2/
├── .env.example               # Configuration template
├── .gitignore                 # Git ignore rules
├── config.py                  # App settings (pydantic)
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Docker compose
├── start-backend.bat          # Windows: start backend
├── start-frontend.bat         # Windows: start frontend
│
├── core/                      # Core RAG modules
│   ├── __init__.py            # Package init
│   ├── ingestion.py           # Document loading + chunking
│   ├── retriever.py           # Hybrid search + reranking
│   ├── generator.py           # Claude generation + streaming
│   ├── router.py              # Query classification
│   ├── agent.py               # Agentic multi-step RAG
│   ├── multimodal.py          # PDF + image extraction
│   ├── tree_indexer.py        # PDF → hierarchical tree index (local engine)
│   ├── tree_search.py         # LLM reasoning-based tree retrieval
│   ├── pageindex_retriever.py # Local PageIndex orchestrator
│   └── evaluation.py          # RAG quality eval
│
├── api/                       # FastAPI server
│   ├── __init__.py            # Package init
│   ├── server.py              # All API endpoints
│   ├── auth.py                # JWT authentication
│   └── database.py            # SQLite persistence
│
├── frontend/                  # React UI
│   ├── package.json           # Node dependencies
│   ├── public/
│   │   └── index.html         # HTML template
│   └── src/
│       ├── index.js           # React entry point
│       ├── index.css          # Styles (dark/light theme)
│       └── App.js             # Main React app
│
├── tests/
│   ├── __init__.py            # Package init
│   └── eval_cases.json        # Test cases
│
└── data/                      # Created at runtime (gitignored)
    ├── chroma_db/             # Vector store
    └── rag_assistant.db       # SQLite database
```

---

## Windows Setup — Step by Step

### Step 1: Install Prerequisites

1. **Python 3.11+**: Download from https://python.org/downloads
   - IMPORTANT: Check "Add Python to PATH" during install
   - Verify: Open Command Prompt and type `python --version`

2. **Node.js 18+**: Download LTS from https://nodejs.org
   - Verify: `node --version` and `npm --version`

### Step 2: Create Project Folder

Open Command Prompt and run:

```cmd
mkdir D:\RAGv2
```

Copy ALL the downloaded files into `D:\RAGv2` keeping the exact folder structure shown above. Make sure these files exist:

```cmd
dir D:\RAGv2\config.py
dir D:\RAGv2\core\__init__.py
dir D:\RAGv2\api\__init__.py
dir D:\RAGv2\frontend\src\App.js
dir D:\RAGv2\frontend\src\index.css
dir D:\RAGv2\frontend\src\index.js
dir D:\RAGv2\frontend\public\index.html
```

If ANY file is missing, the app will not work. Every file listed above is required.

### Step 3: Create Python Virtual Environment

```cmd
cd D:\RAGv2
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your command prompt.

### Step 4: Install Python Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

This will install about 50+ packages including PyTorch (large download, ~2GB first time). Wait for it to finish completely.

If you get a build error:
- Install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Check "Desktop development with C++" and install
- Then retry `pip install -r requirements.txt`

### Step 5: Configure Environment

```cmd
copy .env.example .env
notepad .env
```

Change this line to your real API key:
```
RAG_ANTHROPIC_API_KEY=sk-ant-api03-your-real-key-here
```

Also change the JWT secret to any random string:
```
RAG_JWT_SECRET=my-super-secret-random-string-12345
```

Save and close notepad.

### Step 6: Install Frontend Dependencies

```cmd
cd frontend
npm install
cd ..
```

This installs React and all frontend packages. Should show 1000+ packages.

If `npm install` shows only ~100 packages:
```cmd
cd frontend
rd /s /q node_modules
del package-lock.json
npm cache clean --force
npm install
cd ..
```

### Step 7: Start the Backend (Terminal 1)

```cmd
cd D:\RAGv2
.venv\Scripts\activate
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Database initialized
INFO:     VectorStore ready: tech_docs (0 docs)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal open.

### Step 8: Start the Frontend (Terminal 2)

Open a NEW Command Prompt window:

```cmd
cd D:\RAGv2\frontend
npm start
```

Wait for it to compile. Your browser should auto-open to http://localhost:3000

### Step 9: Use the App

1. Click **"Index Documents"** in the left sidebar
2. Enter a folder path like `D:\Projects\my-code`
3. Click **Index** and wait for it to finish
4. Start asking questions in the chat!

---

## Troubleshooting

### "python is not recognized"
- Reinstall Python with "Add to PATH" checked
- Or use `py` instead of `python`

### "No module named 'core.retriever'"
- Make sure `core\__init__.py` exists (it can be empty)
- Make sure `api\__init__.py` exists (it can be empty)
- Make sure you're running from `D:\RAGv2\` (the project root)

### "'react-scripts' is not recognized"
```cmd
cd frontend
rd /s /q node_modules
del package-lock.json
npm cache clean --force
npm install
npm start
```

### "CORS error" or "Network Error" in browser
- Make sure the backend is running on port 8000
- Make sure both terminals are open

### Slow first query
Normal! The first query downloads the embedding model (~90MB) and reranker model. Subsequent queries are fast.

### "No documents indexed"
You need to click "Index Documents" first before you can ask questions.

---

## Features

- **Hybrid Search**: BM25 keyword + vector semantic search with rank fusion
- **Local PageIndex Engine**: Vectorless, reasoning-based RAG for PDFs — builds tree indexes, uses Claude tree search (no external API needed)
- **Query Router**: Auto-classifies your question type for optimal retrieval
- **Agent Mode**: Claude searches iteratively until it finds the answer
- **Multi-Modal**: Index PDFs and images alongside code
- **Auth**: JWT login/signup with persistent sessions
- **Analytics**: Query stats, latency charts, category breakdown
- **File Tree**: Browse all indexed files in the right panel
- **Dark/Light Theme**: Toggle in the top bar
- **Streaming**: Watch answers appear in real-time
- **Code Highlighting**: Syntax-colored code blocks with copy buttons

### Local PageIndex Engine

Your project includes a local PageIndex-style engine for PDF documents. Instead of chunking and embedding, it builds a hierarchical tree index and uses Claude to reason through it — like a human expert scanning a table of contents.

**Enable it:**
```
# In .env — that's all you need! Uses your existing Anthropic key.
RAG_PAGEINDEX_ENABLED=true
```

Then in the UI: Settings → enable "PageIndex" → upload a PDF → select it → queries now use tree search.

**Architecture:**
- `core/tree_indexer.py` — Extracts pages with PyMuPDF, uses Claude to detect/generate a hierarchical ToC tree with summaries
- `core/tree_search.py` — Presents tree outline to Claude, which reasons about which branches to explore, then generates cited answers
- `core/pageindex_retriever.py` — Orchestrates uploads, caching, and query routing

---

## Configuration Reference

All settings use `RAG_` prefix in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ANTHROPIC_API_KEY` | — | Required |
| `RAG_LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `RAG_BM25_WEIGHT` | `0.3` | BM25 weight |
| `RAG_VECTOR_WEIGHT` | `0.7` | Vector weight |
| `RAG_TOP_K` | `10` | Initial retrieval |
| `RAG_RERANK_TOP_K` | `5` | After reranking |
| `RAG_AGENT_MAX_STEPS` | `5` | Agent steps |
| `RAG_PAGEINDEX_ENABLED` | `false` | Enable local tree-indexed PDF retrieval |
| `RAG_JWT_SECRET` | `change-me` | Auth secret |
| `RAG_CHUNK_SIZE` | `512` | Chunk tokens |

## Supported File Types

Code: `.py .js .ts .jsx .tsx .go .rs .java .rb .c .cpp .cs .sh`
Docs: `.md .mdx .rst .txt`
Config: `.yaml .yml .json .toml`
Web: `.html .css .scss`
Multi-modal: `.pdf .png .jpg .jpeg .gif .webp`
