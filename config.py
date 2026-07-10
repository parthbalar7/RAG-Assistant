"""
Configuration for the RAG assistant.
All settings use RAG_ prefix as environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.1)

    # --- Ollama (local LLM) ---
    llm_backend: str = Field(default="anthropic", description="'anthropic' or 'ollama'")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Primary Ollama server URL")
    ollama_extra_nodes: str = Field(
        default="", description="Comma-separated extra Ollama URLs for load balancing (e.g. http://192.168.1.50:11434)"
    )
    ollama_model: str = Field(default="qwen2.5-coder:14b", description="Default Ollama chat model")
    ollama_memory_model: str = Field(default="llama3.2:3b", description="Fast model for memory extraction")
    ollama_num_ctx: int = Field(
        default=16384,
        description="Context window (num_ctx) for Ollama calls — Ollama's 4k default silently truncates RAG prompts",
    )
    ollama_keep_alive: str = Field(
        default="1h", description="How long Ollama keeps the chat model loaded after a call (e.g. '5m', '1h')"
    )

    # --- Embeddings ---
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_backend: str = Field(
        default="torch", description="SentenceTransformer backend: 'torch' or 'onnx' (2-3x faster on CPU)"
    )
    # Recommended upgrade: Alibaba-NLP/gte-modernbert-base (768-dim, 8192-token window).
    # Changing the model requires re-embedding: run scripts/migrate_embeddings.py.

    # --- Reranker ---
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model ID; e.g. Alibaba-NLP/gte-reranker-modernbert-base or answerdotai/answerai-colbert-small-v1",
    )
    reranker_type: str = Field(
        default="cross-encoder", description="'cross-encoder' (sigmoid scores) or 'colbert' (MaxSim via rerankers pkg)"
    )

    # --- Vector Store ---
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    collection_name: str = Field(default="tech_docs")

    # --- Chunking ---
    # all-MiniLM-L6-v2 truncates at 256 wordpieces, so chunks must stay under that
    # (incl. the ~25-token breadcrumb prepended at ingest). Re-ingest after changing.
    chunk_size: int = Field(default=224)
    chunk_overlap: int = Field(default=32)
    min_chunk_size: int = Field(default=10)

    # --- Retrieval ---
    top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.25)

    # --- Hybrid search ---
    bm25_weight: float = Field(default=0.3)
    vector_weight: float = Field(default=0.7)

    # --- Progressive indexing ---
    ingest_skip_unchanged: bool = Field(
        default=True, description="Skip re-embedding files whose content hash is unchanged since last ingest"
    )
    sparse_rebuild_debounce_s: float = Field(
        default=2.0,
        description="Rebuild BM25/SPLADE in a background thread, debounced; 0 restores synchronous rebuilds",
    )
    graph_incremental: bool = Field(
        default=True, description="Merge newly ingested chunks into an existing knowledge graph (ner mode only)"
    )

    # --- Query cache ---
    query_cache_ttl_hours: float = Field(
        default=24.0, description="Semantic query-cache entries older than this are ignored (0 disables TTL)"
    )

    # --- SPLADE (learned sparse retrieval) ---
    splade_enabled: bool = Field(
        default=False, description="Build SPLADE index at startup (requires sentence-transformers>=3.0)"
    )
    splade_model: str = Field(
        default="prithivida/Splade_PP_en_v1",
        description="HuggingFace model ID for SPLADE (must be publicly accessible)",
    )

    # --- Agent ---
    agent_max_steps: int = Field(default=5)

    # --- Knowledge graph ---
    graph_extraction: str = Field(
        default="ner",
        description="Graph entity extraction: 'ner' (spaCy+regex, fast) | 'llm' (legacy) | 'hybrid' (ner + LLM for hub entities)",
    )

    # --- Contextual retrieval (Anthropic-style chunk situating) ---
    contextual_enrich: bool = Field(
        default=False, description="LLM-situate each chunk at ingest (slow on CPU Ollama — opt-in)"
    )

    # --- Parent expansion (small-to-big) ---
    parent_expand_budget: int = Field(default=800, description="Token budget for post-rerank parent expansion")

    # --- Gap research loop ---
    research_max_iters: int = Field(default=2, description="Max web-augment iterations per approved gap")

    # --- Learned router ---
    router_model_path: str = Field(default="./data/router.joblib")

    # --- Memory (token-optimized) ---
    memory_enabled: bool = Field(default=True, description="Enable long-term memory")
    memory_top_k: int = Field(default=5, description="Memories to retrieve per query")
    memory_extraction_model: str = Field(
        default="claude-haiku-3-5-20241022", description="Anthropic model for extraction (unused when backend=ollama)"
    )
    memory_auto_extract: bool = Field(default=True, description="Auto-extract after turns")
    memory_auto_summarize: bool = Field(default=True, description="Auto-summarize sessions")
    memory_extract_interval: int = Field(default=3, description="Extract every N turns")
    memory_min_answer_length: int = Field(default=100, description="Skip extraction if answer shorter")
    memory_max_fragments: int = Field(default=500, description="Soft cap; maintenance archives lowest scorers above it")
    memory_maintenance_interval_s: int = Field(default=600, description="Idle-maintenance sweep period (seconds)")
    memory_idle_threshold_s: int = Field(default=1800, description="User idle time before maintenance may run")

    # --- Token Optimization ---
    max_context_chunks: int = Field(default=5, description="Max chunks sent to LLM")
    max_chunk_preview_tokens: int = Field(default=300, description="Truncate each chunk")
    max_history_turns: int = Field(default=6, description="Max conversation history turns")

    # --- PageIndex (local engine) ---
    pageindex_api_key: str = Field(default="")
    pageindex_enabled: bool = Field(default=False)
    pageindex_toc_check_pages: int = Field(default=20)
    pageindex_enrich_summaries: bool = Field(default=True)

    # --- Knowledge Integrity & Risk Radar ---
    integrity_scan_max_chunks: int = Field(default=1200)
    integrity_max_issues: int = Field(default=50)

    # --- Auth ---
    jwt_secret: str = Field(default="change-me-in-production-please")
    jwt_expiry_hours: int = Field(default=72)

    # --- CORS ---
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins (set to ['*'] only for development)",
    )

    # --- Database ---
    database_path: str = Field(default="./data/rag_assistant.db")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Paths ---
    docs_directory: str = Field(default="./docs")

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
