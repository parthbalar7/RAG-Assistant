"""
Configuration for the RAG assistant.
All settings use RAG_ prefix as environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- LLM ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.1)

    # --- Embeddings ---
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # --- Vector Store ---
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    collection_name: str = Field(default="tech_docs")

    # --- Chunking ---
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    min_chunk_size: int = Field(default=10)

    # --- Retrieval ---
    top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.25)

    # --- Hybrid search ---
    bm25_weight: float = Field(default=0.3)
    vector_weight: float = Field(default=0.7)

    # --- Agent ---
    agent_max_steps: int = Field(default=5)

    # --- Memory (token-optimized) ---
    memory_enabled: bool = Field(default=True, description="Enable long-term memory")
    memory_top_k: int = Field(default=5, description="Memories to retrieve per query")
    memory_extraction_model: str = Field(default="claude-haiku-3-5-20241022", description="Cheapest model for extraction")
    memory_auto_extract: bool = Field(default=True, description="Auto-extract after turns")
    memory_auto_summarize: bool = Field(default=True, description="Auto-summarize sessions")
    memory_extract_interval: int = Field(default=3, description="Extract every N turns")
    memory_min_answer_length: int = Field(default=100, description="Skip extraction if answer shorter")

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

    # --- Database ---
    database_path: str = Field(default="./data/rag_assistant.db")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Paths ---
    docs_directory: str = Field(default="./docs")

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
