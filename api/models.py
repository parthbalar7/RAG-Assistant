"""
Pydantic request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field

# ── Auth ──


class AuthReq(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    display_name: str = ""


# ── Sessions ──


class RenameSessionReq(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ── Query ──


class QueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    conversation_history: list | None = None
    top_k: int | None = None
    language_filter: str | None = None
    use_reranking: bool = True
    use_hybrid: bool = True
    use_routing: bool = True
    use_agent: bool = False
    use_pageindex: bool = False
    pageindex_doc_id: str | None = None
    use_memory: bool = True


# ── Ingestion ──


class IngestReq(BaseModel):
    directory: str
    contextual: bool | None = None  # override settings.contextual_enrich for this ingest


# ── Integrity ──


class IntegrityScanReq(BaseModel):
    persist: bool = True


# ── LLM ──


class LLMSwitchReq(BaseModel):
    backend: str  # "anthropic" or "ollama"
    model: str | None = None


# ── Memory ──


class MemoryAddReq(BaseModel):
    content: str = Field(..., min_length=1)
    memory_type: str = Field(default="fact")
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


# ── PageIndex ──


class PageIndexSubmitReq(BaseModel):
    filepath: str = Field(..., description="Path to a PDF file on the server")
    mode: str | None = None


class PageIndexQueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    doc_id: str | None = None
    doc_ids: list | None = None
    conversation_history: list | None = None
    session_id: str | None = None
    enable_citations: bool = False
    temperature: float | None = None
    use_streaming: bool = False


class PageIndexRetrievalReq(BaseModel):
    doc_id: str
    query: str
    thinking: bool = False
