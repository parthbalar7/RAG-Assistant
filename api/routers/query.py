"""Core query endpoints: health, stats, query, query/stream."""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import require_auth
from api.dependencies import get_user_store
from api.helpers import (
    auto_title_session,
    convert_history,
    format_sources,
    process_memories_background,
    safe_retrieve_memories,
    save_messages,
)
from api.models import QueryReq
from config import settings
from core import pageindex_retriever as pindex
from core.agent import run_agent
from core.generator import generate, generate_stream
from core.memory import optimize_context_chunks
from core.retriever import retrieve
from core.router import route_query_fast

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/stats")
def stats(user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    return {
        "collection_name": f"docs_{uid}",
        "document_count": s.count,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "pageindex_enabled": pindex.is_available(),
        "memory_enabled": settings.memory_enabled,
    }


@router.post("/query")
def query(req: QueryReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if not req.use_pageindex and s.count == 0:
        raise HTTPException(400, "No documents indexed")
    start = time.perf_counter()
    route = route_query_fast(req.query) if req.use_routing else None

    if req.use_agent:
        result = run_agent(
            query=req.query, store=s, retrieve_fn=retrieve, conversation_history=req.conversation_history
        )
        ms = (time.perf_counter() - start) * 1000
        save_messages(req.session_id, req.query, result.answer, result.sources)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "model": settings.llm_model,
            "usage": {"total_tokens": result.total_tokens},
            "retrieval_count": len(result.sources),
            "latency_ms": round(ms, 1),
            "memories_used": 0,
            "route": {"category": "agent", "strategy": "agent", "steps": len(result.steps)},
        }

    top_k = req.top_k or (route.suggested_top_k if route else settings.top_k)
    hits = retrieve(
        store=s,
        query=req.query,
        top_k=top_k,
        use_reranking=req.use_reranking,
        use_hybrid=req.use_hybrid,
        language_filter=req.language_filter or (route.language_hint if route else None),
    )
    hist = convert_history(req.conversation_history)
    optimized_hits = optimize_context_chunks(hits)

    memory_ctx = safe_retrieve_memories(uid, req.query) if req.use_memory else None
    resp = generate(query=req.query, hits=optimized_hits, conversation_history=hist, memory_context=memory_ctx)
    process_memories_background(uid, req.query, resp.answer, req.session_id or "")

    ms = (time.perf_counter() - start) * 1000
    sources = format_sources(hits)

    save_messages(req.session_id, req.query, resp.answer, sources, resp.usage)
    return {
        "answer": resp.answer,
        "sources": sources,
        "model": resp.model,
        "usage": resp.usage,
        "retrieval_count": len(hits),
        "latency_ms": round(ms, 1),
        "memories_used": memory_ctx.count if memory_ctx else 0,
        "route": {"category": route.category, "strategy": route.retrieval_strategy, "confidence": route.confidence}
        if route
        else None,
    }


@router.post("/query/stream")
def query_stream(req: QueryReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if not req.use_pageindex and s.count == 0:
        raise HTTPException(400, "No documents indexed")
    route = route_query_fast(req.query) if req.use_routing else None
    top_k = req.top_k or (route.suggested_top_k if route else settings.top_k)
    hits = []
    sources = []
    if not req.use_pageindex:
        hits = retrieve(
            store=s, query=req.query, top_k=top_k, use_reranking=req.use_reranking, use_hybrid=req.use_hybrid
        )
        sources = format_sources(hits)
    hist = convert_history(req.conversation_history)
    optimized_hits = optimize_context_chunks(hits)

    memory_ctx = safe_retrieve_memories(uid, req.query) if req.use_memory else None

    collected_answer = []
    is_first_message = not req.conversation_history

    def stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        if route:
            yield f"data: {json.dumps({'type': 'route', 'route': {'category': route.category, 'strategy': route.retrieval_strategy}})}\n\n"
        if memory_ctx and memory_ctx.count > 0:
            yield f"data: {json.dumps({'type': 'memories', 'count': memory_ctx.count})}\n\n"
        for chunk in generate_stream(
            query=req.query, hits=optimized_hits, conversation_history=hist, memory_context=memory_ctx
        ):
            collected_answer.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
        full_answer = "".join(collected_answer)
        save_messages(req.session_id, req.query, full_answer, sources)
        title = auto_title_session(req.session_id, req.query, is_first_message)
        if title:
            yield f"data: {json.dumps({'type': 'session_renamed', 'title': title})}\n\n"
        process_memories_background(uid, req.query, full_answer, req.session_id or "")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
