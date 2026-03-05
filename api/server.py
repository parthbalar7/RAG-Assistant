import logging
import os
import tempfile
import time
import json
import asyncio
import threading
import queue as stdlib_queue
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List

from config import settings
from core.ingestion import ingest_directory, load_single_file, chunk_document
from core.retriever import VectorStore, retrieve
from core.generator import generate, generate_stream, Message, RAGResponse
from core.router import route_query_fast
from core.agent import run_agent
from core import pageindex_retriever as pindex
from core.integrity import run_integrity_scan
from core.compliance import run_compliance_scan, FRAMEWORKS as COMPLIANCE_FRAMEWORKS
from core.memory import (
    get_memory_store, retrieve_memories, process_turn_memories,
    process_session_summary, MemoryFragment, optimize_context_chunks,
)
from api import database as db
from api.auth import hash_password, verify_password, create_token, get_current_user, require_auth, decode_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_user_stores: dict = {}
_user_stores_lock = threading.Lock()


def get_user_store(uid: str) -> VectorStore:
    with _user_stores_lock:
        if uid not in _user_stores:
            _user_stores[uid] = VectorStore(collection_name=f"docs_{uid}")
        return _user_stores[uid]


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    logger.info("RAG server ready")
    yield


app = FastAPI(title="RAG Assistant v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request Models ──

class AuthReq(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    display_name: str = ""


class QueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    conversation_history: Optional[List] = None
    top_k: Optional[int] = None
    language_filter: Optional[str] = None
    use_reranking: bool = True
    use_hybrid: bool = True
    use_routing: bool = True
    use_agent: bool = False
    use_pageindex: bool = False
    pageindex_doc_id: Optional[str] = None
    use_memory: bool = True


class IngestReq(BaseModel):
    directory: str


class IntegrityScanReq(BaseModel):
    persist: bool = True


class ComplianceScanReq(BaseModel):
    framework: str = Field(..., description="HIPAA | PCI_DSS | GDPR | SOC2 | OWASP")
    sample_size: int = Field(default=30, ge=5, le=200)


# ── Auth Endpoints ──

@app.post("/api/auth/register")
async def register(req: AuthReq):
    if db.get_user_by_username(req.username):
        raise HTTPException(400, "Username exists")
    u = db.create_user(req.username, hash_password(req.password), req.display_name)
    return {"token": create_token(u["id"], req.username), "user": u}


@app.post("/api/auth/login")
async def login(req: AuthReq):
    u = db.get_user_by_username(req.username)
    if not u or not verify_password(req.password, u["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_token(u["id"], u["username"]),
            "user": {"id": u["id"], "username": u["username"], "display_name": u["display_name"]}}


@app.get("/api/auth/me")
async def me(user=Depends(get_current_user)):
    if not user:
        return {"user": None}
    return {"user": {"id": user["id"], "username": user["username"], "display_name": user["display_name"]}}


# ── Session Endpoints ──

@app.post("/api/sessions")
async def create_session(user=Depends(require_auth)):
    return db.create_session(user["id"])


@app.get("/api/sessions")
async def list_sessions(user=Depends(require_auth)):
    return {"sessions": db.get_user_sessions(user["id"])}


@app.get("/api/sessions/{sid}/messages")
async def get_messages(sid: str):
    return {"messages": db.get_session_messages(sid)}


@app.put("/api/sessions/{sid}")
async def rename_session(sid: str, body: dict):
    db.update_session_title(sid, body.get("title", "Untitled"))
    return {"status": "ok"}


@app.delete("/api/sessions/{sid}")
async def del_session(sid: str):
    db.delete_session(sid)
    return {"status": "deleted"}


# ── Core Endpoints ──

@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/stats")
async def stats(user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    return {"collection_name": f"docs_{uid}", "document_count": s.count,
            "embedding_model": settings.embedding_model, "llm_model": settings.llm_model,
            "bm25_weight": settings.bm25_weight, "vector_weight": settings.vector_weight,
            "pageindex_enabled": pindex.is_available(),
            "memory_enabled": settings.memory_enabled}


@app.post("/api/ingest")
async def ingest(req: IngestReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    chunks = ingest_directory(req.directory)
    if not chunks:
        from pathlib import Path
        p = Path(req.directory)
        if not p.exists():
            raise HTTPException(400, f"Directory not found: {req.directory}")
        all_files = list(p.rglob("*"))
        file_count = sum(1 for f in all_files if f.is_file())
        exts = set(f.suffix.lower() for f in all_files if f.is_file() and f.suffix)
        from core.ingestion import SUPPORTED_EXTENSIONS
        supported_found = exts & SUPPORTED_EXTENSIONS
        raise HTTPException(400, f"No supported documents found in {req.directory}. Found {file_count} files with extensions: {', '.join(sorted(exts)[:20]) or 'none'}. Supported matches: {', '.join(sorted(supported_found)) or 'none'}")
    added = s.add_chunks(chunks)
    return {"chunks_indexed": added, "documents_processed": len(set(c.document_path for c in chunks)),
            "collection_total": s.count, "files": sorted(set(c.document_path for c in chunks))}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    total, processed = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            safe_name = f.filename.replace("\\", "/")
            p = os.path.join(tmp, safe_name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(await f.read())
        chunks = ingest_directory(tmp)
        if chunks:
            total = s.add_chunks(chunks)
            processed = sorted(set(c.document_path for c in chunks))
    return {"chunks_indexed": total, "files_processed": processed,
            "documents_processed": len(processed), "collection_total": s.count}


@app.post("/api/query")
async def query(req: QueryReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if not req.use_pageindex and s.count == 0:
        raise HTTPException(400, "No documents indexed")
    start = time.perf_counter()
    route = route_query_fast(req.query) if req.use_routing else None

    if req.use_agent:
        result = run_agent(query=req.query, store=s, retrieve_fn=retrieve,
                           conversation_history=req.conversation_history)
        ms = (time.perf_counter() - start) * 1000
        if req.session_id:
            db.add_message(req.session_id, "user", req.query)
            db.add_message(req.session_id, "assistant", result.answer, result.sources)
        db.log_query(uid, req.query, "agent", len(result.sources), ms, settings.llm_model,
                     result.total_tokens // 2, result.total_tokens // 2, "agent")
        return {"answer": result.answer, "sources": result.sources, "model": settings.llm_model,
                "usage": {"total_tokens": result.total_tokens}, "retrieval_count": len(result.sources),
                "latency_ms": round(ms, 1), "memories_used": 0,
                "route": {"category": "agent", "strategy": "agent", "steps": len(result.steps)}}

    top_k = req.top_k or (route.suggested_top_k if route else settings.top_k)
    hits = retrieve(store=s, query=req.query, top_k=top_k,
                    use_reranking=req.use_reranking, use_hybrid=req.use_hybrid,
                    language_filter=req.language_filter or (route.language_hint if route else None))
    hist = [Message(role=m["role"], content=m["content"]) for m in req.conversation_history] if req.conversation_history else None

    # Optimize chunks to reduce tokens sent to LLM
    optimized_hits = optimize_context_chunks(hits)

    # Memory retrieval step
    memory_ctx = None
    if req.use_memory and settings.memory_enabled:
        try:
            mem_store = get_memory_store(uid or "anonymous")
            memory_ctx = retrieve_memories(mem_store, req.query)
            if memory_ctx.count > 0:
                logger.info("Retrieved {} memories for query".format(memory_ctx.count))
        except Exception as me:
            logger.warning("Memory retrieval failed: {}".format(me))

    resp = generate(query=req.query, hits=optimized_hits, conversation_history=hist, memory_context=memory_ctx)

    # Memory extraction (after response, respects interval)
    if settings.memory_enabled and settings.memory_auto_extract:
        try:
            process_turn_memories(uid or "anonymous", req.query, resp.answer, req.session_id or "")
        except Exception as me:
            logger.warning("Memory extraction failed: {}".format(me))

    ms = (time.perf_counter() - start) * 1000

    sources = [{"file": h["metadata"].get("document_path", ""),
                "lines": f"{h['metadata'].get('start_line','?')}-{h['metadata'].get('end_line','?')}",
                "language": h["metadata"].get("language", ""),
                "score": round(h.get("rerank_score", h.get("score", 0)), 4),
                "search_type": h.get("search_type", "vector"),
                "preview": h["content"][:200]} for h in hits]

    if req.session_id:
        db.add_message(req.session_id, "user", req.query)
        db.add_message(req.session_id, "assistant", resp.answer, sources, resp.usage)
    db.log_query(uid, req.query, route.category if route else "general", len(hits), ms,
                 resp.model, resp.usage.get("input_tokens", 0), resp.usage.get("output_tokens", 0),
                 "hybrid" if req.use_hybrid else "vector")
    return {"answer": resp.answer, "sources": sources, "model": resp.model, "usage": resp.usage,
            "retrieval_count": len(hits), "latency_ms": round(ms, 1),
            "memories_used": memory_ctx.count if memory_ctx else 0,
            "route": {"category": route.category, "strategy": route.retrieval_strategy,
                      "confidence": route.confidence} if route else None}


@app.post("/api/query/stream")
async def query_stream(req: QueryReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if not req.use_pageindex and s.count == 0:
        raise HTTPException(400, "No documents indexed")
    route = route_query_fast(req.query) if req.use_routing else None
    top_k = req.top_k or (route.suggested_top_k if route else settings.top_k)
    hits = []
    sources = []
    if not req.use_pageindex:
        hits = retrieve(store=s, query=req.query, top_k=top_k,
                        use_reranking=req.use_reranking, use_hybrid=req.use_hybrid)
        sources = [{"file": h["metadata"].get("document_path", ""),
                    "lines": f"{h['metadata'].get('start_line','?')}-{h['metadata'].get('end_line','?')}",
                    "language": h["metadata"].get("language", ""),
                    "score": round(h.get("rerank_score", h.get("score", 0)), 4),
                    "search_type": h.get("search_type", ""), "preview": h["content"][:200]} for h in hits]
    hist = [Message(role=m["role"], content=m["content"]) for m in req.conversation_history] if req.conversation_history else None

    optimized_hits = optimize_context_chunks(hits)

    # Memory retrieval for streaming
    memory_ctx = None
    if req.use_memory and settings.memory_enabled:
        try:
            mem_store = get_memory_store(uid)
            memory_ctx = retrieve_memories(mem_store, req.query)
        except Exception as me:
            logger.warning("Memory retrieval failed: {}".format(me))

    collected_answer = []
    is_first_message = not req.conversation_history

    def stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        if route:
            yield f"data: {json.dumps({'type': 'route', 'route': {'category': route.category, 'strategy': route.retrieval_strategy}})}\n\n"
        if memory_ctx and memory_ctx.count > 0:
            yield f"data: {json.dumps({'type': 'memories', 'count': memory_ctx.count})}\n\n"
        for chunk in generate_stream(query=req.query, hits=optimized_hits, conversation_history=hist, memory_context=memory_ctx):
            collected_answer.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
        full_answer = "".join(collected_answer)
        # Save conversation to DB
        if req.session_id:
            db.add_message(req.session_id, "user", req.query)
            db.add_message(req.session_id, "assistant", full_answer, sources)
            if is_first_message:
                title = req.query[:50] + ('...' if len(req.query) > 50 else '')
                db.update_session_title(req.session_id, title)
                yield f"data: {json.dumps({'type': 'session_renamed', 'title': title})}\n\n"
        # Memory extraction after streaming (respects interval)
        if settings.memory_enabled and settings.memory_auto_extract:
            try:
                process_turn_memories(uid, req.query, full_answer, req.session_id or "")
            except Exception as me:
                logger.warning("Memory extraction failed: {}".format(me))
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.websocket("/api/ws")
async def ws_query_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()

            # Authenticate via token in message payload
            token_str = raw.get("token", "")
            user = None
            if token_str:
                payload = decode_token(token_str)
                if payload:
                    user = db.get_user_by_id(payload["sub"])

            if not user:
                await websocket.send_json({"type": "error", "message": "Authentication required"})
                continue

            uid = user["id"]
            qd = raw.get("query_data", {})
            query_text = qd.get("query", "")
            session_id = qd.get("session_id")
            conv_history = qd.get("conversation_history") or []
            use_pi = bool(qd.get("use_pageindex") and qd.get("pageindex_doc_id"))
            pi_doc = qd.get("pageindex_doc_id")
            use_reranking = qd.get("use_reranking", True)
            use_hybrid = qd.get("use_hybrid", True)
            use_routing = qd.get("use_routing", True)
            use_memory = qd.get("use_memory", True)
            top_k_req = qd.get("top_k")
            is_first = not conv_history

            if not query_text:
                await websocket.send_json({"type": "error", "message": "Empty query"})
                continue

            s = get_user_store(uid)
            if not use_pi and s.count == 0:
                await websocket.send_json({"type": "error", "message": "No documents indexed"})
                continue

            try:
                if use_pi and pi_doc:
                    # ── PageIndex path ──────────────────────────────────────
                    await websocket.send_json({"type": "sources", "sources": []})
                    await websocket.send_json({"type": "route", "route": {"category": "pageindex", "strategy": "tree_reasoning"}})
                    result = await asyncio.to_thread(
                        pindex.chat_query, query_text,
                        doc_id=pi_doc, conversation_history=conv_history
                    )
                    answer = result["answer"]
                    # Send word-by-word to simulate streaming
                    for tok in answer.split(' '):
                        await websocket.send_json({"type": "token", "token": tok + ' '})
                    if session_id:
                        db.add_message(session_id, "user", query_text)
                        db.add_message(session_id, "assistant", answer)
                        if is_first:
                            title = query_text[:50] + ('...' if len(query_text) > 50 else '')
                            db.update_session_title(session_id, title)
                            await websocket.send_json({"type": "session_renamed", "title": title})
                else:
                    # ── Standard RAG path ───────────────────────────────────
                    route = route_query_fast(query_text) if use_routing else None
                    top_k = top_k_req or (route.suggested_top_k if route else settings.top_k)

                    hits = await asyncio.to_thread(
                        retrieve, store=s, query=query_text, top_k=top_k,
                        use_reranking=use_reranking, use_hybrid=use_hybrid
                    )
                    sources = [{"file": h["metadata"].get("document_path", ""),
                                "lines": f"{h['metadata'].get('start_line','?')}-{h['metadata'].get('end_line','?')}",
                                "language": h["metadata"].get("language", ""),
                                "score": round(h.get("rerank_score", h.get("score", 0)), 4),
                                "search_type": h.get("search_type", ""), "preview": h["content"][:200]} for h in hits]
                    hist = [Message(role=m["role"], content=m["content"]) for m in conv_history] if conv_history else None
                    optimized_hits = optimize_context_chunks(hits)

                    # Memory retrieval
                    memory_ctx = None
                    if use_memory and settings.memory_enabled:
                        try:
                            mem_store = get_memory_store(uid)
                            memory_ctx = retrieve_memories(mem_store, query_text)
                        except Exception as me:
                            logger.warning("Memory retrieval failed: {}".format(me))

                    await websocket.send_json({"type": "sources", "sources": sources})
                    if route:
                        await websocket.send_json({"type": "route", "route": {"category": route.category, "strategy": route.retrieval_strategy}})
                    if memory_ctx and memory_ctx.count > 0:
                        await websocket.send_json({"type": "memories", "count": memory_ctx.count})

                    # Stream tokens via thread + queue bridge with cancel support
                    token_queue: stdlib_queue.Queue = stdlib_queue.Queue()
                    stop_event = threading.Event()

                    def producer():
                        try:
                            for chunk in generate_stream(query=query_text, hits=optimized_hits,
                                                         conversation_history=hist, memory_context=memory_ctx):
                                if stop_event.is_set():
                                    break
                                token_queue.put(("token", chunk))
                        except Exception as e:
                            token_queue.put(("error", str(e)))
                        finally:
                            token_queue.put(("done", None))

                    threading.Thread(target=producer, daemon=True).start()

                    collected_answer = []
                    while True:
                        item_type, item_val = await asyncio.to_thread(token_queue.get)
                        if item_type == "done":
                            break
                        elif item_type == "error":
                            await websocket.send_json({"type": "error", "message": item_val})
                            break
                        else:
                            collected_answer.append(item_val)
                            try:
                                await websocket.send_json({"type": "token", "token": item_val})
                            except Exception:
                                stop_event.set()
                                break

                    full_answer = "".join(collected_answer)

                    if session_id:
                        db.add_message(session_id, "user", query_text)
                        db.add_message(session_id, "assistant", full_answer, sources)
                        if is_first:
                            title = query_text[:50] + ('...' if len(query_text) > 50 else '')
                            db.update_session_title(session_id, title)
                            await websocket.send_json({"type": "session_renamed", "title": title})

                    if settings.memory_enabled and settings.memory_auto_extract:
                        try:
                            await asyncio.to_thread(process_turn_memories, uid, query_text, full_answer, session_id or "")
                        except Exception as me:
                            logger.warning("Memory extraction failed: {}".format(me))

                await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error(f"WS query processing error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")


@app.get("/api/files")
async def file_tree(user=Depends(require_auth)):
    return {"files": get_user_store(user["id"]).get_all_files()}


@app.delete("/api/files")
async def delete_file(path: str, user=Depends(require_auth)):
    s = get_user_store(user["id"])
    deleted = s.delete_file(path)
    if deleted == 0:
        raise HTTPException(404, "File not found in index")
    return {"deleted_chunks": deleted, "path": path}


@app.get("/api/analytics")
async def analytics(days: int = 7):
    return db.get_analytics(days)


# ── Evaluation ──

class EvalCaseReq(BaseModel):
    query: str
    expected_sources: Optional[List[str]] = None


class EvalRunReq(BaseModel):
    cases: List[EvalCaseReq] = Field(default_factory=list)


@app.post("/api/eval/run")
async def eval_run(req: EvalRunReq, user=Depends(require_auth)):
    from core.evaluation import evaluate_response
    uid = user["id"]
    s = get_user_store(uid)
    if s.count == 0:
        raise HTTPException(400, "No documents indexed")
    if not req.cases:
        raise HTTPException(400, "Provide at least one test case")

    def _run():
        case_results = []
        for tc in req.cases:
            hits = retrieve(store=s, query=tc.query, top_k=settings.top_k,
                            use_reranking=True, use_hybrid=True)
            from core.generator import generate
            resp = generate(query=tc.query, hits=hits)
            ev = evaluate_response(tc.query, resp.answer, hits, tc.expected_sources)
            case_results.append({
                "query": ev.query,
                "retrieval_hit": ev.retrieval_hit,
                "mrr": ev.mrr,
                "faithfulness": ev.faithfulness_score,
                "relevance": ev.relevance_score,
                "details": ev.details,
            })
        n = len(case_results)
        vf = [r["faithfulness"] for r in case_results if r["faithfulness"] >= 0]
        vr = [r["relevance"] for r in case_results if r["relevance"] >= 0]
        return {
            "total_cases": n,
            "retrieval_hit_rate": sum(r["retrieval_hit"] for r in case_results) / n,
            "avg_mrr": sum(r["mrr"] for r in case_results) / n,
            "avg_faithfulness": sum(vf) / len(vf) if vf else -1,
            "avg_relevance": sum(vr) / len(vr) if vr else -1,
            "cases": case_results,
        }

    metrics = await asyncio.to_thread(_run)
    db.save_eval_run(uid, metrics)
    return metrics


@app.get("/api/eval/history")
async def eval_history(user=Depends(require_auth)):
    return {"runs": db.get_eval_history(user["id"])}


# ── Knowledge Integrity & Risk Radar ──

@app.post("/api/integrity/scan")
async def integrity_scan(req: IntegrityScanReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if s.count == 0:
        raise HTTPException(400, "No documents indexed")
    prev = db.get_latest_integrity_fingerprints()
    result = run_integrity_scan(s, previous_fingerprints=prev)
    if req.persist:
        scan_id = db.save_integrity_scan(uid, result)
        result["scan_id"] = scan_id
    return result


@app.get("/api/integrity/history")
async def integrity_history(days: int = 30, limit: int = 30):
    return db.get_integrity_history(days=days, limit=limit)


@app.get("/api/integrity/scan/{scan_id}")
async def integrity_scan_detail(scan_id: str):
    res = db.get_integrity_scan(scan_id)
    if not res:
        raise HTTPException(404, "Scan not found")
    return res


@app.delete("/api/collection")
async def clear(user=Depends(require_auth)):
    get_user_store(user["id"]).clear()
    return {"status": "cleared"}


# ── Compliance ──

@app.get("/api/compliance/frameworks")
async def compliance_frameworks():
    return {"frameworks": {k: {"name": v["name"], "full_name": v["full_name"]} for k, v in COMPLIANCE_FRAMEWORKS.items()}}


@app.post("/api/compliance/scan")
async def compliance_scan(req: ComplianceScanReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    if s.count == 0:
        raise HTTPException(400, "No documents indexed")
    if req.framework not in COMPLIANCE_FRAMEWORKS:
        raise HTTPException(400, f"Unknown framework. Choose from: {list(COMPLIANCE_FRAMEWORKS.keys())}")
    try:
        result = await asyncio.to_thread(run_compliance_scan, s, req.framework, req.sample_size)
    except Exception as e:
        logger.error("Compliance scan error: %s", e)
        raise HTTPException(500, str(e))
    scan_id = db.save_compliance_scan(uid, result)
    result["scan_id"] = scan_id
    return result


@app.get("/api/compliance/history")
async def compliance_history(user=Depends(require_auth)):
    return {"scans": db.get_compliance_history(user["id"])}


# ── Memory Endpoints ──

@app.get("/api/memory")
async def memory_list(user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    mem_store = get_memory_store(uid)
    return {"fragments": mem_store.get_all(), "count": mem_store.count,
            "enabled": settings.memory_enabled}


@app.get("/api/memory/search")
async def memory_search(q: str, top_k: int = 5, user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    mem_store = get_memory_store(uid)
    results = mem_store.search(q, top_k=top_k)
    return {"results": results, "query": q}


@app.post("/api/memory")
async def memory_add(body: dict, user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    mem_store = get_memory_store(uid)
    frag = MemoryFragment(
        content=body.get("content", ""),
        memory_type=body.get("memory_type", "fact"),
        importance=body.get("importance", 0.7),
        tags=body.get("tags", []),
        source_query="manual",
    )
    fid = mem_store.add_fragment(frag)
    return {"fragment_id": fid, "status": "stored"}


@app.delete("/api/memory/{fragment_id}")
async def memory_delete(fragment_id: str, user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    mem_store = get_memory_store(uid)
    return {"deleted": mem_store.delete_fragment(fragment_id)}


@app.delete("/api/memory")
async def memory_clear(user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    mem_store = get_memory_store(uid)
    mem_store.clear()
    return {"status": "cleared"}


@app.post("/api/memory/summarize-session/{session_id}")
async def memory_summarize(session_id: str, user=Depends(get_current_user)):
    uid = user["id"] if user else "anonymous"
    messages = db.get_session_messages(session_id)
    summary = process_session_summary(uid, session_id, messages)
    return {"summary": summary.to_dict() if summary else None}


# ── PageIndex Endpoints ──

class PageIndexSubmitReq(BaseModel):
    filepath: str = Field(..., description="Path to a PDF file on the server")
    mode: Optional[str] = None


class PageIndexQueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    doc_id: Optional[str] = None
    doc_ids: Optional[List] = None
    conversation_history: Optional[List] = None
    session_id: Optional[str] = None
    enable_citations: bool = False
    temperature: Optional[float] = None
    use_streaming: bool = False


class PageIndexRetrievalReq(BaseModel):
    doc_id: str
    query: str
    thinking: bool = False


@app.get("/api/pageindex/status")
async def pageindex_status():
    return {"enabled": pindex.is_available(), "has_api_key": bool(settings.anthropic_api_key), "engine": "local"}


@app.post("/api/pageindex/submit")
async def pageindex_submit(req: PageIndexSubmitReq):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled. Set RAG_PAGEINDEX_ENABLED=true in .env")
    try:
        return pindex.submit_document(req.filepath, mode=req.mode)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/pageindex/upload")
async def pageindex_upload(file: UploadFile = File(...)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled. Set RAG_PAGEINDEX_ENABLED=true in .env")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PageIndex only accepts PDF files")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = pindex.submit_document(tmp_path)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/pageindex/documents")
async def pageindex_list_docs(limit: int = 50, offset: int = 0):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.list_documents(limit, offset)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/pageindex/document/{doc_id}")
async def pageindex_doc_status(doc_id: str):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_document_status(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/pageindex/document/{doc_id}/metadata")
async def pageindex_doc_metadata(doc_id: str):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_document_metadata(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/pageindex/document/{doc_id}")
async def pageindex_doc_delete(doc_id: str):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.delete_document(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/pageindex/tree/{doc_id}")
async def pageindex_tree(doc_id: str, flat: bool = False, summary: bool = False):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        tree = pindex.get_tree(doc_id, include_summary=summary)
        if flat:
            return {"nodes": pindex.flatten_tree(tree), "doc_id": doc_id}
        return {"tree": tree, "outline": pindex.tree_to_outline(tree), "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/pageindex/ocr/{doc_id}")
async def pageindex_ocr(doc_id: str, format: str = "page"):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_ocr_results(doc_id, fmt=format)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/pageindex/query")
async def pageindex_query(req: PageIndexQueryReq, user=Depends(get_current_user)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    start = time.perf_counter()
    uid = user["id"] if user else None
    try:
        history = req.conversation_history or []
        doc_target = req.doc_ids if (req.doc_ids and len(req.doc_ids) > 1) else req.doc_id

        if req.use_streaming:
            def stream():
                for ev in pindex.chat_query_stream(
                    req.query, doc_id=doc_target,
                    conversation_history=history,
                    enable_citations=req.enable_citations,
                    temperature=req.temperature,
                ):
                    yield f"data: {json.dumps(ev)}\n\n"
            return StreamingResponse(stream(), media_type="text/event-stream")

        result = pindex.chat_query(
            req.query, doc_id=doc_target,
            conversation_history=history,
            enable_citations=req.enable_citations,
            temperature=req.temperature,
        )
        ms = (time.perf_counter() - start) * 1000
        if req.session_id:
            db.add_message(req.session_id, "user", req.query)
            db.add_message(req.session_id, "assistant", result["answer"])
        db.log_query(uid, req.query, "pageindex", 0, ms, "pageindex",
                     result.get("usage", {}).get("prompt_tokens", 0),
                     result.get("usage", {}).get("completion_tokens", 0), "pageindex")
        return {
            "answer": result["answer"], "method": "pageindex_chat",
            "doc_id": doc_target, "latency_ms": round(ms, 1),
            "usage": result.get("usage", {}), "sources": [],
            "route": {"category": "pageindex", "strategy": "tree_reasoning"},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/pageindex/retrieve")
async def pageindex_retrieve(req: PageIndexRetrievalReq):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        result = pindex.retrieve_and_wait(req.doc_id, req.query, thinking=req.thinking)
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/pageindex/markdown-to-tree")
async def pageindex_md_tree(file: UploadFile = File(...)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    if not file.filename.lower().endswith((".md", ".markdown")):
        raise HTTPException(400, "Only .md files accepted")
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="wb") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return pindex.markdown_to_tree(tmp_path)
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp_path)


# Serve frontend build if available
_fb = Path(__file__).parent.parent / "frontend" / "build"
if not _fb.exists():
    _fb = Path(__file__).parent.parent / "build"
if _fb.exists():
    app.mount("/", StaticFiles(directory=str(_fb), html=True), name="frontend")