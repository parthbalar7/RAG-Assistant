"""PageIndex (local PDF tree-based retrieval) endpoints."""

import asyncio
import json
import logging
import os
import tempfile
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api import database as db
from api.auth import require_auth
from api.models import PageIndexQueryReq, PageIndexRetrievalReq, PageIndexSubmitReq
from config import settings
from core import pageindex_retriever as pindex

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pageindex", tags=["pageindex"])


@router.get("/status")
def pageindex_status():
    return {"enabled": pindex.is_available(), "backend": settings.llm_backend, "engine": "local"}


@router.post("/submit")
async def pageindex_submit(req: PageIndexSubmitReq, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled. Set RAG_PAGEINDEX_ENABLED=true in .env")
    try:
        return await asyncio.to_thread(pindex.submit_document, req.filepath, mode=req.mode)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/upload")
async def pageindex_upload(file: UploadFile = File(...), user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled. Set RAG_PAGEINDEX_ENABLED=true in .env")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PageIndex only accepts PDF files")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        result = await asyncio.to_thread(pindex.submit_document, tmp_path)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    finally:
        os.unlink(tmp_path)


@router.get("/documents")
def pageindex_list_docs(limit: int = 50, offset: int = 0, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.list_documents(limit, offset)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.get("/document/{doc_id}")
def pageindex_doc_status(doc_id: str, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_document_status(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.get("/document/{doc_id}/metadata")
def pageindex_doc_metadata(doc_id: str, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_document_metadata(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/document/{doc_id}")
def pageindex_doc_delete(doc_id: str, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.delete_document(doc_id)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.get("/tree/{doc_id}")
def pageindex_tree(doc_id: str, flat: bool = False, summary: bool = False, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        tree = pindex.get_tree(doc_id, include_summary=summary)
        if flat:
            return {"nodes": pindex.flatten_tree_nodes(tree), "doc_id": doc_id}
        return {"tree": tree, "outline": pindex.tree_to_outline(tree), "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.get("/ocr/{doc_id}")
def pageindex_ocr(doc_id: str, format: str = "page", user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        return pindex.get_ocr_results(doc_id, fmt=format)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/query")
async def pageindex_query(req: PageIndexQueryReq, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    start = time.perf_counter()
    try:
        history = req.conversation_history or []
        doc_target = req.doc_ids if (req.doc_ids and len(req.doc_ids) > 1) else req.doc_id

        if req.use_streaming:

            def stream():
                for ev in pindex.chat_query_stream(
                    req.query,
                    doc_id=doc_target,
                    conversation_history=history,
                    enable_citations=req.enable_citations,
                    temperature=req.temperature,
                ):
                    yield f"data: {json.dumps(ev)}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        result = await asyncio.to_thread(
            pindex.chat_query,
            req.query,
            doc_id=doc_target,
            conversation_history=history,
            enable_citations=req.enable_citations,
            temperature=req.temperature,
        )
        ms = (time.perf_counter() - start) * 1000
        if req.session_id:
            db.add_message(req.session_id, "user", req.query)
            db.add_message(req.session_id, "assistant", result["answer"])
        return {
            "answer": result["answer"],
            "method": "pageindex_chat",
            "doc_id": doc_target,
            "latency_ms": round(ms, 1),
            "usage": result.get("usage", {}),
            "sources": [],
            "route": {"category": "pageindex", "strategy": "tree_reasoning"},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.post("/retrieve")
async def pageindex_retrieve(req: PageIndexRetrievalReq, user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    try:
        result = await asyncio.to_thread(pindex.retrieve_and_wait, req.doc_id, req.query, thinking=req.thinking)
        return result
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/markdown-to-tree")
async def pageindex_md_tree(file: UploadFile = File(...), user=Depends(require_auth)):
    if not pindex.is_available():
        raise HTTPException(503, "PageIndex not enabled")
    if not file.filename.lower().endswith((".md", ".markdown")):
        raise HTTPException(400, "Only .md files accepted")
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="wb") as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        return await asyncio.to_thread(pindex.markdown_to_tree, tmp_path)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    finally:
        os.unlink(tmp_path)
