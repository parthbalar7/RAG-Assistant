"""Memory management endpoints."""

from fastapi import APIRouter, Depends

from api import database as db
from api.auth import require_auth
from api.models import MemoryAddReq
from config import settings
from core.memory import MemoryFragment, consolidate_memories, get_memory_store, process_session_summary

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("")
def memory_list(user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    return {"fragments": mem_store.get_all(), "count": mem_store.count, "enabled": settings.memory_enabled}


@router.get("/search")
def memory_search(q: str, top_k: int = 5, user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    results = mem_store.search(q, top_k=top_k)
    return {"results": results, "query": q}


@router.post("")
def memory_add(body: MemoryAddReq, user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    frag = MemoryFragment(
        content=body.content,
        memory_type=body.memory_type,
        importance=body.importance,
        tags=body.tags,
        source_query="manual",
    )
    # Manual adds are authoritative (user explicitly typed them) — skip the synchronous
    # reconcile LLM round trip so this endpoint doesn't block its worker on the LLM.
    fid = mem_store.add_fragment(frag, reconcile=False)
    return {"fragment_id": fid, "status": "stored"}


@router.delete("/{fragment_id}")
def memory_delete(fragment_id: str, user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    return {"deleted": mem_store.delete_fragment(fragment_id)}


@router.delete("")
def memory_clear(user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    mem_store.clear()
    return {"status": "cleared"}


@router.post("/consolidate")
def memory_consolidate(user=Depends(require_auth)):
    uid = user["id"]
    mem_store = get_memory_store(uid)
    result = consolidate_memories(mem_store)
    return result


@router.post("/summarize-session/{session_id}")
def memory_summarize(session_id: str, user=Depends(require_auth)):
    uid = user["id"]
    messages = db.get_session_messages(session_id)
    summary = process_session_summary(uid, session_id, messages)
    return {"summary": summary.to_dict() if summary else None}
