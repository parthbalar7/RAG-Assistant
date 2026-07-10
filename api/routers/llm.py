"""LLM backend status and switching endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_auth
from api.models import LLMSwitchReq
from config import settings
from core import llm_client as _llm_client

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/status")
async def llm_status():
    backend = _llm_client.get_backend()
    nodes = await asyncio.to_thread(_llm_client.get_node_status) if backend == "ollama" else None
    result = {
        "backend": backend,
        "model": _llm_client.get_model(),
        "memory_model": _llm_client.get_memory_model(),
        "ollama_url": settings.ollama_base_url,
        "ollama_reachable": any(n["reachable"] for n in nodes) if nodes is not None else None,
        "anthropic_key_set": bool(settings.anthropic_api_key),
    }
    if nodes is not None:
        result["nodes"] = nodes
    return result


@router.get("/models")
async def llm_models():
    models = await asyncio.to_thread(_llm_client.list_ollama_models)
    return {"models": models, "ollama_url": settings.ollama_base_url}


@router.post("/switch")
async def llm_switch(req: LLMSwitchReq, _user=Depends(require_auth)):
    try:
        _llm_client.set_backend(req.backend, req.model)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "backend": _llm_client.get_backend(),
        "model": _llm_client.get_model(),
        "memory_model": _llm_client.get_memory_model(),
    }
