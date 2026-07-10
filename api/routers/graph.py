"""Knowledge graph endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_auth
from api.dependencies import get_user_store
from core.knowledge_graph import get_user_graph, save_user_graph

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.post("/build")
async def graph_build(user=Depends(require_auth)):
    uid = user["id"]
    store = await asyncio.to_thread(get_user_store, uid)
    graph = await asyncio.to_thread(get_user_graph, uid)
    graph.reset()
    result = await asyncio.to_thread(graph.build_from_store, store)
    await asyncio.to_thread(save_user_graph, uid)
    return result


@router.get("")
def graph_data(max_nodes: int = 200, user=Depends(require_auth)):
    uid = user["id"]
    graph = get_user_graph(uid)
    return graph.get_viz_data(max_nodes=max_nodes)


@router.get("/stats")
def graph_stats(user=Depends(require_auth)):
    uid = user["id"]
    graph = get_user_graph(uid)
    return graph.get_stats()


@router.get("/communities")
def graph_communities(user=Depends(require_auth)):
    """Community roster — pure graph math, no LLM."""
    uid = user["id"]
    graph = get_user_graph(uid)
    return graph.get_communities()


@router.post("/communities/{community_id}/summary")
async def graph_community_summary(community_id: int, user=Depends(require_auth)):
    """Lazy LLM summary of one community — generated on first request, cached in the graph JSON."""
    uid = user["id"]
    graph = await asyncio.to_thread(get_user_graph, uid)
    if not any(nd.get("community") == community_id for _, nd in graph.graph.nodes(data=True)):
        raise HTTPException(status_code=404, detail=f"Unknown community: {community_id}")
    store = await asyncio.to_thread(get_user_store, uid)
    summary = await asyncio.to_thread(graph.summarize_community, community_id, store)
    return {"community_id": community_id, "summary": summary}
