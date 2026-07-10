"""WebSocket endpoint for real-time streaming queries."""

import asyncio
import json
import logging
import queue as stdlib_queue
import re
import threading

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api import database as db
from api.auth import decode_token
from api.dependencies import get_user_store
from api.helpers import (
    auto_title_session,
    convert_history,
    format_sources,
    process_memories_background,
    safe_retrieve_memories,
    save_messages,
)
from config import settings
from core import llm_client
from core import pageindex_retriever as pindex
from core.decomposer import decompose, is_multi_part, merge_hits
from core.gap_analyzer import analyze as analyze_gap
from core.gap_analyzer import analyze_pre
from core.generator import generate_stream
from core.hyde import generate_hypothetical_doc
from core.knowledge_graph import get_user_graph
from core.memory import optimize_context_chunks
from core.provenance import compute_provenance
from core.query_cache import get_user_cache, invalidate_user_cache, is_cache_eligible
from core.retriever import embed_texts, retrieve
from core.router import route_query
from core.web_augmenter import augment as web_augment

try:  # parent expansion (small-to-big) may not have landed yet — flag degrades to a no-op
    from core.retriever import expand_parents
except ImportError:
    expand_parents = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


async def _stream_text(websocket: WebSocket, text: str):
    """Stream pre-generated text to a WebSocket preserving newlines for markdown rendering."""
    for line in text.split("\n"):
        if line.strip():
            words = line.split(" ")
            chunk, CHUNK_SIZE = [], 12
            for i, w in enumerate(words):
                chunk.append(w)
                if len(chunk) >= CHUNK_SIZE or i == len(words) - 1:
                    await websocket.send_json({"type": "token", "token": " ".join(chunk) + " "})
                    chunk = []
        # Always send the newline to preserve markdown structure
        await websocket.send_json({"type": "token", "token": "\n"})


async def _stream_generation(websocket, query_text, hits, hist, memory_ctx):
    """Run generate_stream() in a producer thread and relay tokens over the socket.

    Returns (full_answer, stream_failed) — stream_failed is True when generation
    raised or the client went away mid-stream (truncated answers must not seed
    memories or the semantic cache).
    """
    token_queue: stdlib_queue.Queue = stdlib_queue.Queue()
    stop_event = threading.Event()

    def producer():
        try:
            for chunk in generate_stream(
                query=query_text, hits=hits, conversation_history=hist, memory_context=memory_ctx
            ):
                if stop_event.is_set():
                    break
                token_queue.put(("token", chunk))
        except Exception as e:
            token_queue.put(("error", str(e)))
        finally:
            token_queue.put(("done", None))

    threading.Thread(target=producer, daemon=True).start()

    collected_answer = []
    stream_failed = False
    while True:
        item_type, item_val = await asyncio.to_thread(token_queue.get)
        if item_type == "done":
            break
        elif item_type == "error":
            stream_failed = True
            await websocket.send_json({"type": "error", "message": item_val})
            break
        else:
            collected_answer.append(item_val)
            try:
                await websocket.send_json({"type": "token", "token": item_val})
            except Exception:
                stop_event.set()
                stream_failed = True
                break

    return "".join(collected_answer), stream_failed


_REFINE_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

_REFINE_SYSTEM = (
    "You refine web search queries. Given a user question, the previous search query, and snippets of what that "
    "search retrieved, produce ONE better web search query targeting the still-missing information. "
    'Respond with JSON only: {"query": "..."}'
)


def _refine_search_query(question, prev_search, hits):
    """One LLM call producing a refined web-search query. Returns None on any failure (fail-open)."""
    snippets = "\n".join(f"- {h.get('content', '')[:150]}" for h in hits[:5]) or "(nothing retrieved)"
    prompt = (
        f"Question: {question}\n"
        f"Previous search query: {prev_search}\n"
        f"Snippets retrieved so far:\n{snippets}\n\n"
        "The retrieved content still does not answer the question. Give a refined web search query."
    )
    try:
        raw = llm_client.chat(
            [{"role": "user", "content": prompt}],
            system=_REFINE_SYSTEM,
            max_tokens=100,
            temperature=0.3,
            json_schema=_REFINE_QUERY_SCHEMA,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        refined = str(data.get("query", "")).strip()
        return refined or None
    except Exception as e:
        logger.warning("Search-query refinement failed: %s", e)
        return None


@router.websocket("/api/ws")
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
                await websocket.send_json({"type": "done"})
                continue

            uid = user["id"]

            # ── Web search approval (gap-driven bounded research loop) ──
            if raw.get("type") == "web_search_approved":
                try:
                    await _handle_web_research(websocket, uid, raw)
                except WebSocketDisconnect:
                    raise
                except Exception as we:
                    logger.error("Web research failed: %s", we, exc_info=True)
                    try:
                        await websocket.send_json({"type": "error", "message": str(we)})
                        await websocket.send_json({"type": "done"})
                    except Exception:
                        pass
                continue

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
            use_graph = qd.get("use_graph", False)
            use_hyde = qd.get("use_hyde", False)
            use_splade = qd.get("use_splade", False)
            use_multiquery = qd.get("use_multiquery", False)
            use_parent_expand = qd.get("use_parent_expand", False)
            top_k_req = qd.get("top_k")
            is_first = not conv_history

            if not query_text:
                await websocket.send_json({"type": "error", "message": "Empty query"})
                await websocket.send_json({"type": "done"})
                continue

            s = await asyncio.to_thread(get_user_store, uid)
            # An empty store no longer hard-errors: the RAG path forces a direct
            # (memory-only) answer so users can chat before ingesting documents.
            force_direct = not use_pi and s.count == 0

            try:
                # ── Semantic query cache ──
                qcache = get_user_cache(uid)
                if qcache.size > 0 and is_cache_eligible(query_text):
                    q_vec = (await asyncio.to_thread(embed_texts, [query_text]))[0]
                    hit = qcache.lookup(np.array(q_vec, dtype="float32"), query_text=query_text)
                    if hit is not None:
                        await websocket.send_json(
                            {"type": "cache_hit", "sim": hit["sim"], "matched": hit["query_text"][:80]}
                        )
                        await websocket.send_json({"type": "sources", "sources": hit["sources"]})
                        await _stream_text(websocket, hit["answer"])
                        save_messages(session_id, query_text, hit["answer"], hit["sources"])
                        title = auto_title_session(session_id, query_text, is_first)
                        if title:
                            await websocket.send_json({"type": "session_renamed", "title": title})
                        await websocket.send_json({"type": "done"})
                        continue

                if use_pi and pi_doc:
                    await _handle_pageindex_query(
                        websocket,
                        uid,
                        query_text,
                        pi_doc,
                        session_id,
                        conv_history,
                        is_first,
                        use_memory,
                        use_routing,
                        use_hyde,
                    )
                else:
                    await _handle_rag_query(
                        websocket,
                        uid,
                        s,
                        query_text,
                        session_id,
                        conv_history,
                        is_first,
                        use_reranking,
                        use_hybrid,
                        use_routing,
                        use_memory,
                        use_graph,
                        use_hyde,
                        use_splade,
                        use_multiquery,
                        use_parent_expand,
                        top_k_req,
                        force_direct=force_direct,
                    )

                await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error("WS query processing error: %s", e, exc_info=True)
                await websocket.send_json({"type": "error", "message": str(e)})
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket session error: %s", e)


async def _handle_web_research(websocket, uid, raw):
    """Bounded research loop: augment → verify gap → refine query (≤ research_max_iters), then auto-regenerate.

    Payload fields beyond topic/query/token are read defensively — an older
    GapPrompt may send only those three, in which case regeneration runs without
    conversation history, persistence no-ops (session_id is None), and
    verification uses default retrieval toggles.
    """
    topic = raw.get("topic", "") or ""
    query = raw.get("query") or topic
    session_id = raw.get("session_id")
    conv_history = raw.get("conversation_history") or []
    opts = raw.get("opts") or {}
    use_memory = bool(raw.get("use_memory", True))
    # Verify/regenerate under the same retrieval config that produced the gap.
    retrieve_kwargs = {
        "use_reranking": bool(opts.get("use_reranking", True)),
        "use_hybrid": bool(opts.get("use_hybrid", True)),
        "use_hyde": bool(opts.get("use_hyde", False)),
        "use_splade": bool(opts.get("use_splade", False)),
        "use_multiquery": bool(opts.get("use_multiquery", False)),
    }

    store = await asyncio.to_thread(get_user_store, uid)
    await websocket.send_json({"type": "web_search_started", "topic": topic})

    search_topic = topic or query
    final_hits = []
    augmented = False
    max_iters = max(1, settings.research_max_iters)

    for iteration in range(1, max_iters + 1):
        try:
            result = await asyncio.to_thread(web_augment, search_topic, store, query)
            invalidate_user_cache(uid)
            # web_augment reports failures via result.error without raising —
            # only actually-ingested chunks justify a regeneration pass.
            augmented = augmented or result.chunks_added > 0
            await websocket.send_json(
                {
                    "type": "web_ingested",
                    "iteration": iteration,
                    "topic": result.topic,
                    "chunks_added": result.chunks_added,
                    "urls": result.urls_fetched,
                    "failed": result.urls_failed,
                    "error": result.error,
                }
            )
        except Exception as we:
            await websocket.send_json(
                {"type": "web_ingested", "iteration": iteration, "error": str(we), "chunks_added": 0, "urls": []}
            )
            break

        # Verify: did the newly ingested chunks close the gap?
        try:
            final_hits = await asyncio.to_thread(retrieve, store=store, query=query, **retrieve_kwargs)
            gap = analyze_pre(query, final_hits)
        except Exception as ge:
            logger.warning("Post-augment verification failed: %s", ge)
            break
        if not gap.is_gap or iteration >= max_iters:
            break

        refined = await asyncio.to_thread(_refine_search_query, query, search_topic, final_hits)
        if not refined or refined.lower() == search_topic.lower():
            break
        search_topic = refined
        await websocket.send_json({"type": "research_iteration", "iteration": iteration + 1, "query": refined})

    if not augmented or not query:
        await websocket.send_json({"type": "done"})
        return

    # Auto-regenerate from the final hit set instead of making the user re-ask.
    if not final_hits:
        try:
            final_hits = await asyncio.to_thread(retrieve, store=store, query=query, **retrieve_kwargs)
        except Exception as rerr:
            logger.warning("Post-augment retrieval failed: %s", rerr)
            final_hits = []

    sources = format_sources(final_hits)
    hist = convert_history(conv_history)
    memory_ctx = await asyncio.to_thread(safe_retrieve_memories, uid, query) if use_memory else None
    await websocket.send_json({"type": "sources", "sources": sources})
    optimized_hits = optimize_context_chunks(final_hits)
    full_answer, stream_failed = await _stream_generation(websocket, query, optimized_hits, hist, memory_ctx)

    save_messages(session_id, query, full_answer, sources)
    if not stream_failed:
        process_memories_background(uid, query, full_answer, session_id or "")
    await websocket.send_json({"type": "done"})


async def _handle_pageindex_query(
    websocket,
    uid,
    query_text,
    pi_doc,
    session_id,
    conv_history,
    is_first,
    use_memory,
    use_routing,
    use_hyde,
):
    """PageIndex path with full feature parity."""
    # 1. Memory retrieval
    pi_memory_ctx = await asyncio.to_thread(safe_retrieve_memories, uid, query_text) if use_memory else None

    # 2. Query decomposition
    pi_sub_queries = None
    if use_routing and is_multi_part(query_text):
        try:
            pi_sub_queries = await asyncio.to_thread(decompose, query_text)
            if len(pi_sub_queries) <= 1:
                pi_sub_queries = None
        except Exception as de:
            logger.warning("Query decomposition failed (PageIndex): %s", de)

    # 3. HyDE
    pi_hyde_query = None
    if use_hyde:
        try:
            pi_hyde_query = await asyncio.to_thread(generate_hypothetical_doc, query_text)
        except Exception as he:
            logger.warning("HyDE failed (PageIndex): %s", he)

    # 4. Build augmented history with memory context
    aug_history = list(conv_history)
    if pi_memory_ctx and pi_memory_ctx.count > 0:
        mem_block = "\n".join("- " + f.content for f in pi_memory_ctx.fragments)
        aug_history = [
            {"role": "user", "content": f"[Memory context]\n{mem_block}\n[/Memory context]"},
            {"role": "assistant", "content": "Noted the memory context."},
            *aug_history,
        ]

    # 5. Route + memories events
    await websocket.send_json({"type": "route", "route": {"category": "pageindex", "strategy": "tree_reasoning"}})
    if pi_memory_ctx and pi_memory_ctx.count > 0:
        await websocket.send_json({"type": "memories", "count": pi_memory_ctx.count})

    # 6. Tree search (decomposed parallel or single)
    if pi_sub_queries:
        pi_tasks = [
            asyncio.to_thread(
                pindex.chat_query, sq, doc_id=pi_doc, conversation_history=aug_history, search_query=pi_hyde_query
            )
            for sq in pi_sub_queries
        ]
        pi_results = await asyncio.gather(*pi_tasks)
        answer_parts, pi_retrieved_nodes = [], []
        for sq, r in zip(pi_sub_queries, pi_results):
            answer_parts.append(f"**{sq}**\n{r['answer']}")
            pi_retrieved_nodes.extend(r.get("retrieved_nodes", []))
        answer = "\n\n".join(answer_parts)
        await websocket.send_json({"type": "decomposed", "sub_queries": pi_sub_queries})
    else:
        pi_result = await asyncio.to_thread(
            pindex.chat_query, query_text, doc_id=pi_doc, conversation_history=aug_history, search_query=pi_hyde_query
        )
        answer = pi_result["answer"]
        pi_retrieved_nodes = pi_result.get("retrieved_nodes", [])

    # 7. Build sources from retrieved nodes
    pi_sources = [
        {
            "file": pi_doc,
            "lines": f"p{n.get('start_page', '?')}-{n.get('end_page', '?')}",
            "language": "pdf",
            "score": 0.6,
            "search_type": "tree_search",
            "preview": n.get("text", "")[:200],
        }
        for n in pi_retrieved_nodes
    ]
    await websocket.send_json({"type": "sources", "sources": pi_sources})

    # 8. Stream answer in chunks (preserving newlines for markdown)
    await _stream_text(websocket, answer)

    # 9. Provenance (before gap analysis — Signal 5 consumes the map)
    pi_prov = None
    if pi_retrieved_nodes and answer:
        try:
            pi_chunks = [
                {
                    "content": n.get("text", ""),
                    "metadata": {
                        "document_path": pi_doc,
                        "start_line": n.get("start_page", "?"),
                        "end_line": n.get("end_page", "?"),
                    },
                }
                for n in pi_retrieved_nodes
                if n.get("text")
            ]
            pi_mem_frags = pi_memory_ctx.fragments if pi_memory_ctx else []
            pi_hist_dicts = [{"role": m["role"], "content": m["content"]} for m in (conv_history or [])]
            pi_prov = await asyncio.to_thread(
                compute_provenance, answer, pi_chunks, pi_mem_frags, query_text, pi_hist_dicts
            )
            if pi_prov:
                await websocket.send_json({"type": "provenance", "map": pi_prov.to_dict()})
        except Exception as pe:
            logger.warning("Provenance failed (PageIndex): %s", pe)

    # 10. Gap analysis
    gap = None
    try:
        pi_hits = [{"score": 0.6, "content": n.get("text", ""), "metadata": {}} for n in pi_retrieved_nodes]
        gap = analyze_gap(query_text, pi_hits, answer, provenance=pi_prov)
        if gap.is_gap:
            await websocket.send_json(
                {
                    "type": "gap_detected",
                    "stage": gap.stage,
                    "topic": gap.topic,
                    "reason": gap.reason,
                    "top_score": round(gap.top_score, 3),
                }
            )
    except Exception as ge:
        logger.warning("Gap analysis failed (PageIndex): %s", ge)

    # 11. Cache store — skip empty/gap answers and cache-ineligible queries
    if (
        answer.strip()
        and gap is not None
        and not gap.is_gap
        and not is_multi_part(query_text)
        and is_cache_eligible(query_text)
    ):
        try:
            q_vec = await asyncio.to_thread(lambda: embed_texts([query_text])[0])
            get_user_cache(uid).store(np.array(q_vec, dtype="float32"), query_text, answer, pi_sources, [])
        except Exception as ce:
            logger.warning("Cache store failed (PageIndex): %s", ce)

    # 12. Memory extraction
    process_memories_background(uid, query_text, answer, session_id or "")

    # 13. Persist to DB
    save_messages(session_id, query_text, answer, pi_sources)
    title = auto_title_session(session_id, query_text, is_first)
    if title:
        await websocket.send_json({"type": "session_renamed", "title": title})


async def _handle_rag_query(
    websocket,
    uid,
    s,
    query_text,
    session_id,
    conv_history,
    is_first,
    use_reranking,
    use_hybrid,
    use_routing,
    use_memory,
    use_graph,
    use_hyde,
    use_splade,
    use_multiquery,
    use_parent_expand,
    top_k_req,
    force_direct=False,
):
    """Standard RAG path."""
    route = route_query(query_text) if use_routing else None
    top_k = top_k_req or (route.suggested_top_k if route else settings.top_k)

    eval_config = {
        "use_reranking": use_reranking,
        "use_hybrid": use_hybrid,
        "use_routing": use_routing,
        "use_memory": use_memory,
        "use_graph": use_graph,
        "use_hyde": use_hyde,
        "use_splade": use_splade,
        "use_multiquery": use_multiquery,
        "use_parent_expand": use_parent_expand,
        "top_k": top_k,
        "route": route.category if route else None,
    }

    # ── Direct path: learned router says no retrieval, or the store is empty ──
    # Memory-only chat: no retrieval/graph/rerank, no gap analysis, no cache store.
    if force_direct or (route is not None and not route.needs_retrieval):
        memory_ctx = await asyncio.to_thread(safe_retrieve_memories, uid, query_text) if use_memory else None
        await websocket.send_json({"type": "sources", "sources": []})
        await websocket.send_json({"type": "route", "route": {"category": "direct", "strategy": "no_retrieval"}})
        if memory_ctx and memory_ctx.count > 0:
            await websocket.send_json({"type": "memories", "count": memory_ctx.count})

        hist = convert_history(conv_history)
        full_answer, stream_failed = await _stream_generation(websocket, query_text, [], hist, memory_ctx)

        eval_config["route"] = "direct"
        save_messages(
            session_id, query_text, full_answer, [], metadata={"eval": {"config": eval_config, "contexts": []}}
        )
        title = auto_title_session(session_id, query_text, is_first)
        if title:
            await websocket.send_json({"type": "session_renamed", "title": title})
        if not stream_failed:
            process_memories_background(uid, query_text, full_answer, session_id or "")
        return

    # ── Query decomposition ──
    sub_queries = None
    if use_routing and is_multi_part(query_text):
        try:
            sub_queries = await asyncio.to_thread(decompose, query_text)
            if len(sub_queries) <= 1:
                sub_queries = None
        except Exception as de:
            logger.warning("Query decomposition failed: %s", de)
            sub_queries = None

    if sub_queries:
        # Sub-queries are already expansions — multiquery on top would add one
        # LLM call and up to 8 searches per sub-query for no recall gain.
        tasks = [
            asyncio.to_thread(
                retrieve,
                store=s,
                query=sq,
                top_k=top_k,
                use_reranking=use_reranking,
                use_hybrid=use_hybrid,
                use_hyde=use_hyde,
                use_splade=use_splade,
            )
            for sq in sub_queries
        ]
        hits_per_query = list(await asyncio.gather(*tasks))
        hits = merge_hits(hits_per_query, max_total=top_k * 2)
        await websocket.send_json({"type": "decomposed", "sub_queries": sub_queries})
    else:
        hits = await asyncio.to_thread(
            retrieve,
            store=s,
            query=query_text,
            top_k=top_k,
            use_reranking=use_reranking,
            use_hybrid=use_hybrid,
            use_hyde=use_hyde,
            use_splade=use_splade,
            use_multiquery=use_multiquery,
        )

    # Graph-walk augmentation
    graph_traversal = None
    if use_graph:
        try:
            kg = get_user_graph(uid)
            if kg.graph.number_of_nodes() > 0:
                graph_hits, graph_traversal = await asyncio.to_thread(
                    kg.graph_retrieve,
                    query_text,
                    s,
                    top_k,
                    seed_chunk_ids=[h.get("id") for h in hits if h.get("id")],
                )
                existing_ids = {h.get("id") for h in hits}
                for gh in graph_hits:
                    if gh.get("id") not in existing_ids:
                        hits.append(gh)
                        existing_ids.add(gh["id"])
        except Exception as ge:
            logger.warning("Graph retrieval failed: %s", ge)

    # Small-to-big parent expansion (post-rerank, post-graph-merge): generation
    # sees stitched parent context; sources keep the original chunk line ranges.
    gen_hits = hits
    if use_parent_expand and expand_parents is not None and hits:
        try:
            gen_hits = await asyncio.to_thread(expand_parents, s, hits)
        except Exception as ee:
            logger.warning("Parent expansion failed: %s", ee)
            gen_hits = hits

    sources = format_sources(hits)
    hist = convert_history(conv_history)
    optimized_hits = optimize_context_chunks(gen_hits)

    # Memory retrieval
    memory_ctx = await asyncio.to_thread(safe_retrieve_memories, uid, query_text) if use_memory else None

    await websocket.send_json({"type": "sources", "sources": sources})
    if route:
        await websocket.send_json(
            {"type": "route", "route": {"category": route.category, "strategy": route.retrieval_strategy}}
        )
    if memory_ctx and memory_ctx.count > 0:
        await websocket.send_json({"type": "memories", "count": memory_ctx.count})
    if graph_traversal is not None:
        await websocket.send_json({"type": "graph_path", "traversal": graph_traversal})

    # Pre-generation retrieval-confidence gate (CRAG-style): surface the gap
    # prompt before a weak answer streams, but never abort generation.
    pre_gap_sent = False
    try:
        pre_gap = analyze_pre(query_text, hits)
        if pre_gap.is_gap:
            pre_gap_sent = True
            await websocket.send_json(
                {
                    "type": "gap_detected",
                    "stage": pre_gap.stage,
                    "topic": pre_gap.topic,
                    "reason": pre_gap.reason,
                    "top_score": round(pre_gap.top_score, 3),
                }
            )
    except Exception as ge:
        logger.warning("Pre-generation gap analysis failed: %s", ge)

    # Stream tokens via thread + queue bridge
    full_answer, stream_failed = await _stream_generation(websocket, query_text, optimized_hits, hist, memory_ctx)

    # Eval corpus (offline harness): the toggle config + the contexts generation saw.
    eval_meta = {"eval": {"config": eval_config, "contexts": [h["content"][:800] for h in optimized_hits]}}
    save_messages(session_id, query_text, full_answer, sources, metadata=eval_meta)
    title = auto_title_session(session_id, query_text, is_first)
    if title:
        await websocket.send_json({"type": "session_renamed", "title": title})

    # A truncated answer must not seed memories or the semantic cache.
    if not stream_failed:
        process_memories_background(uid, query_text, full_answer, session_id or "")

    # Provenance (before gap analysis — Signal 5 consumes the map). Scored
    # against gen_hits: with parent expansion the answer derives from parent
    # text, which original chunks would falsely flag as novel.
    prov = None
    if gen_hits and full_answer:
        try:

            def _prov():
                mem_frags = memory_ctx.fragments if memory_ctx else []
                hist_dicts = [{"role": m.role, "content": m.content} for m in (hist or [])]
                return compute_provenance(full_answer, gen_hits, mem_frags, query_text, hist_dicts)

            prov = await asyncio.to_thread(_prov)
            if prov:
                await websocket.send_json({"type": "provenance", "map": prov.to_dict()})
        except Exception as pe:
            logger.warning("Provenance computation failed: %s", pe)

    # Gap analysis
    gap = None
    try:
        gap = analyze_gap(query_text, hits, full_answer, provenance=prov)
        if gap.is_gap and not pre_gap_sent:
            await websocket.send_json(
                {
                    "type": "gap_detected",
                    "stage": gap.stage,
                    "topic": gap.topic,
                    "reason": gap.reason,
                    "top_score": round(gap.top_score, 3),
                }
            )
    except Exception as ge:
        logger.warning("Gap analysis failed: %s", ge)

    # Cache store — after gap analysis so gap answers are never replayed; skip
    # multi-part and context-dependent queries whose answers don't transfer.
    # Fail closed: gap analysis raising (gap is None) or a pre-gen gap also skip.
    if (
        full_answer
        and not stream_failed
        and gap is not None
        and not gap.is_gap
        and not pre_gap_sent
        and not is_multi_part(query_text)
        and is_cache_eligible(query_text)
    ):
        try:
            q_vec = await asyncio.to_thread(lambda: embed_texts([query_text])[0])
            get_user_cache(uid).store(np.array(q_vec, dtype="float32"), query_text, full_answer, sources, hits)
        except Exception as ce:
            logger.warning("Query cache store failed: %s", ce)
