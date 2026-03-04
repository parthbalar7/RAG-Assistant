"""
Tree Search — LLM reasoning-based retrieval over tree indexes.

Implements the core PageIndex retrieval algorithm:
1. Present the tree outline to Claude
2. Claude reasons about which branches are relevant to the query
3. Drill down into selected branches, loading actual page content
4. Extract the most relevant content
5. Generate a final answer with page citations

This replaces vector similarity with LLM REASONING — the model
navigates the document like a human expert would.
"""

import json
import logging
import re
from typing import Optional

import anthropic

from config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  LLM Call Helper
# ═══════════════════════════════════════════

def _call_claude(system: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.content[0].text


def _call_claude_stream(system: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.1):
    """Streaming version — yields text chunks."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ═══════════════════════════════════════════
#  Step 1: Tree Navigation (Branch Selection)
# ═══════════════════════════════════════════

NAVIGATE_SYSTEM = """You are a document retrieval expert. Given a document's hierarchical structure and a user query, determine which sections are most likely to contain the answer.

You must REASON about where the information would be located — like a human expert scanning a table of contents.

Output ONLY a JSON object — no markdown fences, no explanation."""

NAVIGATE_PROMPT = """QUERY: {query}

DOCUMENT: "{doc_title}"
{doc_description}

DOCUMENT STRUCTURE:
{outline}

TASK: Identify which sections (by node_id) are most likely to contain information relevant to the query.

Think step-by-step:
1. What kind of information is the query asking for?
2. Which top-level sections could contain this?
3. Which specific subsections are most relevant?

Output JSON:
{{
  "reasoning": "Brief explanation of why these sections are relevant",
  "selected_nodes": ["node_id_1", "node_id_2", ...],
  "confidence": "high" | "medium" | "low"
}}

Select 1-5 nodes. Prefer specific subsections over broad parent sections."""


def navigate_tree(query: str, tree_data: dict) -> dict:
    """Step 1: LLM reasons about which tree branches to explore.
    Returns: {"reasoning": "...", "selected_nodes": [...], "confidence": "..."}
    """
    outline = _tree_to_navigable_outline(tree_data.get("nodes", []))
    prompt = NAVIGATE_PROMPT.format(
        query=query,
        doc_title=tree_data.get("title", "Unknown"),
        doc_description=tree_data.get("description", ""),
        outline=outline,
    )

    raw = _call_claude(NAVIGATE_SYSTEM, prompt, max_tokens=1024)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: select first 3 nodes
        all_nodes = _collect_all_node_ids(tree_data.get("nodes", []))
        result = {
            "reasoning": "Failed to parse navigation result, selecting top nodes",
            "selected_nodes": all_nodes[:3],
            "confidence": "low",
        }

    logger.info(f"Tree navigation: selected {len(result.get('selected_nodes', []))} nodes "
                f"(confidence: {result.get('confidence', '?')})")
    return result


def _tree_to_navigable_outline(nodes: list, indent: int = 0) -> str:
    """Create an outline showing node_ids for the LLM to reference."""
    lines = []
    for node in nodes:
        prefix = "  " * indent
        nid = node.get("node_id", "?")
        title = node.get("title", "Untitled")
        sp = node.get("start_page", "?")
        ep = node.get("end_page", "?")
        summary = node.get("summary", "")
        summary_str = f" — {summary}" if summary else ""
        lines.append(f"{prefix}[{nid}] {title} (pp. {sp}–{ep}){summary_str}")
        if node.get("children"):
            lines.append(_tree_to_navigable_outline(node["children"], indent + 1))
    return "\n".join(lines)


def _collect_all_node_ids(nodes: list) -> list:
    """Collect all node_ids from a tree."""
    ids = []
    for node in nodes:
        if node.get("node_id"):
            ids.append(node["node_id"])
        if node.get("children"):
            ids.extend(_collect_all_node_ids(node["children"]))
    return ids


# ═══════════════════════════════════════════
#  Step 2: Content Extraction
# ═══════════════════════════════════════════

def extract_content_for_nodes(tree_data: dict, node_ids: list) -> list[dict]:
    """Extract actual page text for the selected nodes.
    Returns: [{"node_id": "...", "title": "...", "pages": [...], "text": "..."}]
    """
    pages = tree_data.get("pages", [])
    all_nodes = tree_data.get("nodes", [])

    # Build node lookup
    node_map = {}
    _build_node_map(all_nodes, node_map)

    results = []
    seen_pages = set()

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            continue

        start = node.get("start_page", 1)
        end = node.get("end_page", start)

        # Collect page texts
        node_text = ""
        node_pages = []
        for p in pages:
            if start <= p["page"] <= end and p["page"] not in seen_pages:
                node_text += f"\n--- Page {p['page']} ---\n{p['text']}\n"
                node_pages.append(p["page"])
                seen_pages.add(p["page"])

        if node_text.strip():
            results.append({
                "node_id": nid,
                "title": node.get("title", "Untitled"),
                "start_page": start,
                "end_page": end,
                "pages": node_pages,
                "text": node_text.strip(),
            })

    return results


def _build_node_map(nodes: list, node_map: dict):
    """Recursively build a flat map of node_id → node."""
    for node in nodes:
        if node.get("node_id"):
            node_map[node["node_id"]] = node
        if node.get("children"):
            _build_node_map(node["children"], node_map)


# ═══════════════════════════════════════════
#  Step 3: Answer Generation
# ═══════════════════════════════════════════

ANSWER_SYSTEM = """You are a knowledgeable assistant that answers questions using retrieved document sections.

RULES:
1. Answer based ONLY on the provided document content
2. Cite specific page numbers when possible, e.g. (p. 15)
3. If the retrieved content doesn't contain the answer, say so clearly
4. Be precise and thorough
5. Structure your answer clearly"""

ANSWER_PROMPT = """QUESTION: {query}

DOCUMENT: "{doc_title}"

RETRIEVED SECTIONS:
{sections}

RETRIEVAL REASONING: {reasoning}

Answer the question using the retrieved content. Cite page numbers."""


def generate_answer(
    query: str,
    tree_data: dict,
    retrieved_sections: list[dict],
    navigation_result: dict,
    stream: bool = False,
):
    """Step 3: Generate answer using retrieved content.
    
    If stream=True, returns a generator yielding text chunks.
    If stream=False, returns a dict with answer and metadata.
    """
    sections_text = ""
    all_pages = []
    for sec in retrieved_sections:
        sections_text += f"\n### [{sec['node_id']}] {sec['title']} (pp. {sec['start_page']}–{sec['end_page']})\n"
        sections_text += sec["text"][:8000]  # Cap per section
        sections_text += "\n"
        all_pages.extend(sec["pages"])

    if not sections_text.strip():
        no_result = "I couldn't find relevant content in the document for this question. The tree search didn't identify matching sections."
        if stream:
            def empty_gen():
                yield no_result
            return empty_gen()
        return {
            "answer": no_result,
            "retrieved_nodes": [],
            "pages_referenced": [],
            "confidence": "low",
        }

    prompt = ANSWER_PROMPT.format(
        query=query,
        doc_title=tree_data.get("title", "Unknown"),
        sections=sections_text[:24000],  # Keep within context
        reasoning=navigation_result.get("reasoning", ""),
    )

    if stream:
        return _call_claude_stream(ANSWER_SYSTEM, prompt, max_tokens=settings.llm_max_tokens)

    answer_text = _call_claude(ANSWER_SYSTEM, prompt, max_tokens=settings.llm_max_tokens, temperature=0.1)

    return {
        "answer": answer_text.strip(),
        "retrieved_nodes": [
            {"node_id": s["node_id"], "title": s["title"],
             "start_page": s["start_page"], "end_page": s["end_page"]}
            for s in retrieved_sections
        ],
        "pages_referenced": sorted(set(all_pages)),
        "confidence": navigation_result.get("confidence", "medium"),
        "reasoning": navigation_result.get("reasoning", ""),
    }


# ═══════════════════════════════════════════
#  Full Pipeline: Query → Answer
# ═══════════════════════════════════════════

def tree_search_query(
    query: str,
    tree_data: dict,
    stream: bool = False,
    conversation_history: list = None,
) -> dict:
    """Complete tree search pipeline:
    1. Navigate tree (LLM reasons about which branches)
    2. Extract content from selected nodes
    3. Generate answer with citations

    Returns: {
        "answer": "...",
        "method": "tree_search",
        "retrieved_nodes": [...],
        "pages_referenced": [...],
        "confidence": "high|medium|low",
        "reasoning": "Why these sections were selected"
    }
    """
    # Add conversation context to query if available
    full_query = query
    if conversation_history:
        context = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in conversation_history[-4:]  # Last 4 messages
        )
        full_query = f"Conversation context:\n{context}\n\nCurrent question: {query}"

    # Step 1: Navigate
    nav_result = navigate_tree(full_query, tree_data)
    selected_ids = nav_result.get("selected_nodes", [])

    if not selected_ids:
        return {
            "answer": "The tree search couldn't identify relevant sections for this query.",
            "method": "tree_search",
            "retrieved_nodes": [],
            "pages_referenced": [],
            "confidence": "low",
            "reasoning": nav_result.get("reasoning", ""),
        }

    # Step 2: Extract content
    sections = extract_content_for_nodes(tree_data, selected_ids)

    # Step 3: Generate answer
    result = generate_answer(query, tree_data, sections, nav_result, stream=stream)

    if stream:
        # For streaming, return the generator plus metadata
        return {
            "stream": result,
            "method": "tree_search",
            "retrieved_nodes": [
                {"node_id": s["node_id"], "title": s["title"],
                 "start_page": s["start_page"], "end_page": s["end_page"]}
                for s in sections
            ],
            "pages_referenced": sorted(set(
                p for s in sections for p in s["pages"]
            )),
            "confidence": nav_result.get("confidence", "medium"),
            "reasoning": nav_result.get("reasoning", ""),
        }

    result["method"] = "tree_search"
    return result
