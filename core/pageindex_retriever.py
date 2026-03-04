"""
Local PageIndex Engine — orchestrates tree indexing + reasoning-based retrieval.

This replaces the external PageIndex API with a fully local engine:
  - PDF → hierarchical tree index (using Claude for structure detection)
  - LLM tree search (Claude reasons over tree to find relevant sections)
  - Answer generation with page citations

No external API needed. Uses your existing Anthropic API key.
Inspired by the PageIndex framework (https://pageindex.ai).
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

from config import settings
from core.tree_indexer import (
    generate_tree_index,
    get_stored_trees,
    get_tree_by_id,
    delete_tree,
    flatten_tree_nodes,
    tree_to_outline,
    extract_pages,
    TREES_DIR,
)
from core.tree_search import tree_search_query

logger = logging.getLogger(__name__)

# Upload storage for PDFs processed via the UI
UPLOAD_DIR = Path(settings.docs_directory).parent / "data" / "pageindex_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════
#  Availability
# ═══════════════════════════════════════════

def is_available() -> bool:
    """Local engine is available if Anthropic key is set and feature enabled."""
    return settings.pageindex_enabled and bool(settings.anthropic_api_key)


# ═══════════════════════════════════════════
#  Document Management
# ═══════════════════════════════════════════

def submit_document(filepath: str, mode: str = None) -> dict:
    """Process a PDF: copy to uploads dir, generate tree index.
    Returns: {"doc_id": "local-xxx", "status": "completed", "filename": "..."}
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Copy PDF to upload storage
    filename = os.path.basename(filepath)
    dest = UPLOAD_DIR / filename
    if str(Path(filepath).resolve()) != str(dest.resolve()):
        shutil.copy2(filepath, dest)

    # Generate tree index
    tree = generate_tree_index(str(dest))

    return {
        "doc_id": tree["doc_id"],
        "filename": filename,
        "status": "completed",
        "total_pages": tree.get("total_pages", 0),
        "title": tree.get("title", filename),
    }


def get_document_status(doc_id: str) -> dict:
    """Check if a tree index exists for this doc_id."""
    tree = get_tree_by_id(doc_id)
    if tree:
        return {
            "doc_id": doc_id,
            "status": "completed",
            "total_pages": tree.get("total_pages", 0),
            "title": tree.get("title", ""),
        }
    return {"doc_id": doc_id, "status": "not_found"}


def get_document_metadata(doc_id: str) -> dict:
    tree = get_tree_by_id(doc_id)
    if not tree:
        return {"doc_id": doc_id, "status": "not_found"}
    return {
        "doc_id": doc_id,
        "status": "completed",
        "title": tree.get("title", ""),
        "total_pages": tree.get("total_pages", 0),
        "source_file": tree.get("source_file", ""),
        "created_at": tree.get("created_at", ""),
        "generation_time_s": tree.get("generation_time_s", 0),
        "node_count": len(flatten_tree_nodes(tree.get("nodes", []))),
    }


def list_documents(limit: int = 50, offset: int = 0) -> dict:
    trees = get_stored_trees()
    return {
        "documents": trees[offset:offset + limit],
        "total": len(trees),
    }


def delete_document(doc_id: str) -> dict:
    deleted = delete_tree(doc_id)
    return {"doc_id": doc_id, "deleted": deleted}


# ═══════════════════════════════════════════
#  Tree Access
# ═══════════════════════════════════════════

def get_tree(doc_id: str, include_summary: bool = False) -> list:
    """Get the hierarchical tree nodes for a document."""
    tree = get_tree_by_id(doc_id)
    if not tree:
        raise ValueError(f"No tree index found for {doc_id}")
    return tree.get("nodes", [])


def get_ocr_results(doc_id: str, fmt: str = "page") -> dict:
    """Get extracted text (our equivalent of OCR).
    fmt: 'page' = per-page, 'raw' = concatenated, 'node' = by tree node
    """
    tree = get_tree_by_id(doc_id)
    if not tree:
        raise ValueError(f"No tree index found for {doc_id}")

    pages = tree.get("pages", [])

    if fmt == "raw":
        return {
            "doc_id": doc_id,
            "format": "raw",
            "text": "\n\n".join(p["text"] for p in pages),
        }
    elif fmt == "node":
        nodes = flatten_tree_nodes(tree.get("nodes", []))
        node_texts = []
        for node in nodes:
            sp = node.get("start_page", 1)
            ep = node.get("end_page", sp)
            text = "\n".join(
                p["text"] for p in pages if sp <= p["page"] <= ep
            )
            node_texts.append({
                "node_id": node["node_id"],
                "title": node["title"],
                "text": text,
            })
        return {"doc_id": doc_id, "format": "node", "nodes": node_texts}
    else:  # page
        return {"doc_id": doc_id, "format": "page", "pages": pages}


# ═══════════════════════════════════════════
#  Chat / Query — Reasoning-based Retrieval
# ═══════════════════════════════════════════

def chat_query(
    query: str,
    doc_id: Union[str, list, None] = None,
    conversation_history: list = None,
    enable_citations: bool = False,
    temperature: float = None,
) -> dict:
    """Query documents using LLM tree search (non-streaming).
    Returns: {"answer": "...", "usage": {...}, ...}
    """
    if not doc_id:
        raise ValueError("doc_id is required for tree search queries")

    # Handle single or multi-doc
    doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
    all_answers = []
    all_nodes = []
    all_pages = []

    for did in doc_ids:
        tree = get_tree_by_id(did)
        if not tree:
            all_answers.append(f"[Document {did} not found]")
            continue

        result = tree_search_query(
            query=query,
            tree_data=tree,
            stream=False,
            conversation_history=conversation_history,
        )
        all_answers.append(result.get("answer", ""))
        all_nodes.extend(result.get("retrieved_nodes", []))
        all_pages.extend(result.get("pages_referenced", []))

    combined = "\n\n".join(all_answers) if len(all_answers) > 1 else all_answers[0] if all_answers else ""

    return {
        "answer": combined,
        "usage": {},
        "retrieved_nodes": all_nodes,
        "pages_referenced": sorted(set(all_pages)),
        "method": "local_tree_search",
    }


def chat_query_stream(
    query: str,
    doc_id: Union[str, list, None] = None,
    conversation_history: list = None,
    enable_citations: bool = False,
    temperature: float = None,
):
    """Streaming tree search. Yields SSE dicts."""
    if not doc_id:
        yield {"type": "token", "token": "Error: doc_id is required for tree search."}
        yield {"type": "done"}
        return

    did = doc_id if isinstance(doc_id, str) else doc_id[0]
    tree = get_tree_by_id(did)
    if not tree:
        yield {"type": "token", "token": f"Document {did} not found."}
        yield {"type": "done"}
        return

    result = tree_search_query(
        query=query,
        tree_data=tree,
        stream=True,
        conversation_history=conversation_history,
    )

    stream_gen = result.get("stream")
    if stream_gen:
        for chunk in stream_gen:
            yield {"type": "token", "token": chunk}
    yield {"type": "done"}


# ═══════════════════════════════════════════
#  Legacy Retrieval (returns raw nodes)
# ═══════════════════════════════════════════

def retrieve_and_wait(doc_id: str, query: str, thinking: bool = False, timeout: int = 60) -> dict:
    """Retrieve relevant tree nodes without generating an answer."""
    tree = get_tree_by_id(doc_id)
    if not tree:
        raise ValueError(f"No tree index found for {doc_id}")

    from core.tree_search import navigate_tree, extract_content_for_nodes

    nav = navigate_tree(query, tree)
    selected = nav.get("selected_nodes", [])
    sections = extract_content_for_nodes(tree, selected)

    return {
        "doc_id": doc_id,
        "status": "completed",
        "query": query,
        "reasoning": nav.get("reasoning", ""),
        "retrieved_nodes": [
            {
                "node_id": s["node_id"],
                "title": s["title"],
                "start_page": s["start_page"],
                "end_page": s["end_page"],
                "text": s["text"][:2000],
            }
            for s in sections
        ],
    }


# ═══════════════════════════════════════════
#  Markdown → Tree
# ═══════════════════════════════════════════

def markdown_to_tree(filepath: str) -> dict:
    """Convert a markdown file to a tree structure based on headings."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    nodes = []
    stack = [{"children": nodes, "level": 0}]

    for i, line in enumerate(lines):
        match = None
        for level in range(1, 7):
            if line.startswith("#" * level + " "):
                match = (level, line[level + 1:].strip())
                break

        if match:
            level, title = match
            node = {
                "title": title,
                "node_id": f"{len(nodes) + 1:04d}",
                "start_page": i + 1,
                "end_page": i + 1,
                "summary": "",
                "children": [],
            }

            # Find parent at appropriate level
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append({"children": node["children"], "level": level, **node})

    return {
        "title": Path(filepath).stem,
        "total_pages": len(lines),
        "nodes": nodes,
        "source_file": os.path.basename(filepath),
    }
