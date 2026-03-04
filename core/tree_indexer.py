"""
Tree Indexer — Local PageIndex-style tree generation engine.

Converts PDFs into hierarchical tree-structured indexes by:
1. Extracting text per page using PyMuPDF
2. Using Claude to detect/generate a table-of-contents tree
3. Enriching each node with LLM-generated summaries
4. Persisting the tree index as JSON for fast retrieval

Inspired by the PageIndex framework (https://pageindex.ai).
Runs 100% locally — only needs your existing Anthropic API key.
"""

import json
import logging
import os
import re
import hashlib
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import anthropic

from config import settings

logger = logging.getLogger(__name__)

TREES_DIR = Path(settings.docs_directory).parent / "data" / "tree_indexes"
TREES_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════
#  PDF Text Extraction
# ═══════════════════════════════════════════

def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF.
    Returns: [{"page": 1, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append({
            "page": i + 1,
            "text": text,
            "char_count": len(text),
        })
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


def pages_to_tagged_text(pages: list[dict], start: int = 0, end: int = None) -> str:
    """Convert pages to tagged text with <page_N> markers for LLM consumption."""
    end = end or len(pages)
    parts = []
    for p in pages[start:end]:
        parts.append(f"<page_{p['page']}>\n{p['text']}\n</page_{p['page']}>")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════
#  LLM Helpers
# ═══════════════════════════════════════════

def _call_claude(system: str, prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude for tree generation tasks."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.content[0].text


# ═══════════════════════════════════════════
#  Phase 1: Detect or Generate ToC
# ═══════════════════════════════════════════

TOC_DETECTION_SYSTEM = """You are an expert document analyst. Your task is to examine the first pages of a document and determine if there is an explicit Table of Contents (ToC).

If a ToC exists, extract it as a structured JSON tree.
If no ToC exists, generate one by analyzing the document structure (headings, sections, topic changes).

Output ONLY valid JSON — no markdown fences, no explanation."""

TOC_DETECTION_PROMPT = """Analyze this document and produce a hierarchical tree structure (like a table of contents).

DOCUMENT TEXT (first {n_pages} pages):
{text}

RULES:
1. Each node must have: "title" (string), "start_page" (int), "end_page" (int), "summary" (1-2 sentence description)
2. Nodes can have "children" (array of child nodes) for subsections
3. Page numbers must be actual page numbers from the <page_N> tags
4. Every page in the document must be covered by at least one node
5. Keep the hierarchy 2-4 levels deep maximum
6. Title should be extracted from the actual document text when possible

Output this exact JSON structure:
{{
  "title": "Document Title",
  "total_pages": {total_pages},
  "description": "Brief document description",
  "nodes": [
    {{
      "title": "Section Title",
      "start_page": 1,
      "end_page": 5,
      "summary": "What this section covers",
      "children": [
        {{
          "title": "Subsection Title",
          "start_page": 1,
          "end_page": 3,
          "summary": "What this subsection covers",
          "children": []
        }}
      ]
    }}
  ]
}}"""


def _detect_or_generate_toc(pages: list[dict], toc_check_pages: int = 20) -> dict:
    """Use Claude to detect an existing ToC or generate one from the document structure."""
    check_pages = min(toc_check_pages, len(pages))
    text = pages_to_tagged_text(pages, 0, check_pages)

    prompt = TOC_DETECTION_PROMPT.format(
        n_pages=check_pages,
        text=text,
        total_pages=len(pages),
    )

    raw = _call_claude(TOC_DETECTION_SYSTEM, prompt, max_tokens=8192)

    # Clean and parse JSON
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        tree = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse ToC JSON from first attempt, retrying with full doc...")
        tree = _generate_toc_from_full_doc(pages)

    return tree


# ═══════════════════════════════════════════
#  Phase 1b: Full-doc fallback for long docs
# ═══════════════════════════════════════════

def _generate_toc_from_full_doc(pages: list[dict]) -> dict:
    """For docs where the first N pages don't have a clear ToC,
    sample pages throughout the doc and generate a structure."""
    # Sample: first 5, every 10th page, last 5
    sampled_indices = set(range(min(5, len(pages))))
    sampled_indices.update(range(0, len(pages), max(1, len(pages) // 15)))
    sampled_indices.update(range(max(0, len(pages) - 5), len(pages)))
    sampled = sorted(sampled_indices)

    text_parts = []
    for i in sampled:
        p = pages[i]
        text_parts.append(f"<page_{p['page']}>\n{p['text'][:2000]}\n</page_{p['page']}>")
    text = "\n\n".join(text_parts)

    prompt = f"""Analyze these sampled pages from a {len(pages)}-page document and generate a hierarchical tree structure.

SAMPLED PAGES:
{text}

Generate a JSON tree covering ALL {len(pages)} pages. Infer section boundaries from content changes.
Output ONLY valid JSON with this structure:
{{"title": "...", "total_pages": {len(pages)}, "description": "...", "nodes": [
  {{"title": "...", "start_page": N, "end_page": M, "summary": "...", "children": []}}
]}}"""

    raw = _call_claude(TOC_DETECTION_SYSTEM, prompt, max_tokens=8192)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Ultimate fallback: one node per ~10 pages
        return _fallback_flat_tree(pages)


def _fallback_flat_tree(pages: list[dict]) -> dict:
    """Absolute fallback: create a flat tree with sections every ~10 pages."""
    nodes = []
    chunk_size = min(10, max(1, len(pages) // 5))
    for i in range(0, len(pages), chunk_size):
        end = min(i + chunk_size, len(pages))
        nodes.append({
            "title": f"Section (Pages {i+1}-{end})",
            "start_page": i + 1,
            "end_page": end,
            "summary": f"Content from pages {i+1} to {end}",
            "children": [],
        })
    return {
        "title": "Document",
        "total_pages": len(pages),
        "description": "Auto-generated flat structure",
        "nodes": nodes,
    }


# ═══════════════════════════════════════════
#  Phase 2: Enrich nodes with summaries
# ═══════════════════════════════════════════

SUMMARY_SYSTEM = """You are a document analyst. Summarize the given section concisely in 2-3 sentences.
Focus on the key topics, findings, or arguments. Output ONLY the summary text."""


def _enrich_node_summaries(tree: dict, pages: list[dict], max_depth: int = 3, depth: int = 0) -> dict:
    """Recursively enrich tree nodes with better summaries using actual page content."""
    if depth >= max_depth:
        return tree

    nodes = tree.get("nodes", [])
    for node in nodes:
        start = node.get("start_page", 1) - 1
        end = node.get("end_page", start + 1)
        # Get actual text (limited to ~3000 chars for speed)
        section_text = ""
        for p in pages[start:end]:
            section_text += p["text"][:1500] + "\n"
            if len(section_text) > 4000:
                break

        if section_text.strip() and (not node.get("summary") or len(node.get("summary", "")) < 20):
            try:
                summary = _call_claude(
                    SUMMARY_SYSTEM,
                    f"Section: {node.get('title', 'Untitled')}\n\nContent:\n{section_text[:4000]}",
                    max_tokens=200,
                )
                node["summary"] = summary.strip()
            except Exception as e:
                logger.warning(f"Failed to summarize node '{node.get('title')}': {e}")

        # Recurse into children
        if node.get("children"):
            _enrich_node_summaries(node, pages, max_depth, depth + 1)

    return tree


# ═══════════════════════════════════════════
#  Phase 3: Assign node IDs
# ═══════════════════════════════════════════

def _assign_node_ids(nodes: list, prefix: str = "") -> None:
    """Assign hierarchical node IDs like 0001, 0002, etc."""
    for i, node in enumerate(nodes):
        node["node_id"] = f"{prefix}{i+1:04d}" if prefix else f"{i+1:04d}"
        if node.get("children"):
            _assign_node_ids(node["children"], prefix=node["node_id"] + ".")


# ═══════════════════════════════════════════
#  Main: Generate Tree Index
# ═══════════════════════════════════════════

def generate_tree_index(
    pdf_path: str,
    enrich_summaries: bool = True,
    toc_check_pages: int = 20,
    force_rebuild: bool = False,
) -> dict:
    """Full pipeline: PDF → tree-structured index.

    Returns: {
        "doc_id": "local-xxx",
        "title": "...",
        "total_pages": N,
        "description": "...",
        "nodes": [...],
        "pages": [...],  # raw page texts
        "source_file": "...",
        "created_at": "..."
    }
    """
    pdf_path = str(pdf_path)
    doc_id = _doc_id_from_path(pdf_path)

    # Check cache
    cache_path = TREES_DIR / f"{doc_id}.json"
    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading cached tree index for {doc_id}")
        with open(cache_path) as f:
            return json.load(f)

    logger.info(f"Generating tree index for {pdf_path}...")
    start = time.time()

    # 1. Extract pages
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")

    # 2. Generate/detect ToC tree
    tree = _detect_or_generate_toc(pages, toc_check_pages)

    # 3. Assign node IDs
    _assign_node_ids(tree.get("nodes", []))

    # 4. Optionally enrich summaries with actual content
    if enrich_summaries and len(pages) <= 200:
        tree = _enrich_node_summaries(tree, pages)

    # 5. Assemble final result
    result = {
        "doc_id": doc_id,
        "title": tree.get("title", Path(pdf_path).stem),
        "total_pages": len(pages),
        "description": tree.get("description", ""),
        "nodes": tree.get("nodes", []),
        "pages": [{"page": p["page"], "text": p["text"]} for p in pages],
        "source_file": os.path.basename(pdf_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation_time_s": round(time.time() - start, 1),
    }

    # 6. Cache
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Tree index generated in {result['generation_time_s']}s → {cache_path}")

    return result


def _doc_id_from_path(pdf_path: str) -> str:
    """Generate a stable doc_id from file path + modification time."""
    stat = os.stat(pdf_path)
    key = f"{pdf_path}:{stat.st_size}:{stat.st_mtime}"
    return "local-" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════
#  Tree Utilities
# ═══════════════════════════════════════════

def get_stored_trees() -> list[dict]:
    """List all cached tree indexes."""
    trees = []
    for f in TREES_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
                trees.append({
                    "doc_id": data.get("doc_id", f.stem),
                    "title": data.get("title", "Unknown"),
                    "total_pages": data.get("total_pages", 0),
                    "source_file": data.get("source_file", ""),
                    "created_at": data.get("created_at", ""),
                })
        except Exception:
            continue
    return trees


def get_tree_by_id(doc_id: str) -> Optional[dict]:
    """Load a tree index by doc_id."""
    cache_path = TREES_DIR / f"{doc_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    # Search all files
    for f in TREES_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
                if data.get("doc_id") == doc_id:
                    return data
        except Exception:
            continue
    return None


def delete_tree(doc_id: str) -> bool:
    """Delete a cached tree index."""
    cache_path = TREES_DIR / f"{doc_id}.json"
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False


def flatten_tree_nodes(nodes: list, depth: int = 0) -> list:
    """Flatten tree into a list with depth info."""
    flat = []
    for node in nodes:
        flat.append({
            "title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
            "start_page": node.get("start_page"),
            "end_page": node.get("end_page"),
            "summary": node.get("summary", ""),
            "depth": depth,
        })
        if node.get("children"):
            flat.extend(flatten_tree_nodes(node["children"], depth + 1))
    return flat


def tree_to_outline(nodes: list, indent: int = 0) -> str:
    """Convert tree to human-readable outline."""
    lines = []
    for node in nodes:
        prefix = "  " * indent
        title = node.get("title", "Untitled")
        sp = node.get("start_page", "?")
        ep = node.get("end_page", "?")
        lines.append(f"{prefix}• {title} (pp. {sp}–{ep})")
        if node.get("children"):
            lines.append(tree_to_outline(node["children"], indent + 1))
    return "\n".join(lines)
