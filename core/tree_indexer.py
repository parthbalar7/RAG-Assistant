"""
Tree Indexer — 100% Local Tree Generation (Zero LLM Calls)
═══════════════════════════════════════════════════════════

Converts PDFs into hierarchical tree-structured indexes using ONLY local analysis:

1. PyMuPDF built-in ToC extraction (PDF bookmarks / outline)
2. Font-size based heading detection (analyzes text blocks for large/bold text)
3. Regex pattern matching for numbered sections (1.0, 1.1, Chapter 1, etc.)
4. Text-density based summarization (first sentences of each section)

The synchronous path makes NO API calls and NO LLM calls — it runs instantly.
Optionally (settings.pageindex_enrich_summaries), a daemon background thread
upgrades node summaries via LLM after the tree is cached, since tree_search's
navigation prompt quality gates branch selection on these summaries.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from config import settings

logger = logging.getLogger(__name__)

TREES_DIR = Path(settings.docs_directory).parent / "data" / "tree_indexes"
TREES_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════
#  PDF Text Extraction
# ═══════════════════════════════════════════


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append(
            {
                "page": i + 1,
                "text": text,
                "char_count": len(text),
            }
        )
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


def _extract_text_blocks(pdf_path: str) -> list[dict]:
    """Extract text blocks with font info for heading detection.
    Returns list of: {page, text, font_size, is_bold, y_pos, block_idx}
    """
    doc = fitz.open(pdf_path)
    blocks = []

    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:  # text block only
                continue

            for line in block.get("lines", []):
                line_text = ""
                max_font_size = 0
                is_bold = False

                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    font_size = span.get("size", 12)
                    font_name = span.get("font", "").lower()

                    if font_size > max_font_size:
                        max_font_size = font_size

                    if "bold" in font_name or "black" in font_name or "heavy" in font_name:
                        is_bold = True

                    # Some PDFs use font flags instead of name
                    flags = span.get("flags", 0)
                    if flags & 2**4:  # bit 4 = bold
                        is_bold = True

                line_text = line_text.strip()
                if not line_text:
                    continue

                blocks.append(
                    {
                        "page": page_idx + 1,
                        "text": line_text,
                        "font_size": round(max_font_size, 1),
                        "is_bold": is_bold,
                        "y_pos": line.get("bbox", [0, 0, 0, 0])[1],
                        "block_idx": block_idx,
                    }
                )

    doc.close()
    return blocks


# ═══════════════════════════════════════════
#  Strategy 1: PDF Built-in ToC (Bookmarks)
# ═══════════════════════════════════════════


def _extract_pdf_toc(pdf_path: str) -> list[dict]:
    """Extract table of contents from PDF bookmarks/outline.
    PyMuPDF returns: [[level, title, page_number], ...]
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    total_pages = doc.page_count
    doc.close()

    if not toc or len(toc) < 2:
        return []

    entries = []
    for level, title, page in toc:
        title = title.strip()
        if not title or page < 1:
            continue
        entries.append(
            {
                "level": level,
                "title": title,
                "start_page": page,
            }
        )

    if not entries:
        return []

    for i, entry in enumerate(entries):
        end_page = total_pages
        for j in range(i + 1, len(entries)):
            if entries[j]["level"] <= entry["level"]:
                end_page = entries[j]["start_page"] - 1
                break
        entry["end_page"] = max(entry["start_page"], end_page)

    logger.info(f"Extracted {len(entries)} ToC entries from PDF bookmarks")
    return entries


def _toc_entries_to_tree(entries: list[dict], pages: list[dict]) -> list[dict]:
    """Convert flat ToC entries with levels into a nested tree."""
    if not entries:
        return []

    root_nodes = []
    stack = []  # (level, node)

    for entry in entries:
        node = {
            "title": entry["title"],
            "start_page": entry["start_page"],
            "end_page": entry["end_page"],
            "summary": _extract_summary_local(pages, entry["start_page"], entry["end_page"]),
            "children": [],
        }

        level = entry["level"]

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root_nodes.append(node)

        stack.append((level, node))

    return root_nodes


# ═══════════════════════════════════════════
#  Strategy 2: Font-Size Based Heading Detection
# ═══════════════════════════════════════════


def _detect_headings_by_font(text_blocks: list[dict], pages: list[dict]) -> list[dict]:
    """Detect headings by analyzing font sizes across the document."""
    if not text_blocks:
        return []

    font_sizes = [b["font_size"] for b in text_blocks if len(b["text"]) > 10]
    if not font_sizes:
        return []

    size_counts = Counter(round(s, 0) for s in font_sizes)
    body_size = size_counts.most_common(1)[0][0]

    heading_sizes = sorted(
        set(
            round(b["font_size"], 0)
            for b in text_blocks
            if b["font_size"] > body_size * 1.15 and len(b["text"].strip()) > 2 and len(b["text"].strip()) < 200
        ),
        reverse=True,
    )

    if not heading_sizes:
        heading_sizes = [body_size]
        bold_only = True
    else:
        bold_only = False

    size_to_level = {}
    for i, size in enumerate(heading_sizes[:4]):
        size_to_level[size] = i + 1

    headings = []
    for block in text_blocks:
        text = block["text"].strip()
        font_rounded = round(block["font_size"], 0)

        is_heading = False
        level = 3

        if font_rounded in size_to_level:
            is_heading = True
            level = size_to_level[font_rounded]
        elif bold_only and block["is_bold"] and font_rounded >= body_size:
            is_heading = True
            level = 2

        if is_heading:
            if len(text) > 150:
                is_heading = False
            if text.isdigit() or len(text) < 2:
                is_heading = False

        if is_heading:
            headings.append(
                {
                    "level": level,
                    "title": text[:120],
                    "start_page": block["page"],
                    "font_size": block["font_size"],
                    "is_bold": block["is_bold"],
                }
            )

    if not headings:
        return []

    seen = set()
    unique_headings = []
    for h in headings:
        key = (h["title"][:50], h["start_page"])
        if key not in seen:
            seen.add(key)
            unique_headings.append(h)

    total_pages = len(pages)
    for i, h in enumerate(unique_headings):
        end_page = total_pages
        for j in range(i + 1, len(unique_headings)):
            if unique_headings[j]["level"] <= h["level"]:
                end_page = unique_headings[j]["start_page"] - 1
                break
        h["end_page"] = max(h["start_page"], end_page)

    logger.info(f"Detected {len(unique_headings)} headings by font analysis (body={body_size}pt)")
    return unique_headings


# ═══════════════════════════════════════════
#  Strategy 3: Regex Pattern Matching
# ═══════════════════════════════════════════

HEADING_PATTERNS = [
    (r"^(?:CHAPTER|Chapter)\s+(\d+)\s*[:\--—.]\s*(.+)", 1),
    (r"^(?:PART|Part)\s+([IVXLCDM]+|\d+)\s*[:\--—.]\s*(.+)", 1),
    (r"^(\d{1,2})\.\s+([A-Z].{2,80})$", 1),
    (r"^(\d{1,2}\.\d{1,2})\.?\s+(.{3,80})$", 2),
    (r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\.?\s+(.{3,80})$", 3),
    (r"^(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[:\--—.]\s*(.+)", 2),
    (r"^([IVXLCDM]{1,5})\.\s+([A-Z].{2,80})$", 1),
    (r"^([A-Z])\.\s+([A-Z].{2,80})$", 2),
    (r"^(?:Appendix|APPENDIX)\s+([A-Z])\s*[:\--—.]\s*(.+)", 1),
    (r"^([A-Z][A-Z\s]{8,60})$", 1),
]


def _detect_headings_by_pattern(pages: list[dict]) -> list[dict]:
    """Detect headings using regex patterns on page text."""
    headings = []

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3 or len(line) > 200:
                continue

            for pattern, level in HEADING_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    title = groups[-1].strip() if len(groups) >= 2 else line.strip()

                    if len(title) < 3 or title.isdigit():
                        continue

                    if pattern == HEADING_PATTERNS[-1][0]:
                        words = title.split()
                        if len(words) < 2 or len(title) > 50:
                            continue

                    headings.append(
                        {
                            "level": level,
                            "title": title[:120],
                            "start_page": page["page"],
                        }
                    )
                    break

    if not headings:
        return []

    seen = set()
    unique = []
    for h in headings:
        key = (h["title"][:50], h["start_page"])
        if key not in seen:
            seen.add(key)
            unique.append(h)

    total = len(pages)
    for i, h in enumerate(unique):
        end = total
        for j in range(i + 1, len(unique)):
            if unique[j]["level"] <= h["level"]:
                end = unique[j]["start_page"] - 1
                break
        h["end_page"] = max(h["start_page"], end)

    logger.info(f"Detected {len(unique)} headings by pattern matching")
    return unique


# ═══════════════════════════════════════════
#  Strategy 4: Fallback — Even Page Splits
# ═══════════════════════════════════════════


def _fallback_flat_tree(pages: list[dict]) -> list[dict]:
    """Fallback: split document into even sections based on page count."""
    total = len(pages)
    if total <= 5:
        chunk_size = 1
    elif total <= 20:
        chunk_size = 5
    elif total <= 100:
        chunk_size = 10
    else:
        chunk_size = max(10, total // 10)

    nodes = []
    for i in range(0, total, chunk_size):
        end = min(i + chunk_size, total)
        first_page_text = pages[i]["text"]
        title = _guess_section_title(first_page_text, i + 1, end)
        nodes.append(
            {
                "title": title,
                "start_page": i + 1,
                "end_page": end,
                "summary": _extract_summary_local(pages, i + 1, end),
                "children": [],
            }
        )

    logger.info(f"Generated fallback flat tree with {len(nodes)} sections")
    return nodes


def _guess_section_title(text: str, start_page: int, end_page: int) -> str:
    """Try to extract a reasonable title from the first few lines of text."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:5]:
        if 5 <= len(line) <= 80:
            alpha_ratio = sum(1 for c in line if c.isalpha()) / max(len(line), 1)
            if alpha_ratio > 0.5:
                return line
    return f"Pages {start_page}-{end_page}"


# ═══════════════════════════════════════════
#  Local Summary Extraction (No LLM)
# ═══════════════════════════════════════════


def _extract_summary_local(pages: list[dict], start_page: int, end_page: int) -> str:
    """Extract a summary by taking the first meaningful sentences from a section."""
    text_parts = []
    for p in pages:
        if start_page <= p["page"] <= end_page:
            text_parts.append(p["text"])

    full_text = " ".join(text_parts)
    if not full_text.strip():
        return ""

    full_text = re.sub(r"\s+", " ", full_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    summary_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:
            summary_sentences.append(sent)
        if len(summary_sentences) >= 2:
            break

    summary = " ".join(summary_sentences)
    if len(summary) > 200:
        summary = summary[:197] + "..."
    return summary


# ═══════════════════════════════════════════
#  Background LLM Summary Enrichment
# ═══════════════════════════════════════════

_SUMMARY_SYSTEM = (
    "You summarize sections of documents for a table-of-contents index used to navigate the document. "
    "Given a section title and its text, reply with a 2-3 sentence summary of what the section covers. "
    "Reply with the summary only — no preamble, no markdown, no bullet points."
)

_ENRICH_MAX_NODES = 40
_ENRICH_MAX_SECTION_CHARS = 6000
_ENRICH_MAX_SUMMARY_CHARS = 400

_enrich_lock = threading.Lock()
_enrich_in_flight: set[str] = set()


def _iter_nodes(nodes: list):
    """Yield node dicts by reference (depth-first), so edits mutate the tree."""
    for node in nodes:
        yield node
        if node.get("children"):
            yield from _iter_nodes(node["children"])


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to a temp file in the same directory, then os.replace into place."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    # On Windows, os.replace raises PermissionError if a concurrent reader holds the file open.
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def _summarize_node_llm(title: str, section_text: str) -> str:
    """One LLM call → 2-3 sentence summary. Uses the lightweight model with a
    short keep_alive so enrichment never evicts the resident chat model."""
    from core import llm_client

    text = llm_client.chat(
        [{"role": "user", "content": f"Section: {title}\n\n{section_text[:_ENRICH_MAX_SECTION_CHARS]}"}],
        system=_SUMMARY_SYSTEM,
        model=llm_client.get_memory_model(),
        max_tokens=160,
        temperature=0.3,
        keep_alive="1m",
    )
    summary = re.sub(r"\s+", " ", str(text)).strip()
    if len(summary) > _ENRICH_MAX_SUMMARY_CHARS:
        summary = summary[: _ENRICH_MAX_SUMMARY_CHARS - 3] + "..."
    return summary


def _enrich_tree_summaries(doc_id: str, cache_path: Path) -> None:
    """Background worker: upgrade local extractive node summaries to LLM summaries,
    then atomically rewrite the cached tree JSON with summaries_enriched stamped."""
    try:
        # Stat BEFORE reading: if the file is replaced between stat and read we
        # get new content with an old mtime, and the pre-write re-stat aborts.
        loaded_mtime = os.stat(cache_path).st_mtime_ns
        with open(cache_path) as f:
            tree = json.load(f)
        if tree.get("summaries_enriched"):
            return

        pages = tree.get("pages", [])
        # Largest page-span first: big sections carry the most navigation weight.
        candidates = sorted(
            _iter_nodes(tree.get("nodes", [])),
            key=lambda n: (n.get("end_page") or 0) - (n.get("start_page") or 0),
            reverse=True,
        )[:_ENRICH_MAX_NODES]

        enriched = 0
        attempted = 0
        for node in candidates:
            sp = node.get("start_page") or 1
            ep = node.get("end_page") or sp
            section_text = " ".join(p["text"] for p in pages if sp <= p["page"] <= ep).strip()
            if not section_text:
                continue
            attempted += 1
            try:
                summary = _summarize_node_llm(node.get("title", "Untitled"), section_text)
                if summary:
                    node["summary"] = summary
                    enriched += 1
            except Exception as exc:
                logger.debug(f"Summary enrichment failed for node {node.get('node_id')}: {exc}")

        if attempted and not enriched:
            # LLM backend likely unreachable — keep the un-stamped cache so a later rebuild can retry.
            logger.warning(f"Summary enrichment produced no summaries for {doc_id}; cache left unchanged")
            return
        try:
            current_mtime = os.stat(cache_path).st_mtime_ns
        except OSError:  # document deleted while enriching
            return
        if current_mtime != loaded_mtime:
            # A concurrent force_rebuild rewrote the cache while we held a
            # minutes-stale copy — the newer tree wins; enrichment re-fires on next load.
            logger.info(f"Tree cache for {doc_id} was rewritten during enrichment — skipping stale summary write")
            return

        tree["summaries_enriched"] = True
        tree["summaries_enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tree["summaries_enriched_count"] = enriched
        _atomic_write_json(cache_path, tree)
        logger.info(f"Enriched {enriched}/{attempted} node summaries for {doc_id}")
    except Exception as exc:
        logger.warning(f"Tree summary enrichment failed for {doc_id}: {exc}")
    finally:
        with _enrich_lock:
            _enrich_in_flight.discard(doc_id)


def _spawn_summary_enrichment(doc_id: str, cache_path: Path) -> None:
    """Start enrichment in a daemon thread; never raises into the submit path."""
    try:
        with _enrich_lock:
            if doc_id in _enrich_in_flight:
                return
            _enrich_in_flight.add(doc_id)
        threading.Thread(
            target=_enrich_tree_summaries,
            args=(doc_id, cache_path),
            name=f"tree-enrich-{doc_id[-8:]}",
            daemon=True,
        ).start()
    except Exception as exc:
        with _enrich_lock:
            _enrich_in_flight.discard(doc_id)
        logger.warning(f"Could not start summary enrichment for {doc_id}: {exc}")


# ═══════════════════════════════════════════
#  Node ID Assignment
# ═══════════════════════════════════════════


def _assign_node_ids(nodes: list, prefix: str = "") -> None:
    """Assign hierarchical node IDs like 0001, 0002, 0001.0001."""
    for i, node in enumerate(nodes):
        node["node_id"] = f"{prefix}{i + 1:04d}" if prefix else f"{i + 1:04d}"
        if node.get("children"):
            _assign_node_ids(node["children"], prefix=node["node_id"] + ".")


# ═══════════════════════════════════════════
#  Main Pipeline: Generate Tree Index
# ═══════════════════════════════════════════


def generate_tree_index(
    pdf_path: str,
    enrich_summaries: bool | None = None,
    toc_check_pages: int = 20,
    force_rebuild: bool = False,
) -> dict:
    """Full pipeline: PDF → tree-structured index.

    Tries strategies in order:
    1. PDF built-in ToC (bookmarks/outline) — best quality, instant
    2. Font-size heading detection — analyzes text block sizes
    3. Regex pattern matching — detects "Chapter 1", "1.1", etc.
    4. Fallback even splits — divides by page count

    When enrich_summaries (defaults to settings.pageindex_enrich_summaries), a
    daemon thread upgrades node summaries via LLM after the tree is cached and
    rewrites the cache with summaries_enriched=true; the synchronous return is
    never delayed or affected by enrichment.

    Returns: {doc_id, title, total_pages, description, nodes, pages, ...}
    """
    pdf_path = str(pdf_path)
    doc_id = _doc_id_from_path(pdf_path)
    if enrich_summaries is None:
        enrich_summaries = settings.pageindex_enrich_summaries

    cache_path = TREES_DIR / f"{doc_id}.json"
    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading cached tree index for {doc_id}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Older caches (or a previously unreachable LLM) may still hold local summaries.
        if enrich_summaries and not cached.get("summaries_enriched"):
            _spawn_summary_enrichment(doc_id, cache_path)
        return cached

    logger.info(f"Generating tree index for {pdf_path} (local analysis, zero LLM calls)...")
    start = time.time()

    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")

    total_pages = len(pages)
    tree_nodes = []
    method_used = "none"

    # Strategy 1: PDF Built-in ToC
    toc_entries = _extract_pdf_toc(pdf_path)
    if len(toc_entries) >= 3:
        tree_nodes = _toc_entries_to_tree(toc_entries, pages)
        method_used = "pdf_toc"
        logger.info(f"Strategy 1 (PDF ToC): {len(tree_nodes)} top-level nodes")

    # Strategy 2: Font-size heading detection
    if not tree_nodes:
        text_blocks = _extract_text_blocks(pdf_path)
        font_headings = _detect_headings_by_font(text_blocks, pages)
        if len(font_headings) >= 3:
            tree_nodes = _toc_entries_to_tree(font_headings, pages)
            method_used = "font_analysis"
            logger.info(f"Strategy 2 (Font analysis): {len(tree_nodes)} top-level nodes")

    # Strategy 3: Regex pattern matching
    if not tree_nodes:
        pattern_headings = _detect_headings_by_pattern(pages)
        if len(pattern_headings) >= 3:
            tree_nodes = _toc_entries_to_tree(pattern_headings, pages)
            method_used = "pattern_matching"
            logger.info(f"Strategy 3 (Pattern matching): {len(tree_nodes)} top-level nodes")

    # Strategy 4: Fallback even splits
    if not tree_nodes:
        tree_nodes = _fallback_flat_tree(pages)
        method_used = "fallback_split"
        logger.info(f"Strategy 4 (Fallback split): {len(tree_nodes)} sections")

    _assign_node_ids(tree_nodes)

    doc_title = _extract_document_title(pages, pdf_path)
    doc_description = _extract_summary_local(pages, 1, min(3, total_pages))

    result = {
        "doc_id": doc_id,
        "title": doc_title,
        "total_pages": total_pages,
        "description": doc_description,
        "nodes": tree_nodes,
        "pages": [{"page": p["page"], "text": p["text"]} for p in pages],
        "source_file": os.path.basename(pdf_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation_time_s": round(time.time() - start, 2),
        "method": method_used,
        "summaries_enriched": False,
    }

    _atomic_write_json(cache_path, result)
    logger.info(f"Tree index generated in {result['generation_time_s']}s via {method_used} → {cache_path}")

    if enrich_summaries:
        _spawn_summary_enrichment(doc_id, cache_path)

    return result


def _extract_document_title(pages: list[dict], pdf_path: str) -> str:
    """Try to extract document title from first page or filename."""
    if pages:
        first_lines = [ln.strip() for ln in pages[0]["text"].split("\n") if ln.strip()]
        for line in first_lines[:5]:
            if 5 <= len(line) <= 100:
                alpha_ratio = sum(1 for c in line if c.isalpha()) / max(len(line), 1)
                if alpha_ratio > 0.6:
                    return line
    return Path(pdf_path).stem.replace("_", " ").replace("-", " ").title()


def _doc_id_from_path(pdf_path: str) -> str:
    """Generate a stable doc_id from file path + size + mtime."""
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
                trees.append(
                    {
                        "doc_id": data.get("doc_id", f.stem),
                        "title": data.get("title", "Unknown"),
                        "total_pages": data.get("total_pages", 0),
                        "source_file": data.get("source_file", ""),
                        "created_at": data.get("created_at", ""),
                        "method": data.get("method", "unknown"),
                        "summaries_enriched": bool(data.get("summaries_enriched", False)),
                    }
                )
        except Exception:
            continue
    return trees


def get_tree_by_id(doc_id: str) -> dict | None:
    """Load a tree index by doc_id."""
    cache_path = TREES_DIR / f"{doc_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
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
        flat.append(
            {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "start_page": node.get("start_page"),
                "end_page": node.get("end_page"),
                "summary": node.get("summary", ""),
                "depth": depth,
            }
        )
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
        lines.append(f"{prefix}• {title} (pp. {sp}-{ep})")
        if node.get("children"):
            lines.append(tree_to_outline(node["children"], indent + 1))
    return "\n".join(lines)
