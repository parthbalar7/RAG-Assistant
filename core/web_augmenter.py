"""
core/web_augmenter.py — Web Search → Ingest Pipeline

When the user approves a knowledge gap suggestion, this module:
  1. Searches DuckDuckGo for the topic (free, no API key)
  2. Fetches the top N result pages
  3. Strips HTML to plain text
  4. Chunks + embeds via the existing ingestion pipeline
  5. Upserts into the user's VectorStore

Everything reuses existing infrastructure — no new storage or models needed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_RESULTS = 4  # DDG results to fetch
FETCH_TIMEOUT = 8  # seconds per HTTP request
MAX_PAGE_CHARS = 30_000  # chars to keep per page before chunking


@dataclass
class AugmentResult:
    topic: str
    urls_fetched: list[str] = field(default_factory=list)
    urls_failed: list[str] = field(default_factory=list)
    chunks_added: int = 0
    error: str | None = None


# ── search ────────────────────────────────────────────────────────────────────


def _ddg_search(topic: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Return list of {title, href, body} from DuckDuckGo text search."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_results))
        return results
    except ImportError as ie:
        raise RuntimeError("ddgs is not installed. Add 'ddgs>=1.0' to requirements.txt and rebuild.") from ie
    except Exception as e:
        logger.warning(f"DDG search failed for '{topic}': {e}")
        return []


# ── fetch + strip ─────────────────────────────────────────────────────────────


def _fetch_text(url: str) -> str | None:
    """Fetch a URL and return clean plain text, or None on failure."""
    try:
        import requests

        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGv2-bot/1.0)"}
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers)
        resp.raise_for_status()
        raw_html = resp.text
    except Exception as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        return None

    return _html_to_text(raw_html)[:MAX_PAGE_CHARS]


def _html_to_text(html: str) -> str:
    """Minimal HTML → plain text without external dependencies."""
    # Remove script / style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    for ent, ch in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        html = html.replace(ent, ch)
    # Collapse whitespace
    html = re.sub(r"\s{2,}", "\n", html)
    return html.strip()


# ── chunk + ingest ────────────────────────────────────────────────────────────


def _ingest_text(text: str, url: str, topic: str, store) -> int:
    """Chunk raw text and upsert into VectorStore. Returns chunks added."""
    from core.ingestion import Document, chunk_document

    # Wrap as a Document so existing chunker handles it
    doc = Document(
        content=text,
        filepath=f"web:{url}",
        language="text",
        metadata={"source": "web", "url": url, "topic": topic, "fetched_at": time.time()},
    )
    chunks = chunk_document(doc)
    if not chunks:
        return 0
    store.add_chunks(chunks)
    return len(chunks)


# ── public entry point ────────────────────────────────────────────────────────


def augment(topic: str, store, query: str = "") -> AugmentResult:
    """
    Search the web for *topic*, fetch pages, chunk, and ingest into *store*.
    Returns an AugmentResult summary for the WS response.
    """
    search_query = query if query else topic
    result = AugmentResult(topic=topic)

    logger.info(f"Web augment: searching for '{search_query}'")
    search_hits = _ddg_search(search_query, max_results=MAX_RESULTS)

    if not search_hits:
        result.error = f"No search results returned for '{topic}'."
        logger.warning(result.error)
        return result

    for hit in search_hits:
        url = hit.get("href") or hit.get("url", "")
        if not url:
            continue

        text = _fetch_text(url)
        if not text or len(text) < 200:
            result.urls_failed.append(url)
            continue

        try:
            added = _ingest_text(text, url, topic, store)
            if added > 0:
                result.urls_fetched.append(url)
                result.chunks_added += added
                logger.info(f"Ingested {added} chunks from {url}")
            else:
                result.urls_failed.append(url)
        except Exception as e:
            logger.warning(f"Ingest failed for {url}: {e}")
            result.urls_failed.append(url)

    if result.chunks_added == 0:
        result.error = f"Fetched {len(search_hits)} page(s) but could not extract usable content."

    return result
