"""Document ingestion, upload, file management, and collection clearing."""

import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.auth import require_auth
from api.dependencies import get_user_store
from api.models import IngestReq
from config import settings
from core.ingestion import chunk_document, file_hash_for, load_documents
from core.query_cache import invalidate_user_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])

_SITUATE_TIMEOUT_S = 30
_ENRICH_LOG_EVERY = 25
_ENRICH_MAX_CONSECUTIVE_TIMEOUTS = 3
_SKIPPED_FILES_RESPONSE_CAP = 50


def _contextual_enrich(chunks, base_dir, override=None):
    """Contextual retrieval (Anthropic-style): insert an LLM situating line into each
    chunk's content directly below the breadcrumb line. Opt-in and strictly best-effort —
    ingest never fails or blocks indefinitely because of this pass."""
    enabled = settings.contextual_enrich if override is None else override
    if not enabled or not chunks:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor

        from core.contextualizer import situate
        from core.ingestion import _COMMENT_PREFIX

        by_path = {}
        for c in chunks:
            by_path.setdefault(c.document_path, []).append(c)
        doc_texts = {}
        for path, path_chunks in by_path.items():
            text = ""
            if not path_chunks[0].metadata.get("multimodal"):
                try:
                    text = (Path(base_dir) / path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
            if not text.strip():
                # PDFs/images or unreadable files: rebuild document context from the
                # chunks themselves (minus each chunk's breadcrumb/header line).
                text = "\n".join(ch.content.split("\n", 1)[-1] for ch in path_chunks)
            doc_texts[path] = text

        total = len(chunks)
        done = enriched = consecutive_timeouts = 0
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            futures = [(pool.submit(situate, doc_texts[c.document_path], c.content), c) for c in chunks]
            for future, c in futures:
                try:
                    ctx = (future.result(timeout=_SITUATE_TIMEOUT_S) or "").strip()
                    consecutive_timeouts = 0
                except TimeoutError:
                    ctx = ""
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= _ENRICH_MAX_CONSECUTIVE_TIMEOUTS:
                        logger.warning(
                            "Contextual enrich: %d consecutive timeouts — aborting pass", consecutive_timeouts
                        )
                        break
                except Exception:
                    ctx = ""
                done += 1
                if ctx:
                    head, sep, rest = c.content.partition("\n")
                    if sep:
                        prefix = f"{_COMMENT_PREFIX.get(c.language, '#')} " if c.chunk_type == "code" else ""
                        c.content = f"{head}\n{prefix}{ctx}\n{rest}"
                        enriched += 1
                if done % _ENRICH_LOG_EVERY == 0:
                    logger.info("Contextual enrich: %d/%d chunks (%d situated)", done, total, enriched)
        finally:
            # wait=False so a hung LLM call can't block ingest past the per-chunk timeout.
            pool.shutdown(wait=False, cancel_futures=True)
        logger.info("Contextual enrich complete: %d/%d chunks situated", enriched, total)
    except Exception as e:
        logger.warning("Contextual enrichment skipped after error: %s", e)


def _partition_unchanged(store, docs):
    """Split loaded documents into (changed, skipped_paths) by comparing each file's
    content hash against the file_hash stored in the collection's chunk metadata.

    Returns everything as changed when skip-unchanged is disabled, the collection is
    empty, or the lookup fails — the worst case is a redundant re-embed, never a miss."""
    if not settings.ingest_skip_unchanged or not docs or store.count == 0:
        return list(docs), []

    stored = {}
    try:
        paths = sorted({d.filepath for d in docs})
        raw = store.collection.get(where={"document_path": {"$in": paths}}, include=["metadatas"])
        for meta in raw.get("metadatas") or []:
            meta = meta or {}
            path = meta.get("document_path")
            if path and path not in stored:
                stored[path] = meta.get("file_hash", "")
    except Exception as e:
        logger.warning(f"Skip-unchanged hash lookup failed (re-indexing everything): {e}")
        return list(docs), []

    changed, skipped = [], []
    for d in docs:
        # file_hash_for folds in the chunking fingerprint: a chunk_size/enrichment
        # config change must re-chunk even byte-identical files.
        if d.content_hash and stored.get(d.filepath) == file_hash_for(d):
            skipped.append(d.filepath)
        else:
            changed.append(d)
    return changed, sorted(skipped)


def _incremental_graph_update(uid, chunks):
    """Merge freshly ingested chunks into an existing knowledge graph on a daemon
    thread so ingest latency is unaffected. Strictly best-effort: never creates a
    graph, and any failure (including the API not existing yet) logs and continues."""
    if not settings.graph_incremental or not chunks:
        return
    payload = [
        (
            c.chunk_id,
            c.content,
            {
                **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
                "document_path": c.document_path,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_type": c.chunk_type,
                "source": c.display_source,
            },
        )
        for c in chunks
    ]

    def _run():
        try:
            from core.knowledge_graph import get_user_graph

            kg = get_user_graph(uid)
            if kg.graph.number_of_nodes() == 0:
                return  # never force graph creation — full builds stay explicit
            kg.update_from_chunks(payload)
        except (ImportError, AttributeError) as e:
            logger.debug(f"Incremental graph API unavailable, skipping: {e}")
        except Exception as e:
            logger.warning(f"Incremental graph update failed (continuing): {e}")

    threading.Thread(target=_run, name=f"kg-incremental-{uid}", daemon=True).start()


def _ingest_docs(uid, store, docs, base_dir, contextual_override=None):
    """Shared skip-unchanged -> chunk -> enrich -> index -> graph-hook pipeline for
    /ingest and /upload. Returns (chunks_indexed, processed_paths, skipped_paths)."""
    changed, skipped = _partition_unchanged(store, docs)
    chunks = []
    for doc in changed:
        chunks.extend(chunk_document(doc))

    added = 0
    if chunks:
        _contextual_enrich(chunks, base_dir, override=contextual_override)
        added = store.add_chunks(chunks)
        invalidate_user_cache(uid)
        _incremental_graph_update(uid, chunks)

    processed = sorted({c.document_path for c in chunks})
    logger.info(
        "Ingest summary: %d files processed, %d skipped (unchanged), %d chunks indexed",
        len(processed),
        len(skipped),
        added,
    )
    return added, processed, skipped


@router.post("/ingest")
def ingest(req: IngestReq, user=Depends(require_auth)):
    uid = user["id"]
    s = get_user_store(uid)
    docs = load_documents(req.directory)
    if not docs:
        p = Path(req.directory)
        if not p.exists():
            raise HTTPException(400, f"Directory not found: {req.directory}")
        all_files = list(p.rglob("*"))
        file_count = sum(1 for f in all_files if f.is_file())
        exts = set(f.suffix.lower() for f in all_files if f.is_file() and f.suffix)
        from core.ingestion import SUPPORTED_EXTENSIONS

        supported_found = exts & SUPPORTED_EXTENSIONS
        raise HTTPException(
            400,
            f"No supported documents found in {req.directory}. Found {file_count} files with extensions: "
            f"{', '.join(sorted(exts)[:20]) or 'none'}. Supported matches: {', '.join(sorted(supported_found)) or 'none'}",
        )
    added, processed, skipped = _ingest_docs(uid, s, docs, req.directory, contextual_override=req.contextual)
    return {
        "chunks_indexed": added,
        "documents_processed": len(processed),
        "collection_total": s.count,
        "files": processed,
        "files_processed": processed,
        "files_skipped": {"count": len(skipped), "files": skipped[:_SKIPPED_FILES_RESPONSE_CAP]},
    }


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...), user=Depends(require_auth)):
    uid = user["id"]
    s = await asyncio.to_thread(get_user_store, uid)
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            safe_name = f.filename.replace("\\", "/")
            p = os.path.join(tmp, safe_name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        docs = await asyncio.to_thread(load_documents, tmp)
        added, processed, skipped = await asyncio.to_thread(_ingest_docs, uid, s, docs, tmp)
    return {
        "chunks_indexed": added,
        "files_processed": processed,
        "documents_processed": len(processed),
        "collection_total": s.count,
        "files_skipped": {"count": len(skipped), "files": skipped[:_SKIPPED_FILES_RESPONSE_CAP]},
    }


@router.get("/files")
def file_tree(user=Depends(require_auth)):
    return {"files": get_user_store(user["id"]).get_all_files()}


@router.delete("/files")
def delete_file(path: str, user=Depends(require_auth)):
    # Path traversal protection
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise HTTPException(400, "Invalid file path")
    uid = user["id"]
    s = get_user_store(uid)
    deleted = s.delete_file(path)
    if deleted == 0:
        raise HTTPException(404, "File not found in index")
    invalidate_user_cache(uid)
    return {"deleted_chunks": deleted, "path": path}


@router.delete("/collection")
def clear(user=Depends(require_auth)):
    uid = user["id"]
    get_user_store(uid).clear()
    invalidate_user_cache(uid)
    return {"status": "cleared"}
