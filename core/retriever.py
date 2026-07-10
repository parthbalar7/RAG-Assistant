"""
Hybrid retrieval: Vector (ChromaDB) + BM25 keyword search with cross-encoder reranking.
Pipeline: query -> [vector search + BM25] -> reciprocal rank fusion -> rerank -> top-K
"""

import heapq
import logging
import math
import re
import threading
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

try:
    import bm25s
except ImportError:
    bm25s = None

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Singleton models
_embedding_model = None
_reranker_model = None
_reranker_kind = None  # "cross-encoder" | "colbert" (set alongside _reranker_model)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        backend = settings.embedding_backend
        logger.info(f"Loading embedding model: {settings.embedding_model} (backend={backend})")
        if backend != "torch":
            try:
                _embedding_model = SentenceTransformer(settings.embedding_model, backend=backend)
            except Exception as e:
                logger.warning(f"Embedding backend '{backend}' failed, falling back to torch: {e}")
                _embedding_model = SentenceTransformer(settings.embedding_model)
        else:
            _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_reranker():
    """Load the configured reranker once. Returns (model, kind) with kind in
    {'cross-encoder', 'colbert'}; degrades to the default cross-encoder on any failure."""
    global _reranker_model, _reranker_kind
    if _reranker_model is None:
        model_name = settings.reranker_model or DEFAULT_RERANKER_MODEL
        if settings.reranker_type == "colbert":
            try:
                from rerankers import Reranker

                logger.info(f"Loading ColBERT reranker: {model_name}")
                _reranker_model = Reranker(model_name, model_type="colbert", verbose=0)
                _reranker_kind = "colbert"
                return _reranker_model, _reranker_kind
            except Exception as e:
                logger.warning(f"ColBERT reranker '{model_name}' unavailable, falling back to cross-encoder: {e}")
                model_name = DEFAULT_RERANKER_MODEL
        logger.info(f"Loading reranker: {model_name}")
        try:
            _reranker_model = CrossEncoder(model_name)
        except Exception as e:
            if model_name == DEFAULT_RERANKER_MODEL:
                raise
            logger.warning(f"Reranker '{model_name}' failed to load, falling back to {DEFAULT_RERANKER_MODEL}: {e}")
            _reranker_model = CrossEncoder(DEFAULT_RERANKER_MODEL)
        _reranker_kind = "cross-encoder"
    return _reranker_model, _reranker_kind


def check_embedding_dimension(collection, model=None):
    """Fail fast when *collection* was embedded with a different-dimension model.

    Peeks one stored embedding and compares it against the active embedding model's
    output dimension. Raises RuntimeError pointing at scripts/migrate_embeddings.py
    on mismatch; any peek failure is non-fatal (check is skipped).
    """
    try:
        if collection.count() == 0:
            return
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return
        stored_dim = len(embeddings[0])
    except Exception as e:
        logger.warning(f"Embedding dimension check skipped for '{getattr(collection, 'name', '?')}': {e}")
        return
    model = model or get_embedding_model()
    model_dim = model.get_sentence_embedding_dimension()
    if model_dim and stored_dim != model_dim:
        raise RuntimeError(
            f"Collection '{collection.name}' stores {stored_dim}-dim embeddings but model "
            f"'{settings.embedding_model}' produces {model_dim}-dim vectors. "
            f"Re-embed all collections first: .venv\\Scripts\\python.exe scripts/migrate_embeddings.py"
        )


def embed_texts(texts, persist=True):
    from core.embed_cache import get_embed_cache

    cache = get_embed_cache()
    model = get_embedding_model()

    result: list = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            result[i] = cached.tolist()
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        new_vecs = model.encode(uncached_texts, show_progress_bar=False, normalize_embeddings=True)
        cache.put_batch(uncached_texts, new_vecs)
        if persist:
            cache.save()
        for list_pos, original_idx in enumerate(uncached_indices):
            result[original_idx] = new_vecs[list_pos].tolist()

    return result


# ── BM25 Index ──


class BM25Index:
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.doc_contents = []
        self.doc_metadatas = []
        self.bm25 = None

    def build_from_collection(self, collection):
        count = collection.count()
        if count == 0:
            self.bm25 = None
            return

        all_docs = collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        self.doc_contents = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]

        # Pre-tokenized corpus (no bm25s stemming/stopwords — code identifiers must survive)
        tokenized = [self._tokenize(doc) for doc in self.doc_contents]
        if bm25s is not None:
            try:
                index = bm25s.BM25()
                index.index(tokenized, show_progress=False)
                self.bm25 = index
                logger.info(f"BM25 index built with {len(self.doc_ids)} documents (bm25s)")
                return
            except Exception as e:
                logger.warning(f"bm25s index build failed (falling back to rank_bm25): {e}")
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self.doc_ids)} documents")

    def search(self, query, top_k=10):
        if not self.bm25 or not self.doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        # Both BM25Okapi and bm25s.BM25 expose get_scores(list[str]) -> per-doc scores
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "content": self.doc_contents[idx],
                        "metadata": self.doc_metadatas[idx],
                        "score": float(scores[idx]),
                        "search_type": "bm25",
                    }
                )
        return results

    @staticmethod
    def _tokenize(text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > 1]


def _hit_key(hit):
    """Stable dedup key for RRF fusion. Breadcrumb prefixes make content[:100] collide,
    so prefer metadata identity and only fall back to the content prefix."""
    meta = hit.get("metadata") or {}
    source = meta.get("source")
    if source:
        return source
    path = meta.get("document_path")
    if path:
        return (path, meta.get("start_line"))
    return hit.get("content", "")[:100]


# ── Vector Store ──


class VectorStore:
    def __init__(self, persist_dir=None, collection_name=None):
        persist_dir = persist_dir or settings.chroma_persist_dir
        collection_name = collection_name or settings.collection_name

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        check_embedding_dimension(self.collection)
        self.bm25_index = BM25Index()
        self.splade_index = None  # built only when settings.splade_enabled=True
        self._file_cache = []
        self._file_ids_by_path = {}
        # Progressive indexing: debounced background sparse rebuilds (see _schedule_index_rebuild)
        self._index_lock = threading.Lock()  # guards generation counters + timer handle
        self._rebuild_exec_lock = threading.Lock()  # serializes rebuild passes
        self._rebuild_timer = None  # pending threading.Timer, if any
        self._index_generation = 0  # bumped on every collection mutation
        self._built_generation = 0  # generation the live sparse indexes reflect
        self._last_built = None
        self._rebuild_indexes()  # startup build stays synchronous — queries work immediately after boot
        self._last_built = time.time()

        logger.info(f"VectorStore ready: {collection_name} ({self.collection.count()} docs)")

    def _build_sparse_indexes(self):
        """Construct NEW BM25/SPLADE index objects from the collection without touching the live ones.

        Returns (bm25_index, splade_index); splade_index is None when disabled or on build failure."""
        bm25 = BM25Index()
        splade = None
        if self.collection.count() > 0:
            bm25.build_from_collection(self.collection)
            if settings.splade_enabled:
                try:
                    from core.splade_index import SPLADEIndex

                    splade = SPLADEIndex(settings.splade_model)
                    splade.build_from_collection(self.collection)
                except Exception as e:
                    logger.warning(f"SPLADE index build failed (falling back to BM25): {e}")
                    splade = None
        return bm25, splade

    def _rebuild_indexes(self):
        """Synchronously rebuild BM25 (always) and SPLADE (when enabled), then swap them in atomically."""
        bm25, splade = self._build_sparse_indexes()
        self.bm25_index = bm25
        self.splade_index = splade
        self._refresh_metadata_cache()

    # Keep the old name as an alias so any external callers aren't broken
    def _rebuild_bm25(self):
        self._rebuild_indexes()

    def _schedule_index_rebuild(self):
        """Called after every collection mutation.

        With sparse_rebuild_debounce_s > 0, the expensive BM25/SPLADE rebuild runs on a background
        timer that fires after the LAST mutation — repeated mutations within the window coalesce
        into one rebuild, and queries keep serving the previous index objects meanwhile.
        A value of 0 restores the old fully synchronous rebuild."""
        debounce = settings.sparse_rebuild_debounce_s
        if debounce <= 0:
            with self._index_lock:
                self._index_generation += 1
                target = self._index_generation
            self._rebuild_indexes()
            with self._index_lock:
                self._built_generation = max(self._built_generation, target)
                self._last_built = time.time()
            return
        self._refresh_metadata_cache()  # cheap — file listing stays fresh while the rebuild is pending
        with self._index_lock:
            self._index_generation += 1
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
            timer = threading.Timer(debounce, lambda: self._on_rebuild_timer(timer))
            timer.daemon = True
            self._rebuild_timer = timer
            timer.start()

    def _mark_index_mutation_start(self):
        """Bump the generation BEFORE mutating the collection so an in-flight rebuild that
        reads a torn snapshot (mid-delete/mid-upsert) fails its swap check and is discarded."""
        with self._index_lock:
            self._index_generation += 1

    def _arm_retry_timer(self):
        """Re-arm a rebuild after a failed build so staleness is never silently permanent."""
        delay = max(settings.sparse_rebuild_debounce_s, 1.0)
        with self._index_lock:
            if self._rebuild_timer is not None:
                return
            timer = threading.Timer(delay, lambda: self._on_rebuild_timer(timer))
            timer.daemon = True
            self._rebuild_timer = timer
            timer.start()

    def _on_rebuild_timer(self, timer):
        with self._index_lock:
            if self._rebuild_timer is timer:
                self._rebuild_timer = None
        try:
            self._run_rebuild_once()
        except Exception as e:  # never let the timer thread die with an unhandled error
            logger.warning(f"Background sparse index rebuild failed: {e}")

    def _run_rebuild_once(self):
        """One rebuild pass: build new index objects, then swap them in via single attribute
        assignments — unless a newer mutation arrived mid-build, in which case the stale build
        is discarded (the newer mutation's own scheduled rebuild supersedes it).

        Returns False only when the build itself failed (a retry timer is armed); a superseded
        build returns True so flush loops re-run against the newer generation."""
        with self._rebuild_exec_lock:
            with self._index_lock:
                target = self._index_generation
                if self._built_generation >= target:
                    return True
            try:
                bm25, splade = self._build_sparse_indexes()
            except Exception as e:
                # Do NOT record the generation as built — that would report stale
                # indexes as consistent. Keep serving the old ones and retry.
                logger.warning(f"Sparse index rebuild failed (keeping previous indexes; retry armed): {e}")
                self._arm_retry_timer()
                return False
            with self._index_lock:
                if self._index_generation != target:
                    logger.info("Sparse index rebuild superseded by a newer mutation; discarding stale build")
                    return True
                self.bm25_index = bm25
                self.splade_index = splade
                self._built_generation = target
                self._last_built = time.time()
                return True

    def flush_index_rebuild(self, timeout=None):
        """Block until the sparse indexes reflect every mutation issued before this call.

        Cancels any pending debounce timer and runs the rebuild inline instead of waiting out the
        window. Returns True once up to date, False if *timeout* (seconds) expired first or the
        rebuild itself failed (a background retry stays armed)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._index_lock:
            target = self._index_generation
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
                self._rebuild_timer = None
        while True:
            with self._index_lock:
                if self._built_generation >= target:
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if not self._run_rebuild_once():
                return False

    def index_status(self):
        """Observability for progressive indexing: pending rebuild flag, last build time, indexed docs."""
        with self._index_lock:
            pending = self._rebuild_timer is not None or self._built_generation < self._index_generation
            return {"pending": pending, "last_built": self._last_built, "docs": len(self.bm25_index.doc_ids)}

    @property
    def count(self):
        return self.collection.count()

    def _refresh_metadata_cache(self):
        """Rebuild the file-listing / chunk-id caches from the live collection (metadata-only, cheap).

        Reads the collection directly rather than the BM25 arrays so the file tree stays accurate
        while a debounced sparse rebuild is still pending. New dicts are swapped in atomically."""
        if self.collection.count() == 0:
            self._file_cache = []
            self._file_ids_by_path = {}
            return
        try:
            got = self.collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning(f"Metadata cache refresh failed (keeping previous listing): {e}")
            return
        files = {}
        ids_by_path = {}
        for doc_id, meta in zip(got["ids"], got["metadatas"] or []):
            meta = meta or {}
            path = meta.get("document_path", "")
            if not path:
                continue
            if path not in files:
                files[path] = {
                    "path": path,
                    "language": meta.get("language", ""),
                    "chunk_count": 0,
                }
            files[path]["chunk_count"] += 1
            ids_by_path.setdefault(path, []).append(doc_id)
        self._file_cache = sorted(files.values(), key=lambda x: x["path"])
        self._file_ids_by_path = ids_by_path

    def add_chunks(self, chunks, batch_size=64):
        if not chunks:
            return 0

        self._mark_index_mutation_start()
        # Chunk-boundary changes produce new chunk_ids, so upsert alone would leave
        # stale-boundary chunks behind — delete existing chunks for incoming paths first.
        if self.collection.count() > 0:
            paths = sorted({c.document_path for c in chunks if c.document_path})
            if paths:
                try:
                    self.collection.delete(where={"document_path": {"$in": paths}})
                except Exception as e:
                    logger.warning(f"Stale-chunk delete by path failed (continuing with upsert): {e}")

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            embeddings = embed_texts(documents, persist=False)
            # Chunk.metadata (heading_path, page, ast_chunk) rides along so parent
            # expansion can stop at section boundaries; ChromaDB only takes scalars.
            metadatas = [
                {
                    **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
                    "document_path": c.document_path,
                    "language": c.language,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "chunk_type": c.chunk_type,
                    "source": c.display_source,
                }
                for c in batch
            ]
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            added += len(batch)
            logger.info(f"Indexed {added}/{len(chunks)} chunks")

        from core.embed_cache import get_embed_cache

        get_embed_cache().save()
        self._schedule_index_rebuild()
        return added

    def vector_search(self, query_text, top_k=10, language_filter=None):
        if self.count == 0:
            return []

        query_embedding = embed_texts([query_text])[0]
        where_filter = {"language": language_filter} if language_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["documents"] and results["documents"][0]:
            ids = (results.get("ids") or [[]])[0] or [None] * len(results["documents"][0])
            for cid, doc, meta, dist in zip(
                ids, results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                similarity = 1 - (dist / 2)
                if similarity >= settings.similarity_threshold:
                    hits.append(
                        {
                            "id": cid,
                            "content": doc,
                            "metadata": meta,
                            "score": round(similarity, 4),
                            "search_type": "vector",
                        }
                    )
        return hits

    def _rrf_merge(self, vector_hits, bm25_hits=None, top_k=10, weights=None):
        """Reciprocal Rank Fusion of pre-computed ranked hit lists.

        Accepts either the classic two-list form (vector_hits, bm25_hits) — weighted
        by settings.vector_weight/bm25_weight — or a list-of-lists as the first
        argument with bm25_hits=None, with optional per-list `weights` (default 1.0).
        """
        if bm25_hits is not None:
            hit_lists = [vector_hits, bm25_hits]
            weights = weights or [settings.vector_weight, settings.bm25_weight]
        else:
            hit_lists = vector_hits
            weights = weights or [1.0] * len(hit_lists)

        k = 60
        fused_scores = {}

        for hit_list, weight in zip(hit_lists, weights):
            for rank, hit in enumerate(hit_list):
                key = _hit_key(hit)
                if key not in fused_scores:
                    fused_scores[key] = dict(hit)
                    fused_scores[key]["rrf_score"] = 0
                    fused_scores[key]["search_types"] = []
                fused_scores[key]["rrf_score"] += weight / (k + rank + 1)
                fused_scores[key]["search_types"].append(hit.get("search_type", "unknown"))

        results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        for hit in results:
            hit["score"] = hit["rrf_score"]
            hit["search_type"] = "+".join(set(hit.get("search_types", ["unknown"])))
        return results[:top_k]

    def hybrid_search(self, query_text, top_k=10, language_filter=None):
        vector_hits = self.vector_search(query_text, top_k=top_k, language_filter=language_filter)
        bm25_hits = self.bm25_index.search(query_text, top_k=top_k)
        return self._rrf_merge(vector_hits, bm25_hits, top_k)

    def get_all_files(self):
        if self.count == 0:
            return []
        return list(self._file_cache)

    def delete_file(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific file."""
        if self.count == 0:
            return 0
        # Query the live collection for ids — the cached mapping can lag within the debounce window
        try:
            got = self.collection.get(where={"document_path": file_path}, include=["metadatas"])
            ids_to_delete = list(got.get("ids") or [])
        except Exception as e:
            logger.warning(f"Chunk lookup by path failed for {file_path} (using cached ids): {e}")
            ids_to_delete = list(self._file_ids_by_path.get(file_path, []))
        if ids_to_delete:
            self._mark_index_mutation_start()
            self.collection.delete(ids=ids_to_delete)
            self._schedule_index_rebuild()
            # Deletions are rare and correctness-sensitive: rebuild inline so the
            # sparse indexes never keep serving a deleted file's content.
            self.flush_index_rebuild(timeout=60)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {file_path}")
        return len(ids_to_delete)

    def clear(self):
        # Invalidate pending/in-flight rebuilds so a build of the old collection can't swap in later
        with self._index_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
                self._rebuild_timer = None
            self._index_generation += 1
            target = self._index_generation
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25_index = BM25Index()
        self.splade_index = None
        self._file_cache = []
        self._file_ids_by_path = {}
        with self._index_lock:
            self._built_generation = max(self._built_generation, target)
            self._last_built = time.time()
        logger.info("Vector store cleared")


# ── Reranking ──


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function mapping raw logits into (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _colbert_scores(reranker, query, hits):
    """Score hits with a rerankers-package ColBERT model, min-max normalized per batch.

    MaxSim scores are unbounded, so normalization into [0, 1] is mandatory for
    downstream consumers calibrated on sigmoid cross-encoder probabilities.
    Returns None on failure so the caller can keep the fused order.
    """
    try:
        docs = [hit["content"] for hit in hits]
        ranked = reranker.rank(query=query, docs=docs, doc_ids=list(range(len(docs))))
        raw = [0.0] * len(docs)
        for r in ranked.results:
            doc_id = getattr(r, "doc_id", None)
            if doc_id is None:
                doc_id = r.document.doc_id
            raw[int(doc_id)] = float(r.score)
        lo, hi = min(raw), max(raw)
        if hi - lo < 1e-9:
            return [0.5] * len(raw)
        return [(s - lo) / (hi - lo) for s in raw]
    except Exception as e:
        logger.warning(f"ColBERT rerank failed (keeping fused order): {e}")
        return None


def rerank(query, hits, top_k=None):
    top_k = top_k or settings.rerank_top_k
    if not hits or len(hits) <= top_k:
        return hits

    reranker, kind = get_reranker()
    if kind == "colbert":
        scores = _colbert_scores(reranker, query, hits)
        if scores is None:
            return hits[:top_k]
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = score
    else:
        pairs = [(query, hit["content"]) for hit in hits]
        scores = reranker.predict(pairs)
        # Sigmoid so downstream consumers (gap analyzer thresholds, UI) see probabilities
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = _sigmoid(float(score))

    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ── Parent expansion (small-to-big) ──

# Chunk bodies open with a location line from ingestion: bracketed breadcrumbs
# for prose ("[path > section]") or a comment header for code ("# path:1-40 (python)").
_BREADCRUMB_LINE_RE = re.compile(r"^(\[.*\]|(#|//)\s+\S+:\d+-\d+\s+\(\w+\))\s*$")


def _strip_breadcrumb(text):
    """Drop the leading bracketed breadcrumb line, if present."""
    head, _sep, rest = text.partition("\n")
    if _BREADCRUMB_LINE_RE.match(head.strip()):
        return rest
    return text


def _fetch_siblings(store, doc_path, cache):
    """All chunks for *doc_path* sorted by start_line (cached per expand_parents call)."""
    if doc_path in cache:
        return cache[doc_path]
    sibs = []
    try:
        raw = store.collection.get(where={"document_path": doc_path}, include=["documents", "metadatas"])
        for doc, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
            meta = meta or {}
            if meta.get("start_line") is None or meta.get("end_line") is None:
                continue
            sibs.append(
                {
                    "content": doc,
                    "start_line": int(meta["start_line"]),
                    "end_line": int(meta["end_line"]),
                    "heading_path": meta.get("heading_path"),
                }
            )
        sibs.sort(key=lambda s: (s["start_line"], s["end_line"]))
    except Exception as e:
        logger.warning(f"Parent expansion: sibling fetch failed for {doc_path}: {e}")
        sibs = []
    cache[doc_path] = sibs
    return sibs


def _sibling_index(siblings, start_line, end_line=None):
    for i, sib in enumerate(siblings):
        if sib["start_line"] == start_line and (end_line is None or sib["end_line"] == int(end_line)):
            return i
    for i, sib in enumerate(siblings):  # boundary drift: fall back to the containing chunk
        if sib["start_line"] <= start_line <= sib["end_line"]:
            return i
    return None


def _stitch_parent(siblings, idx, budget, count_tokens):
    """Grow [lo, hi] around siblings[idx] while neighbours are line-adjacent, share the
    seed chunk's heading_path, and fit *budget* tokens. Returns (lo, hi, stitched_text)
    with every breadcrumb after the first stripped."""
    heading = siblings[idx]["heading_path"]
    lo = hi = idx
    used = count_tokens(siblings[idx]["content"])
    while used < budget:
        grew = False
        if lo > 0:
            cand = siblings[lo - 1]
            if cand["heading_path"] == heading and siblings[lo]["start_line"] - cand["end_line"] <= 1:
                tokens = count_tokens(_strip_breadcrumb(cand["content"]))
                if used + tokens <= budget:
                    lo -= 1
                    used += tokens
                    grew = True
        if hi + 1 < len(siblings):
            cand = siblings[hi + 1]
            if cand["heading_path"] == heading and cand["start_line"] - siblings[hi]["end_line"] <= 1:
                tokens = count_tokens(_strip_breadcrumb(cand["content"]))
                if used + tokens <= budget:
                    hi += 1
                    used += tokens
                    grew = True
        if not grew:
            break
    parts = [siblings[lo]["content"]]
    parts += [_strip_breadcrumb(siblings[i]["content"]) for i in range(lo + 1, hi + 1)]
    return lo, hi, "\n".join(parts)


def expand_parents(store, hits, budget_tokens=None):
    """Small-to-big parent expansion — a POST-rerank step, not wired into retrieve().

    For each hit (order preserved), fetch its sibling chunks by document_path and
    stitch contiguous neighbours (start_line adjacency) into one expanded parent,
    stopping when heading_path changes or the hit's share of *budget_tokens*
    (default settings.parent_expand_budget) is spent. A hit whose expanded line
    range is contained in an already-emitted parent from the same file is dropped,
    so two hits from one function never emit the same text twice. Expanded hits
    keep rerank_score and get metadata parent_expanded=True.
    """
    if not hits:
        return []
    from core.ingestion import count_tokens  # lazy: ingestion pulls in heavy optional deps

    budget = budget_tokens or settings.parent_expand_budget
    per_hit_budget = max(budget // len(hits), 1)

    sibling_cache = {}
    emitted = {}  # doc_path -> [(start_line, end_line)] already emitted
    out = []

    def _contained(doc_path, start, end):
        return any(s0 <= start and end <= e0 for s0, e0 in emitted.get(doc_path, []))

    for hit in hits:
        meta = hit.get("metadata") or {}
        doc_path = meta.get("document_path")
        start_line = meta.get("start_line")
        if not doc_path or start_line is None:
            out.append(hit)
            continue

        siblings = _fetch_siblings(store, doc_path, sibling_cache)
        idx = _sibling_index(siblings, int(start_line), meta.get("end_line"))
        if idx is None:
            out.append(hit)
            continue

        lo, hi, content = _stitch_parent(siblings, idx, per_hit_budget, count_tokens)
        new_start, new_end = siblings[lo]["start_line"], siblings[hi]["end_line"]
        if _contained(doc_path, new_start, new_end):
            continue
        emitted.setdefault(doc_path, []).append((new_start, new_end))

        if lo == hi:  # nothing stitched — pass the original hit through
            out.append(hit)
            continue

        expanded = dict(hit)
        expanded_meta = dict(meta)
        expanded_meta.update(
            start_line=new_start,
            end_line=new_end,
            source=f"{doc_path}:{new_start}-{new_end}",
            parent_expanded=True,
        )
        expanded["metadata"] = expanded_meta
        expanded["content"] = content
        out.append(expanded)

    return out


# ── High-level retrieval ──


def retrieve(
    store,
    query,
    top_k=None,
    rerank_top_k=None,
    use_reranking=True,
    use_hybrid=True,
    language_filter=None,
    use_hyde=False,
    use_splade=False,
    use_multiquery=False,
):
    """Hybrid retrieval with optional HyDE or multi-query (RAG-Fusion) expansion.

    HyDE and multi-query are mutually exclusive: when both flags are set,
    multi-query wins and HyDE is skipped (HyDE only runs as a fallback if
    variant generation fails). Reranking always uses the original query.
    """
    top_k = top_k or settings.top_k

    # Multi-query (RAG-Fusion): retrieve per variant, fuse all ranked lists via RRF
    variants = None
    if use_multiquery:
        from core.multi_query import generate_query_variants

        variants = generate_query_variants(query)

    # HyDE: embed a hypothetical answer instead of the raw query for vector search.
    # Sparse retrieval (BM25/SPLADE) always uses the original query text.
    vector_query = query
    if use_hyde and not variants:
        from core.hyde import generate_hypothetical_doc

        hypo = generate_hypothetical_doc(query)
        if hypo:
            vector_query = hypo

    hyde_active = use_hyde and vector_query != query

    # Sparse retrieval: prefer SPLADE when requested and available, else BM25
    splade_available = use_splade and getattr(store, "splade_index", None) is not None

    def sparse_search(q, k):
        if splade_available:
            # Re-read the attribute: a background index swap may have dropped SPLADE mid-retrieve
            splade = getattr(store, "splade_index", None)
            if splade is not None:
                return splade.search(q, top_k=k)
        return store.bm25_index.search(q, top_k=k)

    if variants:
        queries = [query, *variants]
        hit_lists = []
        weights = []
        for q in queries:
            hit_lists.append(store.vector_search(q, top_k=top_k, language_filter=language_filter))
            weights.append(settings.vector_weight)
            if use_hybrid:
                hit_lists.append(sparse_search(q, top_k))
                weights.append(settings.bm25_weight)
        hits = store._rrf_merge(hit_lists, top_k=top_k, weights=weights)
    elif use_hybrid:
        if hyde_active:
            # Split paths: hypothetical doc → vector, original query → sparse
            vector_hits = store.vector_search(vector_query, top_k=top_k, language_filter=language_filter)
            sparse_hits = sparse_search(query, top_k)
            hits = store._rrf_merge(vector_hits, sparse_hits, top_k)
        else:
            if splade_available:
                # Manual hybrid so SPLADE replaces BM25 in the RRF merge
                vector_hits = store.vector_search(query, top_k=top_k, language_filter=language_filter)
                sparse_hits = sparse_search(query, top_k)
                hits = store._rrf_merge(vector_hits, sparse_hits, top_k)
            else:
                hits = store.hybrid_search(query, top_k=top_k, language_filter=language_filter)
    else:
        hits = store.vector_search(vector_query, top_k=top_k, language_filter=language_filter)

    tag = ""
    if variants:
        tag += f" [MultiQuery x{1 + len(variants)}]"
    if hyde_active:
        tag += " [HyDE]"
    if splade_available:
        tag += " [SPLADE]"
    logger.info(f"Retrieved {len(hits)} hits for: {query[:80]}{tag}")

    if use_reranking and len(hits) > 1:
        hits = rerank(query, hits, top_k=rerank_top_k)  # always rerank against original query
        logger.info(f"Reranked to {len(hits)} hits")

    return hits
