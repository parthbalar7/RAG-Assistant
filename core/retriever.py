"""
Hybrid retrieval: Vector (ChromaDB) + BM25 keyword search with cross-encoder reranking.
Pipeline: query -> [vector search + BM25] -> reciprocal rank fusion -> rerank -> top-K
"""

import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from config import settings

logger = logging.getLogger(__name__)

# Singleton models
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: {}".format(settings.embedding_model))
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info("Loading reranker: {}".format(model_name))
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def embed_texts(texts):
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

        tokenized = [self._tokenize(doc) for doc in self.doc_contents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with {} documents".format(len(self.doc_ids)))

    def search(self, query, top_k=10):
        if not self.bm25 or not self.doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.doc_contents[idx],
                    "metadata": self.doc_metadatas[idx],
                    "score": float(scores[idx]),
                    "search_type": "bm25",
                })
        return results

    @staticmethod
    def _tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 1]


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
        self.bm25_index = BM25Index()
        self.splade_index = None   # built only when settings.splade_enabled=True
        self._rebuild_indexes()

        logger.info("VectorStore ready: {} ({} docs)".format(collection_name, self.collection.count()))

    def _rebuild_indexes(self):
        """Rebuild BM25 (always) and SPLADE (when enabled) from the collection."""
        if self.collection.count() == 0:
            return
        self.bm25_index.build_from_collection(self.collection)
        if settings.splade_enabled:
            try:
                if self.splade_index is None:
                    from core.splade_index import SPLADEIndex
                    self.splade_index = SPLADEIndex(settings.splade_model)
                self.splade_index.build_from_collection(self.collection)
            except Exception as e:
                logger.warning(
                    "SPLADE index build failed (falling back to BM25): {}".format(e)
                )
                self.splade_index = None

    # Keep the old name as an alias so any external callers aren't broken
    def _rebuild_bm25(self):
        self._rebuild_indexes()

    @property
    def count(self):
        return self.collection.count()

    def add_chunks(self, chunks, batch_size=64):
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            embeddings = embed_texts(documents)
            metadatas = [
                {
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
            logger.info("Indexed {}/{} chunks".format(added, len(chunks)))

        self._rebuild_indexes()
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
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                similarity = 1 - (dist / 2)
                if similarity >= settings.similarity_threshold:
                    hits.append({
                        "content": doc,
                        "metadata": meta,
                        "score": round(similarity, 4),
                        "search_type": "vector",
                    })
        return hits

    def _rrf_merge(self, vector_hits, bm25_hits, top_k):
        """Reciprocal Rank Fusion of pre-computed vector and BM25 hit lists."""
        k = 60
        fused_scores = {}

        for rank, hit in enumerate(vector_hits):
            key = hit["content"][:100]
            if key not in fused_scores:
                fused_scores[key] = dict(hit)
                fused_scores[key]["rrf_score"] = 0
                fused_scores[key]["search_types"] = []
            fused_scores[key]["rrf_score"] += settings.vector_weight / (k + rank + 1)
            fused_scores[key]["search_types"].append("vector")

        for rank, hit in enumerate(bm25_hits):
            key = hit["content"][:100]
            if key not in fused_scores:
                fused_scores[key] = dict(hit)
                fused_scores[key]["rrf_score"] = 0
                fused_scores[key]["search_types"] = []
            fused_scores[key]["rrf_score"] += settings.bm25_weight / (k + rank + 1)
            fused_scores[key]["search_types"].append("bm25")

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

        all_docs = self.collection.get(include=["metadatas"])
        files = {}

        for meta in all_docs["metadatas"]:
            path = meta.get("document_path", "")
            if path and path not in files:
                files[path] = {
                    "path": path,
                    "language": meta.get("language", ""),
                    "chunk_count": 0,
                }
            if path in files:
                files[path]["chunk_count"] += 1

        return sorted(files.values(), key=lambda x: x["path"])

    def delete_file(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific file."""
        if self.count == 0:
            return 0
        all_docs = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
            if meta.get("document_path") == file_path
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            self._rebuild_indexes()
            logger.info("Deleted {} chunks for {}".format(len(ids_to_delete), file_path))
        return len(ids_to_delete)

    def clear(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25_index = BM25Index()
        self.splade_index = None
        logger.info("Vector store cleared")


# ── Reranking ──

def rerank(query, hits, top_k=None):
    top_k = top_k or settings.rerank_top_k
    if not hits or len(hits) <= top_k:
        return hits

    reranker = get_reranker()
    pairs = [(query, hit["content"]) for hit in hits]
    scores = reranker.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ── High-level retrieval ──

def retrieve(store, query, top_k=None, rerank_top_k=None, use_reranking=True, use_hybrid=True, language_filter=None, use_hyde=False, use_splade=False):
    top_k = top_k or settings.top_k

    # HyDE: embed a hypothetical answer instead of the raw query for vector search.
    # Sparse retrieval (BM25/SPLADE) always uses the original query text.
    vector_query = query
    if use_hyde:
        from core.hyde import generate_hypothetical_doc
        hypo = generate_hypothetical_doc(query)
        if hypo:
            vector_query = hypo

    hyde_active = use_hyde and vector_query != query

    # Sparse retrieval: prefer SPLADE when requested and available, else BM25
    splade_available = use_splade and getattr(store, "splade_index", None) is not None
    sparse_label = "splade" if splade_available else "bm25"

    def sparse_search(q, k):
        if splade_available:
            return store.splade_index.search(q, top_k=k)
        return store.bm25_index.search(q, top_k=k)

    if use_hybrid:
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
    if hyde_active:
        tag += " [HyDE]"
    if splade_available:
        tag += " [SPLADE]"
    logger.info("Retrieved {} hits for: {}{}".format(len(hits), query[:80], tag))

    if use_reranking and len(hits) > 1:
        hits = rerank(query, hits, top_k=rerank_top_k)  # always rerank against original query
        logger.info("Reranked to {} hits".format(len(hits)))

    return hits
