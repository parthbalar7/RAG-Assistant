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
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


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
        self._rebuild_bm25()

        logger.info("VectorStore ready: {} ({} docs)".format(collection_name, self.collection.count()))

    def _rebuild_bm25(self):
        if self.collection.count() > 0:
            self.bm25_index.build_from_collection(self.collection)

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

        self._rebuild_bm25()
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

    def hybrid_search(self, query_text, top_k=10, language_filter=None):
        vector_hits = self.vector_search(query_text, top_k=top_k, language_filter=language_filter)
        bm25_hits = self.bm25_index.search(query_text, top_k=top_k)

        # Reciprocal Rank Fusion
        k = 60
        fused_scores = {}

        for rank, hit in enumerate(vector_hits):
            key = hit["content"][:100]
            rrf_score = settings.vector_weight / (k + rank + 1)
            if key not in fused_scores:
                fused_scores[key] = dict(hit)
                fused_scores[key]["rrf_score"] = 0
                fused_scores[key]["search_types"] = []
            fused_scores[key]["rrf_score"] += rrf_score
            fused_scores[key]["search_types"].append("vector")

        for rank, hit in enumerate(bm25_hits):
            key = hit["content"][:100]
            rrf_score = settings.bm25_weight / (k + rank + 1)
            if key not in fused_scores:
                fused_scores[key] = dict(hit)
                fused_scores[key]["rrf_score"] = 0
                fused_scores[key]["search_types"] = []
            fused_scores[key]["rrf_score"] += rrf_score
            fused_scores[key]["search_types"].append("bm25")

        results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        for hit in results:
            hit["score"] = hit["rrf_score"]
            hit["search_type"] = "+".join(set(hit.get("search_types", ["unknown"])))

        return results[:top_k]

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

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25_index = BM25Index()
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

def retrieve(store, query, top_k=None, rerank_top_k=None, use_reranking=True, use_hybrid=True, language_filter=None):
    top_k = top_k or settings.top_k

    if use_hybrid:
        hits = store.hybrid_search(query, top_k=top_k, language_filter=language_filter)
    else:
        hits = store.vector_search(query, top_k=top_k, language_filter=language_filter)

    logger.info("Retrieved {} hits for: {}".format(len(hits), query[:80]))

    if use_reranking and len(hits) > 1:
        hits = rerank(query, hits, top_k=rerank_top_k)
        logger.info("Reranked to {} hits".format(len(hits)))

    return hits
