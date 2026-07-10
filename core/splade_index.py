"""
core/splade_index.py — Learned Sparse Retrieval with SPLADE

Drop-in replacement for BM25Index with the same interface:
  build_from_collection(collection) → None
  search(query, top_k) → list[dict]

How it works
------------
SPLADE uses a BERT masked-language-model head to assign a weight to every
token in the vocabulary (~30K terms) for each document and query.  Most
weights are zero (sparse).  Retrieval is a dot-product over these sparse
vectors — semantically related terms receive non-zero weights even when they
don't appear verbatim, fixing the vocabulary-mismatch problem of BM25.

Storage
-------
Document vectors are encoded once (at index-build time) and stored as a
scipy CSR sparse matrix (shape N x vocab_size).  Typical density is < 1 %,
so 100K docs ≈ 50-150 MB in RAM.  Query encoding at search time is a single
forward pass (~10-50 ms on CPU).

Model default: prithivida/Splade_PP_en_v1 (publicly accessible, no HF gating).
Configurable via settings.splade_model (RAG_SPLADE_MODEL env var).
Note: naver/splade-v3 requires HuggingFace account access approval — use the
default or another ungated model unless you have accepted the naver terms.

Requirements
------------
sentence-transformers >= 3.0  (SparseEncoder class)
scipy                          (CSR sparse matrix — transitive dep of sbert)
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Singleton — loaded once, shared across all SPLADEIndex instances
_sparse_encoder = None
_sparse_encoder_lock = threading.Lock()


def get_sparse_encoder(model_name: str):
    global _sparse_encoder
    if _sparse_encoder is None:
        with _sparse_encoder_lock:
            if _sparse_encoder is None:
                from sentence_transformers import SparseEncoder

                logger.info(f"Loading SPLADE model: {model_name}")
                _sparse_encoder = SparseEncoder(model_name)
                logger.info("SPLADE model loaded")
    return _sparse_encoder


# ─────────────────────────────────────────────────────────────────────────────


class SPLADEIndex:
    """
    In-memory SPLADE index.  Same external interface as BM25Index so it can
    be swapped in transparently inside VectorStore.
    """

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.model_name = model_name
        self.doc_ids: list[str] = []
        self.doc_contents: list[str] = []
        self.doc_metadatas: list[dict] = []
        self._doc_matrix = None  # scipy.sparse.csr_matrix (N, vocab)
        self._vocab_size: int | None = None

    # ── public API (mirrors BM25Index) ───────────────────────────────────────

    def build_from_collection(self, collection) -> None:
        """Encode all documents in the ChromaDB collection into SPLADE vectors."""
        try:
            count = collection.count()
        except Exception as e:
            logger.warning(f"SPLADE: could not read collection count: {e}")
            return
        if count == 0:
            self._doc_matrix = None
            return

        all_docs = collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        self.doc_contents = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]

        model = get_sparse_encoder(self.model_name)

        logger.info(f"SPLADE: encoding {len(self.doc_contents)} documents (this may take a moment)…")

        raw = model.encode(
            self.doc_contents,
            batch_size=32,
            show_progress_bar=False,
        )

        self._doc_matrix = _to_csr(raw)
        self._vocab_size = self._doc_matrix.shape[1] if self._doc_matrix is not None else None

        logger.info(f"SPLADE index built: {len(self.doc_ids)} docs, vocab {self._vocab_size}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Encode query and return top-k docs by SPLADE dot-product score."""
        if self._doc_matrix is None or not self.doc_ids:
            return []

        model = get_sparse_encoder(self.model_name)
        q_raw = model.encode(query)  # single string → 1-D or 2-D
        q_vec = _to_csr(q_raw)  # shape (1, vocab)

        try:
            # Efficient sparse dot product: (N, vocab) · (vocab, 1) → (N, 1)
            scores_mat = self._doc_matrix.dot(q_vec.T)  # (N, 1) sparse
            scores = np.asarray(scores_mat.todense()).flatten().astype(float)
        except Exception as e:
            logger.warning(f"SPLADE dot-product failed: {e}")
            return []

        n = len(scores)
        k = min(top_k, n)
        if k == 0:
            return []

        # Partial sort — faster than full sort for large N
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {
                "id": self.doc_ids[i],
                "content": self.doc_contents[i],
                "metadata": self.doc_metadatas[i],
                "score": float(scores[i]),
                "search_type": "splade",
            }
            for i in top_idx
            if scores[i] > 0
        ]


# ── helpers ──────────────────────────────────────────────────────────────────


def _to_csr(raw):
    """
    Convert whatever SparseEncoder.encode() returns into a scipy CSR matrix.

    sentence-transformers SparseEncoder can return:
      - numpy ndarray  (N, vocab) dense
      - torch.Tensor   (N, vocab) dense or sparse
      - scipy sparse   (N, vocab)
      - A single vector when input was a single string
    """
    import scipy.sparse as sp

    # Already a scipy sparse matrix
    if sp.issparse(raw):
        return sp.csr_matrix(raw, dtype=np.float32)

    # Try converting torch tensor
    try:
        import torch

        if isinstance(raw, torch.Tensor):
            if raw.is_sparse:
                raw = raw.to_dense()
            arr = raw.detach().cpu().numpy()
            # Ensure 2-D
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return sp.csr_matrix(arr, dtype=np.float32)
    except ImportError:
        pass

    # Numpy array or list
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return sp.csr_matrix(arr)
