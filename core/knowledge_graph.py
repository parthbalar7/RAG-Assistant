"""
core/knowledge_graph.py — Memory Palace: Spatial Knowledge Graph

Builds a navigable entity-relationship graph from your indexed documents.
Retrieval becomes graph pathfinding + vector search combined, so the model
can traverse concept clusters instead of doing only flat nearest-neighbour
lookups.

Graph construction (POST /api/graph/build), mode = settings.graph_extraction:
  - "ner" (default): LLM-free — per-chunk spaCy NER for prose plus a
    code-identifier regex (CamelCase/snake_case/dotted) for code chunks,
    exact single-chunk attribution, co_occurs edges between entities that
    share a chunk. Seconds instead of hours on CPU.
  - "llm": legacy batched LLM triple extraction, with per-chunk attribution
    (chunk number in the JSON schema + substring fallback).
  - "hybrid": NER for everything, then an LLM enrichment pass over only the
    chunks referenced by the top ~50 highest-degree entities.
  After extraction, one blocked NxN cosine pass over the entity-embedding
  cache adds alias_of edges (>0.85 similarity), and Louvain community
  detection stamps every node with a `community` id. Community summaries are
  generated lazily on first request and cached into the graph JSON.

Hybrid retrieval:
  1. Embed query → cosine-rank entity names → pick top-K seed nodes
  2. Personalized PageRank from the seeds (HippoRAG 2 style; 2-hop BFS fallback)
  3. Score chunks by summing the PPR mass of the entities referencing them
  4. ChromaDB.get(ids=...) to retrieve the top-scored chunks
  5. Merge with standard vector-search results (deduplicated)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path("data/graphs")

# ── LLM extraction prompt ─────────────────────────────────────────────────────
EXTRACT_SYSTEM = """You are an entity extraction engine.
Extract the most important entities and relationships from the provided text chunks.
Return ONLY a valid JSON object with this exact schema:
{
  "entities": [{"name": "...", "type": "class|function|module|concept|config|topic|api", "chunk": 1}],
  "relations": [{"from": "EntityA", "to": "EntityB", "rel": "uses|extends|calls|imports|defines|configures|related_to|part_of"}]
}
Rules:
- Entity names: 1-4 words, use the exact name from the text (CamelCase, snake_case, etc.)
- "chunk" is the 1-based number of the chunk the entity appears in
- Max 8 entities and 10 relations per response
- Only include a relation when BOTH endpoints appear in the entity list
- If the text has no clear entities, return {"entities": [], "relations": []}
- NEVER add commentary outside the JSON"""

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "chunk": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "rel": {"type": "string"},
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["entities", "relations"],
}

_COMMUNITY_SYSTEM = """You summarize a cluster of related entities from a user's document collection.
Given entity names and text excerpts, describe the common theme in 2-3 plain sentences.
No preamble, no bullet points — just the summary text."""


# ── LLM-free extraction (spaCy NER + code-identifier regex) ───────────────────

# CamelCase/mixedCase with >=2 humps, snake_case with >=1 inner underscore
# (optionally private-prefixed), dotted module paths with >=2-char segments
# (so "e.g."/"i.e." never match).
_CAMEL_RE = re.compile(r"\b[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+\b")
_SNAKE_RE = re.compile(r"\b_{0,2}[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]+)+\b")

# spaCy label → node_type; unmapped labels (dates, quantities, ...) are dropped.
_SPACY_LABEL_TYPES = {
    "PERSON": "topic",
    "ORG": "topic",
    "GPE": "topic",
    "LOC": "topic",
    "NORP": "topic",
    "FAC": "topic",
    "EVENT": "topic",
    "PRODUCT": "concept",
    "WORK_OF_ART": "concept",
    "LAW": "concept",
    "LANGUAGE": "concept",
}

_SPACY_NODE_TYPES = frozenset(_SPACY_LABEL_TYPES.values())

_MAX_ENTITIES_PER_CHUNK = 15
_SPACY_TEXT_CAP = 4000  # chars fed to spaCy per chunk
_MAX_TOTAL_ENTITIES = 5000  # distinct-entity cap for NER extraction; top by chunk frequency kept
_MAX_COOCCUR_ENTITIES = 6  # co_occurs pairs only the first N entities of a chunk (<= 15 pairs)
_ALIAS_MAX_ENTITIES = 10000  # skip the alias NxN cosine pass entirely beyond this many entities
_ALIAS_BLOCK_BYTES = 100 * 1024 * 1024  # ~100MB bound per alias matmul block

_nlp = None
_nlp_failed = False


def _get_nlp():
    """Lazy spaCy loader — returns None (and warns once) when spacy or the model is missing."""
    global _nlp, _nlp_failed
    if _nlp is not None or _nlp_failed:
        return _nlp
    try:
        import spacy

        # NER needs tok2vec; the rest of the sm pipeline is dead weight here
        _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        logger.info("spaCy en_core_web_sm loaded for graph NER")
    except Exception as e:
        _nlp_failed = True
        logger.warning(f"spaCy unavailable ({e}) — graph extraction falls back to regex-only")
    return _nlp


def _clean_entity_name(name: str) -> str:
    """Normalize an NER span; returns '' when it is not a usable entity name."""
    name = " ".join(name.split())
    for det in ("the ", "a ", "an "):
        if name.lower().startswith(det):
            name = name[len(det) :]
    if not (2 <= len(name) <= 60) or len(name.split()) > 4:
        return ""
    if not any(c.isalpha() for c in name):
        return ""
    return name


def _regex_entities(text: str) -> list[tuple[str, str]]:
    """Code-identifier extraction: (name, node_type) pairs in order of appearance."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(name: str, etype: str):
        key = name.lower()
        if key in seen or not (3 <= len(name) <= 60):
            return
        seen.add(key)
        found.append((name, etype))

    for m in _DOTTED_RE.finditer(text):
        name = m.group(0)
        if name.startswith(("self.", "cls.")):
            name = name.split(".", 1)[1]
        if "." in name:
            _add(name, "module")
    # Mask dotted matches so their segments don't re-match as camel/snake
    text = _DOTTED_RE.sub(" ", text)
    for m in _CAMEL_RE.finditer(text):
        name = m.group(0)
        _add(name, "class" if name[0].isupper() else "function")
    for m in _SNAKE_RE.finditer(text):
        _add(m.group(0), "function")
    return found[:_MAX_ENTITIES_PER_CHUNK]


def _spacy_entities(spacy_doc) -> list[tuple[str, str]]:
    """(name, node_type) pairs from a spaCy doc, filtered to useful labels."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ent in spacy_doc.ents:
        etype = _SPACY_LABEL_TYPES.get(ent.label_)
        if etype is None:
            continue
        name = _clean_entity_name(ent.text)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        found.append((name, etype))
    return found


# ── data classes ──────────────────────────────────────────────────────────────


@dataclass
class GraphNode:
    name: str
    node_type: str  # class|function|module|concept|config|topic|api
    chunk_ids: list[str] = field(default_factory=list)
    doc_paths: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    rel_type: str
    weight: int = 1

    def to_dict(self):
        return asdict(self)


# ── main graph class ──────────────────────────────────────────────────────────


class KnowledgeGraph:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.built_at: float | None = None
        self.total_chunks_processed: int = 0
        # RLock: build/summarize hold it across mutation+save, and save() re-acquires it
        self._lock = threading.RLock()
        # Pre-computed entity embedding cache — rebuilt after build/load
        self._entity_keys: list[str] | None = None  # ordered list of node keys
        self._entity_vecs: np.ndarray | None = None  # (N, dim) normalised matrix
        # Lazy community summaries, keyed by str(community_id) for JSON stability
        self._community_summaries: dict[str, str] = {}
        self._path: Path | None = None  # set on save/load — lets summaries persist

    def reset(self):
        """Clear graph data for a fresh rebuild. Safer than calling __init__()."""
        with self._lock:
            self.graph = nx.DiGraph()
            self.built_at = None
            self.total_chunks_processed = 0
            self._entity_keys = None
            self._entity_vecs = None
            self._community_summaries = {}

    # ── entity embedding cache ────────────────────────────────────────────────

    def _build_entity_cache(self):
        """Pre-embed all entity display names into a (N, dim) normalised matrix.
        Called once after build or load. Query time then only needs 1 embedding."""
        from core.retriever import embed_texts

        keys = list(self.graph.nodes())
        if not keys:
            self._entity_keys = []
            self._entity_vecs = np.empty((0, 0), dtype=np.float32)
            return
        display_names = [self.graph.nodes[k].get("display_name", k) for k in keys]
        try:
            vecs = embed_texts(display_names)
            self._entity_keys = keys
            self._entity_vecs = np.array(vecs, dtype=np.float32)
            logger.info(f"Entity embedding cache built: {len(keys)} entities")
        except Exception as e:
            logger.warning(f"Entity cache build failed: {e}")
            self._entity_keys = None
            self._entity_vecs = None

    def _ensure_entity_cache(self):
        if self._entity_vecs is None:
            self._build_entity_cache()

    # ── mutation helpers ──────────────────────────────────────────────────────

    def _canon(self, name: str) -> str:
        """Canonical entity key: lower-strip."""
        return name.strip().lower()

    def add_entity(self, name: str, node_type: str, chunk_id: str, doc_path: str):
        key = self._canon(name)
        if not self.graph.has_node(key):
            self.graph.add_node(key, display_name=name, node_type=node_type, chunk_ids=[], doc_paths=[])
        node = self.graph.nodes[key]
        if chunk_id and chunk_id not in node["chunk_ids"]:
            node["chunk_ids"].append(chunk_id)
        if doc_path and doc_path not in node["doc_paths"]:
            node["doc_paths"].append(doc_path)

    def add_relation(self, src_name: str, tgt_name: str, rel_type: str):
        sk = self._canon(src_name)
        tk = self._canon(tgt_name)
        if not self.graph.has_node(sk) or not self.graph.has_node(tk):
            return
        if self.graph.has_edge(sk, tk):
            self.graph[sk][tk]["weight"] = self.graph[sk][tk].get("weight", 1) + 1
        else:
            self.graph.add_edge(sk, tk, rel_type=rel_type, weight=1)

    # ── build from vector store ───────────────────────────────────────────────

    def build_from_store(self, store, batch_size: int = 3) -> dict:
        """
        Pull all chunks from *store* (VectorStore) and extract entities/relations.
        Extraction mode comes from settings.graph_extraction ('ner'|'llm'|'hybrid').
        Returns a summary dict: {nodes, edges, chunks_processed, ms, ...}.
        """
        if store.count == 0:
            return {"nodes": 0, "edges": 0, "chunks_processed": 0, "ms": 0, "message": "No documents indexed"}

        mode = (settings.graph_extraction or "ner").lower()
        if mode not in ("ner", "llm", "hybrid"):
            logger.warning(f"Unknown graph_extraction mode '{mode}' — using 'ner'")
            mode = "ner"

        t0 = time.time()

        # Fetch all chunks
        raw = store.collection.get(include=["documents", "metadatas"])
        all_ids = raw.get("ids", [])
        all_docs = raw.get("documents", [])
        all_metas = raw.get("metadatas", [])
        total = len(all_ids)
        logger.info(f"Knowledge graph: processing {total} chunks (mode={mode})")

        # Hold the lock across mutation + downstream save(): a concurrent
        # summarize_community persist must not interleave with a rebuild.
        with self._lock:
            if mode == "llm":
                stats = self._extract_llm(all_ids, all_docs, all_metas, batch_size)
            else:
                stats = self._extract_ner(all_ids, all_docs, all_metas)

            if mode == "hybrid":
                hub_idx = self._hub_chunk_indices(all_ids)
                if hub_idx:
                    logger.info(f"Hybrid enrichment: LLM pass over {len(hub_idx)} hub chunks")
                    hub_stats = self._extract_llm(
                        [all_ids[i] for i in hub_idx],
                        [all_docs[i] for i in hub_idx],
                        [all_metas[i] for i in hub_idx],
                        batch_size,
                    )
                    for k in ("entities", "relations", "errors"):
                        stats[k] = stats.get(k, 0) + hub_stats.get(k, 0)

            self.built_at = time.time()
            self.total_chunks_processed = total
            self._build_entity_cache()  # pre-compute embeddings once here
            alias_edges = self._add_alias_edges()
            communities = self._detect_communities()
            self._community_summaries = {}  # summaries are stale after any rebuild
            ms = int((time.time() - t0) * 1000)

            result = {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "chunks_processed": total,
                "entities_extracted": stats.get("entities", 0),
                "relations_extracted": stats.get("relations", 0),
                "extraction_errors": stats.get("errors", 0),
                "alias_edges": alias_edges,
                "communities": communities,
                "mode": mode,
                "ms": ms,
            }
        if mode == "llm" and stats.get("batches") and stats.get("errors") == stats.get("batches"):
            result["warning"] = "All batches failed — check LLM connectivity"
        logger.info(
            "Knowledge graph built (%s): %d nodes, %d edges, %d alias edges, %d communities, %d errors in %dms",
            mode,
            result["nodes"],
            result["edges"],
            alias_edges,
            communities,
            result["extraction_errors"],
            ms,
        )
        return result

    def _extract_ner(self, all_ids: list, all_docs: list, all_metas: list) -> dict:
        """LLM-free extraction: spaCy NER for prose, identifier regex for code.
        Exact single-chunk attribution + co_occurs edges within each chunk."""
        from core.ingestion import CODE_LANGUAGES

        nlp = _get_nlp()
        per_chunk: list[list[tuple[str, str]]] = [[] for _ in all_ids]
        prose_idx: list[int] = []

        for i, (doc, meta) in enumerate(zip(all_docs, all_metas)):
            lang = (meta or {}).get("language", "")
            if lang in CODE_LANGUAGES or nlp is None:
                per_chunk[i] = _regex_entities(doc)
            else:
                prose_idx.append(i)

        if prose_idx and nlp is not None:
            texts = [all_docs[i][:_SPACY_TEXT_CAP] for i in prose_idx]
            try:
                for i, spacy_doc in zip(prose_idx, nlp.pipe(texts, batch_size=64)):
                    ents = _spacy_entities(spacy_doc)
                    # Technical prose is full of code identifiers spaCy's NER
                    # misses; the identifier patterns don't fire on natural text
                    seen = {n.lower() for n, _ in ents}
                    for name, etype in _regex_entities(all_docs[i]):
                        if name.lower() not in seen:
                            ents.append((name, etype))
                    per_chunk[i] = ents[:_MAX_ENTITIES_PER_CHUNK]
            except Exception as e:
                logger.warning(f"spaCy NER pass failed ({e}) — regex-only fallback")
                for i in prose_idx:
                    per_chunk[i] = _regex_entities(all_docs[i])

        # Cap distinct entities: big code corpora yield a node per identifier,
        # which blows up the entity-embedding cache and the alias matmul. Keep
        # the top by chunk frequency, always preferring spaCy-NER-typed
        # entities over regex identifiers when trimming.
        freq: dict[str, int] = {}
        etype_of: dict[str, str] = {}
        for ents in per_chunk:
            for name, etype in ents:
                key = self._canon(name)
                freq[key] = freq.get(key, 0) + 1
                etype_of.setdefault(key, etype)
        if len(freq) > _MAX_TOTAL_ENTITIES:
            ranked = sorted(freq, key=lambda k: (etype_of[k] in _SPACY_NODE_TYPES, freq[k]), reverse=True)
            keep = set(ranked[:_MAX_TOTAL_ENTITIES])
            logger.warning(
                f"NER extraction found {len(freq)} distinct entities — keeping top {_MAX_TOTAL_ENTITIES} "
                "by chunk frequency"
            )
            per_chunk = [[(n, t) for n, t in ents if self._canon(n) in keep] for ents in per_chunk]

        total_entities = 0
        total_relations = 0
        for i, ents in enumerate(per_chunk):
            cid = all_ids[i]
            doc_path = (all_metas[i] or {}).get("document_path", "")
            names = []
            for name, etype in ents:
                self.add_entity(name, etype, cid, doc_path)
                names.append(name)
            total_entities += len(names)
            # co_occurs edges between entities sharing a chunk — capped to the
            # first N so a 15-entity chunk emits <=15 pairs instead of 105;
            # canonical pair order so the weight-increment dedup in add_relation applies
            co_names = names[:_MAX_COOCCUR_ENTITIES]
            for a_pos in range(len(co_names)):
                for b_pos in range(a_pos + 1, len(co_names)):
                    a, b = co_names[a_pos], co_names[b_pos]
                    if self._canon(a) > self._canon(b):
                        a, b = b, a
                    self.add_relation(a, b, "co_occurs")
            total_relations += len(co_names) * (len(co_names) - 1) // 2

        return {"entities": total_entities, "relations": total_relations, "errors": 0}

    def _extract_llm(self, all_ids: list, all_docs: list, all_metas: list, batch_size: int = 3) -> dict:
        """Batched LLM triple extraction with per-chunk attribution: the schema
        asks for a chunk number; a substring scan is the fallback."""
        from core import llm_client as _llm

        total = len(all_ids)
        entities = relations = errors = 0
        num_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch_num = i // batch_size + 1
            batch_ids = all_ids[i : i + batch_size]
            batch_docs = all_docs[i : i + batch_size]
            batch_metas = all_metas[i : i + batch_size]

            # Format chunks into a single prompt block
            chunk_block = ""
            for j, (doc, meta) in enumerate(zip(batch_docs, batch_metas)):
                doc_path = (meta or {}).get("document_path", "")
                chunk_block += f"\n--- Chunk {j + 1} (from: {doc_path}) ---\n{doc[:800]}\n"

            try:
                resp = _llm.chat(
                    messages=[{"role": "user", "content": chunk_block.strip()}],
                    system=EXTRACT_SYSTEM,
                    max_tokens=350,
                    temperature=0.0,
                    stream=False,
                    json_schema=_EXTRACT_SCHEMA,
                )
                extracted = _parse_extraction(resp)
            except Exception as e:
                logger.warning("Extraction failed for batch %d/%d: %s", batch_num, num_batches, e)
                errors += 1
                continue

            batch_ents = extracted.get("entities", [])
            batch_rels = extracted.get("relations", [])

            for ent in batch_ents:
                # Handle both dict {"name": "...", "type": "..."} and plain string formats
                if isinstance(ent, str):
                    name, etype, chunk_no = ent.strip(), "concept", None
                elif isinstance(ent, dict):
                    name = str(ent.get("name", "")).strip()
                    etype = ent.get("type") or "concept"
                    chunk_no = ent.get("chunk")
                else:
                    continue
                if not name:
                    continue
                # Per-chunk attribution: LLM-reported chunk number, else the
                # chunks whose text contains the name; only smear across the
                # whole batch when both signals are absent.
                if isinstance(chunk_no, int) and 1 <= chunk_no <= len(batch_ids):
                    targets = [chunk_no - 1]
                else:
                    targets = [j for j, doc in enumerate(batch_docs) if name.lower() in doc.lower()]
                    if not targets:
                        targets = list(range(len(batch_ids)))
                for j in targets:
                    self.add_entity(name, etype, batch_ids[j], (batch_metas[j] or {}).get("document_path", ""))

            for rel in batch_rels:
                if not isinstance(rel, dict):
                    continue
                src = str(rel.get("from", rel.get("source", ""))).strip()
                tgt = str(rel.get("to", rel.get("target", ""))).strip()
                rtype = rel.get("rel", rel.get("relation", "related_to"))
                if src and tgt:
                    self.add_relation(src, tgt, rtype)

            entities += len(batch_ents)
            relations += len(batch_rels)
            logger.info(
                "Graph batch %d/%d: +%d entities, +%d relations",
                batch_num,
                num_batches,
                len(batch_ents),
                len(batch_rels),
            )

        return {"entities": entities, "relations": relations, "errors": errors, "batches": num_batches}

    def _hub_chunk_indices(self, all_ids: list, top_n: int = 50, per_hub: int = 2, max_chunks: int = 100) -> list:
        """Indices (into all_ids) of the chunks referenced by the top-degree
        entities — the subset the hybrid mode sends through the LLM."""
        degree_map = dict(self.graph.degree())
        hubs = sorted(degree_map, key=lambda k: degree_map[k], reverse=True)[:top_n]
        id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
        picked: list[int] = []
        seen: set[int] = set()
        for key in hubs:
            for cid in self.graph.nodes[key].get("chunk_ids", [])[:per_hub]:
                idx = id_to_idx.get(cid)
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    picked.append(idx)
                    if len(picked) >= max_chunks:
                        return picked
        return picked

    def _add_alias_edges(self, threshold: float = 0.85, max_edges: int = 2000) -> int:
        """Blocked NxN cosine pass over the entity-vector cache; near-identical
        names get alias_of edges so PPR can bridge synonyms _canon() misses."""
        if self._entity_vecs is None or self._entity_keys is None or self._entity_vecs.shape[0] < 2:
            return 0
        vecs = self._entity_vecs
        keys = self._entity_keys
        n = vecs.shape[0]
        if n > _ALIAS_MAX_ENTITIES:
            logger.warning(f"Skipping alias-edge pass: {n} entities exceeds cap of {_ALIAS_MAX_ENTITIES}")
            return 0
        # Size row blocks so each sims block (block_rows x n float32) stays under ~100MB
        block_rows = max(1, _ALIAS_BLOCK_BYTES // (n * vecs.itemsize))
        added = 0
        for start in range(0, n, block_rows):  # blocked matmul caps peak memory
            sims = vecs[start : start + block_rows] @ vecs.T
            rows, cols = np.where(sims >= threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                i = start + r
                if c <= i:  # upper triangle only (also skips self-similarity)
                    continue
                a, b = keys[i], keys[c]
                if self.graph.has_edge(a, b) or self.graph.has_edge(b, a):
                    continue
                self.graph.add_edge(a, b, rel_type="alias_of", weight=1)
                added += 1
                if added >= max_edges:
                    logger.info(f"Alias edge cap reached ({max_edges})")
                    return added
        if added:
            logger.info(f"Added {added} alias_of edges (cos >= {threshold})")
        return added

    # ── communities (Louvain) ─────────────────────────────────────────────────

    def _detect_communities(self) -> int:
        """Louvain community detection; stamps each node with a 'community' int
        attribute (persists via the node_link save). Returns community count."""
        if self.graph.number_of_nodes() == 0:
            return 0
        try:
            comms = nx.community.louvain_communities(self.graph.to_undirected(), weight="weight", seed=42)
        except Exception as e:
            logger.warning(f"Louvain community detection failed: {e}")
            return 0
        # Largest community gets id 0 — stable given the fixed seed
        for cid, members in enumerate(sorted(comms, key=len, reverse=True)):
            for key in members:
                self.graph.nodes[key]["community"] = cid
        logger.info(f"Louvain: {len(comms)} communities")
        return len(comms)

    def get_communities(self, top_entities: int = 5) -> list[dict]:
        """Community roster — pure graph math, no LLM. Sorted largest-first."""
        groups: dict[int, list[str]] = {}
        for key, nd in self.graph.nodes(data=True):
            cid = nd.get("community")
            if isinstance(cid, int):
                groups.setdefault(cid, []).append(key)
        out = []
        for cid, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            ranked = sorted(members, key=lambda k: self.graph.degree(k), reverse=True)
            out.append(
                {
                    "community_id": cid,
                    "size": len(members),
                    "top_entities": [self.graph.nodes[k].get("display_name", k) for k in ranked[:top_entities]],
                    "has_summary": str(cid) in self._community_summaries,
                }
            )
        return out

    def summarize_community(self, community_id: int, store) -> str:
        """Lazy LLM summary of one community from its top entities' chunks.
        Cached into the graph JSON on first call so repeats are free.
        Returns '' when the community is unknown or the LLM is unreachable."""
        from core import llm_client as _llm

        cached = self._community_summaries.get(str(community_id))
        if cached:
            return cached

        members = [k for k, nd in self.graph.nodes(data=True) if nd.get("community") == community_id]
        if not members:
            return ""
        ranked = sorted(members, key=lambda k: self.graph.degree(k), reverse=True)
        top_names = [self.graph.nodes[k].get("display_name", k) for k in ranked[:10]]

        chunk_ids: list[str] = []
        for key in ranked[:10]:
            for cid in self.graph.nodes[key].get("chunk_ids", []):
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
                if len(chunk_ids) >= 6:
                    break
            if len(chunk_ids) >= 6:
                break

        excerpts = ""
        if chunk_ids and store is not None:
            try:
                raw = store.collection.get(ids=chunk_ids, include=["documents"])
                for doc in raw.get("documents", []):
                    excerpts += f"\n---\n{doc[:600]}"
            except Exception as e:
                logger.warning(f"Community chunk fetch failed: {e}")

        prompt = (
            f"Entities in this cluster: {', '.join(top_names)}\n\nExcerpts from their documents:{excerpts or ' (none)'}"
        )
        try:
            resp = _llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system=_COMMUNITY_SYSTEM,
                max_tokens=200,
                temperature=0.3,
                stream=False,
            )
        except Exception as e:
            logger.warning(f"Community summary LLM call failed: {e}")
            return ""

        summary = (resp or "").strip()
        if summary:
            with self._lock:  # never persist mid-rebuild — build resets summaries under the same lock
                self._community_summaries[str(community_id)] = summary
                if self._path is not None:  # persist so repeat calls survive restarts
                    try:
                        self.save(self._path)
                    except Exception as e:
                        logger.warning(f"Community summary persist failed: {e}")
        return summary

    # ── hybrid retrieval ──────────────────────────────────────────────────────

    def graph_retrieve(
        self, query: str, store, top_k: int = 10, seed_chunk_ids: list[str] | None = None
    ) -> tuple[list[dict], dict]:
        """
        Personalized-PageRank + vector search hybrid retrieval.

        1. Embed query and all entity names → cosine-rank to find seed nodes
        2. Run personalized PageRank from the seeds (falls back to 2-hop BFS)
        3. Score chunks by summing the PPR mass of the entities referencing them
        4. Fetch the top-scored chunks from ChromaDB

        *seed_chunk_ids*: chunk ids already surfaced by vector search — entities
        attached to them get a personalization boost (HippoRAG 2 passage integration).
        """
        from core.retriever import embed_texts

        if self.graph.number_of_nodes() == 0:
            return [], {}

        # Use pre-cached entity embeddings — only embed the query (1 text vs N texts)
        self._ensure_entity_cache()
        if self._entity_vecs is None or self._entity_vecs.shape[0] == 0:
            return [], {}

        try:
            q_vecs = embed_texts([query])
        except Exception as e:
            logger.warning(f"Graph retrieve query embedding failed: {e}")
            return [], {}

        q_vec = np.array(q_vecs[0], dtype=np.float32)
        # _entity_vecs is already L2-normalised; q_vec from embed_texts is also normalised
        sims = (self._entity_vecs @ q_vec).tolist()
        entity_keys = self._entity_keys
        ranked = sorted(zip(sims, entity_keys), reverse=True)

        # Top-3 seed nodes (with their similarity scores) — take top-3 unconditionally
        seed_pairs = [(sim, k) for sim, k in ranked[:3] if sim > 0.1]
        if not seed_pairs:
            return [], {"seeds": [], "nodes": [], "edges": [], "chunks_found": 0, "message": "No matching entities"}

        seeds = [k for _, k in seed_pairs]
        seed_info = [
            {"key": k, "display": self.graph.nodes[k].get("display_name", k), "sim": round(s, 3)} for s, k in seed_pairs
        ]

        personalization = {k: max(sim, 0.0) for sim, k in seed_pairs}
        if seed_chunk_ids:
            seed_cid_set = set(seed_chunk_ids)
            for key, nd in self.graph.nodes(data=True):
                if seed_cid_set.intersection(nd.get("chunk_ids", [])):
                    personalization[key] = personalization.get(key, 0.0) + 0.5

        try:
            # Edge weights increment on repeated relations, so PPR is weight-aware for free
            ppr = nx.pagerank(
                self.graph.to_undirected(),
                alpha=0.85,
                personalization=personalization,
                weight="weight",
            )
        except Exception as e:
            logger.warning(f"Personalized PageRank failed, falling back to BFS: {e}")
            return self._bfs_retrieve(seeds, seed_info, store, top_k)

        # Chunk score = sum of PPR mass over the entities that reference it
        chunk_scores: dict[str, float] = {}
        for key, mass in ppr.items():
            if mass <= 0.0:
                continue
            for cid in self.graph.nodes[key].get("chunk_ids", []):
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + mass

        traversal_info = self._ppr_traversal_info(seeds, seed_info, ppr)

        if not chunk_scores:
            return [], traversal_info

        max_score = max(chunk_scores.values())
        ids_to_fetch = [cid for cid, _ in sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[:50]]

        try:
            raw = store.collection.get(ids=ids_to_fetch, include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Graph chunk fetch failed: {e}")
            return [], traversal_info

        hits = []
        for cid, doc, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
            hits.append(
                {
                    "id": cid,
                    "content": doc,
                    "metadata": meta,
                    "score": round(chunk_scores[cid] / max_score, 4),
                    "search_type": "graph",
                }
            )
        hits.sort(key=lambda h: h["score"], reverse=True)

        traversal_info["chunks_found"] = len(hits)
        return hits[:top_k], traversal_info

    def _ppr_traversal_info(self, seeds: list[str], seed_info: list[dict], ppr: dict[str, float]) -> dict:
        """Derive the graph_path WS payload (nodes + edges) from the top-PPR nodes,
        preserving the shape the frontend GraphPathBadge renders."""
        seed_set = set(seeds)
        top_keys = [k for k, mass in sorted(ppr.items(), key=lambda kv: kv[1], reverse=True) if mass > 0.0][:20]
        for k in seeds:
            if k not in top_keys:
                top_keys.append(k)
        top_set = set(top_keys)

        # Hop distance from the nearest seed — the frontend labels edges "hop N"
        dist = {k: 0 for k in seeds}
        frontier = list(seeds)
        und = self.graph.to_undirected(as_view=True)
        for depth in range(1, 4):
            next_frontier = []
            for node in frontier:
                for nb in und.neighbors(node):
                    if nb not in dist:
                        dist[nb] = depth
                        next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break

        nodes = []
        for k in top_keys:
            nd = self.graph.nodes[k]
            nodes.append(
                {
                    "key": k,
                    "display": nd.get("display_name", k),
                    "type": nd.get("node_type", "concept"),
                    "is_seed": k in seed_set,
                    "degree": self.graph.degree(k),
                    "ppr": round(ppr.get(k, 0.0), 5),
                }
            )

        edges = []
        seen_edges = set()
        for src, tgt, data in self.graph.edges(data=True):
            if src not in top_set or tgt not in top_set:
                continue
            rel = data.get("rel_type", "related_to")
            src_disp = self.graph.nodes[src].get("display_name", src)
            tgt_disp = self.graph.nodes[tgt].get("display_name", tgt)
            ekey = (src_disp, tgt_disp, rel)
            if ekey in seen_edges:
                continue
            seen_edges.add(ekey)
            hop = min(dist.get(src, 3), dist.get(tgt, 3)) + 1
            edges.append({"from": src_disp, "to": tgt_disp, "rel": rel, "hop": hop})
        edges.sort(key=lambda e: e["hop"])

        return {"seeds": seed_info, "nodes": nodes, "edges": edges[:30], "chunks_found": 0}

    def _bfs_retrieve(self, seeds: list[str], seed_info: list[dict], store, top_k: int) -> tuple[list[dict], dict]:
        """Legacy 2-hop BFS traversal — fallback when personalized PageRank fails."""
        visited = set(seeds)
        frontier = list(seeds)
        traversal_edges = []  # {"from_display", "to_display", "rel", "direction"}

        for hop in range(2):
            next_frontier = []
            for node in frontier:
                nd = self.graph.nodes[node]
                for nb in self.graph.successors(node):
                    edge_data = self.graph[node][nb]
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                    traversal_edges.append(
                        {
                            "from": nd.get("display_name", node),
                            "to": self.graph.nodes[nb].get("display_name", nb),
                            "rel": edge_data.get("rel_type", "related_to"),
                            "hop": hop + 1,
                        }
                    )
                for nb in self.graph.predecessors(node):
                    edge_data = self.graph[nb][node]
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                    traversal_edges.append(
                        {
                            "from": self.graph.nodes[nb].get("display_name", nb),
                            "to": nd.get("display_name", node),
                            "rel": edge_data.get("rel_type", "related_to"),
                            "hop": hop + 1,
                        }
                    )
            frontier = next_frontier

        # Deduplicate edges (same from/to/rel)
        seen_edges = set()
        unique_edges = []
        for e in traversal_edges:
            key = (e["from"], e["to"], e["rel"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        # Visited node details (for display)
        visited_nodes = []
        for k in visited:
            nd = self.graph.nodes[k]
            visited_nodes.append(
                {
                    "key": k,
                    "display": nd.get("display_name", k),
                    "type": nd.get("node_type", "concept"),
                    "is_seed": k in seeds,
                    "degree": self.graph.degree(k),
                }
            )

        traversal_info = {
            "seeds": seed_info,
            "nodes": visited_nodes,
            "edges": unique_edges[:30],  # cap for payload size
            "chunks_found": 0,  # filled below
        }

        # Collect chunk_ids from all visited nodes
        chunk_ids_to_fetch = []
        for node_key in visited:
            chunk_ids_to_fetch.extend(self.graph.nodes[node_key].get("chunk_ids", []))

        chunk_ids_to_fetch = list(dict.fromkeys(chunk_ids_to_fetch))[:50]  # dedup, cap

        if not chunk_ids_to_fetch:
            return [], traversal_info

        # Fetch from ChromaDB
        try:
            raw = store.collection.get(ids=chunk_ids_to_fetch, include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Graph chunk fetch failed: {e}")
            return [], traversal_info

        hits = []
        for cid, doc, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
            hits.append(
                {
                    "id": cid,
                    "content": doc,
                    "metadata": meta,
                    "score": 0.7,
                    "search_type": "graph",
                }
            )

        traversal_info["chunks_found"] = len(hits)
        return hits[:top_k], traversal_info

    # ── visualization data ────────────────────────────────────────────────────

    def get_viz_data(self, max_nodes: int = 200) -> dict:
        """Return capped nodes + edges suitable for the frontend force layout."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": [], "edges": [], "stats": self.get_stats()}

        # Pick top nodes by degree centrality (most connected first)
        degree_map = dict(self.graph.degree())
        top_keys = sorted(degree_map, key=lambda k: degree_map[k], reverse=True)[:max_nodes]
        top_set = set(top_keys)

        # Assign a color index per unique doc (stable ordering)
        all_docs = []
        for k in top_keys:
            for dp in self.graph.nodes[k].get("doc_paths", []):
                if dp not in all_docs:
                    all_docs.append(dp)
        doc_color = {dp: i for i, dp in enumerate(all_docs)}

        nodes = []
        for k in top_keys:
            nd = self.graph.nodes[k]
            dp_list = nd.get("doc_paths", [])
            primary_doc = dp_list[0] if dp_list else ""
            nodes.append(
                {
                    "id": k,
                    "label": nd.get("display_name", k),
                    "type": nd.get("node_type", "concept"),
                    "doc": primary_doc,
                    "color_idx": doc_color.get(primary_doc, 0),
                    "degree": degree_map[k],
                    "chunk_count": len(nd.get("chunk_ids", [])),
                    "community": nd.get("community", -1),
                }
            )

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            if src in top_set and tgt in top_set:
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "rel": data.get("rel_type", "related_to"),
                        "weight": data.get("weight", 1),
                    }
                )

        return {"nodes": nodes, "edges": edges, "stats": self.get_stats()}

    def get_stats(self) -> dict:
        all_docs = set()
        communities = set()
        for _, nd in self.graph.nodes(data=True):
            all_docs.update(nd.get("doc_paths", []))
            cid = nd.get("community")
            if isinstance(cid, int):
                communities.add(cid)
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents": len(all_docs),
            "communities": len(communities),
            "built_at": self.built_at,
            "chunks_processed": self.total_chunks_processed,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        with self._lock:  # snapshot the graph consistently; RLock so locked callers can nest
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "built_at": self.built_at,
                "total_chunks_processed": self.total_chunks_processed,
                "community_summaries": self._community_summaries,
                "graph": nx.node_link_data(self.graph),
            }
            path.write_text(json.dumps(data, indent=2))
            self._path = path
            logger.info(f"Knowledge graph saved: {path} ({self.graph.number_of_nodes()} nodes)")

    def load(self, path: Path):
        try:
            data = json.loads(path.read_text())
            self.built_at = data.get("built_at")
            self.total_chunks_processed = data.get("total_chunks_processed", 0)
            self._community_summaries = data.get("community_summaries", {}) or {}
            self.graph = nx.node_link_graph(data["graph"], directed=True, multigraph=False)
            self._path = path
            logger.info(f"Knowledge graph loaded: {path} ({self.graph.number_of_nodes()} nodes)")
            self._build_entity_cache()  # warm the cache on load
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
            self.graph = nx.DiGraph()
            self._community_summaries = {}


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_extraction(text: str) -> dict:
    """Parse LLM response into {entities, relations}, robustly."""
    text = text.strip()
    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Try to decode first valid JSON object using the decoder (handles trailing text)
    decoder = json.JSONDecoder()
    # Find first '{' and try to decode from there
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    logger.warning("Could not parse extraction JSON from LLM response: %s", text[:200])
    return {"entities": [], "relations": []}


# ── per-user graph factory ────────────────────────────────────────────────────

_graph_cache: dict[str, KnowledgeGraph] = {}


def get_user_graph(uid: str) -> KnowledgeGraph:
    if uid not in _graph_cache:
        g = KnowledgeGraph()
        path = GRAPHS_DIR / f"{uid}.json"
        if path.exists():
            g.load(path)
        _graph_cache[uid] = g
    return _graph_cache[uid]


def save_user_graph(uid: str):
    if uid in _graph_cache:
        path = GRAPHS_DIR / f"{uid}.json"
        _graph_cache[uid].save(path)
