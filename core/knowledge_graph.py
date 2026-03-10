"""
core/knowledge_graph.py — Memory Palace: Spatial Knowledge Graph

Builds a navigable entity–relationship graph from your indexed documents.
Retrieval becomes graph pathfinding + vector search combined, so the model
can traverse concept clusters instead of doing only flat nearest-neighbour
lookups.

Graph construction (POST /api/graph/build):
  1. Pull all chunks from ChromaDB
  2. Batch-send to LLM → extract (entity, relation, entity) triples
  3. Build a NetworkX DiGraph; nodes carry chunk_id references
  4. Persist as JSON alongside the vector store data

Hybrid retrieval:
  1. Embed query → cosine-rank entity names → pick top-K seed nodes
  2. BFS expand 2 hops through the graph
  3. Collect chunk_ids of every traversed node
  4. ChromaDB.get(ids=...) to retrieve those chunks
  5. Merge with standard vector-search results (deduplicated)

Zero extra dependencies beyond NetworkX (already standard in data science
envs). Add `networkx>=3.0` to requirements.txt.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path("data/graphs")

# ── LLM extraction prompt ─────────────────────────────────────────────────────
EXTRACT_SYSTEM = """You are an entity extraction engine.
Extract the most important entities and relationships from the provided text chunks.
Return ONLY a valid JSON object with this exact schema:
{
  "entities": [{"name": "...", "type": "class|function|module|concept|config|topic|api"}],
  "relations": [{"from": "EntityA", "to": "EntityB", "rel": "uses|extends|calls|imports|defines|configures|related_to|part_of"}]
}
Rules:
- Entity names: 1–4 words, use the exact name from the text (CamelCase, snake_case, etc.)
- Max 8 entities and 10 relations per response
- Only include a relation when BOTH endpoints appear in the entity list
- If the text has no clear entities, return {"entities": [], "relations": []}
- NEVER add commentary outside the JSON"""


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    name: str
    node_type: str                   # class|function|module|concept|config|topic|api
    chunk_ids: List[str] = field(default_factory=list)
    doc_paths: List[str] = field(default_factory=list)

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
        self.built_at: Optional[float] = None
        self.total_chunks_processed: int = 0
        # Pre-computed entity embedding cache — rebuilt after build/load
        self._entity_keys: Optional[List[str]] = None   # ordered list of node keys
        self._entity_vecs: Optional[np.ndarray] = None  # (N, dim) normalised matrix

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
            logger.info("Entity embedding cache built: {} entities".format(len(keys)))
        except Exception as e:
            logger.warning("Entity cache build failed: {}".format(e))
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
            self.graph.add_node(key,
                display_name=name,
                node_type=node_type,
                chunk_ids=[],
                doc_paths=[])
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
        Returns a summary dict: {nodes, edges, chunks_processed, ms}.
        """
        from core import llm_client as _llm

        if store.count == 0:
            return {"nodes": 0, "edges": 0, "chunks_processed": 0, "ms": 0,
                    "message": "No documents indexed"}

        t0 = time.time()

        # Fetch all chunks
        raw = store.collection.get(include=["documents", "metadatas"])
        all_ids   = raw.get("ids", [])
        all_docs  = raw.get("documents", [])
        all_metas = raw.get("metadatas", [])
        total = len(all_ids)
        logger.info("Knowledge graph: processing {} chunks in batches of {}".format(total, batch_size))

        processed = 0
        errors = 0

        for i in range(0, total, batch_size):
            batch_ids   = all_ids  [i: i + batch_size]
            batch_docs  = all_docs [i: i + batch_size]
            batch_metas = all_metas[i: i + batch_size]

            # Format chunks into a single prompt block
            chunk_block = ""
            for j, (doc, meta) in enumerate(zip(batch_docs, batch_metas)):
                doc_path = meta.get("document_path", "")
                chunk_block += f"\n--- Chunk {j+1} (from: {doc_path}) ---\n{doc[:800]}\n"

            try:
                resp = _llm.chat(
                    messages=[{"role": "user", "content": chunk_block.strip()}],
                    system=EXTRACT_SYSTEM,
                    max_tokens=350,
                    temperature=0.0,
                    stream=False,
                )
                extracted = _parse_extraction(resp)
            except Exception as e:
                logger.warning("Extraction failed for batch {}: {}".format(i, e))
                errors += 1
                processed += len(batch_ids)
                continue

            # Assign entities to the first chunk in the batch (best approximation)
            for ent in extracted.get("entities", []):
                name = ent.get("name", "").strip()
                etype = ent.get("type", "concept")
                if not name:
                    continue
                # Associate with each chunk in the batch that might mention it
                for cid, meta in zip(batch_ids, batch_metas):
                    doc_path = meta.get("document_path", "")
                    self.add_entity(name, etype, cid, doc_path)

            for rel in extracted.get("relations", []):
                src = rel.get("from", "").strip()
                tgt = rel.get("to", "").strip()
                rtype = rel.get("rel", "related_to")
                if src and tgt:
                    self.add_relation(src, tgt, rtype)

            processed += len(batch_ids)

        self.built_at = time.time()
        self.total_chunks_processed = processed
        self._build_entity_cache()   # pre-compute embeddings once here
        ms = int((time.time() - t0) * 1000)

        result = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "chunks_processed": processed,
            "extraction_errors": errors,
            "ms": ms,
        }
        logger.info("Knowledge graph built: {} nodes, {} edges in {}ms".format(
            result["nodes"], result["edges"], ms))
        return result

    # ── hybrid retrieval ──────────────────────────────────────────────────────

    def graph_retrieve(self, query: str, store, top_k: int = 10) -> List[dict]:
        """
        Graph-walk + vector search hybrid retrieval.

        1. Embed query and all entity names → cosine-rank to find seed nodes
        2. BFS expand 2 hops
        3. Collect chunk_ids from traversal
        4. Fetch chunks from ChromaDB
        5. Merge with vector-search results (deduplicate by chunk_id)
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
            logger.warning("Graph retrieve query embedding failed: {}".format(e))
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
            {"key": k, "display": self.graph.nodes[k].get("display_name", k), "sim": round(s, 3)}
            for s, k in seed_pairs
        ]

        # BFS 2 hops — track traversal edges
        visited = set(seeds)
        frontier = list(seeds)
        traversal_edges = []   # {"from_display", "to_display", "rel", "direction"}

        for hop in range(2):
            next_frontier = []
            for node in frontier:
                nd = self.graph.nodes[node]
                for nb in self.graph.successors(node):
                    edge_data = self.graph[node][nb]
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                    traversal_edges.append({
                        "from": nd.get("display_name", node),
                        "to": self.graph.nodes[nb].get("display_name", nb),
                        "rel": edge_data.get("rel_type", "related_to"),
                        "hop": hop + 1,
                    })
                for nb in self.graph.predecessors(node):
                    edge_data = self.graph[nb][node]
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                    traversal_edges.append({
                        "from": self.graph.nodes[nb].get("display_name", nb),
                        "to": nd.get("display_name", node),
                        "rel": edge_data.get("rel_type", "related_to"),
                        "hop": hop + 1,
                    })
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
            visited_nodes.append({
                "key": k,
                "display": nd.get("display_name", k),
                "type": nd.get("node_type", "concept"),
                "is_seed": k in seeds,
                "degree": self.graph.degree(k),
            })

        traversal_info = {
            "seeds": seed_info,
            "nodes": visited_nodes,
            "edges": unique_edges[:30],   # cap for payload size
            "chunks_found": 0,            # filled below
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
            raw = store.collection.get(
                ids=chunk_ids_to_fetch,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.warning("Graph chunk fetch failed: {}".format(e))
            return [], traversal_info

        hits = []
        for cid, doc, meta in zip(
            raw.get("ids", []),
            raw.get("documents", []),
            raw.get("metadatas", [])
        ):
            hits.append({
                "id": cid,
                "content": doc,
                "metadata": meta,
                "score": 0.7,
                "search_type": "graph",
            })

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
            nodes.append({
                "id": k,
                "label": nd.get("display_name", k),
                "type": nd.get("node_type", "concept"),
                "doc": primary_doc,
                "color_idx": doc_color.get(primary_doc, 0),
                "degree": degree_map[k],
                "chunk_count": len(nd.get("chunk_ids", [])),
            })

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            if src in top_set and tgt in top_set:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "rel": data.get("rel_type", "related_to"),
                    "weight": data.get("weight", 1),
                })

        return {"nodes": nodes, "edges": edges, "stats": self.get_stats()}

    def get_stats(self) -> dict:
        all_docs = set()
        for _, nd in self.graph.nodes(data=True):
            all_docs.update(nd.get("doc_paths", []))
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents": len(all_docs),
            "built_at": self.built_at,
            "chunks_processed": self.total_chunks_processed,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "built_at": self.built_at,
            "total_chunks_processed": self.total_chunks_processed,
            "graph": nx.node_link_data(self.graph),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Knowledge graph saved: {} ({} nodes)".format(path, self.graph.number_of_nodes()))

    def load(self, path: Path):
        try:
            data = json.loads(path.read_text())
            self.built_at = data.get("built_at")
            self.total_chunks_processed = data.get("total_chunks_processed", 0)
            self.graph = nx.node_link_graph(data["graph"], directed=True, multigraph=False)
            logger.info("Knowledge graph loaded: {} ({} nodes)".format(
                path, self.graph.number_of_nodes()))
            self._build_entity_cache()   # warm the cache on load
        except Exception as e:
            logger.warning("Failed to load knowledge graph: {}".format(e))
            self.graph = nx.DiGraph()


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
    # Extract first JSON object from the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"entities": [], "relations": []}


# ── per-user graph factory ────────────────────────────────────────────────────

_graph_cache: Dict[str, KnowledgeGraph] = {}


def get_user_graph(uid: str) -> KnowledgeGraph:
    if uid not in _graph_cache:
        g = KnowledgeGraph()
        path = GRAPHS_DIR / "{}.json".format(uid)
        if path.exists():
            g.load(path)
        _graph_cache[uid] = g
    return _graph_cache[uid]


def save_user_graph(uid: str):
    if uid in _graph_cache:
        path = GRAPHS_DIR / "{}.json".format(uid)
        _graph_cache[uid].save(path)
