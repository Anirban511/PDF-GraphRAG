"""
graph_retriever.py — Hybrid GraphRAG retrieval.

WHY THIS EXISTS:
  This is what makes the system "GraphRAG" and not just "RAG with a graph
  sitting next to it." It fuses two retrieval strategies:

    1. VECTOR retrieval  — find chunks semantically similar to the query
                           (good at: paraphrase, fuzzy topic matching)
    2. GRAPH retrieval   — find the entities named in the query, then
                           traverse the knowledge graph to pull in chunks
                           about *connected* entities
                           (good at: multi-hop, relational questions)

  The union of both is deduplicated and handed to the reranker. The result:
  questions that vector search alone would miss ("what did the CEO of the
  company that acquired X say about margins?") now resolve, because the
  graph supplies the connecting chunks.

HOW QUERY ENTITIES ARE FOUND:
  A lightweight LLM call pulls candidate entity names out of the query, which
  are then fuzzy-matched against graph entities to find seed nodes. This is
  cheap (short prompt, deterministic) and avoids a heavy NER dependency.
"""

from __future__ import annotations
import json

from app.config import settings
from app.generation.llm import call_llm
from app.graph.neo4j_store import Neo4jStore
from app.retrieval.retriever import Retriever
from app.utils.logger import logger

_QUERY_ENTITY_SYSTEM = """Extract the named entities (companies, people, products,
metrics) from the user's question. Respond with ONLY a JSON array of strings.
Example: ["Acme Corp", "revenue"]. If none, return []."""


def _extract_query_entities(query: str) -> list[str]:
    try:
        raw = call_llm(system=_QUERY_ENTITY_SYSTEM, user=query,
                       max_tokens=128, temperature=0.0)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        names = json.loads(raw)
        return [n for n in names if isinstance(n, str) and n.strip()]
    except Exception as exc:
        logger.warning(f"Query entity extraction failed: {exc}")
        return []


class GraphRetriever:
    """Combines vector retrieval with Neo4j graph traversal."""

    def __init__(self, vector_retriever: Retriever, graph: Neo4jStore):
        self._vector = vector_retriever
        self._graph = graph

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.top_k

        # 1. Vector retrieval (semantic similarity)
        vector_hits = self._vector.retrieve(query, top_k=top_k)
        for h in vector_hits:
            h["retrieval_source"] = "vector"

        # 2. Graph retrieval (entity traversal)
        graph_hits = []
        query_entities = _extract_query_entities(query)
        if query_entities:
            seeds = self._graph.find_seed_entities(query_entities)
            if seeds:
                graph_hits = self._graph.expand_context(seeds)
                for h in graph_hits:
                    h["retrieval_source"] = "graph"
                    h["score"] = h.get("score", 0.5)  # graph hits lack cosine score
                logger.info(
                    f"GraphRAG: query entities {query_entities} → "
                    f"seeds {seeds} → {len(graph_hits)} graph chunks"
                )

        # 3. Fuse + deduplicate by chunk_id
        seen, fused = set(), []
        for hit in vector_hits + graph_hits:
            cid = hit.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                fused.append(hit)

        logger.debug(
            f"Hybrid retrieval: {len(vector_hits)} vector + "
            f"{len(graph_hits)} graph → {len(fused)} unique chunks"
        )
        return fused
