"""
neo4j_store.py — Neo4j knowledge-graph storage and traversal.

WHY THIS EXISTS:
  This is the heart of the GraphRAG upgrade. It does three jobs:
    1. Writes extracted entities/relationships into Neo4j as nodes/edges.
    2. Traverses the graph at query time to find *connected* context
       (the "graph hops" that plain vector search cannot do).
    3. Exposes the graph for visualization (nodes + edges as JSON).

GRAPH SCHEMA:
    (:Entity {name, type})           — a company, person, metric, money, etc.
    (:Chunk  {chunk_id, text, page, filename, doc_id})
    (:Document {doc_id, filename})

    (Entity)-[:MENTIONED_IN]->(Chunk)         — provenance for citations
    (Chunk)-[:PART_OF]->(Document)
    (Entity)-[:REL {type, page, filename}]->(Entity)   — the extracted facts

WHY MERGE (not CREATE):
  MERGE is idempotent — re-ingesting the same document does not duplicate
  nodes. Entity names are the merge key, so "Acme Corp" mentioned on five
  pages becomes one node connected to five chunks.

WHY GraphRAG beats vector-only retrieval here:
  Vector search finds chunks similar to the *query text*. Graph traversal
  finds chunks connected to the *entities in the query* — even if those
  chunks use completely different wording. For relational financial
  questions ("what did X acquire and for how much"), the answer lives in
  the edges, not in text similarity.
"""

from __future__ import annotations
from neo4j import GraphDatabase

from app.config import settings
from app.graph.entity_extractor import GraphFragment
from app.ingestion.chunker import Chunk
from app.utils.logger import logger


class Neo4jStore:
    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._db = settings.neo4j_database

    def close(self):
        self._driver.close()

    # ── Schema setup ──────────────────────────────────────────────────

    def init_schema(self):
        """Create uniqueness constraints + indexes (idempotent)."""
        stmts = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        ]
        with self._driver.session(database=self._db) as s:
            for stmt in stmts:
                s.run(stmt)
        logger.info("Neo4j schema initialised.")

    # ── Writing ───────────────────────────────────────────────────────

    def write_chunks(self, chunks: list[Chunk]):
        """Create Document + Chunk nodes and link them."""
        with self._driver.session(database=self._db) as s:
            s.run(
                """
                UNWIND $rows AS row
                MERGE (d:Document {doc_id: row.doc_id})
                  SET d.filename = row.filename
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                  SET c.text = row.text, c.page = row.page,
                      c.filename = row.filename, c.doc_id = row.doc_id
                MERGE (c)-[:PART_OF]->(d)
                """,
                rows=[{
                    "doc_id": c.doc_id, "filename": c.filename,
                    "chunk_id": c.chunk_id, "text": c.text, "page": c.page_num,
                } for c in chunks],
            )
        logger.info(f"Wrote {len(chunks)} Chunk nodes to Neo4j.")

    def write_graph(self, fragment: GraphFragment, chunk_lookup: dict[str, str]):
        """
        Write entities + relationships.
        chunk_lookup maps (doc_id, page) -> a representative chunk_id so each
        entity can be linked to the chunk it was mentioned in.
        """
        with self._driver.session(database=self._db) as s:
            # Entities + MENTIONED_IN links
            s.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {name: row.name})
                  SET e.type = row.type
                WITH e, row
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                rows=[{
                    "name": e.name, "type": e.type,
                    "chunk_id": chunk_lookup.get((e.doc_id, e.page_num), ""),
                } for e in fragment.entities
                  if chunk_lookup.get((e.doc_id, e.page_num))],
            )
            # Relationships (entity -> entity)
            s.run(
                """
                UNWIND $rows AS row
                MERGE (a:Entity {name: row.source})
                MERGE (b:Entity {name: row.target})
                MERGE (a)-[r:REL {type: row.type}]->(b)
                  SET r.page = row.page, r.filename = row.filename
                """,
                rows=[{
                    "source": r.source, "target": r.target, "type": r.type,
                    "page": r.page_num, "filename": r.filename,
                } for r in fragment.relationships],
            )
        logger.success(
            f"Wrote {len(fragment.entities)} entities + "
            f"{len(fragment.relationships)} relationships to Neo4j."
        )

    # ── GraphRAG traversal ────────────────────────────────────────────

    def find_seed_entities(self, names: list[str]) -> list[str]:
        """Fuzzy-match query entity names against graph entities."""
        with self._driver.session(database=self._db) as s:
            result = s.run(
                """
                UNWIND $names AS qname
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(qname)
                   OR toLower(qname) CONTAINS toLower(e.name)
                RETURN DISTINCT e.name AS name
                LIMIT 10
                """,
                names=names,
            )
            return [r["name"] for r in result]

    def expand_context(self, seed_names: list[str], hops: int | None = None,
                       max_chunks: int | None = None) -> list[dict]:
        """
        From seed entities, traverse `hops` relationships and collect the
        chunks where the connected entities are mentioned. This is the
        GraphRAG retrieval step.
        """
        hops = hops or settings.graph_hops
        max_chunks = max_chunks or settings.graph_max_chunks
        with self._driver.session(database=self._db) as s:
            result = s.run(
                f"""
                MATCH (seed:Entity)
                WHERE seed.name IN $seeds
                MATCH path = (seed)-[:REL*1..{hops}]-(connected:Entity)
                MATCH (connected)-[:MENTIONED_IN]->(c:Chunk)
                RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text,
                       c.page AS page_num, c.filename AS filename,
                       c.doc_id AS doc_id
                LIMIT $limit
                """,
                seeds=seed_names, limit=max_chunks,
            )
            return [dict(r) for r in result]

    # ── Visualization + analytics support ─────────────────────────────

    def get_subgraph(self, limit: int = 100) -> dict:
        """Return nodes + edges for visualization (e.g. in Streamlit)."""
        with self._driver.session(database=self._db) as s:
            nodes = s.run(
                "MATCH (e:Entity) RETURN e.name AS id, e.type AS type LIMIT $limit",
                limit=limit,
            ).data()
            edges = s.run(
                """
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                RETURN a.name AS source, b.name AS target, r.type AS type
                LIMIT $limit
                """,
                limit=limit,
            ).data()
        return {"nodes": nodes, "edges": edges}

    def entity_stats(self) -> dict:
        """Aggregate counts used by the analytics layer."""
        with self._driver.session(database=self._db) as s:
            by_type = s.run(
                "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count "
                "ORDER BY count DESC"
            ).data()
            top_connected = s.run(
                """
                MATCH (e:Entity)-[r:REL]-()
                RETURN e.name AS name, e.type AS type, count(r) AS degree
                ORDER BY degree DESC LIMIT 10
                """
            ).data()
            totals = s.run(
                "MATCH (e:Entity) WITH count(e) AS ents "
                "MATCH ()-[r:REL]->() RETURN ents, count(r) AS rels"
            ).single()
        return {
            "by_type": by_type,
            "top_connected": top_connected,
            "total_entities": totals["ents"] if totals else 0,
            "total_relationships": totals["rels"] if totals else 0,
        }

    def wipe(self):
        """Delete all nodes/edges (for clean re-ingest)."""
        with self._driver.session(database=self._db) as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.warning("Neo4j graph wiped.")
