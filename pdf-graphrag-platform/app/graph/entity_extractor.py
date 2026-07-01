"""
entity_extractor.py — Pull entities & relationships out of chunks.

WHY THIS EXISTS:
  Plain RAG retrieves chunks that are *semantically similar* to a query.
  But financial questions are often *relational*: "Which subsidiaries did
  Company X acquire, and what did each cost?" The answer is scattered across
  chunks that may not all be similar to the query wording. A knowledge graph
  captures these relationships explicitly so retrieval can *traverse* them.

WHAT IT DOES:
  For each chunk, the local LLM extracts a small set of typed entities
  (ORG, PERSON, MONEY, METRIC, DATE, PRODUCT) and the relationships between
  them (e.g. ACQUIRED, REPORTED, OWNS, PARTNERED_WITH). These become nodes
  and edges in Neo4j.

DESIGN NOTES:
  • Output is strict JSON so it can be parsed deterministically.
  • Every entity/relationship keeps a back-reference to its source chunk
    (doc_id + page) so graph answers remain citable — the same grounding
    guarantee as vector RAG.
  • Extraction is best-effort: a malformed JSON response for one chunk is
    logged and skipped, never crashing the ingest.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field

from app.generation.llm import call_llm
from app.ingestion.chunker import Chunk
from app.utils.logger import logger

# Entity types we care about for financial / business documents
ENTITY_TYPES = ["ORG", "PERSON", "MONEY", "METRIC", "DATE", "PRODUCT", "LOCATION"]

EXTRACTION_SYSTEM = """You extract a knowledge graph from business/financial text.
Identify entities and the relationships between them.

Entity types: ORG, PERSON, MONEY, METRIC, DATE, PRODUCT, LOCATION
Relationship examples: ACQUIRED, OWNS, REPORTED, PARTNERED_WITH, INVESTED_IN,
  COMPETES_WITH, LED_BY, LOCATED_IN, INCREASED, DECREASED

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "entities": [{"name": "Acme Corp", "type": "ORG"}, ...],
  "relationships": [{"source": "Acme Corp", "target": "$2.5M", "type": "REPORTED"}, ...]
}
If nothing relevant is found, return {"entities": [], "relationships": []}."""

EXTRACTION_USER = """Extract the knowledge graph from this text:

{text}"""


@dataclass
class Entity:
    name: str
    type: str
    doc_id: str
    filename: str
    page_num: int


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    doc_id: str
    filename: str
    page_num: int


@dataclass
class GraphFragment:
    """Entities + relationships extracted from a single chunk."""
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


def _normalise(name: str) -> str:
    """Canonicalise an entity name so 'Acme Corp.' and 'Acme Corp' merge."""
    return name.strip().rstrip(".").strip()


def extract_from_chunk(chunk: Chunk) -> GraphFragment:
    """Run LLM extraction on one chunk. Returns a GraphFragment (may be empty)."""
    try:
        raw = call_llm(
            system=EXTRACTION_SYSTEM,
            user=EXTRACTION_USER.format(text=chunk.text),
            max_tokens=1024,
            temperature=0.0,   # deterministic extraction
        )
        # The model sometimes wraps JSON in ```; strip fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as exc:
        logger.warning(f"Extraction failed on {chunk.chunk_id}: {exc}")
        return GraphFragment()

    ents = [
        Entity(
            name=_normalise(e["name"]), type=e.get("type", "UNKNOWN"),
            doc_id=chunk.doc_id, filename=chunk.filename, page_num=chunk.page_num,
        )
        for e in data.get("entities", [])
        if e.get("name")
    ]
    rels = [
        Relationship(
            source=_normalise(r["source"]), target=_normalise(r["target"]),
            type=r.get("type", "RELATED_TO"),
            doc_id=chunk.doc_id, filename=chunk.filename, page_num=chunk.page_num,
        )
        for r in data.get("relationships", [])
        if r.get("source") and r.get("target")
    ]
    return GraphFragment(entities=ents, relationships=rels)


def extract_from_chunks(chunks: list[Chunk]) -> GraphFragment:
    """Extract and merge graph fragments from many chunks."""
    from tqdm import tqdm
    from app.config import settings
    if settings.max_extraction_chunks and len(chunks) > settings.max_extraction_chunks:
        logger.warning(
            f"Capping entity extraction at {settings.max_extraction_chunks} "
            f"of {len(chunks)} chunks (set MAX_EXTRACTION_CHUNKS=0 to disable)."
        )
        chunks = chunks[: settings.max_extraction_chunks]
    merged = GraphFragment()
    for chunk in tqdm(chunks, desc="Extracting entities", unit="chunk"):
        frag = extract_from_chunk(chunk)
        merged.entities.extend(frag.entities)
        merged.relationships.extend(frag.relationships)
    logger.success(
        f"Extracted {len(merged.entities)} entities, "
        f"{len(merged.relationships)} relationships from {len(chunks)} chunks"
    )
    return merged
