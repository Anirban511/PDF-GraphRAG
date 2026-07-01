"""
ingestion_service.py — Ingestion business logic, decoupled from HTTP.

WHY THIS EXISTS (SDE pattern):
  In the previous version, all ingestion logic lived inside the FastAPI route
  handler. That mixes transport (HTTP) with business logic, making it hard to
  test and impossible to reuse from a background worker.

  Here the logic lives in a service function that takes plain arguments and
  returns plain data. It is called from TWO places:
    • the background job worker (async path)
    • a synchronous fallback (when Redis/queue is unavailable)

  This separation (route -> service -> stores) is the standard layered
  architecture and the thing that makes the async refactor clean.

CACHING:
  Before doing any expensive work, we check whether this exact document hash
  was already ingested. If so, we skip the whole pipeline — the single
  biggest cost saving when hosted LLM APIs are used.
"""

from __future__ import annotations
from pathlib import Path

from app.config import settings
from app.core.cache import cache_exists_doc, mark_doc_ingested
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.loader import load_pdf
from app.ingestion.vector_store import VectorStore
from app.utils.helpers import file_hash
from app.utils.logger import logger

# Shared singletons (set by the app on startup)
_store: VectorStore | None = None
_graph = None


def configure(store: VectorStore, graph=None):
    global _store, _graph
    _store = store
    _graph = graph


def ingest_document(pdf_path: str, filename: str) -> dict:
    """
    Full ingestion pipeline for one PDF. Returns a summary dict.
    Safe to call from a worker thread or synchronously.
    """
    path = Path(pdf_path)
    doc_hash = file_hash(path)

    # Cache check — skip everything if already ingested
    if cache_exists_doc(doc_hash):
        logger.info(f"Document {filename} already ingested (cache hit) — skipping.")
        return {"status": "cached", "filename": filename, "doc_id": doc_hash,
                "chunks": 0, "graph": {"entities": 0, "relationships": 0}}

    # 1. Extract + chunk
    pages = load_pdf(path)
    chunks = chunk_pages(pages)

    # 2. Embed + index
    vectors = embed_chunks(chunks)
    if _store.is_ready:
        _store.load()
        _store.add(chunks, vectors)
    else:
        _store.build(chunks, vectors)

    # 3. Knowledge graph (optional)
    graph_info = {"entities": 0, "relationships": 0}
    if _graph is not None:
        from app.graph.entity_extractor import extract_from_chunks
        _graph.write_chunks(chunks)
        frag = extract_from_chunks(chunks)
        lookup = {(c.doc_id, c.page_num): c.chunk_id for c in chunks}
        _graph.write_graph(frag, lookup)
        graph_info = {"entities": len(frag.entities),
                      "relationships": len(frag.relationships)}

    summary = {
        "status": "ok", "filename": filename, "doc_id": doc_hash,
        "pages": len(pages), "chunks": len(chunks), "graph": graph_info,
    }
    mark_doc_ingested(doc_hash, summary)
    logger.success(f"Ingested {filename}: {len(chunks)} chunks, "
                   f"{graph_info['entities']} entities")
    return summary
