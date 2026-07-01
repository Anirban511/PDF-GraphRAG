"""
retriever.py — Retrieval Stage 1: query → candidate chunks.

WHY THIS EXISTS:
  Keeps the "embed query + search FAISS" logic in one place.
  The Retriever owns no state beyond a reference to the shared VectorStore,
  so it is cheap to instantiate and trivial to swap in tests.

WHAT IT DOES:
  1. Calls embed_query() to turn the user string into a float32 vector
     using the same model used at ingest time (critical: mismatched models
     produce meaningless similarity scores).
  2. Passes the vector to VectorStore.search(), which runs an exact inner-
     product search and returns the top-K chunks with cosine similarity scores.

TOP-K SIZING:
  We deliberately retrieve more candidates here (default 5) than the
  final number shown to the LLM (default 3 after reranking).  The excess
  gives the cross-encoder reranker room to promote better results that
  bi-encoder retrieval may have ranked lower.
"""

from __future__ import annotations
from app.config import settings
from app.ingestion.embedder import embed_query
from app.ingestion.vector_store import VectorStore
from app.utils.logger import logger


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self._store = store or VectorStore()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Embed *query* and return top-k chunks from the vector store."""
        top_k = top_k or settings.top_k
        results = self._store.search(embed_query(query), top_k=top_k)
        logger.debug(f"Retrieved {len(results)} candidates for: '{query[:80]}'")
        return results
