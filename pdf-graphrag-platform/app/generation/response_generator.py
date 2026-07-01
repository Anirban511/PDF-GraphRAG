"""
response_generator.py — Full RAG pipeline orchestrator.

WHY THIS EXISTS:
  This is the single place that knows the order of operations.  Each step
  is implemented in its own module; ResponseGenerator just wires them together.
  This separation means you can unit-test each stage in isolation and swap
  any step (e.g. a different reranker) without touching this file.

PIPELINE (per query):
  1. Guardrail  — block harmful / injected queries before any retrieval
  2. Retrieve   — embed query, ANN search, get top-K candidates
  3. Rerank     — cross-encoder re-scores candidates for precision
  4. Context    — format chunks into the LLM prompt context block
  5. Generate   — call Claude, get answer with inline citations
  6. Return     — RAGResponse dataclass with answer, citations, and metadata

RAGResponse:
  answer      — final LLM text (may contain inline [Source: …] references)
  citations   — structured list for UI display
  latency_s   — wall-clock time for the full pipeline (useful for monitoring)
  chunks_used — how many chunks were fed to the LLM (transparency)
  error       — non-empty string if the pipeline short-circuited

generate() vs stream():
  generate() — blocks until the full answer is ready; simpler to test.
  stream()   — yields text deltas for a typewriter UI effect, then yields
               the final RAGResponse so callers can access citations.
"""

from __future__ import annotations
from dataclasses import dataclass

from app.config import settings
from app.generation.guardrails import is_query_safe
from app.generation.llm import call_llm, stream_llm
from app.ingestion.vector_store import VectorStore
from app.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from app.retrieval.citation_builder import build_context, extract_citations
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.utils.helpers import Timer
from app.utils.logger import logger


@dataclass
class RAGResponse:
    answer:      str
    citations:   list[dict]
    query:       str
    latency_s:   float
    chunks_used: int
    error:       str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class ResponseGenerator:
    """
    Instantiate once per process; call .generate() or .stream() per query.

    If a GraphRetriever is supplied, retrieval becomes hybrid GraphRAG
    (vector similarity + Neo4j graph traversal). If not, it falls back to
    pure vector retrieval — so the system runs with or without Neo4j.
    """

    def __init__(self, store: VectorStore | None = None, graph_retriever=None):
        self._store     = store or VectorStore()
        self._retriever = Retriever(self._store)
        self._reranker  = Reranker()
        self._graph_retriever = graph_retriever   # optional GraphRetriever

    # ── Shared pipeline steps ─────────────────────────────────────────

    def _prepare(self, query: str) -> tuple[list[dict], str, list[dict]] | RAGResponse:
        """
        Run guardrail → retrieve (hybrid if graph available) → rerank → context.
        Returns (top_chunks, context_str, citations) on success,
        or a RAGResponse sentinel on early exit.
        """
        safe, reason = is_query_safe(query)
        if not safe:
            return RAGResponse(
                answer=f"I can't help with that request. ({reason})",
                citations=[], query=query, latency_s=0.0, chunks_used=0,
                error="guardrail",
            )

        try:
            if self._graph_retriever is not None:
                candidates = self._graph_retriever.retrieve(query, top_k=settings.top_k)
            else:
                candidates = self._retriever.retrieve(query, top_k=settings.top_k)
        except FileNotFoundError as exc:
            return RAGResponse(
                answer=str(exc), citations=[], query=query,
                latency_s=0.0, chunks_used=0, error="no_index",
            )

        if not candidates:
            return RAGResponse(
                answer="No relevant content found in the uploaded documents.",
                citations=[], query=query, latency_s=0.0, chunks_used=0,
            )

        top_chunks = self._reranker.rerank(query, candidates)
        context    = build_context(top_chunks)
        citations  = extract_citations(top_chunks)
        return top_chunks, context, citations

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, query: str) -> RAGResponse:
        """Blocking RAG call — returns a complete RAGResponse."""
        with Timer() as t:
            result = self._prepare(query)
            if isinstance(result, RAGResponse):
                return result
            top_chunks, context, citations = result
            answer = call_llm(
                system=RAG_SYSTEM_PROMPT,
                user=RAG_USER_TEMPLATE.format(context=context, question=query),
            )

        logger.info(f"Query answered in {t} — {len(top_chunks)} chunks")
        return RAGResponse(
            answer=answer, citations=citations, query=query,
            latency_s=t.elapsed, chunks_used=len(top_chunks),
        )

    def stream(self, query: str):
        """
        Generator: yields str deltas, then a final RAGResponse object.
        Callers should check isinstance(item, RAGResponse) for the sentinel.
        """
        result = self._prepare(query)
        if isinstance(result, RAGResponse):
            yield result.answer
            return

        top_chunks, context, citations = result
        full_text = ""
        for delta in stream_llm(
            system=RAG_SYSTEM_PROMPT,
            user=RAG_USER_TEMPLATE.format(context=context, question=query),
        ):
            full_text += delta
            yield delta

        yield RAGResponse(
            answer=full_text, citations=citations, query=query,
            latency_s=0.0, chunks_used=len(top_chunks),
        )
