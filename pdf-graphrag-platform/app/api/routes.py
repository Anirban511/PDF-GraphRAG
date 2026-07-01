"""
routes.py — REST API (SDE/production edition).

KEY DIFFERENCES FROM THE PREVIOUS VERSION:
  • Async ingestion: POST /ingest enqueues a job and returns 202 + job_id
    instead of blocking for minutes. New GET /jobs/{id} polls status.
  • Caching: repeat uploads (same hash) and repeat queries return instantly.
  • Rate limiting: every expensive endpoint is rate-limited per client.
  • Auth: optional API-key dependency on protected endpoints.
  • Service layer: route handlers are thin; logic lives in services/.

ENDPOINT MAP:
  POST /ingest          enqueue ingestion job        -> 202 {job_id}
  GET  /jobs/{id}       poll job status              -> {status, result}
  POST /query           GraphRAG Q&A (cached)        -> {answer, citations}
  POST /query/stream    streaming answer
  POST /report          analytics report (Word)
  GET  /graph           graph nodes + edges
  GET  /graph/stats     graph statistics
  GET  /status          system + dependency health
"""

from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

from fastapi import (APIRouter, Depends, File, HTTPException, Request,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.cache import cache_get, cache_set
from app.core.job_queue import enqueue_job, get_job
from app.core.rate_limiter import RateLimitExceeded, check_rate_limit
from app.core.security import client_id, require_api_key
from app.generation.response_generator import ResponseGenerator
from app.ingestion.vector_store import VectorStore
from app.services import ingestion_service
from app.utils.logger import logger

router = APIRouter()

# ── Shared singletons ──
_store = VectorStore()
_graph = None
_graph_retriever = None
try:
    from app.graph.neo4j_store import Neo4jStore
    from app.graph.graph_retriever import GraphRetriever
    from app.retrieval.retriever import Retriever
    _graph = Neo4jStore()
    _graph.init_schema()
    _graph_retriever = GraphRetriever(Retriever(_store), _graph)
    logger.success("Neo4j connected — GraphRAG enabled.")
except Exception as exc:
    logger.warning(f"Neo4j unavailable ({exc}); vector-only mode.")

_generator = ResponseGenerator(store=_store, graph_retriever=_graph_retriever)
ingestion_service.configure(_store, _graph)


# ── Schemas ──
class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.top_k

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    latency_s: float
    chunks_used: int
    cached: bool = False


# ── Rate-limit helper ──
def _enforce_limit(request: Request):
    cid = client_id(request)
    try:
        return check_rate_limit(cid)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": str(exc.retry_after)})


# ── Ingestion (async) ──
@router.post("/ingest", status_code=202, summary="Enqueue PDF ingestion (async)")
async def ingest_pdf(request: Request, file: UploadFile = File(...),
                     _: str | None = Depends(require_api_key)):
    _enforce_limit(request)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Persist upload to a stable path the worker can read
    dest = settings.raw_pdfs_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    payload = {"pdf_path": str(dest), "filename": file.filename}

    if settings.async_ingestion:
        job_id = enqueue_job("ingest", payload)
        if job_id:
            return JSONResponse(status_code=202,
                                content={"job_id": job_id, "status": "queued",
                                         "poll": f"/api/v1/jobs/{job_id}"})

    # Fallback: no Redis/queue — run synchronously
    result = ingestion_service.ingest_document(**payload)
    return JSONResponse(status_code=200, content=result)


@router.get("/jobs/{job_id}", summary="Poll ingestion job status")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found (or Redis unavailable).")
    import json
    out = {"job_id": job_id, "status": job.get("status"),
           "progress": job.get("progress")}
    if job.get("status") == "completed" and job.get("result"):
        out["result"] = json.loads(job["result"])
    if job.get("status") == "failed":
        out["error"] = job.get("error")
    return out


# ── Query (cached) ──
@router.post("/query", response_model=QueryResponse, summary="GraphRAG Q&A")
def query(request: Request, req: QueryRequest,
          _: str | None = Depends(require_api_key)):
    _enforce_limit(request)
    if not _store.is_ready:
        raise HTTPException(400, "No documents indexed yet.")

    # Cache by query text (+ index size to invalidate when new docs added)
    cache_key = f"{req.query}|{_store.stats().get('total_chunks', 0)}"
    if cached := cache_get("query", cache_key):
        cached["cached"] = True
        return QueryResponse(**cached)

    resp = _generator.generate(req.query)
    if resp.error == "no_index":
        raise HTTPException(400, resp.answer)
    out = {"answer": resp.answer, "citations": resp.citations,
           "latency_s": resp.latency_s, "chunks_used": resp.chunks_used,
           "cached": False}
    cache_set("query", cache_key, out)
    return QueryResponse(**out)


@router.post("/query/stream", summary="Streaming GraphRAG answer")
def query_stream(request: Request, req: QueryRequest,
                 _: str | None = Depends(require_api_key)):
    _enforce_limit(request)
    if not _store.is_ready:
        raise HTTPException(400, "No documents indexed yet.")

    def _gen():
        for item in _generator.stream(req.query):
            if isinstance(item, str):
                yield item
    return StreamingResponse(_gen(), media_type="text/plain")


# ── Analytics report (demoted to one capability, not the headline) ──
@router.post("/report", summary="Generate analytics report (Word)")
def report(request: Request, _: str | None = Depends(require_api_key)):
    _enforce_limit(request)
    if _graph is None:
        raise HTTPException(503, "Neo4j required for analytics reporting.")
    pdfs = sorted(settings.raw_pdfs_dir.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(400, "Ingest a document first.")
    from app.ingestion.loader import load_all_pdfs
    from app.ingestion.chunker import chunk_pages
    from app.analytics.analytics_pipeline import run_analytics
    chunks = chunk_pages(load_all_pdfs())
    report_path = run_analytics(chunks, _graph)
    return FileResponse(
        str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=report_path.name)


# ── Graph ──
@router.get("/graph", summary="Knowledge graph nodes + edges")
def get_graph(limit: int = 100):
    if _graph is None:
        raise HTTPException(503, "Neo4j not available.")
    return _graph.get_subgraph(limit=limit)


@router.get("/graph/stats", summary="Graph statistics")
def graph_stats():
    if _graph is None:
        raise HTTPException(503, "Neo4j not available.")
    return _graph.entity_stats()


# ── Health / status ──
@router.get("/status", summary="System + dependency health")
def status():
    from app.core.redis_client import redis_healthy
    out = {
        "vector_ready": _store.is_ready,
        "graph_enabled": _graph is not None,
        "redis_healthy": redis_healthy(),
        "async_ingestion": settings.async_ingestion,
        "auth_enabled": bool(settings.api_keys),
    }
    if _store.is_ready:
        _store.load()
        out.update(_store.stats())
    return out


# ── Evaluation ────────────────────────────────────────────────────────

@router.get("/evaluate/results", summary="Latest extraction accuracy results")
def eval_results():
    """Return the most recently saved evaluation JSON for the UI."""
    from app.evaluation.eval_report import load_latest_evaluation
    data = load_latest_evaluation()
    if not data:
        raise HTTPException(
            404,
            "No evaluation results yet. "
            "Run: python run_evaluation.py"
        )
    return data


@router.get("/evaluate/ground-truth-count",
            summary="How many ground truth entries are loaded")
def gt_count():
    from app.evaluation.ground_truth import load_all_ground_truth
    gts = load_all_ground_truth()
    entries = sum(len(g.entries) for g in gts)
    return {"files": len(gts), "entries": entries,
            "ready": entries > 0}
