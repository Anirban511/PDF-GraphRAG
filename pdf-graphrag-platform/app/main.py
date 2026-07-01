"""
main.py — FastAPI application factory (SDE/production edition).

DIFFERENCES FROM PREVIOUS VERSION:
  • Starts a background job worker on startup (lifespan) so queued ingestion
    jobs get processed.
  • Registers the ingestion handler with the worker.
  • Graceful shutdown stops the worker.

Run: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.core.job_queue import JobWorker
from app.services import ingestion_service
from app.utils.logger import logger

# Worker handler registry — maps job types to service functions
_worker = JobWorker(handlers={
    "ingest": lambda payload: ingestion_service.ingest_document(**payload),
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — launching background job worker…")
    _worker.start()
    yield
    logger.info("Shutting down — stopping job worker…")
    _worker.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="GraphRAG document platform with async ingestion, caching, "
                "rate limiting, and pluggable LLM backends.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["health"])
def root():
    return {"name": settings.app_name, "version": settings.app_version,
            "status": "running"}


@app.get("/health", tags=["health"])
def health():
    from app.core.redis_client import redis_healthy
    return {"status": "ok", "redis": redis_healthy()}
