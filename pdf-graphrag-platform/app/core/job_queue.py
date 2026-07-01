"""
job_queue.py — Lightweight Redis-backed background job queue.

WHY THIS EXISTS (the key architectural upgrade):
  In the previous version, POST /ingest blocked the HTTP request for the
  entire ingestion — extraction, embedding, graph build, metric extraction —
  which can take minutes. That's bad API design: requests time out, the UI
  freezes, and one upload ties up a worker.

  Here, /ingest instead ENQUEUES a job and returns a job_id immediately
  (HTTP 202 Accepted). A background worker thread pulls jobs and processes
  them. The client polls GET /jobs/{id} for status:
      queued -> processing -> completed | failed

  This is the standard async-processing pattern (the same idea as Celery,
  RQ, or SQS workers, implemented minimally here with Redis lists + hashes so
  the project has no heavy task-queue dependency).

JOB STATE (stored in a Redis hash per job):
    status, progress, result, error, created_at, updated_at

WHY A THREAD, NOT A SEPARATE PROCESS:
  Keeps the demo single-container and simple. The design note in the docs
  explains how you'd swap the in-thread worker for a separate worker process
  / Celery in real production — the queue contract stays identical.
"""

from __future__ import annotations
import json
import time
import uuid
import threading
from typing import Callable

from app.core.redis_client import get_redis
from app.utils.logger import logger

_QUEUE_KEY = "jobs:queue"
_JOB_PREFIX = "jobs:item:"


def _job_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}{job_id}"


def enqueue_job(job_type: str, payload: dict) -> str:
    """Create a job, push it on the queue, return its id."""
    r = get_redis()
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id, "type": job_type, "status": "queued",
        "progress": "0", "result": "", "error": "",
        "created_at": str(time.time()), "updated_at": str(time.time()),
        "payload": json.dumps(payload),
    }
    if not r:
        # No Redis: signal caller to fall back to synchronous processing
        return ""
    r.hset(_job_key(job_id), mapping=job)
    r.expire(_job_key(job_id), 86400)  # keep job record 24h
    r.rpush(_QUEUE_KEY, job_id)
    logger.info(f"Enqueued job {job_id} ({job_type})")
    return job_id


def get_job(job_id: str) -> dict | None:
    r = get_redis()
    if not r:
        return None
    data = r.hgetall(_job_key(job_id))
    return data or None


def update_job(job_id: str, **fields) -> None:
    r = get_redis()
    if not r:
        return
    fields["updated_at"] = str(time.time())
    r.hset(_job_key(job_id), mapping={k: str(v) for k, v in fields.items()})


class JobWorker:
    """Background thread that pulls jobs and runs the registered handler."""

    def __init__(self, handlers: dict[str, Callable[[dict], dict]]):
        self._handlers = handlers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        r = get_redis()
        if not r:
            logger.warning("No Redis — job worker not started (sync fallback in use).")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.success("Background job worker started.")

    def stop(self):
        self._stop.set()

    def _run(self):
        r = get_redis()
        while not self._stop.is_set():
            try:
                popped = r.blpop(_QUEUE_KEY, timeout=2)
                if not popped:
                    continue
                _, job_id = popped
                job = get_job(job_id)
                if not job:
                    continue
                handler = self._handlers.get(job["type"])
                if not handler:
                    update_job(job_id, status="failed", error="No handler")
                    continue
                update_job(job_id, status="processing", progress="10")
                try:
                    payload = json.loads(job.get("payload", "{}"))
                    result = handler(payload)
                    update_job(job_id, status="completed", progress="100",
                               result=json.dumps(result, default=str))
                    logger.success(f"Job {job_id} completed.")
                except Exception as exc:
                    update_job(job_id, status="failed", error=str(exc))
                    logger.error(f"Job {job_id} failed: {exc}")
            except Exception as exc:
                logger.error(f"Worker loop error: {exc}")
                time.sleep(1)
