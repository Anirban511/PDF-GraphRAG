"""
redis_client.py — Centralised Redis connection + health.

WHY REDIS (the SDE rationale):
  The previous version did everything in-process and synchronously. That's
  fine for a demo but has three production problems this module solves:

    1. CACHING — re-uploading the same PDF re-ran the entire expensive
       pipeline. Redis caches results by document hash so repeat work is free.
    2. RATE LIMITING — a public endpoint with no limits is a cost/DoS risk.
       Redis stores per-client request counts with TTL windows.
    3. JOB QUEUE — ingestion is slow (per-chunk LLM calls). Instead of
       blocking the HTTP request for minutes, we enqueue a job in Redis and
       return immediately with a job_id the client can poll.

  One Redis instance backs all three. This is a very standard production
  pattern and a strong thing to be able to explain in an SDE interview.

CONNECTION:
  A single module-level client (connection pool under the hood) shared across
  the app. Fails gracefully — if Redis is down, callers can fall back to
  synchronous, uncached behaviour rather than crashing.
"""

from __future__ import annotations
import redis
from app.config import settings
from app.utils.logger import logger

_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    """Return a shared Redis client, or None if Redis is unreachable."""
    global _client
    if _client is None:
        try:
            _client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            _client.ping()
            logger.success(f"Redis connected at {settings.redis_host}:{settings.redis_port}")
        except Exception as exc:
            logger.warning(f"Redis unavailable ({exc}); caching/queue/rate-limit disabled.")
            _client = None
    return _client


def redis_healthy() -> bool:
    r = get_redis()
    try:
        return bool(r and r.ping())
    except Exception:
        return False
