"""
cache.py — Result caching keyed by content hash.

WHY THIS EXISTS:
  Ingesting a PDF is the most expensive operation (per-chunk LLM calls).
  If the same document is uploaded again — common in any real app — we should
  never pay that cost twice. Each PDF already has a SHA-256 doc_id, so it's a
  natural cache key.

  We also cache query answers: identical questions against the same index
  return instantly from cache instead of re-running retrieval + generation.

DESIGN:
  • JSON-serialised values with a configurable TTL.
  • Namespaced keys (cache:doc:<hash>, cache:query:<hash>) to avoid collisions.
  • All operations are best-effort: a cache miss or Redis outage simply means
    the caller does the work normally. Caching never breaks correctness.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any

from app.config import settings
from app.core.redis_client import get_redis
from app.utils.logger import logger


def _key(namespace: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"cache:{namespace}:{digest}"


def cache_get(namespace: str, raw_key: str) -> Any | None:
    r = get_redis()
    if not r:
        return None
    try:
        val = r.get(_key(namespace, raw_key))
        if val is not None:
            logger.debug(f"Cache HIT [{namespace}]")
            return json.loads(val)
    except Exception as exc:
        logger.debug(f"Cache get failed: {exc}")
    return None


def cache_set(namespace: str, raw_key: str, value: Any,
              ttl: int | None = None) -> None:
    r = get_redis()
    if not r:
        return
    try:
        r.setex(_key(namespace, raw_key),
                ttl or settings.cache_ttl_seconds,
                json.dumps(value, default=str))
        logger.debug(f"Cache SET [{namespace}]")
    except Exception as exc:
        logger.debug(f"Cache set failed: {exc}")


def cache_exists_doc(doc_hash: str) -> bool:
    """Has this exact document already been ingested?"""
    r = get_redis()
    if not r:
        return False
    try:
        return bool(r.exists(_key("doc", doc_hash)))
    except Exception:
        return False


def mark_doc_ingested(doc_hash: str, meta: dict) -> None:
    cache_set("doc", doc_hash, meta, ttl=settings.cache_ttl_seconds)
