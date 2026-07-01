"""
rate_limiter.py — Sliding-window rate limiting backed by Redis.

WHY THIS EXISTS:
  Any endpoint that triggers LLM calls spends money and compute. Without a
  limit, one client (or a bug, or an attacker) can exhaust resources. This is
  a core production concern an SDE interviewer will expect to see addressed.

ALGORITHM — fixed-window counter:
  For each client + window, increment a counter with a TTL equal to the window
  length. If the counter exceeds the limit, reject with HTTP 429. Simple,
  O(1), and good enough for most APIs. (A sliding-log or token-bucket would be
  more precise; this is the pragmatic choice and worth being able to compare.)

GRACEFUL DEGRADATION:
  If Redis is down, the limiter "fails open" — requests are allowed rather than
  blocked. A monitoring system should alert on Redis being down; we don't want
  a cache outage to take down the whole API.
"""

from __future__ import annotations
import time

from app.config import settings
from app.core.redis_client import get_redis
from app.utils.logger import logger


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry in {retry_after}s.")


def check_rate_limit(client_id: str, limit: int | None = None,
                     window: int | None = None) -> dict:
    """
    Raise RateLimitExceeded if the client is over the limit.
    Returns a dict of {limit, remaining, reset_in} on success.
    """
    r = get_redis()
    limit = limit or settings.rate_limit_requests
    window = window or settings.rate_limit_window_seconds

    if not r:
        return {"limit": limit, "remaining": limit, "reset_in": window}  # fail open

    bucket = int(time.time()) // window
    key = f"ratelimit:{client_id}:{bucket}"
    try:
        count = r.incr(key)
        if count == 1:
            r.expire(key, window)
        remaining = max(0, limit - count)
        reset_in = window - (int(time.time()) % window)
        if count > limit:
            logger.warning(f"Rate limit exceeded for {client_id} ({count}/{limit})")
            raise RateLimitExceeded(retry_after=reset_in)
        return {"limit": limit, "remaining": remaining, "reset_in": reset_in}
    except RateLimitExceeded:
        raise
    except Exception as exc:
        logger.debug(f"Rate limiter error ({exc}); failing open.")
        return {"limit": limit, "remaining": limit, "reset_in": window}
