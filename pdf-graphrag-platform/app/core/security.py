"""
security.py — API key authentication + client identification.

WHY THIS EXISTS:
  The previous version had no auth — anyone hitting the API could spend its
  resources. For an SDE-grade service, even a simple API-key check is expected.

  Two functions:
    • require_api_key  — FastAPI dependency; rejects requests without a valid
                         key when auth is enabled.
    • client_id        — derives a stable identifier (the API key, or the
                         client IP) used as the rate-limit bucket.

  Auth is OPTIONAL (toggle via API_KEYS in config). With no keys configured,
  the API is open — convenient for local dev, locked down for deployment.
"""

from __future__ import annotations
from fastapi import Header, HTTPException, Request

from app.config import settings


def _valid_keys() -> set[str]:
    return {k.strip() for k in settings.api_keys.split(",") if k.strip()}


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency. Enforces API key only if keys are configured."""
    keys = _valid_keys()
    if not keys:
        return None  # auth disabled — open API (dev mode)
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return x_api_key


def client_id(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """Stable identifier for rate limiting: API key if present, else client IP."""
    if x_api_key:
        return f"key:{x_api_key[:12]}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"
