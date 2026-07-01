"""Tests for core infrastructure: cache keys, rate-limit logic, job lifecycle.

These test the logic that does not require a live Redis (graceful-degradation
paths) plus key construction. Live-Redis integration is covered manually.
"""
from app.core.cache import _key
from app.core.rate_limiter import RateLimitExceeded


def test_cache_key_is_deterministic():
    assert _key("query", "hello") == _key("query", "hello")

def test_cache_key_namespaced():
    assert _key("query", "x") != _key("doc", "x")

def test_cache_key_format():
    k = _key("doc", "abc")
    assert k.startswith("cache:doc:")

def test_rate_limit_exception_carries_retry():
    exc = RateLimitExceeded(retry_after=42)
    assert exc.retry_after == 42

def test_cache_get_returns_none_without_redis(monkeypatch):
    # When Redis unavailable, cache_get must return None, never raise
    import app.core.cache as c
    monkeypatch.setattr(c, "get_redis", lambda: None)
    assert c.cache_get("query", "anything") is None

def test_rate_limit_fails_open_without_redis(monkeypatch):
    import app.core.rate_limiter as rl
    monkeypatch.setattr(rl, "get_redis", lambda: None)
    result = rl.check_rate_limit("client")
    assert result["remaining"] == result["limit"]
