"""Tests for the financial metric value coercion (pure function)."""
from app.analytics.metrics_extractor import _coerce_value


def test_plain_number():
    assert _coerce_value(2500000) == 2500000.0

def test_million_suffix():
    assert _coerce_value("2.5M") == 2_500_000.0

def test_billion_suffix():
    assert _coerce_value("1.2B") == 1_200_000_000.0

def test_thousand_suffix():
    assert _coerce_value("750K") == 750_000.0

def test_dollar_and_commas():
    assert _coerce_value("$1,250,000") == 1_250_000.0

def test_garbage_returns_none():
    assert _coerce_value("not a number") is None
