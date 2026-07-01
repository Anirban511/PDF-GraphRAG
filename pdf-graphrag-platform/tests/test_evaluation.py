"""
test_evaluation.py — Unit tests for the extraction accuracy evaluator.

Tests the matching logic, metric computation, and edge cases.
All tests run without Ollama, Neo4j, or any PDF.
"""

import pytest
from app.evaluation.evaluator import (
    _entity_match, _metric_match, _value_match,
    evaluate, MatchResult,
)
from app.evaluation.ground_truth import GroundTruthEntry
from app.analytics.metrics_extractor import MetricRecord


# ── Helper factories ──────────────────────────────────────────────────

def make_record(entity="Acme Corp", metric="revenue",
                value=4_200_000_000.0, unit="USD",
                period="FY2024", page=12):
    return MetricRecord(entity=entity, metric=metric, value=value,
                        unit=unit, period=period,
                        filename="test.pdf", page_num=page)


def make_gt(entity="Acme Corp", metric="revenue",
            value=4_200_000_000.0, unit="USD",
            period="FY2024", page=12):
    return GroundTruthEntry(entity=entity, metric=metric,
                            true_value=value, unit=unit,
                            period=period, source_page=page)


# ── Entity matching ───────────────────────────────────────────────────

def test_entity_match_exact():
    assert _entity_match("Acme Corp", "Acme Corp")

def test_entity_match_case_insensitive():
    assert _entity_match("acme corp", "Acme Corp")

def test_entity_match_abbreviation():
    assert _entity_match("Acme Corp", "Acme Corporation")

def test_entity_match_substring():
    assert _entity_match("Acme", "Acme Corp")

def test_entity_no_match():
    assert not _entity_match("Tesla", "Acme Corp")


# ── Metric matching ───────────────────────────────────────────────────

def test_metric_match_exact():
    assert _metric_match("revenue", "revenue")

def test_metric_match_partial():
    assert _metric_match("total revenue", "revenue")

def test_metric_match_case():
    assert _metric_match("Net Income", "net_income")

def test_metric_no_match():
    assert not _metric_match("revenue", "cost_of_goods")


# ── Value matching ────────────────────────────────────────────────────

def test_value_match_exact():
    assert _value_match(4_200_000_000, 4_200_000_000)

def test_value_match_within_1pct():
    assert _value_match(4_200_000_000 * 1.009, 4_200_000_000, tolerance=0.01)

def test_value_match_outside_1pct():
    assert not _value_match(4_200_000_000 * 1.02, 4_200_000_000, tolerance=0.01)

def test_value_match_billion_vs_million():
    # A 3B model classic error: reading $4.2B as $4.2M
    assert not _value_match(4_200_000, 4_200_000_000, tolerance=0.01)

def test_value_match_zero():
    assert _value_match(0.0, 0.0)


# ── Full evaluation ───────────────────────────────────────────────────

def test_perfect_extraction():
    """All extracted = all ground truth → F1 = 1.0"""
    gt = [make_gt(entity="Acme", metric="revenue", value=4_200_000_000)]
    ex = [make_record(entity="Acme", metric="revenue", value=4_200_000_000)]
    result = evaluate(ex, gt)
    assert result.true_positives_strict == 1
    assert result.false_positives_strict == 0
    assert result.false_negatives_strict == 0
    assert result.f1_strict == 1.0
    assert result.precision_strict == 1.0
    assert result.recall_strict == 1.0


def test_wrong_value_billion_vs_million():
    """
    Classic small-model error: right entity/metric, wrong scale.
    The GT entry is consumed (entity+metric matched) but value is wrong.
    This scores as a FP (bad value extracted), F1 = 0.
    FN = 0 because the entry was found — just read incorrectly.
    """
    gt = [make_gt(value=4_200_000_000)]
    ex = [make_record(value=4_200_000)]   # 1000x off
    result = evaluate(ex, gt)
    assert result.true_positives_strict  == 0
    assert result.true_positives_lenient == 0
    assert result.false_positives_strict == 1
    assert result.f1_strict == 0.0


def test_missed_entity():
    """Extracted has no matching entity → all false positives + false negatives."""
    gt  = [make_gt(entity="Acme Corp")]
    ex  = [make_record(entity="Tesla Inc")]
    result = evaluate(ex, gt)
    assert result.true_positives_strict == 0
    assert result.false_positives_strict == 1
    assert result.false_negatives_strict == 1


def test_partial_extraction():
    """Extracted 2 of 3 correct → partial precision/recall."""
    gt = [
        make_gt(entity="Acme", metric="revenue", value=4_200_000_000),
        make_gt(entity="Acme", metric="net_income", value=850_000_000),
        make_gt(entity="Acme", metric="costs", value=2_100_000_000),
    ]
    ex = [
        make_record(entity="Acme", metric="revenue", value=4_200_000_000),
        make_record(entity="Acme", metric="net_income", value=850_000_000),
        # costs missed
    ]
    result = evaluate(ex, gt)
    assert result.true_positives_strict == 2
    assert result.false_negatives_strict == 1
    assert result.precision_strict == 1.0      # both extractions were correct
    assert result.recall_strict < 1.0          # missed one


def test_extra_extraction_false_positive():
    """More extracted than ground truth — precision penalised."""
    gt = [make_gt(entity="Acme", metric="revenue", value=4_200_000_000)]
    ex = [
        make_record(entity="Acme", metric="revenue", value=4_200_000_000),
        make_record(entity="Acme", metric="phantom", value=999_000_000),  # hallucination
    ]
    result = evaluate(ex, gt)
    assert result.true_positives_strict  == 1
    assert result.false_positives_strict == 1
    assert result.precision_strict < 1.0
    assert result.recall_strict == 1.0


def test_f1_balances_precision_recall():
    """F1 is lower than both precision and recall when either is imperfect."""
    gt = [make_gt(entity="Acme", metric="revenue", value=4_200_000_000),
          make_gt(entity="Acme", metric="net_income", value=850_000_000)]
    ex = [make_record(entity="Acme", metric="revenue", value=4_200_000_000)]
    result = evaluate(ex, gt)
    assert 0 < result.f1_strict < 1.0
    # Precision = 1.0 (one hit, no FP), Recall = 0.5, F1 = 0.667
    assert abs(result.f1_strict - 2/3) < 0.01


def test_empty_extraction():
    """No extraction at all → F1 = 0."""
    gt = [make_gt()]
    result = evaluate([], gt)
    assert result.f1_strict == 0.0
    assert result.false_negatives_strict == 1


def test_empty_ground_truth():
    """No ground truth → can't compute meaningful metrics, no crash."""
    ex = [make_record()]
    result = evaluate(ex, [])
    assert result.total_ground_truth == 0
    assert result.false_positives_strict == 1


def test_lenient_catches_minor_rounding():
    """Value off by 3% — strict miss, lenient hit."""
    gt = [make_gt(value=4_200_000_000)]
    ex = [make_record(value=4_200_000_000 * 1.03)]
    result = evaluate(ex, gt)
    assert result.true_positives_strict  == 0  # outside 1%
    assert result.true_positives_lenient == 1  # within 5%


def test_match_label_correct():
    gt = [make_gt()]
    ex = [make_record()]
    result = evaluate(ex, gt)
    assert result.matches[0].label == "CORRECT (strict)"


def test_match_label_wrong_value():
    gt = [make_gt(value=4_200_000_000)]
    ex = [make_record(value=4_200_000)]  # 1000x off
    result = evaluate(ex, gt)
    assert result.matches[0].label == "WRONG VALUE"


def test_dedup_prevents_double_counting():
    """Same ground truth entry cannot match two extracted records."""
    gt = [make_gt(entity="Acme", metric="revenue", value=4_200_000_000)]
    ex = [
        make_record(entity="Acme", metric="revenue", value=4_200_000_000),
        make_record(entity="Acme", metric="revenue", value=4_200_000_000),
    ]
    result = evaluate(ex, gt)
    # Only one can match the single GT entry
    assert result.true_positives_strict == 1
    assert result.false_positives_strict == 1
