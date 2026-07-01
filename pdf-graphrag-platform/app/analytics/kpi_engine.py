"""
kpi_engine.py — Compute business KPIs and insights from extracted metrics.

WHY THIS EXISTS:
  Raw extracted metrics are just rows. A business user wants *insights*:
  which entity grew fastest, what the totals are, where the outliers sit.
  This module aggregates the MetricRecords into KPIs and ranked insights
  that the reporting layer renders into prose + charts.

WHY pandas:
  Aggregation, grouping, and pivoting financial data by entity/period/metric
  is exactly what pandas is built for; doing it by hand would be error-prone.

OUTPUT:
  A KPISummary object — totals, per-entity breakdowns, growth signals, and a
  list of plain-English insight strings — ready for the report generator.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd

from app.analytics.metrics_extractor import MetricRecord
from app.utils.logger import logger


@dataclass
class KPISummary:
    total_by_metric: dict           # {"revenue": 12_500_000, ...}
    by_entity: dict                 # {"Acme": {"revenue": ...}, ...}
    top_entities: list              # ranked by total value
    insights: list[str] = field(default_factory=list)
    record_count: int = 0
    entity_count: int = 0


def compute_kpis(records: list[MetricRecord]) -> KPISummary:
    if not records:
        return KPISummary(total_by_metric={}, by_entity={}, top_entities=[],
                          insights=["No financial metrics were extracted."])

    df = pd.DataFrame([r.__dict__ for r in records])

    # Totals per metric type
    total_by_metric = (
        df.groupby("metric")["value"].sum().sort_values(ascending=False).to_dict()
    )

    # Per-entity, per-metric breakdown
    by_entity = {}
    for entity, grp in df.groupby("entity"):
        by_entity[entity] = grp.groupby("metric")["value"].sum().to_dict()

    # Rank entities by total value mentioned (rough "significance" proxy)
    entity_totals = (
        df.groupby("entity")["value"].sum().sort_values(ascending=False)
    )
    top_entities = [
        {"entity": e, "total_value": float(v)}
        for e, v in entity_totals.head(10).items()
    ]

    # Generate plain-English insights
    insights = _generate_insights(df, total_by_metric, entity_totals)

    summary = KPISummary(
        total_by_metric=total_by_metric,
        by_entity=by_entity,
        top_entities=top_entities,
        insights=insights,
        record_count=len(records),
        entity_count=df["entity"].nunique(),
    )
    logger.success(f"Computed KPIs across {summary.entity_count} entities.")
    return summary


def _generate_insights(df: pd.DataFrame, totals: dict,
                       entity_totals: pd.Series) -> list[str]:
    """Derive a handful of headline insights from the data."""
    insights = []

    if not entity_totals.empty:
        top = entity_totals.index[0]
        insights.append(
            f"{top} accounts for the largest aggregate financial figure "
            f"({entity_totals.iloc[0]:,.0f} across all metrics)."
        )

    for metric, total in list(totals.items())[:3]:
        insights.append(f"Total {metric} across all documents: {total:,.0f}.")

    # Growth signal: same entity+metric across different periods
    if "period" in df.columns:
        for (entity, metric), grp in df.groupby(["entity", "metric"]):
            periods = grp.dropna(subset=["period"])
            if len(periods["period"].unique()) >= 2:
                ordered = periods.sort_values("period")
                first, last = ordered["value"].iloc[0], ordered["value"].iloc[-1]
                if first > 0:
                    change = (last - first) / first * 100
                    direction = "increased" if change >= 0 else "decreased"
                    insights.append(
                        f"{entity}'s {metric} {direction} by {abs(change):.1f}% "
                        f"from {ordered['period'].iloc[0]} to {ordered['period'].iloc[-1]}."
                    )

    # Concentration insight
    if len(entity_totals) >= 3:
        top3_share = entity_totals.head(3).sum() / entity_totals.sum() * 100
        insights.append(
            f"The top 3 entities represent {top3_share:.0f}% of total "
            f"financial value mentioned across the document set."
        )

    return insights
