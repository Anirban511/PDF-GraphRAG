"""
insight_generator.py — Turn KPIs + graph into a narrative insight summary.

WHY THIS EXISTS:
  Numbers and graphs need a story. This module asks the local LLM to write
  an executive-summary narrative *grounded strictly in the computed KPIs and
  graph facts* — never inventing figures. It is the bridge between the
  quantitative analytics layer and the human-readable report.

GROUNDING:
  The LLM receives only the structured KPI summary and graph stats as input
  and is instructed to summarise — not to add outside knowledge. Same
  anti-hallucination discipline as the RAG answer path.
"""

from __future__ import annotations
import json

from app.analytics.kpi_engine import KPISummary
from app.generation.llm import call_llm
from app.utils.logger import logger

_NARRATIVE_SYSTEM = """You are a financial analyst writing an executive summary.
You are given pre-computed metrics and graph statistics. Write a concise,
professional summary (3-4 short paragraphs) of what the data shows.

RULES:
- Use ONLY the numbers provided. Do not invent or estimate any figure.
- Be specific and reference the actual entities and values.
- Write for a business executive: clear, direct, no jargon.
- Do not use markdown headers or bullet points; write flowing prose."""

_NARRATIVE_USER = """Computed KPIs:
{kpis}

Graph statistics:
{graph_stats}

Top insights already derived:
{insights}

Write the executive summary:"""


def generate_narrative(kpi: KPISummary, graph_stats: dict) -> str:
    """Produce an executive-summary narrative from KPIs + graph stats."""
    try:
        narrative = call_llm(
            system=_NARRATIVE_SYSTEM,
            user=_NARRATIVE_USER.format(
                kpis=json.dumps({
                    "total_by_metric": kpi.total_by_metric,
                    "top_entities": kpi.top_entities,
                    "entity_count": kpi.entity_count,
                    "record_count": kpi.record_count,
                }, indent=2),
                graph_stats=json.dumps(graph_stats, indent=2, default=str),
                insights="\n".join(f"- {i}" for i in kpi.insights),
            ),
            max_tokens=1024,
        )
        logger.success("Generated executive narrative.")
        return narrative.strip()
    except Exception as exc:
        logger.error(f"Narrative generation failed: {exc}")
        # Fall back to a deterministic summary from the insights
        return ("Executive summary (auto-generated from extracted metrics):\n\n"
                + " ".join(kpi.insights))
