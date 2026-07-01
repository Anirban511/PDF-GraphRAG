"""
analytics_pipeline.py — Orchestrates the full document-to-report flow.

WHY THIS EXISTS:
  Single entry point that runs the business-analytics half of the system:
    chunks → extract metrics → compute KPIs → pull graph stats →
    generate narrative → build report

  Keeps the API route thin: it just calls run_analytics() and returns the
  report path.

NOTE ON CHUNKING:
  This pipeline receives chunks that were created for Q&A retrieval
  (with overlap). For extraction accuracy, use run_analytics_from_pages()
  which re-chunks with overlap=0. Both entry points are provided so the
  caller can choose.
"""

from __future__ import annotations
from pathlib import Path

from app.analytics.kpi_engine import compute_kpis
from app.analytics.metrics_extractor import extract_metrics_full
from app.graph.neo4j_store import Neo4jStore
from app.ingestion.chunker import Chunk, chunk_pages
from app.ingestion.loader import PageRecord
from app.reporting.insight_generator import generate_narrative
from app.reporting.report_builder import build_report
from app.utils.logger import logger


def run_analytics(chunks: list[Chunk], graph: Neo4jStore) -> Path:
    """
    Run metric extraction → KPIs → narrative → report.
    Accepts pre-chunked data (used by the API route).
    Uses extract_metrics_full which includes deduplication as a safety net.
    """
    logger.info(f"Analytics pipeline starting on {len(chunks)} chunks…")

    metrics = extract_metrics_full(chunks)
    kpi = compute_kpis(metrics)
    graph_stats = graph.entity_stats()
    narrative = generate_narrative(kpi, graph_stats)

    source_files = sorted({c.filename for c in chunks})
    report_path = build_report(narrative, kpi, graph_stats, source_files)

    logger.success(f"Analytics complete → {report_path.name}")
    return report_path


def run_analytics_from_pages(pages: list[PageRecord],
                              graph: Neo4jStore) -> Path:
    """
    Re-chunk with overlap=0 before extraction — the cleanest path.
    Use this when you control the chunking step.
    """
    # Zero overlap: each sentence processed exactly once, no duplicates
    chunks = chunk_pages(pages, overlap=0)
    logger.info(
        f"Re-chunked for analytics: {len(pages)} pages → "
        f"{len(chunks)} chunks (overlap=0)"
    )
    return run_analytics(chunks, graph)
