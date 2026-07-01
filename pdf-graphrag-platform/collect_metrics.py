"""
collect_metrics.py — Run the full pipeline on real PDFs and print every
number you need for your CV summary.

USAGE
-----
1. Make sure Ollama is running:        ollama serve
   and the model is pulled:            ollama pull llama3.2
2. (Optional) Start Neo4j for graph numbers:
       docker run -p7474:7474 -p7687:7687 \
         -e NEO4J_AUTH=neo4j/password123 neo4j:5.26-community
3. Put 1+ financial PDFs in:           data/raw_pdfs/
   (A public company annual report or 10-K works great.)
4. Run:                                python collect_metrics.py

It prints a block of CV-READY NUMBERS at the end. Copy those back.

WHAT YOU OPTIONALLY PROVIDE (for accuracy metrics)
--------------------------------------------------
To get *accuracy* numbers (the strongest kind), open this file and fill in
GROUND_TRUTH below: a few questions you know the answer to, and a few figures
you've verified by hand. The script will grade the pipeline against them.
Leave it empty to still get all the volume/coverage numbers.
"""

from __future__ import annotations
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# OPTIONAL: fill these in for ACCURACY metrics (the strongest numbers).
# Leave as empty lists to skip — you'll still get volume/coverage numbers.
# ─────────────────────────────────────────────────────────────────────

# Questions you know the answer to + the page that answers them.
# Used to measure retrieval recall + citation accuracy.
GROUND_TRUTH_QUESTIONS = [
    # {"question": "What was total revenue in FY2024?",
    #  "expected_page": 12,
    #  "expected_answer_contains": "4.2"},
]

# Figures you've verified by hand from the PDF.
# Used to measure extraction accuracy.
GROUND_TRUTH_FIGURES = [
    # {"entity": "Acme Corp", "metric": "revenue", "true_value": 4_200_000_000},
]

# ─────────────────────────────────────────────────────────────────────

from app.config import settings
from app.ingestion.loader import load_all_pdfs
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.vector_store import VectorStore


def hr(title=""):
    print("\n" + "=" * 64)
    if title:
        print(f"  {title}")
        print("=" * 64)


def main():
    results = {}

    # ── 1. INGESTION VOLUME ───────────────────────────────────────────
    hr("STAGE 1 — INGESTION")
    t0 = time.perf_counter()
    pages = load_all_pdfs()
    if not pages:
        print("!! No PDFs found in data/raw_pdfs/. Add at least one and rerun.")
        return
    chunks = chunk_pages(pages)
    docs = {p.filename for p in pages}
    results["num_documents"] = len(docs)
    results["num_pages"] = len(pages)
    results["num_chunks"] = len(chunks)
    ingest_time = time.perf_counter() - t0
    results["ingest_seconds"] = round(ingest_time, 1)
    print(f"Documents ingested : {len(docs)}")
    print(f"Total pages        : {len(pages)}")
    print(f"Total chunks       : {len(chunks)}")
    print(f"Ingestion time     : {ingest_time:.1f}s")

    # ── 2. EMBEDDING + VECTOR INDEX ───────────────────────────────────
    hr("STAGE 2 — EMBEDDING + VECTOR INDEX")
    t0 = time.perf_counter()
    vectors = embed_chunks(chunks)
    embed_time = time.perf_counter() - t0
    store = VectorStore()
    store.build(chunks, vectors)
    results["embedding_dim"] = int(vectors.shape[1])
    results["embed_seconds"] = round(embed_time, 1)
    results["embed_throughput"] = round(len(chunks) / embed_time, 1) if embed_time else 0
    print(f"Embedding dimension : {vectors.shape[1]}")
    print(f"Vectors indexed     : {vectors.shape[0]}")
    print(f"Embedding time      : {embed_time:.1f}s "
          f"({results['embed_throughput']} chunks/s)")

    # ── 3. METRIC EXTRACTION (analytics) ──────────────────────────────
    hr("STAGE 3 — FINANCIAL METRIC EXTRACTION")
    try:
        from app.analytics.metrics_extractor import extract_metrics
        t0 = time.perf_counter()
        metrics = extract_metrics(chunks)
        extract_time = time.perf_counter() - t0
        results["num_metrics"] = len(metrics)
        results["num_metric_entities"] = len({m.entity for m in metrics})
        results["extract_seconds"] = round(extract_time, 1)
        print(f"Financial metrics extracted : {len(metrics)}")
        print(f"Distinct entities w/ metrics: {results['num_metric_entities']}")
        print(f"Extraction time             : {extract_time:.1f}s")
        if metrics:
            print("\nSample extracted metrics:")
            for m in metrics[:5]:
                print(f"  - {m.entity} | {m.metric} = {m.value:,.0f} {m.unit} "
                      f"({m.period}) [p.{m.page_num}]")
    except Exception as exc:
        print(f"Metric extraction unavailable: {exc}")
        metrics = []

    # ── 4. KPI ENGINE ─────────────────────────────────────────────────
    hr("STAGE 4 — KPI COMPUTATION")
    try:
        from app.analytics.kpi_engine import compute_kpis
        kpi = compute_kpis(metrics)
        results["num_kpi_insights"] = len(kpi.insights)
        results["num_metric_types"] = len(kpi.total_by_metric)
        print(f"Metric types aggregated : {len(kpi.total_by_metric)}")
        print(f"Insights derived        : {len(kpi.insights)}")
        print("\nDerived insights:")
        for ins in kpi.insights:
            print(f"  - {ins}")
    except Exception as exc:
        print(f"KPI engine unavailable: {exc}")
        kpi = None

    # ── 5. KNOWLEDGE GRAPH (if Neo4j up) ──────────────────────────────
    hr("STAGE 5 — KNOWLEDGE GRAPH (Neo4j)")
    graph = None
    try:
        from app.graph.neo4j_store import Neo4jStore
        from app.graph.entity_extractor import extract_from_chunks
        graph = Neo4jStore()
        graph.init_schema()
        graph.wipe()
        graph.write_chunks(chunks)
        frag = extract_from_chunks(chunks)
        lookup = {(c.doc_id, c.page_num): c.chunk_id for c in chunks}
        graph.write_graph(frag, lookup)
        stats = graph.entity_stats()
        results["num_entities"] = stats["total_entities"]
        results["num_relationships"] = stats["total_relationships"]
        print(f"Entities in graph       : {stats['total_entities']}")
        print(f"Relationships in graph  : {stats['total_relationships']}")
        if stats.get("top_connected"):
            print("\nMost connected entities (network hubs):")
            for tc in stats["top_connected"][:5]:
                print(f"  - {tc['name']} ({tc['type']}): {tc['degree']} links")
    except Exception as exc:
        print(f"Neo4j unavailable (skipping graph numbers): {exc}")

    # ── 6. REPORT GENERATION ──────────────────────────────────────────
    hr("STAGE 6 — REPORT GENERATION")
    if graph is not None and kpi is not None:
        try:
            from app.reporting.insight_generator import generate_narrative
            from app.reporting.report_builder import build_report
            narrative = generate_narrative(kpi, graph.entity_stats())
            report_path = build_report(narrative, kpi, graph.entity_stats(),
                                       sorted(docs))
            results["report_path"] = str(report_path)
            print(f"Report generated: {report_path}")
        except Exception as exc:
            print(f"Report generation failed: {exc}")
    else:
        print("Skipped (needs Neo4j + KPIs).")

    # ── 7. ACCURACY (only if ground truth provided) ───────────────────
    hr("STAGE 7 — ACCURACY (optional)")
    if GROUND_TRUTH_QUESTIONS:
        from app.retrieval.retriever import Retriever
        retr = Retriever(store)
        recall_hits = 0
        for gt in GROUND_TRUTH_QUESTIONS:
            cands = retr.retrieve(gt["question"], top_k=settings.top_k)
            pages_found = {c["page_num"] for c in cands}
            if gt["expected_page"] in pages_found:
                recall_hits += 1
        n = len(GROUND_TRUTH_QUESTIONS)
        results["retrieval_recall"] = f"{recall_hits}/{n}"
        results["retrieval_recall_pct"] = round(recall_hits / n * 100)
        print(f"Retrieval recall: {recall_hits}/{n} "
              f"({results['retrieval_recall_pct']}%) — correct page in top-{settings.top_k}")
    else:
        print("No GROUND_TRUTH_QUESTIONS provided — skipping retrieval recall.")
        print("(Fill in GROUND_TRUTH_QUESTIONS at the top of this file to get it.)")

    if GROUND_TRUTH_FIGURES and metrics:
        correct = 0
        for gt in GROUND_TRUTH_FIGURES:
            match = [m for m in metrics
                     if gt["entity"].lower() in m.entity.lower()
                     and gt["metric"].lower() in m.metric.lower()]
            if match and any(abs(m.value - gt["true_value"]) / gt["true_value"] < 0.01
                             for m in match):
                correct += 1
        n = len(GROUND_TRUTH_FIGURES)
        results["extraction_accuracy"] = f"{correct}/{n}"
        results["extraction_accuracy_pct"] = round(correct / n * 100)
        print(f"Extraction accuracy: {correct}/{n} "
              f"({results['extraction_accuracy_pct']}%) — figures read correctly")
    else:
        print("No GROUND_TRUTH_FIGURES provided — skipping extraction accuracy.")

    # ── FINAL: CV-READY NUMBERS ───────────────────────────────────────
    hr("★ CV-READY NUMBERS — COPY THESE BACK ★")
    print(f"""
  Documents ingested ............ {results.get('num_documents', '?')}
  Pages processed ............... {results.get('num_pages', '?')}
  Chunks created ................ {results.get('num_chunks', '?')}
  Embedding dimension ........... {results.get('embedding_dim', '?')}
  Embedding throughput .......... {results.get('embed_throughput', '?')} chunks/s
  Financial metrics extracted ... {results.get('num_metrics', '?')}
  Entities with metrics ......... {results.get('num_metric_entities', '?')}
  Metric types aggregated ....... {results.get('num_metric_types', '?')}
  Insights derived .............. {results.get('num_kpi_insights', '?')}
  Graph entities ................ {results.get('num_entities', 'N/A - Neo4j off')}
  Graph relationships ........... {results.get('num_relationships', 'N/A - Neo4j off')}
  Report sections ............... 5 (fixed)
  Retrieval recall .............. {results.get('retrieval_recall', 'N/A - no ground truth')}
  Extraction accuracy ........... {results.get('extraction_accuracy', 'N/A - no ground truth')}
  Total ingest time ............. {results.get('ingest_seconds', '?')}s
""")
    print("Paste the numbers above back into the chat and I'll lock them")
    print("into your final CV summary.\n")


if __name__ == "__main__":
    main()
