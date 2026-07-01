"""
run_all.py — Single entry point for the entire PDF GraphRAG project.

This is the one file you run. It has three modes:

  python run_all.py pipeline [pdf_path]   Run the FULL pipeline end-to-end in
                                          one process — no servers needed:
                                          ingest -> embed -> graph -> analytics
                                          -> report. Great for a quick demo or
                                          to generate a report from the CLI.

  python run_all.py serve                 Launch the API + Streamlit frontend
                                          together (the interactive app).

  python run_all.py check                 Pre-flight: verify Ollama, Neo4j,
                                          models, and dependencies are ready.

If you run `python run_all.py` with no arguments, it does `check` then prints
this help.

PREREQUISITES
-------------
  • Ollama running:   ollama serve   (and: ollama pull llama3.2)
  • Neo4j running (optional, for graph + analytics):
       docker run -p7474:7474 -p7687:7687 \
         -e NEO4J_AUTH=neo4j/password123 neo4j:5.26-community
  • Dependencies:     pip install -r requirements.txt
"""

from __future__ import annotations
import sys
import time
import subprocess
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────

def banner(text: str):
    line = "═" * 64
    print(f"\n{line}\n  {text}\n{line}")

def step(text: str):
    print(f"\n▶ {text}")

def ok(text: str):
    print(f"  ✓ {text}")

def warn(text: str):
    print(f"  ! {text}")

def fail(text: str):
    print(f"  ✗ {text}")


# ──────────────────────────────────────────────────────────────────────
# MODE: check — pre-flight readiness
# ──────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        import httpx
        from app.config import settings
    except ImportError as exc:
        fail(f"Cannot check Ollama — missing dependency ({exc}). "
             f"Run: pip install -r requirements.txt")
        return False
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        ok(f"Ollama reachable at {settings.ollama_base_url}")
        if any(settings.llm_model in m for m in models):
            ok(f"Model '{settings.llm_model}' is available")
            return True
        warn(f"Model '{settings.llm_model}' not pulled. Run: ollama pull {settings.llm_model}")
        return False
    except Exception:
        fail(f"Ollama not reachable. Start it with: ollama serve")
        return False


def check_neo4j() -> bool:
    try:
        from app.graph.neo4j_store import Neo4jStore
        g = Neo4jStore()
        g.init_schema()
        g.close()
        ok("Neo4j reachable — GraphRAG + analytics enabled")
        return True
    except Exception:
        warn("Neo4j not reachable — system will run in vector-only mode "
             "(graph + analytics disabled)")
        return False


def check_redis() -> bool:
    try:
        from app.core.redis_client import redis_healthy
        if redis_healthy():
            ok("Redis reachable — caching, queue, rate-limiting enabled")
            return True
        warn("Redis not reachable — sync ingestion, no cache (still works)")
        return False
    except Exception:
        warn("Redis not reachable — running without cache/queue")
        return False


def check_deps() -> bool:
    missing = []
    for mod in ("faiss", "sentence_transformers", "fastapi", "streamlit",
                "pandas", "matplotlib", "docx", "neo4j", "httpx"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt")
        return False
    ok("All Python dependencies installed")
    return True


def mode_check() -> dict:
    banner("PRE-FLIGHT CHECK")
    step("Checking dependencies")
    deps = check_deps()
    if not deps:
        warn("Install dependencies first; skipping service checks.")
        return {"deps": False, "ollama": False, "neo4j": False}
    try:
        from app.config import settings
        ok(f"LLM provider configured: {settings.llm_provider}")
    except Exception:
        pass
    step("Checking Ollama (local LLM)")
    ollama = check_ollama()
    step("Checking Neo4j (knowledge graph)")
    neo4j = check_neo4j()
    step("Checking Redis (cache / queue / rate-limit)")
    check_redis()

    print()
    if deps and ollama:
        ok("Core system ready.")
        if not neo4j:
            warn("Graph features off — start Neo4j to enable them.")
    else:
        fail("Core system not ready — fix the items above first.")
    return {"deps": deps, "ollama": ollama, "neo4j": neo4j}


# ──────────────────────────────────────────────────────────────────────
# MODE: pipeline — run everything in one process on a PDF
# ──────────────────────────────────────────────────────────────────────

def mode_pipeline(pdf_path: str | None):
    from app.config import settings
    from app.ingestion.loader import load_pdf, load_all_pdfs
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.embedder import embed_chunks
    from app.ingestion.vector_store import VectorStore

    banner("FULL PIPELINE — END TO END")

    # Resolve input
    if pdf_path:
        pdfs = [Path(pdf_path)]
        if not pdfs[0].exists():
            fail(f"PDF not found: {pdf_path}")
            return
    else:
        pdfs = sorted(settings.raw_pdfs_dir.glob("*.pdf"))
        if not pdfs:
            fail(f"No PDFs in {settings.raw_pdfs_dir}. Pass a path or add files there.")
            return
        ok(f"Found {len(pdfs)} PDF(s) in {settings.raw_pdfs_dir}")

    # 1. Ingest + chunk
    step("Stage 1/6 — Extract + chunk")
    all_pages = []
    for pdf in pdfs:
        all_pages.extend(load_pdf(pdf))
    chunks = chunk_pages(all_pages)
    ok(f"{len(all_pages)} pages → {len(chunks)} chunks")

    # 2. Embed + index
    step("Stage 2/6 — Embed + build vector index")
    vectors = embed_chunks(chunks)
    store = VectorStore()
    store.build(chunks, vectors)
    ok(f"{vectors.shape[0]} vectors indexed (dim={vectors.shape[1]})")

    # 3. Knowledge graph
    graph = None
    step("Stage 3/6 — Build knowledge graph (Neo4j)")
    try:
        from app.graph.neo4j_store import Neo4jStore
        from app.graph.entity_extractor import extract_from_chunks
        graph = Neo4jStore()
        graph.init_schema()
        graph.write_chunks(chunks)
        frag = extract_from_chunks(chunks)
        lookup = {(c.doc_id, c.page_num): c.chunk_id for c in chunks}
        graph.write_graph(frag, lookup)
        ok(f"{len(frag.entities)} entities, {len(frag.relationships)} relationships")
    except Exception as exc:
        warn(f"Graph skipped (Neo4j off): {exc}")

    # 4 + 5. Analytics + report
    step("Stage 4/6 — Extract financial metrics")
    try:
        from app.analytics.metrics_extractor import extract_metrics
        from app.analytics.kpi_engine import compute_kpis
        metrics = extract_metrics(chunks)
        kpi = compute_kpis(metrics)
        ok(f"{len(metrics)} metrics → {len(kpi.insights)} insights")

        step("Stage 5/6 — Compute KPIs")
        for ins in kpi.insights[:4]:
            print(f"    • {ins}")

        step("Stage 6/6 — Generate report")
        if graph is not None:
            from app.reporting.insight_generator import generate_narrative
            from app.reporting.report_builder import build_report
            narrative = generate_narrative(kpi, graph.entity_stats())
            report = build_report(narrative, kpi, graph.entity_stats(),
                                  sorted({c.filename for c in chunks}))
            ok(f"Report saved → {report}")
        else:
            warn("Report skipped — needs Neo4j for graph stats.")
    except Exception as exc:
        warn(f"Analytics skipped: {exc}")

    banner("PIPELINE COMPLETE")
    print("  Try a query:  python run_all.py serve   → open http://localhost:8501")
    if graph:
        graph.close()


# ──────────────────────────────────────────────────────────────────────
# MODE: serve — launch API + frontend together
# ──────────────────────────────────────────────────────────────────────

def mode_serve():
    from app.config import settings
    banner("LAUNCHING API + FRONTEND")

    api = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", settings.api_host, "--port", str(settings.api_port),
    ])
    ok(f"API starting on http://localhost:{settings.api_port}  (docs: /docs)")
    time.sleep(3)

    frontend = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.port", str(settings.streamlit_port),
    ])
    ok(f"Frontend starting on http://localhost:{settings.streamlit_port}")

    print("\n  Both running. Press Ctrl+C to stop both.\n")
    try:
        api.wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
        frontend.terminate()
        api.terminate()


# ──────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    mode = args[0] if args else "help"

    if mode == "check":
        mode_check()
    elif mode == "pipeline":
        ready = mode_check()
        if not (ready["deps"] and ready["ollama"]):
            fail("Cannot run pipeline — fix pre-flight failures above.")
            return
        pdf = args[1] if len(args) > 1 else None
        mode_pipeline(pdf)
    elif mode == "serve":
        mode_serve()
    else:
        print(__doc__)
        print("\nQuick start:")
        print("  python run_all.py check                  # verify setup")
        print("  python run_all.py pipeline report.pdf    # run full pipeline on a PDF")
        print("  python run_all.py serve                  # launch the web app")


if __name__ == "__main__":
    main()
