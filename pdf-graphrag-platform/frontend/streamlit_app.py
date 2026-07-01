"""
Streamlit frontend — GraphRAG + Analytics edition.

Three tabs:
  💬 Chat       — ask questions (hybrid GraphRAG)
  🕸️ Graph      — interactive knowledge-graph visualization
  📊 Analytics  — generate & download the business report

Run: streamlit run frontend/streamlit_app.py
"""

import os
import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"

st.set_page_config(page_title="PDF GraphRAG Analytics", page_icon="🕸️", layout="wide")


def api_status():
    try:
        return httpx.get(f"{API_V1}/status", timeout=5).json()
    except Exception:
        return {"vector_ready": False, "graph_enabled": False, "error": "API unreachable"}


def upload_pdf(file_bytes, filename):
    """Enqueue ingestion, then poll the job until done (async API)."""
    import time
    r = httpx.post(f"{API_V1}/ingest",
                   files={"file": (filename, file_bytes, "application/pdf")},
                   timeout=60)
    r.raise_for_status()
    data = r.json()
    # Synchronous fallback (no Redis): result returned immediately
    if "job_id" not in data:
        return data
    # Async path: poll the job
    job_id = data["job_id"]
    for _ in range(600):  # up to ~10 min
        time.sleep(1)
        jr = httpx.get(f"{API_V1}/jobs/{job_id}", timeout=30).json()
        if jr.get("status") == "completed":
            return jr.get("result", {})
        if jr.get("status") == "failed":
            raise RuntimeError(jr.get("error", "ingestion failed"))
    raise TimeoutError("Ingestion job did not finish in time")


def ask(query):
    r = httpx.post(f"{API_V1}/query", json={"query": query}, timeout=120)
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕸️ GraphRAG Analytics")
    st.caption("PDFs → Knowledge Graph → Insights")

    status = api_status()
    if status.get("vector_ready"):
        st.success(f"✅ {status.get('total_chunks', '?')} chunks indexed")
    else:
        st.warning("No documents indexed yet")
    if status.get("graph_enabled"):
        g = status.get("graph", {})
        st.info(f"🕸️ Graph: {g.get('total_entities', 0)} entities, "
                f"{g.get('total_relationships', 0)} relationships")
    else:
        st.error("Neo4j offline — vector-only mode")

    st.divider()
    st.subheader("📤 Upload PDFs")
    files = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                             label_visibility="collapsed")
    if files and st.button("Ingest", type="primary", use_container_width=True):
        for f in files:
            with st.spinner(f"Ingesting {f.name} (extraction + graph build)…"):
                try:
                    res = upload_pdf(f.read(), f.name)
                    st.success(f"✔ {res['filename']}: {res['chunks']} chunks, "
                               f"{res['graph']['entities']} entities")
                except Exception as exc:
                    st.error(f"{exc}")
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_graph, tab_analytics, tab_eval = st.tabs(["💬 Chat", "🕸️ Graph", "📊 Analytics", "🎯 Accuracy"])

# --- CHAT ---
with tab_chat:
    st.subheader("Ask your documents (Hybrid GraphRAG)")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                with st.expander(f"📚 {len(m['citations'])} sources"):
                    for c in m["citations"]:
                        st.caption(f"**{c['filename']}** — Page {c['page_num']}")

    if q := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            if not status.get("vector_ready"):
                st.warning("Upload documents first.")
            else:
                with st.spinner("Searching graph + vectors…"):
                    try:
                        res = ask(q)
                        st.markdown(res["answer"])
                        cites = res.get("citations", [])
                        if cites:
                            with st.expander(f"📚 {len(cites)} sources"):
                                for c in cites:
                                    st.caption(f"**{c['filename']}** — Page {c['page_num']}")
                        st.caption(f"⏱ {res['latency_s']:.2f}s · 📦 {res['chunks_used']} chunks")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": res["answer"], "citations": cites})
                    except Exception as exc:
                        st.error(f"{exc}")

# --- GRAPH ---
with tab_graph:
    st.subheader("Knowledge Graph")
    if not status.get("graph_enabled"):
        st.error("Neo4j is offline. Start it to view the graph.")
    else:
        if st.button("Load graph", use_container_width=True):
            try:
                data = httpx.get(f"{API_V1}/graph", params={"limit": 80}, timeout=30).json()
                try:
                    from streamlit_agraph import agraph, Node, Edge, Config
                    type_color = {"ORG": "#2E5C8A", "PERSON": "#0F6E56",
                                  "MONEY": "#993C1D", "METRIC": "#854F0B",
                                  "DATE": "#534AB7", "PRODUCT": "#993556"}
                    nodes = [Node(id=n["id"], label=n["id"][:20],
                                  color=type_color.get(n.get("type"), "#888"))
                             for n in data["nodes"]]
                    edges = [Edge(source=e["source"], target=e["target"],
                                  label=e.get("type", ""))
                             for e in data["edges"]
                             if e["source"] and e["target"]]
                    cfg = Config(width=900, height=600, directed=True,
                                 physics=True, nodeHighlightBehavior=True)
                    agraph(nodes=nodes, edges=edges, config=cfg)
                except ImportError:
                    st.json(data)
            except Exception as exc:
                st.error(f"{exc}")

# --- ANALYTICS ---
with tab_analytics:
    st.subheader("Business Analytics Report")
    st.write("Generate an executive report with KPIs, charts, entity rankings, "
             "and graph insights — auto-extracted from your documents.")
    if not status.get("graph_enabled"):
        st.error("Neo4j is required for analytics.")
    elif st.button("📊 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Extracting metrics → computing KPIs → writing report…"):
            try:
                r = httpx.post(f"{API_V1}/report", timeout=900)
                r.raise_for_status()
                st.success("Report generated.")
                st.download_button("⬇️ Download Word Report", data=r.content,
                                   file_name="analytics_report.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   use_container_width=True)
            except Exception as exc:
                st.error(f"{exc}")

    if status.get("graph_enabled") and status.get("graph"):
        st.divider()
        st.caption("Current graph statistics")
        g = status["graph"]
        c1, c2 = st.columns(2)
        c1.metric("Entities", g.get("total_entities", 0))
        c2.metric("Relationships", g.get("total_relationships", 0))
        if g.get("by_type"):
            st.bar_chart({row["type"]: row["count"] for row in g["by_type"]})

# --- ACCURACY EVALUATION ---
with tab_eval:
    st.subheader("Extraction Accuracy Evaluation")
    st.write(
        "Real, measured accuracy of the financial metric extraction — "
        "computed by comparing pipeline output against hand-verified ground truth."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load latest results", use_container_width=True):
            try:
                r = httpx.get(f"{API_V1}/evaluate/results", timeout=15)
                r.raise_for_status()
                data = r.json()
                sm = data.get("summary", {})
                st.success(f"Evaluation run: {data.get('run', '?')}")

                st.divider()
                st.caption("Strict (value within 1% of truth)")
                c1, c2, c3 = st.columns(3)
                c1.metric("F1 Score",    f"{sm.get('f1_strict', 0):.1%}")
                c2.metric("Precision",   f"{sm.get('precision_strict', 0):.1%}")
                c3.metric("Recall",      f"{sm.get('recall_strict', 0):.1%}")

                st.divider()
                st.caption("Lenient (value within 5% of truth)")
                c4, c5, c6 = st.columns(3)
                c4.metric("F1 Score",    f"{sm.get('f1_lenient', 0):.1%}")
                c5.metric("Precision",   f"{sm.get('precision_lenient', 0):.1%}")
                c6.metric("Recall",      f"{sm.get('recall_lenient', 0):.1%}")

                st.divider()
                st.caption("Per-record breakdown")
                records = data.get("per_record", [])
                if records:
                    import pandas as pd
                    df = pd.DataFrame(records)[
                        ["entity", "metric", "extracted_value", "truth_value",
                         "strict_hit", "label"]
                    ]
                    st.dataframe(df, use_container_width=True)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    st.warning("No evaluation run yet. Follow the steps below.")
                else:
                    st.error(str(e))
            except Exception as e:
                st.error(str(e))

    with col2:
        st.info("""
**How to get your accuracy number:**

1. Run the template generator:
```
python run_evaluation.py template
```

2. Open `data/ground_truth/ground_truth_template.json`
   and fill in real figures from your PDF (10-20 entries)

3. Run the evaluation:
```
python run_evaluation.py
```

4. Come back here and click **Load latest results**

The F1 score is your headline metric.
        """)

