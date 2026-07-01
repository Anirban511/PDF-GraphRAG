# GraphRAG + Business Analytics — Architecture Guide

This document explains the two major layers added on top of the base RAG system:
a **Neo4j knowledge-graph layer** (turning RAG into GraphRAG) and a
**business analytics layer** (turning Q&A into a report-generating product).

---

## 1. Why GraphRAG, not just RAG

Plain vector RAG retrieves chunks that are *semantically similar to the query
text*. That works for direct questions but fails on **relational, multi-hop
questions** common in financial documents:

> "Which companies did the firm that acquired Acme later partner with, and what
> were their reported revenues?"

The answer is spread across chunks that may share *no wording* with the query.
The connection lives in the **relationships between entities**, not in text
similarity. A knowledge graph captures those relationships explicitly, so
retrieval can *traverse* them.

### The hybrid retrieval strategy

| Strategy | Finds | Good at | Weak at |
|----------|-------|---------|---------|
| Vector (FAISS) | chunks similar to query text | paraphrase, fuzzy topics | multi-hop, relational |
| Graph (Neo4j) | chunks about *connected* entities | relationships, traversal | open-ended/fuzzy queries |
| **Hybrid (both)** | union of the above | both question types | — |

`GraphRetriever` runs both, deduplicates by chunk ID, and passes the union to
the reranker. This is what makes the system **GraphRAG** rather than "RAG with a
graph sitting beside it."

---

## 2. The knowledge-graph schema

Stored in Neo4j:

```
(:Entity   {name, type})            ← ORG, PERSON, MONEY, METRIC, DATE, PRODUCT, LOCATION
(:Chunk    {chunk_id, text, page, filename, doc_id})
(:Document {doc_id, filename})

(Entity)-[:MENTIONED_IN]->(Chunk)        ← provenance (keeps answers citable)
(Chunk)-[:PART_OF]->(Document)
(Entity)-[:REL {type, page, filename}]->(Entity)   ← the extracted facts
```

The `:REL` edges (ACQUIRED, OWNS, REPORTED, PARTNERED_WITH, INVESTED_IN, …) are
the heart of GraphRAG retrieval. The `MENTIONED_IN` edges preserve the same
citation guarantee as vector RAG — every graph answer still traces to a page.

### Why MERGE, not CREATE
Re-ingesting a document must not duplicate nodes. `MERGE` on `name` means
"Acme Corp" mentioned on five pages becomes **one** node linked to five chunks.

---

## 3. The GraphRAG ingestion flow

```
PDF → chunks (existing pipeline)
        │
        ├──► embed → FAISS              (existing vector path)
        │
        └──► entity_extractor (LLM)     (new graph path)
               │  extracts {entities, relationships} as JSON per chunk
               ▼
             neo4j_store.write_graph()
               creates Entity/Chunk/Document nodes + REL edges
```

Entity extraction uses the **same local LLM** (Llama via Ollama) with
`temperature=0` for deterministic JSON output. A malformed response on one chunk
is logged and skipped — never crashes the ingest.

---

## 4. The GraphRAG query flow

```
question
   │
   ├─ vector arm:  embed query → FAISS top-k
   │
   └─ graph arm:   LLM extracts entity names from query
                   → fuzzy-match to graph seed nodes
                   → traverse REL edges up to GRAPH_HOPS
                   → collect MENTIONED_IN chunks
   │
   ▼
fuse + dedupe → rerank (cross-encoder) → LLM answer + citations
```

`GRAPH_HOPS` (default 2) controls how far the traversal walks from a seed
entity. `GRAPH_MAX_CHUNKS` caps how many chunks the graph arm contributes, so
the graph cannot flood the context.

---

## 5. The business analytics layer

This is the "generates an output" half. It reuses the same chunks and graph but
produces a **structured, quantitative report** instead of a chat answer.

```
chunks ──► metrics_extractor   → MetricRecord[]  (entity, metric, value, unit, period, source)
              │
              ▼
           kpi_engine          → KPISummary      (totals, per-entity, trends, ranked insights)
              │
graph stats ──┤
              ▼
           insight_generator   → executive narrative (LLM, grounded in KPIs only)
              │
              ▼
           report_builder      → report_YYYYMMDD.docx
```

### What each module does

| Module | Responsibility | Why |
|--------|----------------|-----|
| `metrics_extractor.py` | Pull numeric financial facts into a clean table | Figures appear in many phrasings ("$2.5M", "2.5 million"); LLM normalises them |
| `kpi_engine.py` | Aggregate metrics into KPIs + insights (pandas) | Turns rows into decisions: who grew, totals, concentration, outliers |
| `insight_generator.py` | Write an exec summary grounded in the KPIs | Numbers need a narrative; LLM summarises, never invents figures |
| `report_builder.py` | Render Word report with tables + matplotlib charts | A shareable, auditable deliverable a stakeholder can read |

### Grounding discipline (anti-hallucination)
The narrative LLM receives **only** the computed KPI summary and graph stats and
is instructed to summarise — not to add outside knowledge or invent figures.
Same discipline as the RAG answer path. Every metric in the report keeps its
source filename + page, so the analytics remain auditable — non-negotiable for
finance.

---

## 6. The generated report contains

1. **Executive summary** — LLM narrative grounded in the KPIs
2. **Key financial metrics** — totals table + bar chart
3. **Top entities by significance** — ranked table + bar chart
4. **Knowledge-graph overview** — entity/relationship counts + network hubs
5. **Derived insights** — growth %, concentration, headline figures
6. **Source documents** — full provenance list

---

## 7. Graceful degradation

If Neo4j is unavailable, the system logs a warning and runs **vector-only RAG**.
The graph and analytics endpoints return 503, but Q&A still works. This means
the project never hard-fails on a missing graph database — a deliberate
resilience choice.

---

## 8. Endpoints summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/ingest` | PDF → vector index + knowledge graph |
| POST | `/api/v1/query` | Hybrid GraphRAG question answering |
| POST | `/api/v1/report` | Generate the analytics report (Word download) |
| GET | `/api/v1/graph` | Knowledge-graph nodes + edges (visualization) |
| GET | `/api/v1/graph/stats` | Entity/relationship statistics |
| GET | `/api/v1/status` | Vector + graph readiness |

---

## 9. Running it

```bash
# Neo4j + Ollama + API + frontend, all together:
docker-compose up --build

# Pull the model into the Ollama container (first time):
docker exec -it $(docker ps -qf "ancestor=ollama/ollama") ollama pull llama3.2

# Neo4j browser:   http://localhost:7474   (neo4j / password123)
# API docs:        http://localhost:8000/docs
# Frontend:        http://localhost:8501
```

Local (without Docker) additionally requires a running Neo4j instance — either
Neo4j Desktop or `docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5.26-community`.
