# PDF GraphRAG Platform

A production-oriented backend platform that turns unstructured PDFs into a
queryable knowledge base, with **async ingestion**, **Redis caching**,
**rate limiting**, **API-key auth**, and a **pluggable LLM backend**
( local Ollama or hosted OpenAI/Anthropic ).

Built with **Python · FastAPI · Redis · Neo4j · FAISS · Docker**.

> This is the SDE-oriented evolution of the GraphRAG project. The retrieval
> core is the same; everything around it is hardened into a real service.
> See `docs/MIGRATION_FROM_V2.md` for exactly what changed and why.

---

## Architecture at a glance

```
                   ┌─────────── FastAPI (REST API) ───────────┐
   client ──upload──▶ /ingest ──enqueue──▶ Redis job queue     │
                   │                          │                │
                   │              background worker thread     │
                   │                          ▼                │
                   │        extract → chunk → embed → graph    │
                   │              │                 │          │
   client ──poll───▶ /jobs/{id}   ▼                 ▼          │
                   │           FAISS index      Neo4j graph    │
   client ──ask────▶ /query ◀──cache◀── hybrid retrieval ──────┘
                   └───────────────────────────────────────────┘
   cross-cutting: Redis cache · rate limiting · API-key auth
```

---

## Key engineering features

- **Async ingestion** — slow per-chunk LLM work runs in a background worker; the API returns a job id instantly ( HTTP 202 ) and the client polls for status
- **Content-hash caching** — re-uploading the same PDF or repeating a query is served from Redis, never recomputed
- **Rate limiting** — per-client fixed-window limits protect against cost blowups and abuse
- **API-key auth** — optional, toggled via config; open for local dev, locked for deployment
- **Graceful degradation** — if Redis is down: sync ingestion, no cache, fail-open limiting; if Neo4j is down: vector-only retrieval. The system never hard-fails on a missing dependency
- **Pluggable LLM** — one config switch between local Ollama and hosted OpenAI/Anthropic
- **Layered architecture** — thin HTTP handlers, business logic in services, shared by both the worker and the sync fallback

---

## Single Entry Point — `run_all.py`

One file runs everything. Three modes:

```bash
python run_all.py check                  # verify Ollama, Neo4j, deps are ready
python run_all.py pipeline report.pdf    # run the FULL pipeline on a PDF, no servers
python run_all.py serve                  # launch API + Streamlit frontend together
```

- `check` — pre-flight: confirms the local LLM, knowledge graph, and packages are ready
- `pipeline` — ingest → embed → graph → analytics → report, end-to-end in one process (great for a CLI demo or generating a report without the web UI)
- `serve` — starts the FastAPI backend and Streamlit frontend together

Run `python run_all.py` with no arguments to see help.

---

## Quick Start

```bash
# 1. Install Ollama and pull a model
#    https://ollama.com/download
ollama pull llama3.2

# 2. Install Python deps
git clone <repo>
cd pdf-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the API (terminal 1)
uvicorn app.main:app --reload

# 4. Start the frontend (terminal 2)
streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501**, upload a PDF, ask questions.

No `.env` changes needed — defaults work out of the box.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/RUNNING_GUIDE.md`](docs/RUNNING_GUIDE.md) | Setup, Docker, CLI, troubleshooting |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Every module explained |
| [`docs/PIPELINE_DIAGRAM.svg`](docs/PIPELINE_DIAGRAM.svg) | Full pipeline flowchart |

---

## Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Vector Store
                                                              ↕
User Query → Guardrail → Query Embedding → Similarity Search → Reranking
                                                              ↓
                               Context Assembly → Ollama LLM → Answer + Citations
```

---

## Project Structure

```
pdf-rag-assistant/
├── app/
│   ├── main.py                  FastAPI entry point
│   ├── config.py                Settings (no API key needed)
│   ├── prompts.py               LLM prompt templates
│   ├── ingestion/
│   │   ├── loader.py            PDF → PageRecord
│   │   ├── chunker.py           PageRecord → Chunk (sliding window)
│   │   ├── embedder.py          Chunk → float32 vectors
│   │   └── vector_store.py      FAISS IndexFlatIP
│   ├── retrieval/
│   │   ├── retriever.py         Query → top-K candidates
│   │   ├── reranker.py          Cross-encoder reranking
│   │   └── citation_builder.py  Chunks → context + citations
│   ├── generation/
│   │   ├── llm.py               Ollama HTTP wrapper
│   │   ├── guardrails.py        Rule-based safety filter
│   │   └── response_generator.py  RAG orchestrator
│   ├── utils/
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   └── pdf_utils.py
│   └── api/routes.py            FastAPI endpoints
├── data/
│   ├── raw_pdfs/
│   ├── processed/
│   └── vector_db/
├── frontend/streamlit_app.py
├── docs/
│   ├── RUNNING_GUIDE.md
│   ├── ARCHITECTURE.md
│   └── PIPELINE_DIAGRAM.svg
├── tests/
├── .env                         No secrets — safe to inspect
├── requirements.txt             No anthropic package
├── Dockerfile
└── docker-compose.yml           Includes Ollama service
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.2` | Model name (must be pulled first) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | FAISS retrieval candidates |
| `RERANK_TOP_K` | `3` | Chunks fed to the LLM |
| `TEMPERATURE` | `0.2` | LLM temperature |

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Upload & index a PDF |
| `POST` | `/api/v1/query` | Ask a question → JSON |
| `POST` | `/api/v1/query/stream` | Streaming answer |
| `GET` | `/api/v1/status` | Index stats |

Interactive docs: **http://localhost:8000/docs**

---

## Models

```bash
ollama pull llama3.2     # default, ~2 GB
ollama pull mistral      # stronger reasoning, ~4 GB
ollama pull phi3         # fastest on CPU, ~2 GB
```

Change `LLM_MODEL=` in `.env` to switch.

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker-compose up --build
# First time: pull the model into the Ollama container
docker exec -it $(docker ps -qf "ancestor=ollama/ollama") ollama pull llama3.2
```
