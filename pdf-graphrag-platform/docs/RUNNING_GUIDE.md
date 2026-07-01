# Running Guide — PDF RAG Assistant (No API Key)

Everything runs locally. No Anthropic account, no API key, no internet after setup.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |
| RAM | 8 GB+ | For llama3.2 (4 GB model + system overhead) |

---

## Step 1 — Install Ollama

Ollama runs open-weight LLMs locally as a background service.

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from https://ollama.com/download/windows

After install, Ollama starts automatically as a background service on port 11434.

---

## Step 2 — Pull a model

```bash
ollama pull llama3.2
```

This downloads ~2 GB once and caches it locally. Subsequent runs use the cache.

**Alternative models** (pick one based on your hardware):

| Model | Size | RAM needed | Best for |
|-------|------|-----------|----------|
| `llama3.2` | 2 GB | 8 GB | Default — good balance |
| `mistral` | 4 GB | 10 GB | Better reasoning |
| `phi3` | 2 GB | 6 GB | Fastest on CPU |
| `gemma2:2b` | 2 GB | 6 GB | Alternative to llama |

To use a different model, change `LLM_MODEL=` in `.env` to match.

---

## Step 3 — Clone and install Python deps

```bash
git clone <repo-url>
cd pdf-rag-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

No API keys. No additional configuration needed — `.env` is ready to use as-is.

---

## Step 4 — Verify Ollama is reachable

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON list of your downloaded models. If you get "connection refused", start Ollama manually:

```bash
ollama serve
```

---

## Step 5 — Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/health**

---

## Step 6 — Start the frontend (new terminal)

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501**

---

## Step 7 — Upload a PDF and ask questions

1. In the sidebar, click **"Browse files"** and select a PDF.
2. Click **"Ingest selected files"** — wait for the green success messages.
3. Type a question in the chat input.
4. The answer appears with source citations and latency metrics.

---

## Docker (all-in-one)

Docker Compose starts Ollama, the API, and the frontend together.

```bash
docker-compose up --build
```

Then pull the model into the running Ollama container (first time only):

```bash
docker exec -it $(docker ps -qf "ancestor=ollama/ollama") ollama pull llama3.2
```

Services:
- Ollama: http://localhost:11434
- API: http://localhost:8000
- Frontend: http://localhost:8501

---

## CLI Usage

### Upload a PDF
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main conclusions?"}'
```

### Streaming answer
```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the methodology"}' \
  --no-buffer
```

### Index status
```bash
curl http://localhost:8000/api/v1/status
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Resetting the Index

```bash
rm -rf data/vector_db/ data/processed/ data/raw_pdfs/
```

Directories are auto-recreated on next startup.

---

## Troubleshooting

### "Cannot reach Ollama at http://localhost:11434"
Ollama is not running. Fix:
```bash
ollama serve          # starts the Ollama server
```
Or on macOS, open the Ollama app from your Applications folder.

### "model not found"
You haven't pulled the model yet:
```bash
ollama pull llama3.2
```
Or the model name in `.env` doesn't match what you pulled. Check with:
```bash
ollama list
```

### Slow first response
The model loads into RAM on the first request (~5–15 s). Subsequent requests in the same session are faster because the model stays loaded.

### Out of memory
Switch to a smaller model:
```bash
ollama pull phi3
# then set LLM_MODEL=phi3 in .env
```

### "Vector store not found"
No PDFs have been indexed yet. Upload at least one PDF via the UI or `/ingest` endpoint before querying.

### Port already in use
```bash
# Kill port 8000
lsof -ti:8000 | xargs kill -9   # macOS/Linux
# Kill port 8501
lsof -ti:8501 | xargs kill -9
```
