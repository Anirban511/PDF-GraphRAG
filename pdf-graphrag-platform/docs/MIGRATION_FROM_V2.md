# What Changed — v2 (Analytics) → v3 (SDE Platform)

This version is an independent, SDE-oriented evolution of the GraphRAG project.
The retrieval core (FAISS + Neo4j + reranking + LLM) is unchanged; what's new
is everything around it that turns a working pipeline into production software.

This doc explains, stage by stage, how the pipeline now differs.

---

## TL;DR — the five structural changes

| Area | v2 (previous) | v3 (this version) |
|------|---------------|-------------------|
| Ingestion | Blocking HTTP request (minutes) | **Async job queue** — returns instantly, poll for status |
| Repeat work | Re-ran full pipeline every time | **Redis cache** — same doc/query served from cache |
| Abuse / cost control | None | **Rate limiting** per client (Redis) |
| Auth | Open API | **Optional API-key auth** |
| Code structure | Logic inside route handlers | **Service layer** — routes thin, logic reusable |
| Analytics | Headline feature | Demoted to one endpoint among many |

---

## 1. Ingestion is now asynchronous (the biggest change)

### Before (v2)
`POST /ingest` ran the entire pipeline inside the request:
```
client → POST /ingest → [extract → chunk → embed → graph → extract metrics] → response
         (client waits minutes; request may time out; UI frozen)
```

### After (v3)
`POST /ingest` enqueues a job and returns immediately:
```
client → POST /ingest → enqueue job → 202 {job_id}        (instant)
                              ↓
                   Redis queue → background worker thread
                              ↓
client → GET /jobs/{id} → {status: processing → completed, result}
```

**Why this matters (SDE framing):** blocking a web request on a multi-minute
job is an anti-pattern — it ties up workers, times out, and gives no progress
feedback. The job-queue pattern (the same idea behind Celery / RQ / SQS) is
how real systems handle slow work. The worker is implemented as a Redis-backed
queue + a background thread (`app/core/job_queue.py`).

**New endpoint:** `GET /jobs/{job_id}` — poll status (`queued → processing →
completed | failed`).

**Graceful fallback:** if Redis is down, ingestion runs synchronously inline,
so the system still works without the queue.

---

## 2. Redis caching — no more paying twice

### New: `app/core/cache.py`

Two caches, both keyed by content hash:

- **Document cache** — every PDF has a SHA-256 `doc_id`. Before ingesting, the
  service checks `cache_exists_doc(hash)`. If the same document was already
  ingested, the **entire expensive pipeline is skipped**. This is the single
  biggest cost saving once hosted LLM APIs are used.
- **Query cache** — identical questions against the same index return the
  cached answer instantly, skipping retrieval + generation.

**Why this matters:** the previous version re-did all work on every repeat
upload or query. With hosted LLMs that's real money; even locally it's wasted
minutes. Caching by hash is the standard fix.

---

## 3. Rate limiting — cost & abuse protection

### New: `app/core/rate_limiter.py`

A fixed-window counter in Redis caps requests per client per time window
(default 30/min). Over the limit → HTTP 429 with a `Retry-After` header.

**Why this matters:** any endpoint that triggers LLM calls spends money and
compute. An open, unlimited endpoint is a cost and DoS risk. Rate limiting is
expected in any production API. The client is identified by API key (if
present) or IP (`app/core/security.py`).

**Fail-open design:** if Redis is down, the limiter allows requests rather than
blocking everything — a cache outage shouldn't take down the API.

---

## 4. API-key authentication

### New: `app/core/security.py`

`require_api_key` is a FastAPI dependency on the protected endpoints. Keys are
configured via `API_KEYS` in `.env` (comma-separated). If no keys are set, the
API is open (convenient for local dev); set keys to lock it down for deployment.

**Why this matters:** the previous version had no auth — anyone could spend its
resources. Even simple key auth is the baseline for a deployable service.

---

## 5. Service layer — separation of concerns

### New: `app/services/ingestion_service.py`

In v2, all ingestion logic lived inside the route handler, mixing HTTP concerns
with business logic. Now:

```
route handler (HTTP only) → ingestion_service (business logic) → stores
```

The service function takes plain arguments and returns plain data, so it can be
called from **two** places: the background worker (async) and the synchronous
fallback. This is the layered architecture that made the async refactor clean,
and it's far more testable.

**Why this matters:** "thin controllers, logic in services" is a core
backend-design principle. Route handlers should translate HTTP ↔ function
calls, nothing more.

---

## 6. Analytics demoted (not removed)

The business-analytics layer (metrics, KPIs, Word report) still exists as
`POST /report`, but it is no longer the headline. For an SDE profile the story
is the **platform** — async pipeline, caching, queue, rate limiting — not the
financial reporting. The analytics endpoint remains as one capability among
many rather than the project's identity.

---

## New project structure

```
app/
├── core/                  ← NEW: cross-cutting infrastructure
│   ├── redis_client.py    Redis connection + health
│   ├── cache.py           Document + query caching
│   ├── rate_limiter.py    Per-client rate limiting
│   ├── job_queue.py       Background job queue + worker
│   └── security.py        API-key auth + client identification
├── services/              ← NEW: business logic layer
│   └── ingestion_service.py
├── api/routes.py          ← thin handlers (async ingest, cached query)
├── main.py                ← starts background worker via lifespan
├── ingestion/             (unchanged)
├── retrieval/             (unchanged)
├── generation/            (unchanged — pluggable LLM provider)
├── graph/                 (unchanged — Neo4j GraphRAG)
├── analytics/             (unchanged — demoted in API)
└── reporting/             (unchanged)
```

---

## New dependency

```
redis==5.2.1     # caching, rate limiting, job queue
```

And a new container in `docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]
  healthcheck: redis-cli ping
```

---

## Running it

```bash
docker-compose up --build      # redis + neo4j + ollama + api + frontend
docker exec -it $(docker ps -qf "ancestor=ollama/ollama") ollama pull llama3.2
```

The async flow in action:
```bash
# Enqueue ingestion → get a job id back instantly
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@report.pdf"
# → {"job_id": "a1b2c3", "status": "queued", "poll": "/api/v1/jobs/a1b2c3"}

# Poll the job
curl http://localhost:8000/api/v1/jobs/a1b2c3
# → {"status": "processing", "progress": "10"}
# → {"status": "completed", "result": {...}}
```

---

## What an SDE interviewer can now probe (and you can answer)

- **"How do you handle a slow operation in a web request?"** → async job queue,
  return 202, poll for status. Explain why blocking is bad.
- **"How do you avoid redundant work?"** → content-hash caching of documents
  and queries in Redis.
- **"How do you protect the API?"** → API-key auth + per-client rate limiting.
- **"What happens if Redis goes down?"** → graceful degradation: sync ingestion,
  no cache, fail-open rate limiting. The system stays up.
- **"How is this structured?"** → layered: routes (HTTP) → services (logic) →
  stores (data). Worker and sync paths share the same service function.
- **"How would you scale the worker?"** → the queue contract is unchanged if you
  swap the in-thread worker for a separate worker process or Celery; that's the
  point of the queue abstraction.
