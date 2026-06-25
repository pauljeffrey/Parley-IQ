# Parley-IQ

A **data pipeline** and **AI analytics agent** that turn clinical chat transcripts/conversations into **structured analysis**, store them in your database, and answer **natural-language questions** and perform tasks (i.e data analysis, plot graphs) about the results.

Point it at your own relational database, align table and column names via config, and run batch enrichment plus an HTTP API for on-demand insights.

---

## What it does

| Layer | Responsibility |
|--------|----------------|
| **Ingestion pipeline** | Reads user-AI conversations from configurable tables, sends them to an LLM in batch, parses structured outputs. |
| **Persistence** | Writes one row per topic segment (extracted information from the ingestion pipeline) to an analysis table (schema and names are configurable). |
| **Analytics agent** | Answers questions in plain English using safe SQL, guarded Python, and chart generation via dedicated plot tools. |

---

## Key capabilities
- **Structured enrichment** — Conversation transcripts become validated, typed records—not loose summaries.
- **Pluggable datastore** — MySQL or PostgreSQL via SQLAlchemy; connection and table names from environment config.
- **Natural language analytics** — Ask questions in English; the agent picks tools instead of one-off scripts.
- **Operational API** — FastAPI chat endpoint with history continuity and optional zipped plot downloads.

---

## Decision & idea choices

- **Analyze once, query many times**: Batch enrichment runs offline; the analytics layer reads stored rows. Keeps cost predictable and answers fast.
- **Structured output over prose**: A rich Pydantic schema (clinical category, urgency, symptoms, SDoH, intent scores) makes downstream SQL and dashboards reliable.
- **Segment-level grain**: Each topic within a conversation gets its own row, so trends like referral rates or urgency mix are easy to aggregate.
- **Honest extraction**: The model is instructed not to invent facts; uncertain fields stay empty rather than guessed.
- **Bring your own database**: Table and column names are env-driven so the pipeline fits existing deployments without code forks.

---

## Architectural choices

```
Conversations (DB)  →  Batch pipeline (run.py)  →  Analysis table (DB)
                                                          ↓
                                              Analytics agent (FastAPI)
                                                          ↓
                                              SQL / Python / plots
```

- **Two clear stages**: `batch_processor.py` handles ingestion; `analytics_agent.py` handles interactive Q&A and tasks. Each stage has a single job.
- **Sharded batch jobs**: Large backfills split into shards (configurable size), submitted in parallel, persisted as each shard completes.
- **Guarded analytics tools**: Read-only SQL is validated (no writes, no joins). Python runs in a sandbox with capped output. Plots save to a per-request temp directory.
- **Config-first**: DB engine, schema, table names, batch limits, and model settings all live in `.env`.

---

## Workflow

Parley-IQ uses batch processing for information extraction across all user-AI conversations and an analytic gents, not one general-purpose bot:

| Agent | When it runs | Role |
|--------|----------------|------|
| **Extraction agent** | Batch pipeline | Reads full conversationspts, returns JSON matching the clinical schema. |
| **Analytics agent** | HTTP API | Chooses among SQL, Python, and plotting tools to answer a user question. |

The batch orchestrator (`run.py`) coordinates shard submission, polling, and persistence. The analytics agent runs a tool loop per chat turn—fetch data, compute, optionally chart, then reply.

---

## Evaluation of User-AI conversations by Parley IQ

Quality is built into the schema and the pipeline:

- **Per-intent scoring** — Each segment can carry accuracy, safety, helpfulness, and literacy scores, plus flags like missed red-flag checks.
- **Validation at ingest** — Batch output is parsed against the schema; malformed or empty segments are skipped, not silently stored.
- **Run stats** — Each batch reports inserted vs skipped rows; a local cache tracks successfully persisted sessions.
- **End-to-end test** — `test.py` exercises the full path: pending sessions → batch → wait → DB insert.

---

## Edge cases

- **Short conversations** — Sessions below a minimum turn count are skipped.
- **Already processed** — Completed sessions are skipped via DB lookup and a file cache; failed rows are *not* cached so they can retry.
- **Schema drift from the model** — Common alias keys (e.g. `segments` → `topic_segments`) are normalized before validation.
- **Strict JSON failures** — A fallback coercion step tries to salvage partial output; otherwise the row is dropped.
- **Missing metadata** — Rows without topic segments or a conversation start time are not inserted.
- **Chat history formats** — The API accepts native agent message JSON or simple `{role, content}` objects.

---

## Engineering bottlenecks

- **Batch latency** — OpenAI Batch jobs can take hours; the pipeline polls and persists incrementally per shard.
- **Schema strictness** — Tight JSON validation improves data quality but increases skip rate when the model drifts.
- **Transcript size** — Long multi-turn chats consume more tokens per session and slow large backfills.
- **Sandbox limits** — Analytics Python is restricted for safety, so very heavy stats may need pre-aggregated SQL instead.

---

## Trade-offs

| Choice | Upside | Downside |
|--------|--------|----------|
| Batch API vs real-time | Lower cost, scales to large backfills | Slower turnaround |
| Strict typed schema | Consistent analytics and SQL | Some valid extractions get dropped |
| Segment-level rows | Rich, filterable data | More rows than one-row-per-chat |
| Sandboxed Python | Safer in production | Not a full notebook environment |
| Single-table SQL gateway | Simple and safe | No cross-table joins in agent queries |

---

## Repository layout

- `run.py` — Batch orchestrator entry point.
- `batch_processor.py` — Shard building, OpenAI Batch submit/poll, persist.
- `analytics_agent.py` — Tool-equipped agent for NL analytics.
- `api/api.py` — FastAPI HTTP layer.
- `output.py` — Clinical extraction schema (Pydantic).
- `data_analysis/` — Exploratory pipeline material.

---

## Configuration

Configure via `.env` (see `.env.example`). Typical areas:

1. **Database** — Connection URL, schema, conversation and analysis table names.
2. **Model provider** — API key and model name for batch and analytics.
3. **Batch behavior** — Shard size, date filters, min turns, polling, work directory.

Treat API keys as production secrets.

---

## Running the analytics API

Install dependencies (`requirements.txt`), set environment variables, then:

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000
```

POST to `/analytics/chat` with a message and optional chat history.

**Response fields**

| Field | Purpose |
|--------|---------|
| `output` | Agent's natural-language answer |
| `new_messages` | Turn history for the next request |
| `figures` | Saved charts as `{ filename, content_type, data_base64, … }` for inline display |
| `figures_zip_base64` | Zip of all saved charts (when `figures` is non-empty) |

---

## Running the batch pipeline

```bash
python run.py
```

Processes pending sessions (respecting skip rules and date filters), submits sharded batch jobs, and writes results to the analysis table.

---

## Extending Parley-IQ

- Add SQLAlchemy-supported database drivers.
- Tighten or relax analytics guardrails to match your org's policies.
- Add auth, rate limits, or tenancy middleware in front of FastAPI.

---

*Built for teams that already store conversation traffic—repeatable enrichment plus a natural-language layer on structured results.*
