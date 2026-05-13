# Parley-IQ

A comprehensive **data pipeline** and **AI analytics agent** that turn clinical chat (conversation) transcripts into **structured analysis**, persist them to your database, and deliver **real-time, dynamic, data-driven insights** from analyzed conversations via **natural language** (through an analytics agent).

The project is designed so you can point it at **your own relational database**, align table and column conventions with your schema, and run batch enrichment plus an HTTP API that answers analytic questions grounded in stored results.

---

## What it does

| Layer | Responsibility |
|--------|----------------|
| **Ingestion pipeline** | Reads conversation data from configurable tables, submits work to an LLM provider in batch shape, parses structured outputs. |
| **Persistence** | Writes analysis rows to a designated analysis table (schema and names are configurable). |
| **Analytics agent** | Exposes tools for SQL‑safe querying, guarded Python for deeper stats, and plot generation; optional packaged figure downloads over the API. |

---

## Key capabilities

- **Structured enrichment** — Turn transcripts into validated, typed analysis records rather than loose prose.
- **Pluggable datastore** — Use MySQL or PostgreSQL via SQLAlchemy; connection strings and identifiers come from configuration.
- **Natural language analytics** — Ask questions in plain English; the agent uses tools instead of brittle one-off scripts.
- **Operational API** — FastAPI endpoints for conversational analytics with chat history continuity and optional zipped plot artifacts.

---

## Repository layout (high level)

- Root scripts drive **batch orchestration**, **database access**, and **API hosting**.
- `data_analysis/` holds exploratory or notebook-oriented pipeline material.
- Environment-driven settings (including model keys and DB URLs) keep deployments environment-specific without code changes.

---

## Configuration

Configure your environment via `.env` (or your host’s secret store). Typical areas include:

1. **Database** — Connection URL plus schema/table identifiers that match **your** deployment.
2. **Model provider** — API key and model name for batch jobs and/or the analytics agent.
3. **Batch behavior** — Limits, polling, paths for batch input/output when running large runs.

Treat API keys like production secrets: rotate if exposed and restrict file permissions.

---

## Running the analytics API

Install dependencies (`requirements.txt`), set environment variables, then start the HTTP server—for example:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Use the documented chat endpoint to send a message and conversation history; when the agent produces charts, encoded archive payloads can be returned for download workflows.

---

## Extending Parley-IQ

- Swap or add **drivers** while staying within SQLAlchemy-supported backends.
- Tighten or relax **guardrails** on the analytics tools to match your org’s SQL and execution policies.
- Add **middleware** (auth, rate limits, tenancy) in front of the FastAPI layer for enterprise rollouts.

---

*Parley-IQ suits teams that already own conversation traffic and storage—repeatable enrichment plus an NL layer on structured results.*
