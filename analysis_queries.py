"""Read-side helpers for `aisha_conversation_analysis` (safe SELECT + dataframe loading)."""

from __future__ import annotations

import io
import os
import re
from typing import Any, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db import ANALYSIS_TABLE_NAME, SCHEMA_NAME, qualified_table_name


def analysis_table_must_match_substrings() -> tuple[str, str]:
    """Two substrings SQL must contain (schema + logical table name); matches typical qualified names."""
    return SCHEMA_NAME.strip(), ANALYSIS_TABLE_NAME.strip()


ANALYSIS_SELECTABLE_COLUMNS = frozenset(
    {
        "id",
        "session_id",
        "user_phone",
        "model_name",
        "created_at",
        "conversation_id",
        "analysis_timestamp",
        "user_persona",
        "segments",
        "sdoh_economic_barrier",
        "sdoh_geographic_barrier",
        "sdoh_social_barrier",
        "outcome_referral",
    }
)

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"EXEC|EXECUTE|CALL|COMMENT|COPY|DETACH)\b",
    re.IGNORECASE,
)

_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)


def schema_documentation(engine: Engine) -> str:
    """Human-readable DDL-style description for prompting."""
    fq = qualified_table_name(engine, ANALYSIS_TABLE_NAME)
    return f"""Data lives in PostgreSQL/MySQL qualified table fragment: {fq}

Expected columns:
- id: optional BIGINT auto key (may be absent depending on DDL)
- session_id: BIGINT
- user_phone: TEXT/VARCHAR — caller phone identifier
- model_name: TEXT — model label used during analysis ingestion
- created_at: TIMESTAMP — row insert time (UTC-ish)
- conversation_id: BIGINT — echoed analysis conversation id from extract
- analysis_timestamp: TIMESTAMP — analysis event time stored in structured output
- user_persona: TEXT — synthesized persona snapshot
- segments: JSON/Text — serialized segment analyses (opaque JSON blob)
- sdoh_economic_barrier: BOOLEAN
- sdoh_geographic_barrier: BOOLEAN
- sdoh_social_barrier: BOOLEAN
- outcome_referral: BOOLEAN — whether referral was suggested

Prefer aggregate queries (`COUNT`, grouped metrics) rather than dumping very large blobs.
"""


def validate_read_select(sql: str) -> str:
    """
    Lightweight guardrails for a single SELECT (or SELECT wrapped in WITH) against this table only.
    """
    trimmed = sql.strip().rstrip(";")
    flattened = " ".join(trimmed.split())
    lowered = flattened.lower()

    schema_token, table_token = analysis_table_must_match_substrings()
    if schema_token.lower() not in lowered or table_token.lower() not in lowered:
        raise ValueError(
            f"SQL must reference schema {schema_token!r} and analysis table "
            f"{table_token!r} (identifiers may be quoted)."
        )

    starts_ok = lowered.startswith("select") or lowered.startswith("with")
    if not starts_ok:
        raise ValueError("Only read-only SELECT (optionally prefixed with WITH) is allowed.")

    if ";" in trimmed:
        raise ValueError("Chaining statements with ';' is not allowed.")

    if _JOIN_PATTERN.search(flattened):
        raise ValueError("JOIN is not allowed via this gateway; derive metrics in pandas or narrower queries.")

    if _FORBIDDEN_SQL.search(flattened):
        raise ValueError("Detected a forbidden SQL verb (writes / DDL).")

    for frag in ("information_schema", "pg_catalog", "\\copy", " pg_", "sqlite_master"):
        if frag.lower() in lowered:
            raise ValueError(f"Subsystem {frag!r} lookups are blocked.")

    return flattened


def run_validated_select(
    *,
    sql: str,
    engine: Engine,
    bind_params: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    cap = max_rows or int(os.environ.get("ANALYTICS_MAX_QUERY_ROWS", "2000"))
    validated = validate_read_select(sql)
    lim = validated.lower().rstrip()
    limited_sql = validated
    if " limit " not in lim[-80:]:
        limited_sql = f"{validated.rstrip()} LIMIT {cap}"
    stmt = text(limited_sql)
    bind_params = bind_params or {}
    with engine.connect() as conn:
        result = conn.execute(stmt, bind_params)
        rows = result.mappings().fetchall()
    return [dict(r) for r in rows]


def load_analysis_dataframe(
    *,
    engine: Engine,
    limit: int | None = None,
) -> pd.DataFrame:
    cap = limit or int(os.environ.get("ANALYTICS_MAX_DATAFRAME_ROWS", "5000"))
    fq = qualified_table_name(engine, ANALYSIS_TABLE_NAME)
    sql = f"SELECT * FROM {fq} LIMIT {cap}"
    validate_read_select(sql)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def structured_fetch(
    *,
    engine: Engine,
    columns: Sequence[str],
    limit: int | None = None,
    where_equal: dict[str, Any] | None = None,
    order_by: str | None = None,
    descending: bool = False,
) -> list[dict[str, Any]]:
    """Equality filters only; column names are allow-listed."""
    cap = limit or int(os.environ.get("ANALYTICS_MAX_QUERY_ROWS", "2000"))
    where_equal = where_equal or {}
    if not columns:
        raise ValueError("columns must be non-empty")
    bad = set(columns) - ANALYSIS_SELECTABLE_COLUMNS
    if bad:
        raise ValueError(f"Unknown columns: {sorted(bad)}")

    for k in where_equal:
        if k not in ANALYSIS_SELECTABLE_COLUMNS:
            raise ValueError(f"Unknown filter column: {k!r}")

    if order_by and order_by not in ANALYSIS_SELECTABLE_COLUMNS:
        raise ValueError(f"Invalid order_by column: {order_by!r}")

    fq = qualified_table_name(engine, ANALYSIS_TABLE_NAME)
    col_sql = ", ".join(columns)

    binds: dict[str, Any] = {}
    clauses: list[str] = []
    for i, (col, val) in enumerate(where_equal.items()):
        key = f"w{i}"
        clauses.append(f"{col} = :{key}")
        binds[key] = val

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    dir_sql = "DESC" if descending else "ASC"
    order_sql = f"ORDER BY {order_by} {dir_sql}" if order_by else ""
    stmt = (
        f"SELECT {col_sql} FROM {fq} "
        f"{where_sql} {order_sql} LIMIT :_lim"
    ).replace("  ", " ")
    binds["_lim"] = cap
    return run_validated_select(sql=stmt, engine=engine, bind_params=binds, max_rows=cap)


def format_rows_preview(rows: list[dict[str, Any]], *, max_chars: int = 12000) -> str:
    if not rows:
        return "(no rows)"
    buf = io.StringIO()
    for i, row in enumerate(rows[:200]):
        buf.write(str(row))
        buf.write("\n")
    preview = buf.getvalue()
    if len(preview) > max_chars:
        return preview[: max_chars // 2] + "\n…[truncated]…\n" + preview[-max_chars // 2 :]
    return preview.strip()
