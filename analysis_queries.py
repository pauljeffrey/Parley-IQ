"""
Read-side helpers for the configured analysis table (safe SELECT + dataframe loading).

The analysis table stores **one row per topic segment** produced by conversation extraction.
Column names and meanings match `db.insert_conversation_analysis`.
"""

from __future__ import annotations

import io
import os
import re
from enum import Enum
from typing import Any, NamedTuple, Sequence, get_args

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db import ANALYSIS_TABLE_NAME, SCHEMA_NAME, qualified_table_name
from output import (
    AnalysisSegment,
    ClinicalTaxonomy,
    OutcomeReferral,
    SDoHBarrier,
    UrgencyLevel,
)


def _enum_values(enum_cls: type[Enum]) -> tuple[str, ...]:
    return tuple(m.value for m in enum_cls)


def _unique_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


OUTCOME_REFERRAL_VALUES: tuple[str, ...] = get_args(OutcomeReferral)
CLINICAL_CATEGORY_VALUES: tuple[str, ...] = _unique_preserve_order(_enum_values(ClinicalTaxonomy))
URGENCY_LEVEL_VALUES: tuple[str, ...] = _enum_values(UrgencyLevel)
SDOH_BARRIER_VALUES: tuple[str, ...] = _enum_values(SDoHBarrier)

_literacy_field = AnalysisSegment.model_fields["literacy_score"]
LITERACY_SCORE_MIN: int = int(_literacy_field.ge) if _literacy_field.ge is not None else 1
LITERACY_SCORE_MAX: int = int(_literacy_field.le) if _literacy_field.le is not None else 5

CATEGORICAL_COLUMN_VALUES: dict[str, frozenset[str]] = {
    "clinical_category": frozenset(CLINICAL_CATEGORY_VALUES),
    "urgency_level": frozenset(URGENCY_LEVEL_VALUES),
    "outcome_referral": frozenset(OUTCOME_REFERRAL_VALUES),
}


class ColumnSpec(NamedTuple):
    """Human-readable column metadata for agents and allow-lists."""

    name: str
    sql_type: str
    description: str


# Authoritative column catalog (order = suggested SELECT list for exploration).
ANALYSIS_COLUMN_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec("id", "BIGINT", "Optional surrogate primary key."),
    ColumnSpec("session_id", "BIGINT", "Source conversation session identifier."),
    ColumnSpec("user_phone", "TEXT", "Caller / user phone or external id."),
    ColumnSpec("model_name", "TEXT", "LLM model name used during extraction."),
    ColumnSpec(
        "segment_index",
        "INT",
        "1-based index of this topic segment within the session analysis.",
    ),
    ColumnSpec(
        "clinical_category",
        "TEXT",
        f"Clinical taxonomy (`output.ClinicalTaxonomy`). "
        f"{len(CLINICAL_CATEGORY_VALUES)} allowed labels; see schema vocabularies.",
    ),
    ColumnSpec(
        "intent",
        "JSON/TEXT",
        "IntentPerformance object: accuracy, safety, helpfulness, empathy, literacy scores.",
    ),
    ColumnSpec(
        "pharmacology_profiles",
        "JSON/TEXT",
        "Drug-related profiles for the segment (classes, compliance, side effects).",
    ),
    ColumnSpec(
        "mental_health_profiles",
        "JSON/TEXT",
        "Mental-health crisis indicators when applicable.",
    ),
    ColumnSpec(
        "suspected_condition",
        "JSON/TEXT",
        "ICD-11–style condition labels, ordered by likelihood.",
    ),
    ColumnSpec(
        "symptoms_reported",
        "JSON/TEXT",
        "Structured symptom objects reported in the segment.",
    ),
    ColumnSpec(
        "urgency_level",
        "TEXT",
        f"Triage urgency (`output.UrgencyLevel`). Use exact stored strings in SQL.",
    ),
    ColumnSpec(
        "barriers",
        "JSON/TEXT",
        f"JSON list of `output.SDoHBarrier` values for the segment.",
    ),
    ColumnSpec(
        "cultural_tags",
        "JSON/TEXT",
        "Cultural context tags for the segment.",
    ),
    ColumnSpec(
        "outcome_referral",
        "TEXT",
        "How the interaction concluded for this segment. "
        f"Allowed values: {', '.join(repr(v) for v in OUTCOME_REFERRAL_VALUES)}. "
        "Use for referral-rate and escalation analytics.",
    ),
    ColumnSpec(
        "literacy_score",
        "INT",
        f"User health literacy (`output.AnalysisSegment`): "
        f"{LITERACY_SCORE_MIN} (lowest) through {LITERACY_SCORE_MAX} (highest).",
    ),
    ColumnSpec(
        "cultural_notes",
        "JSON/TEXT",
        "Conversation-level cultural notes (`output.CulturalNotes`, repeated on each segment row).",
    ),
    ColumnSpec(
        "topics_enquired",
        "JSON/TEXT",
        "Conversation-level topics the user inquired about (`output.ConversationAnalysis.topics_enquired`).",
    ),
    ColumnSpec(
        "diseases_enquired",
        "JSON/TEXT",
        "Conversation-level diseases the user inquired about (`output.ConversationAnalysis.diseases_enquired`).",
    ),
    ColumnSpec(
        "sdoh_profiles",
        "JSON/TEXT",
        "Full list of SDoH profile objects for the conversation.",
    ),
    ColumnSpec(
        "sdoh_economic_barrier",
        "BOOLEAN",
        "True if economic/cost barriers appear in sdoh_profiles.",
    ),
    ColumnSpec(
        "sdoh_geographic_barrier",
        "BOOLEAN",
        "True if transportation/distance barriers appear.",
    ),
    ColumnSpec(
        "sdoh_social_barrier",
        "BOOLEAN",
        "True if stigma, security, childcare, or similar social barriers appear.",
    ),
    ColumnSpec(
        "created_at",
        "TIMESTAMP",
        "When this analysis row was inserted (UTC).",
    ),
    ColumnSpec(
        "analysis_timestamp",
        "TIMESTAMP",
        "Start time of the original conversation (from source turns).",
    ),
    ColumnSpec(
        "conversation_day_of_week",
        "TEXT",
        "Day name derived from analysis_timestamp (e.g. Monday).",
    ),
    ColumnSpec(
        "conversation_month",
        "INT",
        "Month number (1–12) from analysis_timestamp.",
    ),
    ColumnSpec(
        "conversation_year",
        "INT",
        "Four-digit year from analysis_timestamp.",
    ),
)

ANALYSIS_SELECTABLE_COLUMNS: frozenset[str] = frozenset(spec.name for spec in ANALYSIS_COLUMN_SPECS)

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"EXEC|EXECUTE|CALL|COMMENT|COPY|DETACH)\b",
    re.IGNORECASE,
)

_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)


def analysis_table_must_match_substrings() -> tuple[str, str]:
    """Schema and table tokens that validated SQL must reference."""
    return SCHEMA_NAME.strip(), ANALYSIS_TABLE_NAME.strip()


def conversation_analysis_columns_table() -> str:
    """Column name and SQL type grid for analytics_agent instructions."""
    header = "| column | type |\n| --- | --- |"
    rows = "\n".join(f"| {spec.name} | {spec.sql_type} |" for spec in ANALYSIS_COLUMN_SPECS)
    return f"{header}\n{rows}"


def analytics_agent_instructions(*, qualified_table: str | None = None) -> str:
    """System prompt for analytics_agent (columns, types, vocabularies, query rules)."""
    schema_token, table_token = analysis_table_must_match_substrings()
    table_ref = qualified_table or f"{schema_token}.{table_token}"
    return f"""You are **analytics_agent**. Analyze the conversation analysis table `{table_ref}`.
Grain: one row per `output.AnalysisSegment`.

## Table columns

{conversation_analysis_columns_table()}

{_format_standard_vocabularies()}

## Custom SQL (primary)

- Compose read-only SQL as one string (`SELECT` or `WITH … SELECT`) using only columns above.
- Execute it with **`execute_custom_query(sql)`** — the string is validated and run as-is.
- The query must reference schema `{schema_token}` and table `{table_token}`.
- Use exact vocabulary strings for categorical filters; prefer aggregates over wide row pulls.

## Other tools

- `structured_table_fetch` — simple equality filters on allow-listed columns.
- `run_python_for_analysis` / `plot_and_save_figure` — small result sets or capped samples only.

## PHI

Summarize aggregates; avoid unnecessary verbatim clinical content.
"""


def qualified_analysis_table(engine: Engine) -> str:
    return qualified_table_name(engine, ANALYSIS_TABLE_NAME)


def _format_value_list(values: Sequence[str], *, bullet: bool = True) -> str:
    prefix = "- " if bullet else ""
    return "\n".join(f"{prefix}`{v}`" for v in values)


def _format_standard_vocabularies() -> str:
    return f"""## Standard vocabularies (`output.py`)

Persisted segment fields use the same strings as extraction enums / literals.
Use **exact** values in `WHERE`, `GROUP BY`, and `CASE` (including punctuation and parentheses).

### clinical_category (`ClinicalTaxonomy`)

{_format_value_list(CLINICAL_CATEGORY_VALUES)}

### urgency_level (`UrgencyLevel`)

{_format_value_list(URGENCY_LEVEL_VALUES)}

### outcome_referral (`OutcomeReferral`)

{_format_value_list(OUTCOME_REFERRAL_VALUES)}

### barriers (JSON array of `SDoHBarrier`)

{_format_value_list(SDOH_BARRIER_VALUES)}

Segment-level `barriers` is JSON; session-level rollup flags: `sdoh_economic_barrier`,
`sdoh_geographic_barrier`, `sdoh_social_barrier` (see `db._sdoh_barrier_flags`).

### literacy_score (`AnalysisSegment`)

Integer {LITERACY_SCORE_MIN}–{LITERACY_SCORE_MAX} per segment.

### JSON columns (nested `output` models)

- `intent`: `IntentPerformance` (scores 0–1, boolean rubrics; `intent` uses `FunctionalIntent` labels)
- `pharmacology_profiles`: `PharmacologyProfiles` / `PharmacologyProfile`
- `mental_health_profiles`: list of `MentalHealthCrisis`
- `symptoms_reported`: list of `SymptomNature` (`StandardSymptom`, `SymptomCategory`, …)
- `cultural_tags`: `CulturalTag` with `Tag` enum values
- `sdoh_profiles`: list of `SDoHProfile` (conversation-level; repeated on each segment row)
"""


def _validate_categorical_filters(where_equal: dict[str, Any]) -> None:
    for col, val in where_equal.items():
        allowed = CATEGORICAL_COLUMN_VALUES.get(col)
        if allowed is not None and val not in allowed:
            raise ValueError(
                f"Invalid {col!r} filter {val!r}. "
                f"Use an exact value from output.py ({len(allowed)} allowed)."
            )
    if "literacy_score" in where_equal:
        score = where_equal["literacy_score"]
        if not isinstance(score, int) or not (LITERACY_SCORE_MIN <= score <= LITERACY_SCORE_MAX):
            raise ValueError(
                f"literacy_score must be an integer "
                f"between {LITERACY_SCORE_MIN} and {LITERACY_SCORE_MAX}."
            )


def validate_read_select(sql: str) -> str:
    """Lightweight guardrails for a single SELECT against the analysis table only."""
    trimmed = sql.strip().rstrip(";")
    flattened = " ".join(trimmed.split())
    lowered = flattened.lower()

    schema_token, table_token = analysis_table_must_match_substrings()
    if schema_token.lower() not in lowered or table_token.lower() not in lowered:
        raise ValueError(
            f"SQL must reference schema {schema_token!r} and analysis table "
            f"{table_token!r} (identifiers may be quoted)."
        )

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only read-only SELECT (optionally prefixed with WITH) is allowed.")

    if ";" in trimmed:
        raise ValueError("Chaining statements with ';' is not allowed.")

    if _JOIN_PATTERN.search(flattened):
        raise ValueError("JOIN is not allowed via this gateway.")

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
    """Equality filters only; column names must be in ANALYSIS_SELECTABLE_COLUMNS."""
    cap = limit or int(os.environ.get("ANALYTICS_MAX_QUERY_ROWS", "2000"))
    where_equal = where_equal or {}
    if not columns:
        raise ValueError("columns must be non-empty")

    bad = set(columns) - ANALYSIS_SELECTABLE_COLUMNS
    if bad:
        raise ValueError(
            f"Unknown columns: {sorted(bad)}. "
            f"Allowed: {sorted(ANALYSIS_SELECTABLE_COLUMNS)}"
        )

    for key in where_equal:
        if key not in ANALYSIS_SELECTABLE_COLUMNS:
            raise ValueError(f"Unknown filter column: {key!r}")

    _validate_categorical_filters(where_equal)

    if order_by and order_by not in ANALYSIS_SELECTABLE_COLUMNS:
        raise ValueError(f"Invalid order_by column: {order_by!r}")

    fq = qualified_table_name(engine, ANALYSIS_TABLE_NAME)
    col_sql = ", ".join(columns)

    binds: dict[str, Any] = {}
    clauses: list[str] = []
    for i, (col, val) in enumerate(where_equal.items()):
        bind_key = f"w{i}"
        clauses.append(f"{col} = :{bind_key}")
        binds[bind_key] = val

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    dir_sql = "DESC" if descending else "ASC"
    order_sql = f"ORDER BY {order_by} {dir_sql}" if order_by else ""
    stmt = f"SELECT {col_sql} FROM {fq} {where_sql} {order_sql} LIMIT :_lim".replace("  ", " ")
    binds["_lim"] = cap
    return run_validated_select(sql=stmt, engine=engine, bind_params=binds, max_rows=cap)


def format_rows_preview(rows: list[dict[str, Any]], *, max_chars: int = 12000) -> str:
    if not rows:
        return "(no rows)"
    buf = io.StringIO()
    for row in rows[:200]:
        buf.write(str(row))
        buf.write("\n")
    preview = buf.getvalue()
    if len(preview) > max_chars:
        return preview[: max_chars // 2] + "\n…[truncated]…\n" + preview[-max_chars // 2 :]
    return preview.strip()
