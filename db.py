"""
Database access for `aisha.user_consultation`.

Toggle MySQL vs PostgreSQL with `DB_ENGINE` (or legacy `AISHA_DB_BACKEND`) and
`DATABASE_URL` / `AISHA_DATABASE_URL` or `AISHA_DB_*` components (see `resolve_database_url`).

Expected columns (adjust COLUMN_* constants if your DDL differs):
  id, user_id, session_id, user_message, assistant_message,
  created_at, category, score, evaluation
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from output import AishaConversationAnalysis

# --- Table / column mapping -------------------------------------------------

SCHEMA_NAME = os.environ.get("AISHA_DB_SCHEMA", "aisha")
TABLE_NAME = os.environ.get("AISHA_DB_TABLE", "user_consultation")
ANALYSIS_TABLE_NAME = os.environ.get("AISHA_ANALYSIS_TABLE", "aisha_conversation_analysis")

COLUMN_ID = "id"
COLUMN_USER_ID = "user_id"
COLUMN_SESSION_ID = "session_id"
COLUMN_USER_MESSAGE = "user_message"
COLUMN_ASSISTANT_MESSAGE = "assistant_message"
COLUMN_CREATED_AT = "created_at"
COLUMN_CATEGORY = "category"
COLUMN_SCORE = "score"
COLUMN_EVALUATION = "evaluation"


class DatabaseBackend(str, Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class AishaDBError(RuntimeError):
    """Raised when configuration is invalid or the driver/backend does not match."""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass


_load_dotenv_if_available()


def get_configured_backend() -> DatabaseBackend:
    raw = (
        os.environ.get("DB_ENGINE") or os.environ.get("AISHA_DB_BACKEND") or "mysql"
    ).strip().lower()
    aliases = {
        "mysql": DatabaseBackend.MYSQL,
        "postgres": DatabaseBackend.POSTGRESQL,
        "postgresql": DatabaseBackend.POSTGRESQL,
        "pg": DatabaseBackend.POSTGRESQL,
    }
    if raw not in aliases:
        raise AishaDBError(
            f"DB_ENGINE / AISHA_DB_BACKEND must be one of {set(aliases.keys())}, got {raw!r}"
        )
    return aliases[raw]


def resolve_database_url(backend: Optional[DatabaseBackend] = None) -> str:
    """
    Build a SQLAlchemy URL. Precedence:
    1. DATABASE_URL or AISHA_DATABASE_URL (include dialect, e.g. mysql+pymysql:// or postgresql+psycopg://)
    2. AISHA_DB_USER, AISHA_DB_PASSWORD, AISHA_DB_HOST, AISHA_DB_PORT, AISHA_DB_NAME
    """
    _load_dotenv_if_available()
    backend = backend or get_configured_backend()
    explicit = (
        os.environ.get("DATABASE_URL") or os.environ.get("AISHA_DATABASE_URL") or ""
    ).strip()
    if explicit:
        _validate_url_matches_backend(explicit, backend)
        return explicit

    user = os.environ.get("AISHA_DB_USER", "")
    password = os.environ.get("AISHA_DB_PASSWORD", "")
    host = os.environ.get("AISHA_DB_HOST", "localhost")
    port = os.environ.get("AISHA_DB_PORT", "")
    db_name = os.environ.get("AISHA_DB_NAME", SCHEMA_NAME)

    if not user:
        raise AishaDBError(
            "Set DATABASE_URL (or AISHA_DATABASE_URL) or AISHA_DB_USER + AISHA_DB_* for DB access."
        )

    user_enc = quote_plus(user)
    pw_enc = quote_plus(password) if password else ""
    auth = f"{user_enc}:{pw_enc}@" if password else f"{user_enc}@"
    host_port = f"{host}:{port}" if port else host

    if backend == DatabaseBackend.MYSQL:
        driver = os.environ.get("AISHA_MYSQL_DRIVER", "pymysql")
        return f"mysql+{driver}://{auth}{host_port}/{db_name}"
    driver = os.environ.get("AISHA_PG_DRIVER", "psycopg")
    return f"postgresql+{driver}://{auth}{host_port}/{db_name}"


def _validate_url_matches_backend(url: str, backend: DatabaseBackend) -> None:
    lowered = url.lower()
    if backend == DatabaseBackend.MYSQL and not lowered.startswith("mysql"):
        raise AishaDBError(
            f"DB_ENGINE is mysql but URL does not start with mysql+... ({url[:48]}...)"
        )
    if backend == DatabaseBackend.POSTGRESQL and not (
        lowered.startswith("postgresql") or lowered.startswith("postgres:/")
    ):
        raise AishaDBError(
            f"DB_ENGINE is postgresql but URL is not a postgres SQLAlchemy URL ({url[:48]}...)"
        )


def _safe_ident_fragment(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise AishaDBError(f"Invalid SQL identifier fragment: {name!r}")
    return name


def qualified_table_name(engine: Engine, table: Optional[str] = None) -> str:
    """Dialect-correct `schema.table` for MySQL vs PostgreSQL."""
    schema = _safe_ident_fragment(SCHEMA_NAME)
    tbl = _safe_ident_fragment(table or TABLE_NAME)
    if engine.dialect.name == "mysql":
        return f"`{schema}`.`{tbl}`"
    return f'"{schema}"."{tbl}"'


def _select_columns_sql() -> str:
    parts = [
        COLUMN_ID,
        COLUMN_USER_ID,
        COLUMN_SESSION_ID,
        COLUMN_USER_MESSAGE,
        COLUMN_ASSISTANT_MESSAGE,
        COLUMN_CREATED_AT,
        COLUMN_CATEGORY,
        COLUMN_SCORE,
        COLUMN_EVALUATION,
    ]
    for p in parts:
        _safe_ident_fragment(p)
    return ", ".join(parts)


def get_engine(
    url: Optional[str] = None,
    *,
    backend: Optional[DatabaseBackend] = None,
    pool_pre_ping: bool = True,
) -> Engine:
    """Create a SQLAlchemy engine (callers should `engine.dispose()` when done, or reuse one process-wide)."""
    final_url = url or resolve_database_url(backend)
    return create_engine(final_url, pool_pre_ping=pool_pre_ping)


def _parse_evaluation(raw: Any) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        return json.loads(raw)
    raise TypeError(f"Unexpected evaluation column type: {type(raw)}")


@dataclass(frozen=True)
class ConsultationTurn:
    """One row / turn in a consultation session."""

    id: int
    user_id: int
    session_id: int
    user_message: str
    assistant_message: str
    created_at: datetime
    category: str
    score: float
    evaluation: Optional[dict[str, Any]]


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        raise ValueError("empty created_at")
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {s!r}")


def _row_to_turn(row: Iterable[Any]) -> ConsultationTurn:
    (
        rid,
        user_id,
        session_id,
        user_message,
        assistant_message,
        created_at,
        category,
        score,
        evaluation_raw,
    ) = row
    return ConsultationTurn(
        id=int(rid),
        user_id=int(user_id),
        session_id=int(session_id),
        user_message=str(user_message or ""),
        assistant_message=str(assistant_message or ""),
        created_at=_coerce_datetime(created_at),
        category=str(category or ""),
        score=float(score) if score is not None else 0.0,
        evaluation=_parse_evaluation(evaluation_raw),
    )


def fetch_conversation_by_session_id(
    session_id: int | str,
    *,
    engine: Optional[Engine] = None,
) -> list[ConsultationTurn]:
    """
    Return all turns for a single session, ordered by primary key (conversation order).
    """
    own_engine = engine is None
    eng = engine or get_engine()
    try:
        q = text(
            f"""
            SELECT {_select_columns_sql()}
            FROM {qualified_table_name(eng)}
            WHERE {_safe_ident_fragment(COLUMN_SESSION_ID)} = :session_id
            ORDER BY {_safe_ident_fragment(COLUMN_ID)} ASC
            """
        )
        with eng.connect() as conn:
            result = conn.execute(q, {"session_id": session_id})
            rows = result.fetchall()
        return [_row_to_turn(r) for r in rows]
    finally:
        if own_engine:
            eng.dispose()


def fetch_all_session_ids(
    *,
    engine: Optional[Engine] = None,
) -> list[int]:
    """
    Distinct session ids for all rows, sorted ascending (stable for batch jobs).
    """
    own_engine = engine is None
    eng = engine or get_engine()
    try:
        sid = _safe_ident_fragment(COLUMN_SESSION_ID)
        q = text(
            f"""
            SELECT DISTINCT {sid}
            FROM {qualified_table_name(eng)}
            ORDER BY {sid} ASC
            """
        )
        with eng.connect() as conn:
            rows = conn.execute(q).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        if own_engine:
            eng.dispose()


def insert_conversation_analysis(
    session_id: int | str,
    user_phone: str,
    model_name: str,
    analysis: AishaConversationAnalysis,
    *,
    engine: Optional[Engine] = None,
    created_at: Optional[datetime] = None,
) -> None:
    """
    Persist one analysis row to `schema.aisha_conversation_analysis`.

    Top-level LLM fields map to columns; `segments` is stored as JSON.
    """
    own_engine = engine is None
    eng = engine or get_engine()
    created_at = created_at or datetime.now(timezone.utc)
    segments_payload = json.dumps(analysis.model_dump(mode="json")["segments"])
    sdoh = analysis.sdoh_indicators
    tbl = qualified_table_name(eng, ANALYSIS_TABLE_NAME)
    q = text(
        f"""
        INSERT INTO {tbl} (
            session_id, user_phone, model_name, created_at,
            conversation_id, analysis_timestamp, user_persona, segments,
            sdoh_economic_barrier, sdoh_geographic_barrier, sdoh_social_barrier,
            outcome_referral
        ) VALUES (
            :session_id, :user_phone, :model_name, :created_at,
            :conversation_id, :analysis_timestamp, :user_persona, :segments,
            :sdoh_economic_barrier, :sdoh_geographic_barrier, :sdoh_social_barrier,
            :outcome_referral
        )
        """
    )
    params = {
        "session_id": session_id,
        "user_phone": user_phone,
        "model_name": model_name,
        "created_at": created_at,
        "conversation_id": analysis.conversation_id,
        "analysis_timestamp": analysis.timestamp,
        "user_persona": analysis.user_persona,
        "segments": segments_payload,
        "sdoh_economic_barrier": sdoh.economic_barrier,
        "sdoh_geographic_barrier": sdoh.geographic_barrier,
        "sdoh_social_barrier": sdoh.social_barrier,
        "outcome_referral": analysis.outcome_referral,
    }
    try:
        with eng.begin() as conn:
            conn.execute(q, params)
    finally:
        if own_engine:
            eng.dispose()
