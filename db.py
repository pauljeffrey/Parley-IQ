"""
Database access for conversation ingestion and analysis persistence.

Toggle MySQL vs PostgreSQL with `DB_ENGINE` and `DATABASE_URL` or `DB_*` components
(see `resolve_database_url`).

Table names come from `DB_SCHEMA`, `CONVERSATION_TABLE`, and `ANALYSIS_TABLE`.
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

from output import ConversationAnalysis, SDoHBarrier, SDoHProfile

COLUMN_ID = "id"
COLUMN_USER_ID = "user_id"
COLUMN_SESSION_ID = "session_id"
COLUMN_USER_MESSAGE = "user_message"
COLUMN_ASSISTANT_MESSAGE = "assistant_message"
COLUMN_CREATED_AT = "created_at"


class DatabaseBackend(str, Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class DBError(RuntimeError):
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

SCHEMA_NAME = os.environ.get("DB_SCHEMA", "public")
CONVERSATION_TABLE_NAME = os.environ.get("CONVERSATION_TABLE", "user_consultation")
ANALYSIS_TABLE_NAME = os.environ.get("ANALYSIS_TABLE", "conversation_analysis")


def min_conversation_turns() -> int:
    raw = os.environ.get("BATCH_MIN_TURNS", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def get_configured_backend() -> DatabaseBackend:
    raw = os.environ.get("DB_ENGINE", "mysql").strip().lower()
    aliases = {
        "mysql": DatabaseBackend.MYSQL,
        "postgres": DatabaseBackend.POSTGRESQL,
        "postgresql": DatabaseBackend.POSTGRESQL,
        "pg": DatabaseBackend.POSTGRESQL,
    }
    if raw not in aliases:
        raise DBError(f"DB_ENGINE must be one of {set(aliases.keys())}, got {raw!r}")
    return aliases[raw]


def resolve_database_url(backend: Optional[DatabaseBackend] = None) -> str:
    """
    Build a SQLAlchemy URL. Precedence:
    1. DATABASE_URL when set and non-empty
    2. DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    """
    _load_dotenv_if_available()
    backend = backend or get_configured_backend()
    explicit = os.environ.get("DATABASE_URL", "").strip()
    if explicit:
        _validate_url_matches_backend(explicit, backend)
        return explicit

    user = os.environ.get("DB_USER", "").strip()
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost").strip()
    port = os.environ.get("DB_PORT", "").strip()
    db_name = os.environ.get("DB_NAME", "").strip() or SCHEMA_NAME

    if not user:
        raise DBError(
            "Set DATABASE_URL or DB_USER + DB_PASSWORD + DB_HOST (+ DB_NAME) for DB access."
        )
    if not host:
        raise DBError("DB_HOST must be set when DATABASE_URL is not used.")

    if not port:
        port = "3306" if backend == DatabaseBackend.MYSQL else "5432"

    user_enc = quote_plus(user)
    pw_enc = quote_plus(password) if password else ""
    auth = f"{user_enc}:{pw_enc}@" if password else f"{user_enc}@"
    host_port = f"{host}:{port}"

    if backend == DatabaseBackend.MYSQL:
        driver = os.environ.get("MYSQL_DRIVER", "pymysql")
        return f"mysql+{driver}://{auth}{host_port}/{db_name}"
    driver = os.environ.get("PG_DRIVER", "psycopg")
    return f"postgresql+{driver}://{auth}{host_port}/{db_name}"


def _validate_url_matches_backend(url: str, backend: DatabaseBackend) -> None:
    lowered = url.lower()
    if backend == DatabaseBackend.MYSQL and not lowered.startswith("mysql"):
        raise DBError(
            f"DB_ENGINE is mysql but URL does not start with mysql+... ({url[:48]}...)"
        )
    if backend == DatabaseBackend.POSTGRESQL and not (
        lowered.startswith("postgresql") or lowered.startswith("postgres:/")
    ):
        raise DBError(
            f"DB_ENGINE is postgresql but URL is not a postgres SQLAlchemy URL ({url[:48]}...)"
        )


def _safe_ident_fragment(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise DBError(f"Invalid SQL identifier fragment: {name!r}")
    return name


def qualified_table_name(engine: Engine, table: Optional[str] = None) -> str:
    """Dialect-correct `schema.table` for MySQL vs PostgreSQL."""
    schema = _safe_ident_fragment(SCHEMA_NAME)
    tbl = _safe_ident_fragment(table or CONVERSATION_TABLE_NAME)
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
    final_url = url or resolve_database_url(backend)
    return create_engine(final_url, pool_pre_ping=pool_pre_ping)


@dataclass(frozen=True)
class ConsultationTurn:
    """One row / turn in a consultation session (user + assistant message pair)."""

    id: int
    user_id: int
    session_id: int
    user_message: str
    assistant_message: str
    created_at: datetime


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
    rid, user_id, session_id, user_message, assistant_message, created_at = row
    return ConsultationTurn(
        id=int(rid),
        user_id=int(user_id),
        session_id=int(session_id),
        user_message=str(user_message or ""),
        assistant_message=str(assistant_message or ""),
        created_at=_coerce_datetime(created_at),
    )


def conversation_started_at(turns: list[ConsultationTurn]) -> datetime | None:
    if not turns:
        return None
    return min(t.created_at for t in turns)


def conversation_calendar_parts(dt: datetime) -> dict[str, Any]:
    return {
        "conversation_day_of_week": dt.strftime("%A"),
        "conversation_month": dt.month,
        "conversation_year": dt.year,
    }


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _serialize_db_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        if not value:
            return json.dumps([])
        if hasattr(value[0], "model_dump"):
            return json.dumps([v.model_dump(mode="json") for v in value])
        if isinstance(value[0], Enum):
            return json.dumps([v.value for v in value])
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _sdoh_barrier_flags(profiles: list[SDoHProfile]) -> tuple[bool, bool, bool]:
    economic = geographic = social = False
    economic_barriers = {
        SDoHBarrier.MEDICATION_COST,
        SDoHBarrier.CONSULTATION_COST,
    }
    geographic_barriers = {SDoHBarrier.TRANSPORTATION}
    social_barriers = {
        SDoHBarrier.STIGMA_PRIVACY,
        SDoHBarrier.CULTURAL_CONFLICT,
        SDoHBarrier.SECURITY,
        SDoHBarrier.EMPLOYER_CONSTRAINTS,
        SDoHBarrier.CHILDCARE,
    }
    for profile in profiles:
        for barrier in profile.barriers_to_care or []:
            if barrier in economic_barriers:
                economic = True
            if barrier in geographic_barriers:
                geographic = True
            if barrier in social_barriers:
                social = True
    return economic, geographic, social


def fetch_conversation_by_session_id(
    session_id: int | str,
    *,
    engine: Optional[Engine] = None,
) -> list[ConsultationTurn]:
    """Return all turns for a session, ordered by primary key."""
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
    min_turns: int | None = None,
) -> list[int]:
    """
    Distinct session ids with at least `min_turns` rows (user–assistant pairs).
  """
    min_turns = min_turns if min_turns is not None else min_conversation_turns()
    own_engine = engine is None
    eng = engine or get_engine()
    try:
        sid = _safe_ident_fragment(COLUMN_SESSION_ID)
        q = text(
            f"""
            SELECT {sid}
            FROM {qualified_table_name(eng)}
            GROUP BY {sid}
            HAVING COUNT(*) >= :min_turns
            ORDER BY {sid} ASC
            """
        )
        with eng.connect() as conn:
            rows = conn.execute(q, {"min_turns": min_turns}).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        if own_engine:
            eng.dispose()


def _analysis_insert_params(
    *,
    session_id: int | str,
    user_phone: str,
    model_name: str,
    analysis: ConversationAnalysis,
    created_at: datetime,
    conversation_started: datetime | None,
) -> list[dict[str, Any]]:
    conv_time = conversation_started or created_at
    calendar = conversation_calendar_parts(conv_time)
    sdoh_economic, sdoh_geographic, sdoh_social = _sdoh_barrier_flags(analysis.sdoh_profiles)
    cultural_notes = analysis.cultural_notes or ""

    rows: list[dict[str, Any]] = []
    segments = analysis.topic_segments or []
    if not segments:
        return []

    for i, segment in enumerate(segments):
        row: dict[str, Any] = {
            "session_id": session_id,
            "user_phone": user_phone,
            "model_name": model_name,
            "segment_index": i + 1,
            "clinical_category": _enum_value(getattr(segment, "clinical_category", None)),
            "intent": _serialize_db_value(getattr(segment, "intent", None)),
            "pharmacology_profiles": _serialize_db_value(
                getattr(segment, "pharmacology_profiles", None)
            ),
            "mental_health_profiles": _serialize_db_value(
                getattr(segment, "mental_health_profiles", None)
            ),
            "suspected_condition": _serialize_db_value(
                getattr(segment, "suspected_condition", None)
            ),
            "symptoms_reported": _serialize_db_value(
                getattr(segment, "symptoms_reported", None)
            ),
            "urgency_level": _enum_value(getattr(segment, "urgency_level", None)),
            "barriers": _serialize_db_value(getattr(segment, "barriers", None) or []),
            "cultural_tags": _serialize_db_value(getattr(segment, "cultural_tags", None)),
            "outcome_referral": _enum_value(getattr(segment, "outcome_referral", None)),
            "literacy_score": getattr(segment, "literacy_score", None),
            "cultural_notes": cultural_notes,
            "sdoh_profiles": _serialize_db_value(analysis.sdoh_profiles),
            "sdoh_economic_barrier": sdoh_economic,
            "sdoh_geographic_barrier": sdoh_geographic,
            "sdoh_social_barrier": sdoh_social,
            "created_at": created_at,
            "analysis_timestamp": conv_time,
            **calendar,
        }
        rows.append(row)
    return rows


def insert_conversation_analysis(
    session_id: int | str,
    user_phone: str,
    model_name: str,
    analysis: ConversationAnalysis,
    *,
    engine: Optional[Engine] = None,
    created_at: Optional[datetime] = None,
    conversation_started: Optional[datetime] = None,
) -> None:
    """Persist one analysis as one row per topic segment (flattened columns)."""
    own_engine = engine is None
    eng = engine or get_engine()
    created_at = created_at or datetime.now(timezone.utc)
    params_list = _analysis_insert_params(
        session_id=session_id,
        user_phone=user_phone,
        model_name=model_name,
        analysis=analysis,
        created_at=created_at,
        conversation_started=conversation_started,
    )
    if not params_list:
        return
    tbl = qualified_table_name(eng, ANALYSIS_TABLE_NAME)
    q = text(
        f"""
        INSERT INTO {tbl} (
            session_id, user_phone, model_name, segment_index,
            clinical_category, intent, pharmacology_profiles, mental_health_profiles,
            suspected_condition, symptoms_reported, urgency_level, barriers, cultural_tags,
            outcome_referral, literacy_score, cultural_notes, sdoh_profiles,
            sdoh_economic_barrier, sdoh_geographic_barrier, sdoh_social_barrier,
            created_at, analysis_timestamp,
            conversation_day_of_week, conversation_month, conversation_year
        ) VALUES (
            :session_id, :user_phone, :model_name, :segment_index,
            :clinical_category, :intent, :pharmacology_profiles, :mental_health_profiles,
            :suspected_condition, :symptoms_reported, :urgency_level, :barriers, :cultural_tags,
            :outcome_referral, :literacy_score, :cultural_notes, :sdoh_profiles,
            :sdoh_economic_barrier, :sdoh_geographic_barrier, :sdoh_social_barrier,
            :created_at, :analysis_timestamp,
            :conversation_day_of_week, :conversation_month, :conversation_year
        )
        """
    )
    try:
        with eng.begin() as conn:
            conn.execute(q, params_list)
    finally:
        if own_engine:
            eng.dispose()
