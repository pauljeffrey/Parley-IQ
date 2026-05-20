#!/usr/bin/env python3
"""Seed CONVERSATION_TABLE in DB_SCHEMA (see .env) with dummy sessions. Run after `docker compose up -d`."""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from urllib.parse import quote_plus

from sqlalchemy import create_engine

import db  # noqa: E402


_USER_LINES = (
    "I have had a headache for two days.",
    "The pain is worse in the morning.",
    "No fever but I feel tired.",
    "Can I take something with my blood pressure pill?",
    "Should I see a doctor in person?",
)
_BOT_LINES = (
    "Thanks for sharing. Any vision changes or neck stiffness?",
    "How severe is the pain from 1 to 10?",
    "Any allergies to medications?",
    "If symptoms worsen or you develop fever, seek urgent care.",
    "Hydration and rest often help with tension headaches.",
)


def _root_password_candidates() -> list[str]:
    explicit = os.environ.get("MYSQL_ROOT_PASSWORD")
    if explicit is not None and explicit.strip():
        return [explicit.strip()]
    return ["devroot", ""]


def _bootstrap_schema() -> None:
    """Ensure DB_SCHEMA + user_consultation exist (fixes failed Docker init / missing grants)."""
    host = os.environ.get("DB_HOST", "127.0.0.1").strip()
    if host not in ("127.0.0.1", "localhost"):
        return
    app_user = os.environ.get("DB_USER", "").strip()
    if not app_user:
        return
    sch = db.SCHEMA_NAME
    port = os.environ.get("DB_PORT", "3306").strip() or "3306"
    db._safe_ident_fragment(sch)
    db._safe_ident_fragment(app_user)

    last_err: Exception | None = None
    for root_pw in _root_password_candidates():
        admin = create_engine(
            f"mysql+pymysql://root:{quote_plus(root_pw)}@{host}:{port}/",
            pool_pre_ping=True,
        )
        try:
            with admin.begin() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{sch}`"))
                conn.execute(text(f"GRANT ALL PRIVILEGES ON `{sch}`.* TO '{app_user}'@'%'"))
                conn.execute(text("FLUSH PRIVILEGES"))
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS `{sch}`.user_consultation (
                          id BIGINT AUTO_INCREMENT PRIMARY KEY,
                          user_id VARCHAR(255) NOT NULL,
                          session_id VARCHAR(255) NOT NULL,
                          user_message TEXT,
                          assistant_message TEXT,
                          created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                          INDEX idx_session_created (session_id, created_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                        """
                    )
                )
            return
        except Exception as exc:
            last_err = exc
        finally:
            admin.dispose()
    raise RuntimeError(
        "Could not bootstrap schema as MySQL root. "
        "Add MYSQL_ROOT_PASSWORD=devroot to .env (match docker-compose) and ensure "
        "MySQL is running: docker compose up -d"
    ) from last_err


def main() -> None:
    _bootstrap_schema()
    raw = os.environ.get("SEED_SESSION_COUNT", "").strip()
    if raw:
        n_sess = max(1000, min(3000, int(raw)))
    else:
        n_sess = random.randint(1000, 3000)
    pairs_min = max(5, int(os.environ.get("SEED_MIN_PAIRS_PER_SESSION", "5")))
    pairs_max = max(pairs_min, int(os.environ.get("SEED_MAX_PAIRS_PER_SESSION", "10")))
    rng = random.Random(int(os.environ.get("SEED_RANDOM_SEED", "42")))
    engine = db.get_engine()
    tbl = db.qualified_table_name(engine)

    days_back = max(30, int(os.environ.get("BATCH_CONVERSATION_SINCE_DAYS", "90") or "90"))

    stmt = text(
        f"""
        INSERT INTO {tbl} (
            user_id, session_id, user_message, assistant_message, created_at
        ) VALUES (
            :user_id, :session_id, :user_message, :assistant_message, :created_at
        )
        """
    )
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    for i in range(n_sess):
        sid = f"seed_{i:06d}"
        uid = str(rng.randint(1, 50_000))
        n_pairs = rng.randint(pairs_min, pairs_max)
        start = now - timedelta(days=rng.randint(0, days_back - 1), hours=rng.randint(0, 23))
        for p in range(n_pairs):
            rows.append(
                {
                    "user_id": uid,
                    "session_id": sid,
                    "user_message": rng.choice(_USER_LINES),
                    "assistant_message": rng.choice(_BOT_LINES),
                    "created_at": start + timedelta(minutes=p * 3),
                }
            )

    chunk = 500
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            conn.execute(stmt, rows[i : i + chunk])

    engine.dispose()
    print(f"Inserted {len(rows)} turns across {n_sess} sessions (tbl={tbl}).", flush=True)


def _hint_for_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "can't connect" in msg or "connection refused" in msg or "2003" in msg:
        return (
            "\nMySQL is not reachable. In another terminal run: docker compose up -d\n"
            "(Use -d so Ctrl+C in a foreground terminal does not stop the database.)"
        )
    if "access denied" in msg and "root" in msg:
        return "\nSet MYSQL_ROOT_PASSWORD=devroot in .env (or your compose root password)."
    return ""


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        hint = _hint_for_error(exc)
        if hint:
            print(hint, file=sys.stderr)
        sys.exit(1)
