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
load_dotenv(_ROOT / ".env")

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


def main() -> None:
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
