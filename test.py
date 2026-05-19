#!/usr/bin/env python3
"""
Smoke test: DB conversation retrieval + batch job building (no OpenAI submit).

Run from the project root (not as a module):

    py test.py

Do not use ``py -m test.py`` — that looks for a package named ``test``, not this file.
``py -m test`` would run Python's stdlib test suite, not Parley-IQ.

Prints session ids, turn counts, and prepared job summaries to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

import batch_processor as bp
import db


def main() -> None:
    limit_raw = os.environ.get("BATCH_MAX_SESSIONS", "3").strip()
    limit = int(limit_raw) if limit_raw else 3
    min_turns = db.min_conversation_turns()

    print("=== Parley-IQ retrieval / batch smoke test ===\n")
    print(f"DB_ENGINE={os.environ.get('DB_ENGINE', 'mysql')}")
    print(f"DB_HOST={os.environ.get('DB_HOST', '')}")
    print(f"DB_NAME={os.environ.get('DB_NAME', '')}")
    print(f"DB_SCHEMA={db.SCHEMA_NAME}")
    print(f"CONVERSATION_TABLE={db.CONVERSATION_TABLE_NAME}")
    print(f"ANALYSIS_TABLE={db.ANALYSIS_TABLE_NAME}")
    print(f"BATCH_MIN_TURNS={min_turns}")
    print(f"BATCH_MAX_SESSIONS (limit)={limit}\n")

    engine = db.get_engine()
    try:
        session_ids = db.fetch_all_session_ids(engine=engine, min_turns=min_turns)
        print(f"Eligible session ids (>= {min_turns} turns): {len(session_ids)}")
        if bp.skip_completed_sessions():
            pending = bp.exclude_completed_session_ids(session_ids)
            print(
                f"Completed-session cache ({bp.completed_sessions_cache_path()}): "
                f"{len(session_ids) - len(pending)} skipped, {len(pending)} pending"
            )
        print(f"First ids: {session_ids[:limit]}\n")

        jobs = bp.load_consultation_jobs(engine, limit=limit, min_turns=min_turns)
        print(f"Loaded jobs: {len(jobs)}\n")

        for job in jobs:
            turns = db.fetch_conversation_by_session_id(job.session_id, engine=engine)
            started = db.conversation_started_at(turns)
            calendar = db.conversation_calendar_parts(started) if started else {}
            preview = job.transcript[:400] + ("…" if len(job.transcript) > 400 else "")
            print(
                json.dumps(
                    {
                        "session_id": job.session_id,
                        "turn_count": len(turns),
                        "conversation_started": started.isoformat() if started else None,
                        **calendar,
                        "transcript_preview": preview,
                    },
                    indent=2,
                )
            )
            print()

        if jobs:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as tmp:
                meta = bp.prepare_batch_file(jobs, tmp.name)
                print(f"Batch JSONL written: {tmp.name}")
                print(f"Batch metadata keys: {list(meta.keys())}")
        else:
            print("No jobs to build — check filters and DB contents.", file=sys.stderr)
            sys.exit(1)
    except db.DBError as exc:
        print(f"DB configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
