#!/usr/bin/env python3
"""Integration test: filters, shard preview, sharded batch + persist. Run: py test.py"""

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
    min_turns = db.min_conversation_turns()
    since = db.batch_conversation_since()
    preview = int(os.environ.get("BATCH_TEST_PREVIEW", "3"))

    print("=== Parley-IQ batch integration test ===\n", flush=True)
    print(f"min_turns={min_turns} shard_size={bp.batch_shard_size()}", flush=True)
    print(f"since={since.isoformat() if since else 'none'} submit={bp.env_bool('BATCH_TEST_SUBMIT')}\n", flush=True)

    engine = db.get_engine()
    try:
        ids = bp.collect_pending_session_ids(engine)
        print(f"Pending sessions: {len(ids)}", flush=True)
        if not ids:
            print("Nothing to process.", file=sys.stderr)
            sys.exit(1)

        for job in bp.load_consultation_jobs(engine, session_ids=ids[:preview], min_turns=min_turns):
            turns = db.fetch_conversation_by_session_id(job.session_id, engine=engine)
            print(json.dumps({"session_id": job.session_id, "turns": len(turns)}), flush=True)
        print(flush=True)

        if not bp.env_bool("BATCH_TEST_SUBMIT"):
            print("BATCH_TEST_SUBMIT=false — preview only.", flush=True)
            return

        bp.run_sharded_batch_pipeline(
            engine,
            wait=bp.env_bool("BATCH_WAIT_FOR_COMPLETION"),
            poll_interval_sec=bp.batch_poll_interval_sec(),
            log=print,
        )
        print(f"Manifest: {bp.batch_manifest_path().resolve()}", flush=True)
    except db.DBError as exc:
        print(f"DB error: {exc}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(3)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
