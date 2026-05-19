#!/usr/bin/env python3
"""Full pipeline test: read conversations → OpenAI Batch → wait → insert analysis. Run: py test.py"""

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


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    _log("=== Parley-IQ full batch pipeline test ===\n")
    _log(f"BATCH_MIN_TURNS={db.min_conversation_turns()} BATCH_SHARD_SIZE={bp.batch_shard_size()}")
    _log(f"BATCH_MAX_SESSIONS={os.environ.get('BATCH_MAX_SESSIONS', '') or 'no cap'}")
    since = db.batch_conversation_since()
    _log(f"BATCH_CONVERSATION_SINCE_DAYS/SINCE → {since.isoformat() if since else 'none'}")
    _log(f"BATCH_WAIT_FOR_COMPLETION={bp.env_bool('BATCH_WAIT_FOR_COMPLETION')}")
    _log(f"ANALYSIS_TABLE={db.ANALYSIS_TABLE_NAME}\n")

    engine = db.get_engine()
    try:
        skip = bp.completed_session_ids(engine)
        if skip:
            _log(f"Skipping {len(skip)} session(s) already completed or in {db.ANALYSIS_TABLE_NAME}.")

        ids = bp.collect_pending_session_ids(engine)
        _log(f"Pending sessions to process: {len(ids)}")
        if not ids:
            print("Nothing to process.", file=sys.stderr)
            sys.exit(1)

        preview = int(os.environ.get("BATCH_TEST_PREVIEW", "3"))
        if preview > 0:
            for job in bp.load_consultation_jobs(
                engine, session_ids=ids[:preview], min_turns=db.min_conversation_turns()
            ):
                turns = db.fetch_conversation_by_session_id(job.session_id, engine=engine)
                _log(json.dumps({"session_id": job.session_id, "turns": len(turns)}))
            _log("")

        _log("--- Submit batches, wait for OpenAI, persist to analysis table ---")
        stats = bp.run_sharded_batch_pipeline(
            engine,
            session_ids=ids,
            wait=bp.env_bool("BATCH_WAIT_FOR_COMPLETION", True),
            poll_interval_sec=bp.batch_poll_interval_sec(),
            log=_log,
        )
        _log(f"\nManifest: {bp.batch_manifest_path().resolve()}")
        _log(f"Result: {stats}")
        if stats.get("shards", 0) and not stats.get("inserted") and bp.env_bool("BATCH_WAIT_FOR_COMPLETION", True):
            sys.exit(4)
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
