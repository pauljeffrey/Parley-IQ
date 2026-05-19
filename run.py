"""
Orchestrator: shard consultations, submit all OpenAI batches, persist as each completes.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import batch_processor as bp
import db


def main() -> None:
    engine = db.get_engine()
    try:
        skip = bp.completed_session_ids(engine)
        if skip:
            print(f"Skipping {len(skip)} session(s) (cache + existing analysis rows).")
        since = db.batch_conversation_since()
        if since:
            print(f"Conversation filter: MAX(created_at) >= {since.isoformat()}")
        print(f"Shard size: {bp.batch_shard_size()} | work dir: {bp.batch_work_dir()}")
        bp.run_sharded_batch_pipeline(
            engine,
            wait=bp.env_bool("BATCH_WAIT_FOR_COMPLETION"),
            poll_interval_sec=bp.batch_poll_interval_sec(),
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
