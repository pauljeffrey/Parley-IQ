"""
Orchestrator: read `aisha.user_consultation` (read-only), build Batch JSONL, submit to OpenAI,
download results, insert into `aisha.aisha_conversation_analysis`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

import batch_processor as bp
import db


def _truthy(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    load_dotenv()

    limit_raw = os.environ.get("BATCH_MAX_SESSIONS", "").strip()
    limit = int(limit_raw) if limit_raw else None
    poll = float(os.environ.get("BATCH_POLL_INTERVAL_SEC", "15"))
    wait = _truthy("BATCH_WAIT_FOR_COMPLETION", True)

    input_path = Path(os.environ.get("BATCH_INPUT_JSONL", "batch_input.jsonl"))
    meta_path = Path(os.environ.get("BATCH_META_JSON", "batch_meta.json"))
    output_path = Path(os.environ.get("BATCH_OUTPUT_JSONL", "batch_output.jsonl"))
    error_path = Path(os.environ.get("BATCH_ERROR_JSONL", "batch_errors.jsonl"))

    engine = db.get_engine()
    try:
        jobs = bp.load_consultation_jobs(engine, limit=limit)
        if not jobs:
            print("No consultation sessions to process.")
            return

        custom_meta = bp.prepare_batch_file(jobs, input_path)
        model_name = bp.get_model_name()
        client = bp.get_openai_client()

        file_id = bp.upload_batch_input(client, input_path)
        batch = bp.create_chat_completion_batch(
            client,
            file_id,
            metadata={"source": "run.py"},
        )
        bp.save_batch_metadata(
            meta_path,
            {
                "batch_id": batch.id,
                "input_file_id": file_id,
                "custom_id_meta": custom_meta,
                "model_name": model_name,
            },
        )
        print(f"Submitted batch {batch.id} ({len(jobs)} requests). Metadata: {meta_path}")

        if not wait:
            print("BATCH_WAIT_FOR_COMPLETION is off — exiting after submit.")
            return

        batch = bp.wait_for_batch_completion(client, batch.id, poll_interval_sec=poll)
        print(f"Batch status: {bp.batch_status(batch)}")

        if batch.error_file_id:
            err_text = bp.download_file_text(client, batch.error_file_id)
            error_path.write_text(err_text, encoding="utf-8")
            print(f"Batch errors written to {error_path}")

        if bp.batch_status(batch) != "completed" or not batch.output_file_id:
            print("No output file to persist; fix batch errors or retry.")
            return

        out_text = bp.download_file_text(client, batch.output_file_id)
        output_path.write_text(out_text, encoding="utf-8")
        print(f"Output JSONL saved to {output_path}")

        stats = bp.persist_batch_results(out_text, custom_meta, model_name, engine)
        print(f"Database: {stats}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
