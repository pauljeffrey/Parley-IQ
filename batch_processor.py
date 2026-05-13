"""
OpenAI Batch API utilities: build JSONL from consultations, submit, poll, parse, persist via db.py.
`user_consultation` is only read through db helpers — never written here.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Iterator, Mapping, MutableMapping, Optional, Sequence

from openai import OpenAI
from sqlalchemy.engine import Engine

from db import (
    ConsultationTurn,
    fetch_all_session_ids,
    fetch_conversation_by_session_id,
    insert_conversation_analysis,
)
from output import AishaConversationAnalysis

BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical conversation analyst. Analyze the full transcript into the "
    "required JSON schema. Use only allowed enum string values from the schema. "
    "Set conversation_id to the Session ID given at the start of the user message."
)


def get_openai_client() -> OpenAI:
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=key)


def get_model_name() -> str:
    name = os.environ.get("MODEL_NAME", "").strip()
    if not name:
        raise RuntimeError("Set MODEL_NAME.")
    return name


def _response_format_dict() -> dict[str, Any]:
    use_strict = os.environ.get("OPENAI_JSON_SCHEMA_STRICT", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    schema = AishaConversationAnalysis.model_json_schema()
    rf: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "aisha_conversation_analysis",
            "schema": schema,
        },
    }
    if use_strict:
        rf["json_schema"]["strict"] = True
    return rf


@dataclass(frozen=True)
class ConversationJob:
    """One batch row: a full consultation session."""

    session_id: str
    user_phone: str
    transcript: List[ConsultationTurn]

    @property
    def custom_id(self) -> str:
        return f"session_{self.session_id}"


def format_transcript(turns: Sequence[ConsultationTurn]) -> str:
    parts: list[str] = []
    for t in turns:
        parts.append(f"User: {t.user_message}")
        parts.append(f"Assistant: {t.assistant_message}")
    return "\n".join(parts)


def load_consultation_jobs(
    engine: Engine,
    *,
    session_ids: Optional[Sequence[int | str]] = None,
    limit: Optional[int] = None,
) -> list[ConversationJob]:
    """
    Read-only: load sessions from `aisha.user_consultation` and build jobs.
    `user_phone` is taken as `str(session_id)` when no separate phone column exists.
    """
    ids: list[int | str] = (
        list(session_ids)
        if session_ids is not None
        else fetch_all_session_ids(engine=engine)
    )
    if limit is not None:
        ids = ids[: max(0, limit)]

    jobs: list[ConversationJob] = []
    for sid in ids:
        turns = fetch_conversation_by_session_id(sid, engine=engine)
        if not turns:
            continue
        sid_str = str(sid)
        jobs.append(
            ConversationJob(
                session_id=sid_str,
                user_phone=sid_str,
                transcript=format_transcript(turns),
            )
        )
    return jobs


def _user_content(job: ConversationJob) -> str:
    return f"Session ID: {job.session_id}\n\nConversation transcript:\n{job.transcript}"


def chat_completion_body(job: ConversationJob, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(job)},
        ],
        "response_format": _response_format_dict(),
    }


def prepare_batch_file(
    jobs: Sequence[ConversationJob],
    output_path: str | Path,
) -> dict[str, dict[str, str]]:
    """
    Write a Batch API JSONL file; return map custom_id -> {session_id, user_phone}.
    One model per file (uses get_model_name()).
    """
    model = get_model_name()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict[str, str]] = {}
    with path.open("w", encoding="utf-8") as f:
        for job in jobs:
            cid = job.custom_id
            meta[cid] = {"session_id": job.session_id, "user_phone": job.user_phone}
            row = {
                "custom_id": cid,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": chat_completion_body(job, model),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return meta


def upload_batch_input(client: OpenAI, jsonl_path: str | Path) -> str:
    path = Path(jsonl_path)
    with path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    return uploaded.id


def create_chat_completion_batch(
    client: OpenAI,
    input_file_id: str,
    *,
    metadata: Optional[MutableMapping[str, str]] = None,
) -> Any:
    return client.batches.create(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata=dict(metadata or ()),
    )


def batch_status(batch: Any) -> str:
    s = getattr(batch, "status", "") or ""
    return s if isinstance(s, str) else str(getattr(s, "value", s))


def wait_for_batch_completion(
    client: OpenAI,
    batch_id: str,
    *,
    poll_interval_sec: float = 15.0,
    timeout_sec: Optional[float] = None,
) -> Any:
    """
    Poll until the batch is terminal. Raises TimeoutError if timeout_sec exceeded.
    """
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch_status(batch) in terminal:
            return batch
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"Batch {batch_id} still status={batch_status(batch)} after timeout"
            )
        time.sleep(poll_interval_sec)


def download_file_text(client: OpenAI, file_id: str) -> str:
    return client.files.content(file_id).text


def iter_batch_jsonl_lines(blob: str) -> Iterator[dict[str, Any]]:
    for line in blob.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def parse_successful_analysis(
    record: Mapping[str, Any],
) -> tuple[Optional[AishaConversationAnalysis], Optional[str]]:
    """From one output JSONL object, return (analysis, error_message)."""
    if record.get("error"):
        err = record["error"]
        if isinstance(err, dict):
            return None, err.get("message") or str(err)
        return None, str(err)
    resp = record.get("response") or {}
    code = resp.get("status_code")
    if code is not None and code != 200:
        return None, f"HTTP {code}"
    body = resp.get("body")
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        return None, "invalid response body"
    choices = body.get("choices") or []
    if not choices:
        return None, "no choices in response body"
    msg = (choices[0].get("message") or {}).get("content")
    if not msg:
        return None, "empty message content"
    if isinstance(msg, dict):
        data = msg
    else:
        data = json.loads(msg)
    try:
        return AishaConversationAnalysis.model_validate(data), None
    except Exception as exc:
        return None, f"validate: {exc}"


def save_batch_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_batch_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _session_id_typed(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


def persist_batch_results(
    output_jsonl: str,
    custom_id_meta: Mapping[str, Mapping[str, str]],
    model_name: str,
    engine: Engine,
) -> dict[str, int]:
    """
    Validate each successful line as `AishaConversationAnalysis` and insert into
    `aisha_conversation_analysis` (via db.insert_conversation_analysis).
    """
    inserted = 0
    skipped = 0
    for record in iter_batch_jsonl_lines(output_jsonl):
        cid = record.get("custom_id")
        if not cid or cid not in custom_id_meta:
            skipped += 1
            continue
        meta = custom_id_meta[cid]
        analysis, err = parse_successful_analysis(record)
        if err or analysis is None:
            skipped += 1
            continue
        analysis = analysis.model_copy(
            update={"conversation_id": str(meta["session_id"])}
        )
        insert_conversation_analysis(
            _session_id_typed(meta["session_id"]),
            meta["user_phone"],
            model_name,
            analysis,
            engine=engine,
        )
        inserted += 1
    return {"inserted": inserted, "skipped_or_failed": skipped}
