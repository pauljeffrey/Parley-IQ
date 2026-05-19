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
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

from openai import OpenAI
from sqlalchemy.engine import Engine

from db import (
    ConsultationTurn,
    conversation_started_at,
    fetch_all_session_ids,
    fetch_conversation_by_session_id,
    insert_conversation_analysis,
    min_conversation_turns,
)
from output import ConversationAnalysis

BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical conversation analyst. Analyze the full transcript into the "
    "required JSON schema provided."
)


def skip_completed_sessions() -> bool:
    return os.environ.get("BATCH_SKIP_COMPLETED", "true").lower() in ("1", "true", "yes", "on")


def completed_sessions_cache_path() -> Path:
    return Path(os.environ.get("BATCH_COMPLETED_SESSIONS_CACHE", "batch_completed_sessions.json"))


def load_completed_session_ids() -> set[str]:
    path = completed_sessions_cache_path()
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("completed_session_ids", [])
    return {str(sid) for sid in raw}


def _save_completed_session_ids(ids: set[str]) -> None:
    path = completed_sessions_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed_session_ids": sorted(ids, key=lambda x: (len(x), x))}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_sessions_completed(session_ids: Iterable[int | str]) -> None:
    """Record session ids successfully persisted after a completed batch job."""
    new_ids = {str(sid) for sid in session_ids}
    if not new_ids:
        return
    done = load_completed_session_ids()
    done.update(new_ids)
    _save_completed_session_ids(done)


def exclude_completed_session_ids(ids: Sequence[int | str]) -> list[int | str]:
    if not skip_completed_sessions():
        return list(ids)
    done = load_completed_session_ids()
    return [sid for sid in ids if str(sid) not in done]


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
    schema = ConversationAnalysis.model_json_schema()
    rf: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "conversation_analysis",
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
    transcript: str

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
    min_turns: int | None = None,
) -> list[ConversationJob]:
    """
    Read-only: load sessions from the configured conversation table and build jobs.
    Skips sessions with fewer than `min_turns` user–assistant pairs (default: BATCH_MIN_TURNS).
    """
    min_turns = min_turns if min_turns is not None else min_conversation_turns()
    ids: list[int | str] = (
        list(session_ids)
        if session_ids is not None
        else fetch_all_session_ids(engine=engine, min_turns=min_turns)
    )
    ids = exclude_completed_session_ids(ids)
    if limit is not None:
        ids = ids[: max(0, limit)]

    jobs: list[ConversationJob] = []
    for sid in ids:
        turns = fetch_conversation_by_session_id(sid, engine=engine)
        if len(turns) < min_turns:
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
    return f"Conversation transcript:\n{job.transcript}"


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


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _coerce_conversation_analysis(data: dict[str, Any]) -> tuple[ConversationAnalysis | None, str | None]:
    """Map non-empty LLM fields onto ConversationAnalysis (with alias support)."""
    aliases = {
        "segments": "topic_segments",
        "topic_segment": "topic_segments",
        "sdoh_indicators": "sdoh_profiles",
        "sdoh": "sdoh_profiles",
    }
    payload: dict[str, Any] = dict(data)
    for old_key, new_key in aliases.items():
        if old_key in payload and new_key not in payload and _non_empty(payload[old_key]):
            payload[new_key] = payload.pop(old_key)

    cleaned: dict[str, Any] = {}
    for field_name in ConversationAnalysis.model_fields:
        if field_name in payload and _non_empty(payload[field_name]):
            cleaned[field_name] = payload[field_name]

    if "topic_segments" not in cleaned:
        cleaned["topic_segments"] = []
    if "sdoh_profiles" not in cleaned:
        cleaned["sdoh_profiles"] = []
    if "cultural_notes" not in cleaned:
        cleaned["cultural_notes"] = ""

    try:
        return ConversationAnalysis.model_validate(cleaned), None
    except Exception as strict_exc:
        try:
            partial = ConversationAnalysis.model_construct(**cleaned)
            return ConversationAnalysis.model_validate(partial.model_dump(mode="json")), None
        except Exception:
            return None, f"validate: {strict_exc}"


def parse_successful_analysis(
    record: Mapping[str, Any],
) -> tuple[Optional[ConversationAnalysis], Optional[str]]:
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
    if not isinstance(data, dict):
        return None, "message content is not a JSON object"
    return _coerce_conversation_analysis(data)


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
    Validate each successful line as `ConversationAnalysis` and insert into
    the configured analysis table (via db.insert_conversation_analysis).
    """
    inserted = 0
    skipped = 0
    newly_completed: list[str] = []
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
        sid = _session_id_typed(meta["session_id"])
        turns = fetch_conversation_by_session_id(sid, engine=engine)
        if not analysis.topic_segments:
            skipped += 1
            continue
        insert_conversation_analysis(
            sid,
            meta["user_phone"],
            model_name,
            analysis,
            engine=engine,
            conversation_started=conversation_started_at(turns),
        )
        inserted += 1
        newly_completed.append(str(sid))
    mark_sessions_completed(newly_completed)
    return {
        "inserted": inserted,
        "skipped_or_failed": skipped,
        "cached_completed": len(newly_completed),
    }
