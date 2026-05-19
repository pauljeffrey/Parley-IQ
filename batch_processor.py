"""
OpenAI Batch API utilities: build JSONL from consultations, submit, poll, parse, persist via db.py.
`user_consultation` is only read through db helpers — never written here.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

from openai import OpenAI
from sqlalchemy.engine import Engine

from db import (
    ConsultationTurn,
    batch_conversation_since,
    consultation_user_identifier,
    conversation_started_at,
    fetch_all_session_ids,
    fetch_analyzed_session_ids,
    fetch_conversation_by_session_id,
    insert_conversation_analysis,
    min_conversation_turns,
)
from output import ConversationAnalysis

BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical conversation analyst. Analyze the full transcript into the "
    "required JSON schema. Do not include session_id, user_id, phone, model name, "
    "timestamps, or segment_index. Return only topic_segments, sdoh_profiles, cultural_notes."
)

_TERMINAL_BATCH = frozenset({"completed", "failed", "expired", "cancelled"})


def env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def batch_poll_interval_sec() -> float:
    return float(os.environ.get("BATCH_POLL_INTERVAL_SEC", "15"))


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


def exclude_completed_session_ids(
    ids: Sequence[int | str],
    *,
    engine: Engine | None = None,
) -> list[int | str]:
    if engine is None:
        done = load_completed_session_ids() if skip_completed_sessions() else set()
    else:
        done = completed_session_ids(engine)
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
    schema = ConversationAnalysis.llm_json_schema()
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


def batch_shard_size() -> int:
    raw = os.environ.get("BATCH_SHARD_SIZE", "5000").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5000


def batch_work_dir() -> Path:
    return Path(os.environ.get("BATCH_WORK_DIR", "batch_work"))


def batch_manifest_path() -> Path:
    return batch_work_dir() / os.environ.get("BATCH_MANIFEST_JSON", "manifest.json")


def batch_max_sessions() -> int | None:
    raw = os.environ.get("BATCH_MAX_SESSIONS", "").strip()
    return int(raw) if raw else None


def skip_already_analyzed_sessions() -> bool:
    return env_bool("BATCH_SKIP_ALREADY_ANALYZED", True)


def completed_session_ids(engine: Engine) -> set[str]:
    """Union of file cache and session_ids already in conversation_analysis."""
    done: set[str] = set()
    if skip_completed_sessions():
        done |= load_completed_session_ids()
    if skip_already_analyzed_sessions():
        done |= fetch_analyzed_session_ids(engine=engine)
    return done


def collect_pending_session_ids(engine: Engine) -> list[int]:
    ids = fetch_all_session_ids(
        engine=engine,
        min_turns=min_conversation_turns(),
        since=batch_conversation_since(),
    )
    skip = completed_session_ids(engine)
    ids = [int(i) for i in ids if str(i) not in skip]
    cap = batch_max_sessions()
    return ids[:cap] if cap is not None else ids


def _chunks(items: Sequence[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def load_consultation_jobs(
    engine: Engine,
    *,
    session_ids: Optional[Sequence[int | str]] = None,
    limit: Optional[int] = None,
    min_turns: int | None = None,
) -> list[ConversationJob]:
    """Read-only jobs for given session ids (or pending pool when ids omitted)."""
    min_turns = min_turns if min_turns is not None else min_conversation_turns()
    if session_ids is not None:
        ids: list[int | str] = list(session_ids)
    else:
        ids = collect_pending_session_ids(engine)
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
                user_phone=consultation_user_identifier(turns) or sid_str,
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
        conv_started = conversation_started_at(turns)
        if conv_started is None:
            skipped += 1
            continue
        user_phone = consultation_user_identifier(turns) or meta["user_phone"]
        insert_conversation_analysis(
            sid,
            user_phone,
            model_name,
            analysis,
            engine=engine,
            conversation_started=conv_started,
        )
        inserted += 1
        newly_completed.append(str(sid))
    mark_sessions_completed(newly_completed)
    return {
        "inserted": inserted,
        "skipped_or_failed": skipped,
        "cached_completed": len(newly_completed),
    }


@dataclass
class BatchShard:
    index: int
    input_path: Path
    output_path: Path
    error_path: Path
    meta_path: Path
    session_count: int = 0
    batch_id: str | None = None
    status: str = "pending"
    persist_stats: dict[str, int] | None = None
    custom_id_meta: dict[str, dict[str, str]] = field(default_factory=dict)


def _shard_paths(work: Path, index: int) -> tuple[Path, Path, Path, Path]:
    d = work / f"shard_{index:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "input.jsonl", d / "output.jsonl", d / "errors.jsonl", d / "meta.json"


def build_shards(engine: Engine, session_ids: Sequence[int]) -> list[BatchShard]:
    shards: list[BatchShard] = []
    work = batch_work_dir()
    work.mkdir(parents=True, exist_ok=True)
    for chunk in _chunks(list(session_ids), batch_shard_size()):
        jobs = load_consultation_jobs(engine, session_ids=chunk, min_turns=min_conversation_turns())
        if not jobs:
            continue
        index = len(shards)
        inp, out, err, meta_p = _shard_paths(work, index)
        meta = prepare_batch_file(jobs, inp)
        save_batch_metadata(meta_p, {"custom_id_meta": meta, "session_count": len(jobs)})
        shards.append(
            BatchShard(
                index=index,
                input_path=inp,
                output_path=out,
                error_path=err,
                meta_path=meta_p,
                session_count=len(jobs),
                custom_id_meta=meta,
            )
        )
    return shards


def _shard_to_dict(s: BatchShard) -> dict[str, Any]:
    return {
        "index": s.index,
        "batch_id": s.batch_id,
        "status": s.status,
        "session_count": s.session_count,
        "input_path": str(s.input_path),
        "output_path": str(s.output_path),
        "persist_stats": s.persist_stats,
    }


def save_manifest(shards: Sequence[BatchShard], *, model_name: str) -> None:
    path = batch_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    since = batch_conversation_since()
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "since": since.isoformat() if since else None,
                "shards": [_shard_to_dict(s) for s in shards],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def submit_shards(client: OpenAI, shards: Sequence[BatchShard]) -> None:
    for s in shards:
        if s.batch_id:
            continue
        file_id = upload_batch_input(client, s.input_path)
        batch = create_chat_completion_batch(client, file_id, metadata={"shard": str(s.index)})
        s.batch_id = batch.id
        s.status = batch_status(batch)
        meta = load_batch_metadata(s.meta_path)
        meta.update({"batch_id": s.batch_id, "input_file_id": file_id, "status": s.status})
        save_batch_metadata(s.meta_path, meta)


def _finalize_shard(
    client: OpenAI,
    shard: BatchShard,
    batch: Any,
    *,
    engine: Engine,
    model_name: str,
) -> None:
    shard.status = batch_status(batch)
    if shard.status != "completed" or not batch.output_file_id:
        if batch.error_file_id:
            shard.error_path.write_text(download_file_text(client, batch.error_file_id), encoding="utf-8")
        return
    out_text = download_file_text(client, batch.output_file_id)
    shard.output_path.write_text(out_text, encoding="utf-8")
    meta = load_batch_metadata(shard.meta_path)
    shard.persist_stats = persist_batch_results(
        out_text, meta.get("custom_id_meta", shard.custom_id_meta), model_name, engine
    )


def poll_shards_and_persist(
    client: OpenAI,
    shards: list[BatchShard],
    *,
    engine: Engine,
    model_name: str,
    poll_interval_sec: float,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    totals = {"inserted": 0, "skipped_or_failed": 0, "cached_completed": 0, "shards_done": 0}
    open_shards = [s for s in shards if s.batch_id and s.persist_stats is None]

    while open_shards:
        for s in list(open_shards):
            batch = client.batches.retrieve(s.batch_id)
            st = batch_status(batch)
            if st not in _TERMINAL_BATCH:
                continue
            _finalize_shard(client, s, batch, engine=engine, model_name=model_name)
            open_shards.remove(s)
            if s.persist_stats:
                totals["shards_done"] += 1
                for k in ("inserted", "skipped_or_failed", "cached_completed"):
                    totals[k] += s.persist_stats.get(k, 0)
                log(
                    f"Shard {s.index:04d} {st}: inserted={s.persist_stats.get('inserted', 0)}, "
                    f"skipped={s.persist_stats.get('skipped_or_failed', 0)}"
                )
            else:
                log(f"Shard {s.index:04d} ended {st} (no persist)")
            save_manifest(shards, model_name=model_name)
        if open_shards:
            time.sleep(poll_interval_sec)
    return totals


def run_sharded_batch_pipeline(
    engine: Engine,
    *,
    wait: bool = True,
    poll_interval_sec: float = 15.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build shards, submit all OpenAI batches, persist each shard as it completes."""
    skip = completed_session_ids(engine)
    ids = collect_pending_session_ids(engine)
    since = batch_conversation_since()
    log(f"Skipping {len(skip)} completed/analyzed session(s); pending: {len(ids)}")
    if since:
        log(f"Date filter: MAX(turn created_at) >= {since.date()}")
    if not ids:
        return {"shards": 0, "sessions": 0}

    shards = build_shards(engine, ids)
    if not shards:
        return {"shards": 0, "sessions": 0}

    model_name = get_model_name()
    client = get_openai_client()
    log(f"Prepared {len(shards)} shard(s), {sum(s.session_count for s in shards)} sessions")
    submit_shards(client, shards)
    for s in shards:
        log(f"Submitted shard {s.index:04d} batch_id={s.batch_id} n={s.session_count}")
    save_manifest(shards, model_name=model_name)

    if not wait:
        log("BATCH_WAIT_FOR_COMPLETION=false — exit after submit.")
        return {"shards": len(shards), "sessions": sum(s.session_count for s in shards), "submitted": True}

    totals = poll_shards_and_persist(
        client, shards, engine=engine, model_name=model_name, poll_interval_sec=poll_interval_sec, log=log
    )
    log(f"Done: {totals}")
    return {"shards": len(shards), "sessions": sum(s.session_count for s in shards), **totals}
