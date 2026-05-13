"""HTTP API exposing the pydantic-ai analytics agent."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analytics_agent import (
    AnalyticsAgentDeps,
    coerce_message_history,
    run_analytics_sync,
    zip_b64_optional,
)
from db import get_engine

load_dotenv(Path(__file__).resolve().parent / ".env")


app = FastAPI(title="AISHA Conversation Analytics")


class AnalyticsChatPayload(BaseModel):
    message: str = Field(..., min_length=1, description="User question or instruction for the agent.")
    chat_history: list[Any] = Field(
        default_factory=list,
        description=(
            "Either pydantic-ai ModelMessage JSON (preferred) or simple objects with `role` + `content`."
        ),
    )


class AnalyticsChatResponse(BaseModel):
    output: str
    new_messages: list[Any] = Field(
        default_factory=list,
        description="Messages produced this turn; append to client-side history for continuity.",
    )
    figures_zip_base64: str | None = None
    figures_zip_filename: str = "analytics_figures.zip"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analytics/chat", response_model=AnalyticsChatResponse)
def analytics_chat(body: AnalyticsChatPayload) -> AnalyticsChatResponse:
    history = coerce_message_history(body.chat_history)
    workdir = Path(tempfile.mkdtemp(prefix="aisha_analytics_"))
    engine = None
    try:
        engine = get_engine()
        deps = AnalyticsAgentDeps(engine=engine, work_dir=workdir)
        output_text, new_msgs = run_analytics_sync(
            body.message,
            message_history=history or None,
            deps=deps,
        )
        zip_b64 = zip_b64_optional(deps.work_dir, deps.artifact_relpaths)
        return AnalyticsChatResponse(
            output=output_text,
            new_messages=new_msgs,
            figures_zip_base64=zip_b64,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface DB/model failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if engine is not None:
            engine.dispose()
        shutil.rmtree(workdir, ignore_errors=True)
