"""HTTP API exposing analytics_agent."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import analysis_queries as aq
from analytics_agent import (
    AnalyticsAgentDeps,
    analytics_agent_run,
    coerce_message_history,
    encode_figure_artifacts,
    zip_b64_optional,
)
from db import get_engine

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


app = FastAPI(title="Parley Conversation Analytics")


class AnalyticsChatPayload(BaseModel):
    message: str = Field(..., min_length=1, description="User question or instruction for the agent.")
    chat_history: list[Any] = Field(
        default_factory=list,
        description=(
            "Either pydantic-ai ModelMessage JSON (preferred) or simple objects with `role` + `content`."
        ),
    )


class FigureItem(BaseModel):
    filename: str = Field(..., description="Basename of the saved plot file.")
    relative_path: str = Field(..., description="Path within the agent work directory (also used in the zip archive).")
    content_type: str = Field(..., description="MIME type, e.g. image/png.")
    data_base64: str = Field(..., description="Base64-encoded file bytes for inline display or download.")


class AnalyticsChatResponse(BaseModel):
    output: str
    new_messages: list[Any] = Field(
        default_factory=list,
        description="Messages produced this turn; append to client-side history for continuity.",
    )
    figures: list[FigureItem] = Field(
        default_factory=list,
        description="Plot artifacts produced this turn (inline base64 for each saved figure).",
    )
    figures_zip_base64: str | None = Field(
        default=None,
        description="Optional zip of all figures when more than one chart was saved.",
    )
    figures_zip_filename: str = "analytics_figures.zip"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analytics/chat", response_model=AnalyticsChatResponse)
def analytics_chat(body: AnalyticsChatPayload) -> AnalyticsChatResponse:
    history = coerce_message_history(body.chat_history)
    workdir = Path(tempfile.mkdtemp(prefix="parley_analytics_"))
    engine = None
    try:
        engine = get_engine()
        deps = AnalyticsAgentDeps(
            engine=engine,
            work_dir=workdir,
            qualified_table=aq.qualified_analysis_table(engine),
        )
        output_text, new_msgs = analytics_agent_run(
            body.message,
            message_history=history or None,
            deps=deps,
        )
        figures = [
            FigureItem(
                filename=item.filename,
                relative_path=item.relative_path,
                content_type=item.content_type,
                data_base64=item.data_base64,
            )
            for item in encode_figure_artifacts(deps.work_dir, deps.artifact_relpaths)
        ]
        zip_b64 = zip_b64_optional(deps.work_dir, deps.artifact_relpaths) if figures else None
        return AnalyticsChatResponse(
            output=output_text,
            new_messages=new_msgs,
            figures=figures,
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
