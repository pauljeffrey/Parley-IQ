"""pydantic-ai analytics_agent for the conversation analysis table."""

from __future__ import annotations

import base64
import builtins
import datetime as datetime_mod
import io
import json as json_mod
import os
import pathlib
import re
import traceback
import zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sqlalchemy.engine import Engine

import analysis_queries as aq

_ALLOWED_BUILTINS_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytes",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "False",
        "True",
        "None",
        "Exception",
        "BaseException",
        "ValueError",
        "KeyError",
        "TypeError",
        "ArithmeticError",
    }
)

FILENAME_SAFE = re.compile(r"^[A-Za-z0-9_\-.]+\.(png|jpg|jpeg|pdf|svg)$")


@dataclass
class AnalyticsAgentDeps:
    """Per-request deps for analytics_agent."""

    engine: Engine
    work_dir: pathlib.Path
    qualified_table: str
    artifact_relpaths: list[str] = field(default_factory=list)

    def record_artifact(self, path: pathlib.Path) -> None:
        rel = path.resolve().relative_to(self.work_dir.resolve())
        posix = rel.as_posix()
        if posix not in self.artifact_relpaths:
            self.artifact_relpaths.append(posix)


analytics_agent: Agent[AnalyticsAgentDeps, str] | None = None


def _path_within_root(candidate: pathlib.Path, root_resolved: pathlib.Path) -> bool:
    cand = pathlib.Path(candidate).resolve()
    try:
        cand.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _safe_open_factory(root: pathlib.Path):
    root_resolved = root.resolve()

    def _open(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        raw = pathlib.Path(os.path.abspath(str(file))).resolve()
        if not _path_within_root(raw, root_resolved):
            raise PermissionError("Filesystem access confined to sandbox working directory.")
        return open(raw, mode, *args, **kwargs)  # noqa: SIM115

    return _open


def _restricted_builtins(root: pathlib.Path) -> dict:
    sandbox = {}
    for name in _ALLOWED_BUILTINS_NAMES:
        obj = getattr(builtins, name, None)
        if obj is not None:
            sandbox[name] = obj
    sandbox["open"] = _safe_open_factory(root)
    return sandbox


def _analytics_agent_model() -> OpenAIChatModel:
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY or OPENAI_API_KEY in the environment.")
    raw = (os.environ.get("MODEL_NAME") or "gpt-4o-mini").strip()
    provider = OpenAIProvider(api_key=api_key)
    return OpenAIChatModel(raw.removeprefix("openai:"), provider=provider)


def _analytics_agent_toolset() -> FunctionToolset[AnalyticsAgentDeps]:
    ts = FunctionToolset[AnalyticsAgentDeps]()

    @ts.tool
    def execute_custom_query(ctx: RunContext[AnalyticsAgentDeps], sql: str) -> str:
        """
        Run a validated read-only SQL string against the conversation analysis table.
        Pass the full query exactly as it should execute (`SELECT` or `WITH … SELECT`).
        Must reference the session qualified table (see system prompt). No JOINs.
        """
        rows = aq.run_validated_select(sql=sql, engine=ctx.deps.engine)
        return aq.format_rows_preview(rows)

    @ts.tool
    def structured_table_fetch(
        ctx: RunContext[AnalyticsAgentDeps],
        columns: list[str],
        where_equal_json: dict | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> str:
        """Equality filters on allow-listed columns; use exact vocabulary values from the prompt."""
        rows = aq.structured_fetch(
            engine=ctx.deps.engine,
            columns=columns,
            limit=limit,
            where_equal=where_equal_json,
            order_by=order_by,
            descending=descending,
        )
        return aq.format_rows_preview(rows)

    @ts.tool
    def run_python_for_analysis(ctx: RunContext[AnalyticsAgentDeps], code: str) -> str:
        """Restricted Python (`pd`, `np`); use `run_select(sql)` or `_result` for compact output."""
        root = ctx.deps.work_dir

        def run_select(sql: str) -> list[dict]:
            return aq.run_validated_select(sql=sql, engine=ctx.deps.engine)

        def load_analysis_df(lim: int | None = None) -> pd.DataFrame:
            return aq.load_analysis_dataframe(engine=ctx.deps.engine, limit=lim)

        ns = {
            "__builtins__": _restricted_builtins(root),
            "pd": pd,
            "np": np,
            "pathlib": pathlib,
            "json": json_mod,
            "datetime": datetime_mod,
            "load_analysis_df": load_analysis_df,
            "run_select": run_select,
            "WORK_ROOT": root,
            "zip": builtins.zip,
        }
        return _exec_sandbox(code, ns)

    @ts.tool
    def plot_and_save_figure(ctx: RunContext[AnalyticsAgentDeps], code: str, filename: str) -> str:
        """Matplotlib plot; call `plt.savefig(OUTPUT_PATH)` or rely on auto-save."""
        if not FILENAME_SAFE.match(filename):
            raise ValueError("filename must be ascii-safe with extension png|jpg|jpeg|pdf|svg")

        plots_dir = ctx.deps.work_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = (plots_dir / filename).resolve()
        if plots_dir.resolve() not in out_path.parents:
            raise PermissionError("Invalid plot path.")

        def run_select(sql: str) -> list[dict]:
            return aq.run_validated_select(sql=sql, engine=ctx.deps.engine)

        def load_analysis_df(lim: int | None = None) -> pd.DataFrame:
            return aq.load_analysis_dataframe(engine=ctx.deps.engine, limit=lim)

        plt.close("all")
        ns = {
            "__builtins__": _restricted_builtins(ctx.deps.work_dir),
            "plt": plt,
            "matplotlib": matplotlib,
            "pd": pd,
            "np": np,
            "OUTPUT_PATH": str(out_path),
            "load_analysis_df": load_analysis_df,
            "run_select": run_select,
            "pathlib": pathlib,
        }
        err: BaseException | None = None
        try:
            exec(code, ns, ns)  # noqa: S102
        except BaseException as exc:  # noqa: BLE001
            err = exc

        if err is not None:
            plt.close("all")
            return f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=6)}".strip()

        wrote = out_path.is_file() and out_path.stat().st_size > 0
        if not wrote:
            try:
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(out_path, bbox_inches="tight")
                    wrote = out_path.stat().st_size > 0
            except BaseException:
                pass
        plt.close("all")

        if not wrote:
            return "Figure was not persisted; use plt.savefig(OUTPUT_PATH)."
        ctx.deps.record_artifact(out_path)
        return f"Saved plot artifact at `{out_path.relative_to(ctx.deps.work_dir)}`."

    return ts


def _exec_sandbox(code: str, ns: dict) -> str:
    stdout = io.StringIO()
    err: BaseException | None = None
    try:
        with redirect_stdout(stdout):
            exec(code, ns, ns)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001
        err = exc

    parts: list[str] = []
    if buf := stdout.getvalue().strip():
        parts.append(buf)
    if err is None and "_result" in ns:
        parts.append(str(ns["_result"]))
    elif err is not None:
        parts.append(f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=5)}".strip())

    preview = "\n\n".join(parts).strip()
    cap = int(os.environ.get("ANALYTICS_MAX_PYTHON_OUTPUT_CHARS", "16000"))
    if len(preview) > cap:
        preview = preview[: cap // 2] + "\n…[truncated]…\n" + preview[-cap // 2 :]
    return preview or "(no textual output)"


def _ensure_analytics_agent() -> Agent[AnalyticsAgentDeps, str]:
    global analytics_agent
    if analytics_agent is None:
        analytics_agent = Agent(
            _analytics_agent_model(),
            deps_type=AnalyticsAgentDeps,
            instructions=aq.analytics_agent_instructions(),
            toolsets=[_analytics_agent_toolset()],
        )
    return analytics_agent


def analytics_agent_run(
    user_message: str,
    *,
    message_history: list[ModelMessage] | None,
    deps: AnalyticsAgentDeps,
) -> tuple[str, list[ModelMessage]]:
    """Run analytics_agent for one user turn."""
    agent = _ensure_analytics_agent()
    result = agent.run_sync(
        user_message,
        message_history=message_history or None,
        deps=deps,
        instructions=aq.analytics_agent_instructions(qualified_table=deps.qualified_table),
    )
    serialized_new = ModelMessagesTypeAdapter.dump_python(result.new_messages(), mode="json")
    return str(result.output), serialized_new


def coerce_message_history(history: object | None) -> list[ModelMessage]:
    if history is None:
        return []
    if isinstance(history, list) and history:
        try:
            return ModelMessagesTypeAdapter.validate_python(history)
        except Exception:
            pass
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        out: list[ModelMessage] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower().strip()
            content = "" if item.get("content") is None else str(item.get("content"))
            if role == "user":
                out.append(ModelRequest(parts=[UserPromptPart(content)]))
            elif role == "assistant":
                out.append(ModelResponse(parts=[TextPart(content)]))
        return out
    return []


def zip_artifacts(work_dir: pathlib.Path, artifact_relpaths: list[str]) -> bytes | None:
    if not artifact_relpaths:
        return None
    buf = io.BytesIO()
    base = work_dir.resolve()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in artifact_relpaths:
            fp = base / pathlib.Path(rel)
            if fp.is_file() and fp.resolve().is_relative_to(base):
                zf.write(fp, arcname=rel)
    return buf.getvalue()


def zip_b64_optional(work_dir: pathlib.Path, artifact_relpaths: list[str]) -> str | None:
    raw = zip_artifacts(work_dir, artifact_relpaths)
    if raw is None:
        return None
    return base64.standard_b64encode(raw).decode("ascii")
