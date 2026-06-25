"""pydantic-ai analytics_agent for the conversation analysis table."""

from __future__ import annotations

import base64
import builtins
import datetime as datetime_mod
import io
import json as json_mod
import mimetypes
import os
import pathlib
import re
import traceback
import zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Literal

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

ChartType = Literal["bar", "horizontal_bar", "line", "pie"]

_CONTENT_TYPE_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".svg": "image/svg+xml",
}


@dataclass(frozen=True)
class EncodedFigure:
    """One plot artifact ready for API clients."""

    filename: str
    relative_path: str
    content_type: str
    data_base64: str


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


def content_type_for_artifact(path: pathlib.Path | str) -> str:
    suffix = pathlib.Path(path).suffix.lower()
    return _CONTENT_TYPE_BY_SUFFIX.get(suffix) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _resolve_plot_output(work_dir: pathlib.Path, filename: str) -> pathlib.Path:
    if not FILENAME_SAFE.match(filename):
        raise ValueError("filename must be ascii-safe with extension png|jpg|jpeg|pdf|svg")
    plots_dir = work_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = (plots_dir / filename).resolve()
    if plots_dir.resolve() not in out_path.parents:
        raise PermissionError("Invalid plot path.")
    return out_path


def _plot_query_helpers(engine: Engine) -> dict:
    def run_select(sql: str) -> list[dict]:
        return aq.run_validated_select(sql=sql, engine=engine)

    def load_analysis_df(lim: int | None = None) -> pd.DataFrame:
        return aq.load_analysis_dataframe(engine=engine, limit=lim)

    return {"run_select": run_select, "load_analysis_df": load_analysis_df}


def _save_chart_from_dataframe(
    df: pd.DataFrame,
    *,
    chart_type: ChartType,
    x_column: str,
    y_column: str,
    title: str,
    out_path: pathlib.Path,
) -> None:
    if df.empty:
        raise ValueError("Query returned no rows; cannot chart an empty result.")
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(
            f"Expected columns {x_column!r} and {y_column!r}; got {list(df.columns)}."
        )

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = df[x_column]
    y_vals = pd.to_numeric(df[y_column], errors="coerce")
    if y_vals.isna().all():
        raise ValueError(f"Column {y_column!r} must contain numeric values for charting.")

    if chart_type == "bar":
        ax.bar(x_vals.astype(str), y_vals)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif chart_type == "horizontal_bar":
        ax.barh(x_vals.astype(str), y_vals)
        ax.set_xlabel(y_column)
        ax.set_ylabel(x_column)
    elif chart_type == "line":
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif chart_type == "pie":
        ax.pie(y_vals, labels=x_vals.astype(str), autopct="%1.1f%%")
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type!r}")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _persist_plot_artifact(ctx: RunContext[AnalyticsAgentDeps], out_path: pathlib.Path) -> str:
    if not out_path.is_file() or out_path.stat().st_size <= 0:
        return "Figure was not persisted; ensure the plot was rendered before saving."
    ctx.deps.record_artifact(out_path)
    rel = out_path.relative_to(ctx.deps.work_dir)
    return f"Saved plot artifact at `{rel.as_posix()}`. It will be returned to the user in the API response."


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
        """Restricted Python for stats (`pd`, `np`). For charts, use `create_chart` or `plot_and_save_figure`."""
        root = ctx.deps.work_dir
        helpers = _plot_query_helpers(ctx.deps.engine)
        ns = {
            "__builtins__": _restricted_builtins(root),
            "pd": pd,
            "np": np,
            "pathlib": pathlib,
            "json": json_mod,
            "datetime": datetime_mod,
            "WORK_ROOT": root,
            "zip": builtins.zip,
            **helpers,
        }
        return _exec_sandbox(code, ns)

    @ts.tool
    def create_chart(
        ctx: RunContext[AnalyticsAgentDeps],
        sql: str,
        chart_type: ChartType,
        x_column: str,
        y_column: str,
        title: str,
        filename: str,
    ) -> str:
        """
        Build a chart from a validated SQL aggregate query and save it for the API response.
        Prefer this for standard bar, line, horizontal bar, and pie charts.
        SQL should return small aggregated rows with the given x/y column names.
        """
        rows = aq.run_validated_select(sql=sql, engine=ctx.deps.engine)
        df = pd.DataFrame(rows)
        out_path = _resolve_plot_output(ctx.deps.work_dir, filename)
        _save_chart_from_dataframe(
            df,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            title=title,
            out_path=out_path,
        )
        return _persist_plot_artifact(ctx, out_path)

    @ts.tool
    def plot_and_save_figure(ctx: RunContext[AnalyticsAgentDeps], code: str, filename: str) -> str:
        """
        Advanced matplotlib plotting in a sandbox (`plt`, `pd`, `np`, `OUTPUT_PATH`).
        Use for custom layouts; otherwise prefer `create_chart`.
        Saved files are attached to the API response automatically.
        """
        out_path = _resolve_plot_output(ctx.deps.work_dir, filename)
        helpers = _plot_query_helpers(ctx.deps.engine)
        plt.close("all")
        ns = {
            "__builtins__": _restricted_builtins(ctx.deps.work_dir),
            "plt": plt,
            "matplotlib": matplotlib,
            "pd": pd,
            "np": np,
            "OUTPUT_PATH": str(out_path),
            "pathlib": pathlib,
            **helpers,
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
            return "Figure was not persisted; call plt.savefig(OUTPUT_PATH) or draw on the active axes."
        return _persist_plot_artifact(ctx, out_path)

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


def encode_figure_artifacts(
    work_dir: pathlib.Path,
    artifact_relpaths: list[str],
) -> list[EncodedFigure]:
    """Inline base64 payloads for each saved plot (primary client delivery path)."""
    base = work_dir.resolve()
    encoded: list[EncodedFigure] = []
    for rel in artifact_relpaths:
        fp = (base / pathlib.Path(rel)).resolve()
        if not fp.is_file() or not fp.resolve().is_relative_to(base):
            continue
        encoded.append(
            EncodedFigure(
                filename=fp.name,
                relative_path=rel,
                content_type=content_type_for_artifact(fp),
                data_base64=base64.standard_b64encode(fp.read_bytes()).decode("ascii"),
            )
        )
    return encoded
