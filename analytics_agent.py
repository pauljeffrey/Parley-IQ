"""pydantic-ai analytics agent targeting `aisha_conversation_analysis`."""

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


def _path_within_root(candidate: pathlib.Path, root_resolved: pathlib.Path) -> bool:
    cand = pathlib.Path(candidate).resolve()
    try:
        cand.relative_to(root_resolved)
    except ValueError:
        return False
    else:
        return True


def _safe_open_factory(root: pathlib.Path):
    root_resolved = root.resolve()

    def _open(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        raw = pathlib.Path(os.path.abspath(str(file))).resolve()
        if not _path_within_root(raw, root_resolved):
            raise PermissionError("Filesystem access confined to sandbox working directory.")
        return open(raw, mode, *args, **kwargs)  # noqa: SIM115 — intentional shim

    return _open


def _restricted_builtins(root: pathlib.Path) -> dict:
    sandbox = {}
    for name in _ALLOWED_BUILTINS_NAMES:
        obj = getattr(builtins, name, None)
        if obj is not None:
            sandbox[name] = obj
    sandbox["open"] = _safe_open_factory(root)
    return sandbox


FILENAME_SAFE = re.compile(r"^[A-Za-z0-9_\-.]+\.(png|jpg|jpeg|pdf|svg)$")


def _analytics_model() -> OpenAIChatModel:
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set MODEL_API_KEY or OPENAI_API_KEY in the environment.")

    raw = (os.environ.get("MODEL_NAME") or "gpt-4o-mini").strip()
    name = raw.removeprefix("openai:")
    provider = OpenAIProvider(api_key=api_key)
    return OpenAIChatModel(name, provider=provider)


@dataclass
class AnalyticsAgentDeps:
    """Per-request sandbox + DB."""

    engine: Engine
    work_dir: pathlib.Path
    artifact_relpaths: list[str] = field(default_factory=list)

    def record_artifact(self, path: pathlib.Path) -> None:
        rel = path.resolve().relative_to(self.work_dir.resolve())
        posix = rel.as_posix()
        if posix not in self.artifact_relpaths:
            self.artifact_relpaths.append(posix)


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
            role = item.get("role") or ""
            content = item.get("content") or ""
            role = str(role).lower().strip()
            content = "" if content is None else str(content)
            if role == "user":
                out.append(ModelRequest(parts=[UserPromptPart(content)]))
            elif role == "assistant":
                out.append(ModelResponse(parts=[TextPart(content)]))
        return out
    return []


def build_analytics_toolset() -> FunctionToolset[AnalyticsAgentDeps]:
    ts = FunctionToolset[AnalyticsAgentDeps]()

    @ts.tool
    def fetch_analysis_schema(ctx: RunContext[AnalyticsAgentDeps]) -> str:
        """Return the structured description of columns and the qualified SQL table name."""
        return aq.schema_documentation(ctx.deps.engine)

    @ts.tool
    def query_analysis_sql(ctx: RunContext[AnalyticsAgentDeps], sql: str) -> str:
        """
        Execute a single read-only SELECT (optional WITH…) against ONLY the analytics table.
        The query must literal-match the dialect-qualified identifiers for schema and table names.
        No JOINs are allowed via this pathway.
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
        """
        Safely retrieve rows via allow-listed equality filters against known columns only.
        `where_equal_json` pairs column names to equality values (AND-combined).
        """
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
        """
        Run Python snippets with pandas-like analytics helpers (`pd`, `np`) preloaded plus a DataFrame loader.
        Only stdout is surfaced unless you bind the special `_result` variable (stringifiable), which gets appended.

        Forbidden: imports of new modules besides what is injected. Use `_result = ...` to return compact findings.
        """
        root = ctx.deps.work_dir

        def load_analysis_df(lim: int | None = None) -> pd.DataFrame:
            return aq.load_analysis_dataframe(engine=ctx.deps.engine, limit=lim)

        def run_select(sql: str) -> list[dict]:
            return aq.run_validated_select(sql=sql, engine=ctx.deps.engine)

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
        stdout = io.StringIO()
        err: BaseException | None = None
        try:
            with redirect_stdout(stdout):
                exec(code, ns, ns)  # noqa: S102 — intentional restricted analytics sandbox
        except BaseException as exc:  # noqa: BLE001 — tool must report failures cleanly
            err = exc

        fragments = []
        if stdout_buf := stdout.getvalue().strip():
            fragments.append(stdout_buf)
        if err is None and "_result" in ns:
            fragments.append(str(ns["_result"]))
        elif err is not None:
            fragments.append(f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=5)}".strip())

        preview = "\n\n".join(fragments).strip()
        cap = int(os.environ.get("ANALYTICS_MAX_PYTHON_OUTPUT_CHARS", "16000"))
        if len(preview) > cap:
            preview = preview[: cap // 2] + "\n…[truncated]…\n" + preview[-cap // 2 :]
        return preview or "(no textual output)"

    @ts.tool
    def plot_and_save_figure(ctx: RunContext[AnalyticsAgentDeps], code: str, filename: str) -> str:
        """
        Produce a matplotlib figure using `plt`/`matplotlib.pyplot` helpers.
        The DataFrame helpers match `run_python_for_analysis`.
        Saves into the session plot directory automatically if you call `plt.savefig(OUTPUT_PATH)`.
        If you forget `savefig`, the active figure may be persisted automatically under `OUTPUT_PATH`.
        Filename must stay simple (basename with png/jpg/svg/pdf extension).
        """
        if not FILENAME_SAFE.match(filename):
            raise ValueError("filename must be ascii-safe with extension png|jpg|jpeg|pdf|svg")

        plots_dir = ctx.deps.work_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = (plots_dir / filename).resolve()
        base = plots_dir.resolve()
        if base not in out_path.parents and out_path != base:
            raise PermissionError("Invalid plot path.")

        OUTPUT_PATH_STR = str(out_path)

        def load_analysis_df(lim: int | None = None) -> pd.DataFrame:
            return aq.load_analysis_dataframe(engine=ctx.deps.engine, limit=lim)

        def run_select(sql: str) -> list[dict]:
            return aq.run_validated_select(sql=sql, engine=ctx.deps.engine)

        plt.close("all")

        ns = {
            "__builtins__": _restricted_builtins(ctx.deps.work_dir),
            "plt": plt,
            "matplotlib": matplotlib,
            "pd": pd,
            "np": np,
            "OUTPUT_PATH": OUTPUT_PATH_STR,
            "load_analysis_df": load_analysis_df,
            "run_select": run_select,
            "pathlib": pathlib,
        }
        err: BaseException | None = None
        try:
            exec(code, ns, ns)  # noqa: S102
        except BaseException as exc:  # noqa: BLE001
            err = exc

        wrote = False
        if pathlib.Path(OUTPUT_PATH_STR).exists():
            wrote = pathlib.Path(OUTPUT_PATH_STR).stat().st_size > 0

        if not wrote:
            try:
                fig = plt.gcf()
                if fig.get_axes():
                    fig.savefig(out_path, bbox_inches="tight")
                    wrote = pathlib.Path(out_path).stat().st_size > 0
            except BaseException:
                plt.close("all")
                wrote = wrote

        plt.close("all")

        if err is not None:
            return ("Plot tooling failed:\n" + f"{type(err).__name__}: {err}\n" + traceback.format_exc(limit=6)).strip()

        if not wrote or not pathlib.Path(out_path).is_file():
            return "Figure was not persisted; extend your code using plt.savefig(OUTPUT_PATH)."
        ctx.deps.record_artifact(out_path)
        return f"Saved plot artifact at `{out_path.relative_to(ctx.deps.work_dir)}`."

    return ts


_AGENT_INSTRUCTIONS = """You analyze the medical conversation enrichment table (`aisha_conversation_analysis`).
- Call `fetch_analysis_schema` once when planning SQL so you use the dialect-qualified identifiers correctly.
- Always reason about PHI carefully: summarize counts/aggregates, avoid verbatim messages when not needed,
  and truncate large payloads (JSON `segments`).
- Prefer `structured_table_fetch` for targeted slices, `query_analysis_sql` for aggregates you can express cleanly,
  and `run_python_for_analysis`/`plot_and_save_figure` once you actually need computations or viz.
- When plotting, clearly label axes and titles. Save every figure through the plotting tool."""

_agent_singleton: Agent[AnalyticsAgentDeps, str] | None = None


def get_analytics_agent() -> Agent[AnalyticsAgentDeps, str]:
    global _agent_singleton
    if _agent_singleton is None:
        ts = build_analytics_toolset()
        _agent_singleton = Agent(
            _analytics_model(),
            deps_type=AnalyticsAgentDeps,
            instructions=_AGENT_INSTRUCTIONS,
            toolsets=[ts],
        )
    return _agent_singleton


def run_analytics_sync(
    user_message: str,
    *,
    message_history: list[ModelMessage] | None,
    deps: AnalyticsAgentDeps,
) -> tuple[str, list[ModelMessage]]:
    agent = get_analytics_agent()
    result = agent.run_sync(
        user_message,
        message_history=message_history or None,
        deps=deps,
    )
    serialized_new = ModelMessagesTypeAdapter.dump_python(result.new_messages(), mode="json")
    return str(result.output), serialized_new


def zip_b64_optional(work_dir: pathlib.Path, artifact_relpaths: list[str]) -> str | None:
    raw = zip_artifacts(work_dir, artifact_relpaths)
    if raw is None:
        return None
    return base64.standard_b64encode(raw).decode("ascii")
