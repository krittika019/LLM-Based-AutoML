"""
FastAPI backend — replaces Gradio app.py.
Streams SSE phase events to the React frontend.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import re
import math
import multiprocessing
import logging
import queue as _queue
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from automl_system import config
from automl_system.agents.intent_parser import parse_user_intent
from automl_system.agents.task_classifier import (
    resolve_target_from_data,
    run_task_classification,
)
from automl_system.graph.orchestrator import build_graph
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.memory.episodic import create_session, initialize_session_document
from automl_system.pipeline.unsupervised import UNSUPERVISED_TASK_TYPES as _UNSUPERVISED_TASK_TYPES
from automl_system.state.session import SessionState, _copy_data
from automl_system.utils.dataset_profile import compute_dataset_profile
from automl_system.utils.decision_log import log_decision
from automl_system.utils.phase1_summary import (
    TARGET_CONFIDENCE_THRESHOLD,
    build_phase1_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-pipeline AutoGluon/FLAML comparison runs (AUTOML_BENCHMARK_TIME_LIMIT) are meant
# for benchmark experiments; they add minutes to every app run. Off for the app
# unless APP_RUN_COMPETITORS=true. experiments/benchmark.py is unaffected.
if not config.APP_RUN_COMPETITORS:
    config.AUTOML_BENCHMARK_TIME_LIMIT = 0

app = FastAPI(title="AutoML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory run registry ────────────────────────────────────────────────────
# run_id → {"q": queue.Queue, "script_path": str|None, "pdf_path": str|None,
#            "session": dict|None, "done": bool}
_RUNS: dict[str, dict[str, Any]] = {}

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "automl_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)

# ── Serve built React frontend (production) ───────────────────────────────────
_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

# ── Phase metadata (mirrors graph/orchestrator.py node names) ─────────────────
_PHASE_NODE_META: dict[str, tuple[int, str]] = {
    "planning": (2, "Planning"),
    "basic_cleaning": (3, "Basic Cleaning"),
    "eda": (3, "EDA"),
    "feature_selection": (3, "Feature Selection"),
    "deep_cleaning": (3, "Deep Cleaning"),
    "feature_engineering": (3, "Feature Engineering"),
    "verification": (3, "Verification"),
    "phase4_planning": (4, "Execution Planning"),
    "split_data": (3, "Train/Test Split"),
    "train_evaluate": (5, "Model Training"),
    "unsupervised_train": (5, "Unsupervised Execution"),
    "interpret_results": (5, "Result Interpretation"),
    "threshold_check": (6, "Threshold Check"),
    "diagnose_failure": (6, "Failure Diagnosis"),
    "targeted_revision": (6, "Targeted Revision"),
    "phase7_final_output": (7, "Final Report"),
}

_PHASE_DISPLAY_NAMES: dict[int, str] = {
    1: "Initialization",
    2: "Planning",
    3: "Preprocessing",
    4: "Execution Planning",
    5: "Model Training",
    6: "Threshold & Revision",
    7: "Final Report",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    """NaN/inf → None: json.dumps writes them as bare NaN/Infinity, which the browser's
    JSON.parse rejects — the whole event (e.g. the final one) would be silently dropped."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _sse(data: dict) -> str:
    return f"data: {json.dumps(_json_safe(data), default=str, allow_nan=False)}\n\n"


def _extract_chart_data(df: pd.DataFrame, session: SessionState) -> dict:
    """Build chart-ready JSON from EDA results + dataframe."""
    eda = session.eda_results or {}

    # Missing values (percentage) — only columns that actually have missing data
    missing: dict[str, float] = {}
    for col, pct in (eda.get("missing_patterns") or {}).items():
        if isinstance(pct, (int, float)) and float(pct) > 0:
            missing[col] = round(float(pct) * 100, 2)

    # Correlation matrix — cap at 15 columns for readability
    corr_raw = eda.get("correlation_matrix") or {}
    cols = list(corr_raw.keys())[:15]
    matrix: list[list[float]] = []
    for r in cols:
        row_vals = corr_raw.get(r, {})
        matrix.append([round(float(row_vals.get(c, 0.0)), 3) for c in cols])

    # Numeric distributions (histogram bins) — cap at 8 columns
    distributions: dict[str, list[dict]] = {}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols[:8]:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        counts, edges = np.histogram(s.values, bins=20)
        distributions[col] = [
            {"x": round(float(edges[i]), 4), "count": int(counts[i])}
            for i in range(len(counts))
        ]

    return {
        "missing_values": missing,
        "correlation": {"columns": cols, "matrix": matrix},
        "distributions": distributions,
        "distribution_shapes": eda.get("distribution_shapes") or {},
    }


# ── Graph workers ─────────────────────────────────────────────────────────────

def _inner_graph_worker(graph: Any, initial: dict, out_q: _queue.Queue) -> None:
    """Run LangGraph in a thread; push (node_name, node_out) tuples to out_q."""
    try:
        try:
            stream_iter = graph.stream(initial, {"recursion_limit": 96})
        except TypeError:
            stream_iter = graph.stream(initial)
        for event in stream_iter:
            for node_name, node_out in event.items():
                out_q.put(("node", (node_name, node_out)))
        out_q.put(("done", None))
    except Exception:
        out_q.put(("error", traceback.format_exc()))


def _pipeline_worker(
    run_id: str,
    file_path: str,
    user_instructions: str,
    session_dict: dict | None,
    original_filename: str | None = None,
    task_type: str | None = None,
) -> None:
    """Full pipeline worker; emits SSE-shaped events via _RUNS[run_id]['q']."""
    reg = _RUNS[run_id]

    def emit(event: dict) -> None:
        reg["q"].put(event)

    user_line = (user_instructions or "").strip() or "(no instructions)"
    session = SessionState.from_dict(session_dict)
    session.retry_counts = dict(session.retry_counts or {})

    try:
        # ── File existence guard (before any LLM call) ────────────────────────
        if not file_path or not Path(file_path).exists():
            # Emit a minimal phase_start so the frontend has context
            emit({"type": "phase_start", "phase": 1, "phase_name": "Task & Data Understanding"})
            emit({"type": "message", "phase": 1,
                  "text": "Please upload a dataset before describing your goal."})
            emit({"type": "phase_done", "phase": 1, "status": "failed"})
            emit({"type": "final", "script_available": False,
                  "report_available": False, "session": session.to_dict()})
            return

        df = _load_dataframe(file_path)
        filename = original_filename or Path(file_path).name

        # Fresh session for a new file; re-use session for follow-up messages
        if session.dataset_state_pointer != file_path:
            session = SessionState()
            session.retry_counts = {}
            # Step 3 — Initialize agent memory (session_id, decisions_log, dataset_state_pointer)
            session.decisions_log_id = create_session()
            LLMClient.call_count = 0
            LLMClient.cache_hits = 0
            LLMClient.phase_label = ""

        session.dataset_state_pointer = file_path
        # Step 4 — Preliminary profile (target unknown yet; will be recomputed after target resolved)
        session.dataset_profile = compute_dataset_profile(df)
        session.phase_status_map["phase1"] = "running"

        # UI mode toggle sets task_type explicitly for unsupervised runs — takes
        # precedence over whatever the session already carries from a prior turn.
        if task_type:
            session.task_config = dict(session.task_config or {})
            session.task_config["task_type"] = task_type

        # ── Phase 1 — Task & Data Understanding ────────────────────────────────
        emit({"type": "phase_start", "phase": 1, "phase_name": "Task & Data Understanding"})

        is_unsupervised = str(session.task_config.get("task_type") or "").lower() in _UNSUPERVISED_TASK_TYPES
        if is_unsupervised:
            # UI already picked the mode — no target column to find, so skip
            # intent parsing and target/metric resolution entirely. Falls through
            # to the shared LangGraph invocation below, same as the supervised
            # path — the orchestrator forks on task_type right after basic_cleaning.
            session.task_config["goal"] = user_line
            log_decision(
                session, step="Phase 1", action="unsupervised_mode_selected",
                rationale=(
                    f"UI set task_type={session.task_config['task_type']!r} via the mode toggle — "
                    "skipping intent parsing and target resolution."
                ),
            )
            emit({"type": "message", "phase": 1,
                  "text": (
                      f"Mode: **{session.task_config['task_type']}** — no target column needed. "
                      "Skipping target resolution."
                  )})
            initialize_session_document(
                session_id=session.decisions_log_id or "",
                task_config=session.task_config,
                dataset_state_pointer=file_path,
            )
            session.phase_status_map["phase1"] = "done"
            emit({"type": "phase_done", "phase": 1, "status": "done"})

        else:
            # Step 1: Parse intent — classify state + extract target
            intent = parse_user_intent(user_line, list(df.columns), session)
            session.intent_result = intent
            state = str(intent.get("state") or "in_scope")

            if intent.get("llm_unavailable"):
                emit({"type": "message", "phase": 1,
                      "text": "The AI service is temporarily unavailable. Please try again in a moment."})
                session.phase_status_map["phase1"] = "failed"
                emit({"type": "phase_done", "phase": 1, "status": "failed"})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            # Non-in_scope states: show message and stop (no retry loop for research use)
            if state in ("out_of_scope", "unrecognizable", "ambiguous"):
                msg = str(intent.get("message") or
                          "This system supports supervised ML (classification/regression) on tabular data.")
                emit({"type": "message", "phase": 1, "text": msg})
                session.phase_status_map["phase1"] = "failed"
                emit({"type": "phase_done", "phase": 1, "status": "failed"})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            # Step 2: Resolve target column — intent may have extracted it already.
            # target_confidence stays 1.0 when the user already named a valid column
            # explicitly (nothing to resolve); otherwise it reflects how sure the
            # name-heuristic / LLM inference was.
            target_pre_specified = bool(session.task_config.get("target_column")) and \
                str(session.task_config.get("target_column")) in df.columns
            target_confidence = 1.0
            target_resolution_note: str | None = None
            if not target_pre_specified:
                _resolved, msg, target_confidence, _extra = resolve_target_from_data(session, df)
                target_resolution_note = str(msg) if msg else None

            if not session.task_config.get("target_column") or \
                    str(session.task_config["target_column"]) not in df.columns:
                emit({"type": "message", "phase": 1,
                      "text": "Could not identify a target column. Please include the target column name in your prompt."})
                session.phase_status_map["phase1"] = "failed"
                emit({"type": "phase_done", "phase": 1, "status": "failed"})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            # Ambiguous target: don't silently proceed on a low-confidence guess —
            # stop and ask the user to confirm the column explicitly.
            if target_confidence < TARGET_CONFIDENCE_THRESHOLD:
                candidate = session.task_config.get("target_column")
                emit({"type": "message", "phase": 1,
                      "text": (
                          f"I inferred **`{candidate}`** as the target column, but with low confidence "
                          f"({target_confidence:.0%}). Please confirm by explicitly naming the target "
                          "column in your prompt (e.g. \"predict `column_name`\") and try again."
                      )})
                session.phase_status_map["phase1"] = "failed"
                emit({"type": "phase_done", "phase": 1, "status": "failed"})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            # Step 3: Validate target + select metric
            _res, msg, _conf, _extra = run_task_classification(session, df)
            if session.task_config.get("_pipeline_stop") == "single_value_target":
                emit({"type": "message", "phase": 1, "text": msg})
                session.terminated = True
                session.phase_status_map["phase1"] = "failed"
                emit({"type": "phase_done", "phase": 1, "status": "failed"})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            # Step 4: Full profile with target + initialize memory document
            resolved_target = session.task_config.get("target_column")
            session.dataset_profile = compute_dataset_profile(df, target_column=resolved_target)
            session.task_config["goal"] = user_line
            initialize_session_document(
                session_id=session.decisions_log_id or "",
                task_config=session.task_config,
                dataset_state_pointer=file_path,
            )

            # Step 5: Structured problem/dataset understanding (no new LLM calls —
            # assembled from facts already resolved above) — consumed by the UI
            # and available to Phase 2+ via session.problem_summary.
            summary = build_phase1_summary(
                session=session, df=df,
                target_confidence=target_confidence,
                target_resolution_note=target_resolution_note,
                source_filename=filename,
            )
            session.problem_summary = summary

            session.phase_status_map["phase1"] = "done"
            emit({"type": "phase1_summary", "phase": 1, "data": summary})
            emit({"type": "phase_done", "phase": 1, "status": "done"})

        # ── LangGraph phases 2–7 ───────────────────────────────────────────────
        graph = build_graph()
        initial = {
            "session": session.to_dict(),
            "df": df,
            "messages": [],
            "checkpoints": {},
        }

        inner_q: _queue.Queue = _queue.Queue()
        graph_thread = threading.Thread(
            target=_inner_graph_worker,
            args=(graph, initial, inner_q),
            daemon=True,
        )
        graph_thread.start()

        current_phase: int = 0
        seen_msg_len: int = 0
        latest_split: dict[str, Any] | None = None

        while True:
            try:
                kind, payload = inner_q.get(timeout=1.0)
            except _queue.Empty:
                emit({"type": "heartbeat"})
                continue

            if kind == "done":
                break

            if kind == "error":
                logger.error("Graph worker error:\n%s", payload)
                emit({"type": "error", "message": "Pipeline error occurred. Check server logs."})
                emit({"type": "final", "script_available": False,
                      "report_available": False, "session": session.to_dict()})
                return

            node_name, node_out = payload
            if not isinstance(node_out, dict):
                continue

            # The downloadable script reads this exact split (train.csv / test.csv).
            if node_name == "split_data":
                latest_split = ((node_out.get("checkpoints") or {}).get("after_split")) or latest_split

            meta = _PHASE_NODE_META.get(node_name)
            if not meta:
                continue
            phase_num, node_label = meta

            # Phase transition
            if phase_num != current_phase:
                if current_phase:
                    session.phase_status_map[f"phase{current_phase}"] = "done"
                    emit({"type": "phase_done", "phase": current_phase, "status": "done"})
                current_phase = phase_num
                session.phase_status_map[f"phase{current_phase}"] = "running"
                phase_display = _PHASE_DISPLAY_NAMES.get(phase_num, f"Phase {phase_num}")
                emit({"type": "phase_start", "phase": phase_num, "phase_name": phase_display})

            # Node step label within the phase
            emit({"type": "node_start", "phase": phase_num, "node": node_label})

            # Session update
            if "session" in node_out:
                session = SessionState.from_dict(node_out["session"])
                session.preprocessing_meta["last_completed_phase"] = node_name

            # New text messages (messages list accumulates across all nodes)
            msgs: list = node_out.get("messages") or []
            if len(msgs) > seen_msg_len:
                for m in msgs[seen_msg_len:]:
                    content = m.get("content") if isinstance(m, dict) else str(m)
                    if content and str(content).strip():
                        emit({"type": "message", "phase": phase_num,
                              "text": str(content)})
                seen_msg_len = len(msgs)

            # EDA chart data (numeric distributions + correlation + missing values)
            if node_name == "eda" and "df" in node_out:
                try:
                    chart_data = _extract_chart_data(node_out["df"], session)
                    emit({"type": "charts", "phase": 3, "data": chart_data})
                except Exception as chart_err:
                    logger.warning("Chart extraction failed: %s", chart_err)

            # Phase 5 model comparison table
            if node_name == "train_evaluate":
                exec_res = node_out.get("execution_results") or {}
                tbl = exec_res.get("comparison_table") or []
                if tbl:
                    emit({"type": "model_results", "phase": 5,
                          "data": {
                              "comparison_table": tbl,
                              "interpretation": exec_res.get("interpretation") or "",
                              "task_type": exec_res.get("task_type") or "",
                              "metric": exec_res.get("metric") or "",
                              "metric_display": exec_res.get("metric_display") or "",
                              "metric_is_lower_better": exec_res.get("metric_is_lower_better") or False,
                              "best_model_metrics": exec_res.get("best_model_metrics") or {},
                              "elimination_audit": exec_res.get("elimination_audit") or {},
                              "shap_summary": exec_res.get("shap_summary") or {},
                              "router_stats": session.preprocessing_meta.get("router_stats") or {},
                          }})

        # Close the last open phase
        if current_phase:
            session.phase_status_map[f"phase{current_phase}"] = "done"
            emit({"type": "phase_done", "phase": current_phase, "status": "done"})

        # ── Persist semantic memory (reference: updated only on fully completed successful run) ───
        exec_results = session.execution_results or {}
        best_model = exec_results.get("best_model_name") or session.task_config.get("best_model")

        # ── Save pipeline script ───────────────────────────────────────────────
        script_text = session.final_pipeline_script or ""
        script_path: str | None = None
        if script_text.strip():
            sp = Path(tempfile.gettempdir()) / f"automl_pipeline_{run_id}.py"
            sp.write_text(script_text, encoding="utf-8")
            script_path = str(sp)
            reg["script_path"] = script_path
            for part in ("train", "test"):
                frame = (latest_split or {}).get(part)
                if isinstance(frame, pd.DataFrame):
                    dp = Path(tempfile.gettempdir()) / f"automl_{part}_{run_id}.csv"
                    frame.to_csv(dp, index=False)
                    reg[f"{part}_path"] = str(dp)

        # ── Stakeholder PDF ────────────────────────────────────────────────────
        pdf_path: str | None = None

        reg["session"] = session.to_dict()

        # ── Token cost table ───────────────────────────────────────────────────
        token_cost = dict(session.token_cost_table or {})
        if token_cost:
            emit({"type": "token_cost", "data": token_cost})

        emit({
            "type": "final",
            "script_available": bool(script_path),
            "script_validation": session.script_validation or {},
            "report_available": bool(pdf_path and Path(pdf_path).is_file()),
            "best_model": str(best_model or ""),
            "iterations": int(session.iteration_count or 0),
            "session": session.to_dict(),
        })

    except Exception:
        tb = traceback.format_exc()
        logger.error("Pipeline worker unhandled exception:\n%s", tb)
        emit({"type": "error", "message": "An unexpected error occurred. Check server logs."})
    finally:
        reg["done"] = True
        emit({"type": "__end__"})


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a dataset upload; return preview metadata and file_path for /run."""
    suffix = Path(file.filename or "data.csv").suffix.lower()
    allowed = {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use CSV, TSV, Parquet, or Excel.",
        )

    file_id = str(uuid.uuid4())
    dest = _UPLOAD_DIR / f"{file_id}{suffix}"
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    # Size limit: 200 MB (reference Phase 1 Step 2)
    max_bytes = 200 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum allowed size is 200 MB.",
        )

    dest.write_bytes(content)

    try:
        df = _load_dataframe(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}") from e

    # Minimum size check (reference Phase 1 Step 2)
    if len(df) < 1 or df.shape[1] < 1:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="The file contains no data rows or no columns. Please upload a valid dataset.",
        )

    cols = list(df.columns)
    preview_cols = cols[:16]
    preview_rows = df.head(5).fillna("").astype(str).values.tolist()

    return {
        "file_id": file_id,
        "file_path": str(dest),
        "filename": file.filename,
        "rows": len(df),
        "cols": df.shape[1],
        "size_kb": round(len(content) / 1024, 1),
        "columns": preview_cols,
        "extra_cols": max(0, len(cols) - 16),
        "preview": {
            "columns": preview_cols,
            "rows": preview_rows,
        },
    }


class RunRequest(BaseModel):
    file_path: str
    instructions: str
    session: Optional[dict] = None
    filename: Optional[str] = None
    # Set by the frontend's mode toggle for "Find patterns / reduce dimensions"
    # runs — "clustering" | "dimensionality_reduction". Omitted/None for the
    # default "Predict a target" mode, which keeps today's LLM-inferred flow.
    task_type: Optional[str] = None


@app.post("/run")
async def start_run(request: RunRequest):
    """Start a pipeline run; client then opens GET /stream/{run_id} for SSE events."""
    if not Path(request.file_path).is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found. Please re-upload the dataset.",
        )

    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {
        "q": _queue.Queue(),
        "script_path": None,
        "pdf_path": None,
        "session": None,
        "done": False,
    }

    # Each run executes in its own process: native ML/GPU libraries can crash the
    # process that ran them (seen: silently, seconds after the run finished), and that
    # must never take the API server down. A relay thread forwards the child's events.
    args = (run_id, request.file_path, request.instructions, request.session, request.filename, request.task_type)
    events = _MP.Queue()
    proc = _MP.Process(target=_run_in_child, args=(events, args), daemon=True, name=f"pipeline-{run_id[:8]}")
    proc.start()
    threading.Thread(target=_relay_child_events, args=(run_id, proc, events), daemon=True).start()

    return {"run_id": run_id}


_MP = multiprocessing.get_context("spawn")
_REGISTRY_KEYS = ("script_path", "pdf_path", "train_path", "test_path")


class _ChildEventQueue:
    """Stand-in for the run's queue inside the child process. Forwards events to the
    parent, preceded on "final" by the artifact paths the parent's downloads need."""

    def __init__(self, events: Any, reg: dict):
        self._events, self._reg = events, reg

    def put(self, event: dict) -> None:
        if isinstance(event, dict) and event.get("type") == "final":
            self._events.put({"type": "__registry__",
                              "data": {k: self._reg.get(k) for k in _REGISTRY_KEYS}})
        # Events are pickled across the process boundary: drop private non-data entries
        # (e.g. "_fitted_estimator" model objects in the Phase 5 comparison table).
        self._events.put(_copy_data(event))


def _run_in_child(events: Any, args: tuple) -> None:
    """Child-process entry point: run the normal worker with a forwarding queue."""
    run_id = args[0]
    reg: dict[str, Any] = {"script_path": None, "pdf_path": None, "session": None, "done": False}
    reg["q"] = _ChildEventQueue(events, reg)
    _RUNS[run_id] = reg
    _pipeline_worker(*args)


def _relay_child_events(run_id: str, proc: Any, events: Any) -> None:
    """Parent side: move child events into the run's queue; report an early child death."""
    reg = _RUNS[run_id]
    ended = False
    while not ended:
        try:
            event = events.get(timeout=1.0)
        except _queue.Empty:
            if proc.is_alive():
                continue
            break
        except (EOFError, OSError):
            break
        if event.get("type") == "__registry__":
            reg.update({k: v for k, v in (event.get("data") or {}).items() if v})
            continue
        if event.get("type") in ("final", "__end__"):
            reg["done"] = True
        ended = event.get("type") == "__end__"
        reg["q"].put(event)
    if not ended:
        logger.error("Pipeline process for run %s exited unexpectedly (exit code %s)", run_id, proc.exitcode)
        reg["q"].put({"type": "error",
                      "message": "The pipeline process stopped unexpectedly. Check the server logs."})
        reg["done"] = True
        reg["q"].put({"type": "__end__"})
    proc.join(timeout=10)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str):
    """Server-Sent Events stream: phase updates, messages, charts, model results."""
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="Run not found.")

    reg = _RUNS[run_id]
    q: _queue.Queue = reg["q"]

    async def event_generator():
        loop = asyncio.get_running_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=30))
            except _queue.Empty:
                yield ": keepalive\n\n"
                continue
            except Exception:
                break

            if event.get("type") == "__end__":
                break

            yield _sse(event)

            if event.get("type") in ("final", "error") and reg.get("done"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_RUN_ID_RE = re.compile(r"[0-9a-fA-F-]{36}")


def _run_artifact(run_id: str, key: str, filename: str) -> str | None:
    """Path of a saved run artifact. Falls back to its fixed temp-file name when the
    in-memory registry no longer has the run (e.g. after a server restart)."""
    path = (_RUNS.get(run_id) or {}).get(key)
    if path and Path(path).is_file():
        return path
    if _RUN_ID_RE.fullmatch(run_id):
        candidate = Path(tempfile.gettempdir()) / filename
        if candidate.is_file():
            return str(candidate)
    return None


@app.get("/download/script/{run_id}")
async def download_script(run_id: str):
    """Download the generated automl_pipeline.py for this run."""
    sp = _run_artifact(run_id, "script_path", f"automl_pipeline_{run_id}.py")
    if not sp:
        raise HTTPException(status_code=404, detail="Pipeline script not yet available.")
    return FileResponse(sp, filename="automl_pipeline.py", media_type="text/x-python")


@app.get("/download/split/{run_id}/{part}")
async def download_split(run_id: str, part: str):
    """Download the train.csv / test.csv split the pipeline script expects."""
    if part not in ("train", "test"):
        raise HTTPException(status_code=404, detail="Unknown split part.")
    dp = _run_artifact(run_id, f"{part}_path", f"automl_{part}_{run_id}.csv")
    if not dp:
        raise HTTPException(status_code=404, detail="Split not available for this run.")
    return FileResponse(dp, filename=f"{part}.csv", media_type="text/csv")


@app.get("/download/report/{run_id}")
async def download_report(run_id: str):
    """Download the stakeholder PDF report for this run."""
    reg = _RUNS.get(run_id)
    if not reg:
        raise HTTPException(status_code=404, detail="Run not found.")
    pp = reg.get("pdf_path")
    if not pp or not Path(pp).is_file():
        raise HTTPException(status_code=404, detail="Stakeholder report not yet available.")
    return FileResponse(
        pp, filename="automl_stakeholder_report.pdf", media_type="application/pdf"
    )


# ── SPA catch-all — MUST be last so all API routes match first ───────────────
# Registered here (after all @app.get/@app.post routes) so FastAPI checks
# specific routes before falling back to this wildcard.
if _FRONTEND_DIST.exists():
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        """Catch-all: serve index.html for all non-API routes (SPA routing)."""
        index = _FRONTEND_DIST / "index.html"
        return HTMLResponse(index.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
