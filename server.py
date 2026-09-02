"""
FastAPI backend — replaces Gradio app.py.
Streams SSE phase events to the React frontend.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
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

from automl_system.agents.intent_parser import parse_user_intent
from automl_system.agents.task_classifier import (
    resolve_target_from_data,
    run_task_classification,
)
from automl_system.graph.orchestrator import build_graph
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.memory.episodic import create_session, initialize_session_document
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import compute_dataset_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "split_data": (5, "Data Split"),
    "train_evaluate": (5, "Model Training"),
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


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


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
            emit({"type": "phase_start", "phase": 1, "phase_name": "Initialization"})
            emit({"type": "message", "phase": 1,
                  "text": "Please upload a dataset before describing your goal."})
            emit({"type": "phase_done", "phase": 1, "status": "failed"})
            emit({"type": "final", "script_available": False,
                  "report_available": False, "session": session.to_dict()})
            return

        df = _load_dataframe(file_path)
        filename = Path(file_path).name

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

        # ── Phase 1 — Initialization ──────────────────────────────────────────
        emit({"type": "phase_start", "phase": 1, "phase_name": "Initialization"})

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

        # Step 2: Resolve target column — intent may have extracted it already
        if not session.task_config.get("target_column") or \
                str(session.task_config["target_column"]) not in df.columns:
            _resolved, msg, _conf, _extra = resolve_target_from_data(session, df)
            if msg:
                emit({"type": "message", "phase": 1, "text": str(msg)})

        if not session.task_config.get("target_column") or \
                str(session.task_config["target_column"]) not in df.columns:
            emit({"type": "message", "phase": 1,
                  "text": "Could not identify a target column. Please include the target column name in your prompt."})
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
        if msg:
            emit({"type": "message", "phase": 1, "text": msg})

        # Step 4: Full profile with target + initialize memory document
        resolved_target = session.task_config.get("target_column")
        session.dataset_profile = compute_dataset_profile(df, target_column=resolved_target)
        session.task_config["goal"] = user_line
        initialize_session_document(
            session_id=session.decisions_log_id or "",
            task_config=session.task_config,
            dataset_state_pointer=file_path,
        )

        session.phase_status_map["phase1"] = "done"
        task_type = str(session.task_config.get("task_type") or "")
        metric = str(session.task_config.get("evaluation_metric") or "")
        emit({"type": "message", "phase": 1,
              "text": (
                  f"Task: **{task_type}** | metric: **{metric}**. "
                  f"{session.task_config.get('metric_rationale', '')}"
              )})
        emit({"type": "message", "phase": 1,
              "text": f"Loaded **{len(df):,}** rows × **{df.shape[1]}** columns from `{filename}`."})
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

    t = threading.Thread(
        target=_pipeline_worker,
        args=(run_id, request.file_path, request.instructions, request.session),
        daemon=True,
    )
    t.start()

    return {"run_id": run_id}


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


@app.get("/download/script/{run_id}")
async def download_script(run_id: str):
    """Download the generated automl_pipeline.py for this run."""
    reg = _RUNS.get(run_id)
    if not reg:
        raise HTTPException(status_code=404, detail="Run not found.")
    sp = reg.get("script_path")
    if not sp or not Path(sp).is_file():
        raise HTTPException(status_code=404, detail="Pipeline script not yet available.")
    return FileResponse(sp, filename="automl_pipeline.py", media_type="text/x-python")


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
