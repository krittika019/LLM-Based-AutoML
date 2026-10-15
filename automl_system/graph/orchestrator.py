"""
LangGraph workflow: Phase 2 planning + Phase 3 preprocessing + Phase 4/5 execution.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from automl_system import config
from automl_system.agents.failure_diagnosis import diagnose_failure
from automl_system.llm.client import LLMClient
from automl_system.agents.pipeline_planner import make_execution_plan
from automl_system.agents.preprocessing_planner import make_preprocessing_plan
from automl_system.agents.report_generator import generate_report
from automl_system.pipeline.cleaning import basic_clean
from automl_system.pipeline.code_generator import generate_pipeline_script
from automl_system.pipeline.script_validation import STATUS_VERIFIED, validate_and_repair_script
from automl_system.pipeline.deep_cleaning import deep_clean
from automl_system.pipeline.diagnosis_probes import format_evidence, run_probes
from automl_system.pipeline.eda import run_eda
from automl_system.pipeline.execution import MissingTargetError, run_pipeline, split_dataframe
from automl_system.pipeline.feature_engineering import engineer_features
from automl_system.pipeline.feature_selection import select_features
from automl_system.pipeline.unsupervised import UNSUPERVISED_TASK_TYPES, run_unsupervised_pipeline
from automl_system.pipeline.verification import (
    STEP_BASIC_CLEAN,
    STEP_DEEP_CLEANING,
    STEP_EDA,
    STEP_FEATURE_ENGINEERING,
    STEP_FEATURE_SELECTION,
    parse_verification_route,
    propose_verification_fix,
    verify_preprocessing,
)
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import compute_dataset_profile
from automl_system.utils.decision_log import log_decision

logger = logging.getLogger(__name__)


def _record_phase_calls(session: SessionState, phase: str, calls_at_start: int) -> None:
    """Accumulate LLM call count for this phase into session.token_cost_table."""
    delta = LLMClient.call_count - calls_at_start
    tbl = dict(session.token_cost_table or {})
    tbl[phase] = tbl.get(phase, 0) + delta
    session.token_cost_table = tbl


_REWIND_CHECKPOINT: dict[str, str] = {
    STEP_BASIC_CLEAN: "initial",
    STEP_EDA: "after_split",
    STEP_FEATURE_SELECTION: "after_eda",
    STEP_DEEP_CLEANING: "after_feature_selection",
    STEP_FEATURE_ENGINEERING: "after_deep_cleaning",
}


class OrchestratorState(TypedDict, total=False):
    """Graph state: session dict, dataframe, checkpoints for retry, routing flags."""

    session: dict[str, Any]
    df: Any
    df_test: Any
    messages: list[str]
    checkpoints: dict[str, Any]
    verification_ok: bool
    retry_target: str | None
    retry_exceeded: bool
    execution_results: dict[str, Any]
    threshold_ok: bool
    diagnosis: dict[str, Any]
    loop_target: str | None
    phase7_done: bool


def _session(state: OrchestratorState) -> SessionState:
    return SessionState.from_dict(state.get("session"))


def node_planning(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state.get("df")
    calls0 = LLMClient.call_count
    if df is not None:
        if not session.dataset_profile or session.dataset_profile.get("stub"):
            session.dataset_profile = compute_dataset_profile(
                df, target_column=str((session.task_config or {}).get("target_column") or "") or None
            )
        if not session.initial_dataset_profile:
            session.initial_dataset_profile = dict(session.dataset_profile)
    make_preprocessing_plan(session)
    msgs = list(state.get("messages") or [])
    plan = session.preprocessing_plan or {}
    challenges: list[str] = []
    if plan.get("leakage_risks"):
        challenges.append(
            f"**Potential leakage / ID-like columns:** {plan.get('leakage_risks')} — high-cardinality identifiers can inflate scores if not dropped."
        )
    skewed = plan.get("skewed_columns") or []
    if skewed:
        challenges.append(f"**Skewed columns** (may need log/power transform): {skewed[:8]}.")
    enc = plan.get("encoding_candidates") or []
    if enc:
        enc_summary = ", ".join(
            f"`{e['column']}` ({e.get('method', '?')})" for e in enc[:6]
            if isinstance(e, dict) and e.get("column")
        )
        if enc_summary:
            challenges.append(f"**Encoding candidates:** {enc_summary}.")
    notes = plan.get("notes") or []
    if isinstance(notes, list) and notes:
        challenges.append("**Planning notes:** " + "; ".join(str(n) for n in notes[:6]))
    if plan.get("scaling_needed"):
        challenges.append("**Scaling:** numeric features may need scaling for distance-based models.")
    fe = bool(plan.get("feature_engineering_recommended"))
    challenges.append(
        f"**Feature engineering:** {'will be attempted' if fe else 'is deferred'} "
        f"({'dataset size supports derived features' if fe else 'small or simple dataset — start with baseline features'})."
    )
    block = "### Phase 2 — Planning (Mode 1)\n\n"
    block += "\n".join(f"- {c}" for c in challenges) if challenges else "- Baseline preprocessing plan generated from profile.\n"
    block += f"\n\n{session.user_summary_planning or 'Planning complete.'}\n"
    msgs.append(block)
    _record_phase_calls(session, "planning", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 2 Planning", calls0))
    return {"session": session.to_dict(), "df": df, "messages": msgs}


def node_basic_cleaning(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state.get("df")
    if df is None:
        raise ValueError("basic_cleaning requires state['df']")
    ck = dict(state.get("checkpoints") or {})
    ck["initial"] = df.copy()
    out = basic_clean(df, session)
    ck["after_basic_cleaning"] = out.copy()
    ck["after_eda"] = out.copy()
    msgs = list(state.get("messages") or [])
    rep = session.cleaning_report or {}
    amb = str(rep.get("ambiguous_llm_guidance") or "").strip()
    dup = int(rep.get("duplicate_rows_removed") or 0)
    n_coerce = len(rep.get("coercion_log") or [])
    summary = (
        "### Phase 3 — Basic cleaning\n\n"
        f"- Removed **{dup}** duplicate rows.\n"
        f"- Applied **{n_coerce}** type coercion(s) where columns looked numeric.\n"
    )
    if rep.get("high_dup_warning"):
        initial = (rep.get("initial_shape") or [0])[0]
        after = rep.get("rows_after_dedup") or (initial - dup)
        pct = rep.get("high_dup_pct") or round(dup / initial * 100, 1) if initial else 0
        summary += (
            f"\n> ⚠️ **High deduplication warning:** Removing duplicate rows reduced your dataset "
            f"from **{initial}** to **{after}** rows (**{pct}%** reduction). "
            "This is unusually high and may indicate a data collection issue.\n"
        )
    if amb and amb not in ("no_ambiguous_cases", "no_response"):
        summary += f"- **Ambiguous cases (LLM guidance):** {amb[:1200]}\n"
    msgs.append(summary)
    return {"session": session.to_dict(), "df": out, "checkpoints": ck, "messages": msgs}


def route_after_basic_cleaning(state: OrchestratorState) -> str:
    """Unsupervised runs (no target column) fork here — before split_data, which
    hard-requires a target. They skip straight to their own train node and never
    rejoin the supervised path (no EDA/feature-selection/deep-cleaning/threshold
    loop in v1 — see node_unsupervised_train)."""
    session = _session(state)
    task_type = str((session.task_config or {}).get("task_type") or "").lower()
    if task_type in UNSUPERVISED_TASK_TYPES:
        return "unsupervised_train"
    return "split_data"


def node_unsupervised_train(state: OrchestratorState) -> OrchestratorState:
    """Clustering / dimensionality-reduction path: no split, no CV, no threshold
    loop — runs once on the full (basic-cleaned) dataset and reports internal
    validation metrics. v1 scope; see run_unsupervised_pipeline's docstring."""
    session = _session(state)
    df = state.get("df")
    if df is None:
        raise ValueError("unsupervised_train requires state['df']")
    msgs = list(state.get("messages") or [])
    calls0 = LLMClient.call_count

    results = run_unsupervised_pipeline(df, None, session)
    session.unsupervised_results = results

    if results.get("status") == "ok":
        algo = results.get("algorithm")
        if algo == "kmeans":
            summary = (
                f"### Phase 5 — Unsupervised execution\n\n"
                f"**Algorithm:** K-Means | **k:** {results.get('best_k')} | "
                f"**Silhouette:** {results.get('silhouette_score')} | "
                f"**Davies-Bouldin:** {results.get('davies_bouldin_score')}\n\n"
                f"Cluster sizes: {results.get('cluster_sizes')}"
            )
        else:
            summary = (
                f"### Phase 5 — Unsupervised execution\n\n"
                f"**Algorithm:** PCA | **Components:** {results.get('n_components')} | "
                f"**Variance retained:** {round((results.get('cumulative_variance') or [0])[-1] * 100, 1)}%"
            )
    else:
        summary = (
            f"### Phase 5 — Unsupervised execution\n\n"
            f"> ⚠️ Did not complete: {results.get('reason')}"
        )
    msgs.append(summary)

    _record_phase_calls(session, "unsupervised_train", calls0)
    return {"session": session.to_dict(), "df": df, "messages": msgs}


def node_split_data(state: OrchestratorState) -> OrchestratorState:
    """
    Train/test split, run BEFORE any preprocessing that fits/learns from data
    (feature selection, deep cleaning, feature engineering) — so those steps
    can fit on the train half only and never leak test rows into learned
    parameters (imputation values, scaler bounds, encoder mappings, target
    encodings, engineered-feature accept/reject decisions).
    """
    session = _session(state)
    df = state.get("df")
    if df is None:
        raise ValueError("split_data requires state['df']")
    df_train, df_test = split_dataframe(df, session)
    session.expected_shape = (len(df_train), df_train.shape[1])
    ck = dict(state.get("checkpoints") or {})
    ck["after_split"] = {"train": df_train.copy(), "test": df_test.copy()}
    msgs = list(state.get("messages") or [])
    msgs.append(
        f"### Phase 3 — Train/test split\n\n"
        f"Split before preprocessing so imputation, scaling, encoding, and "
        f"feature engineering fit on the train rows only "
        f"(train={len(df_train)}, test={len(df_test)})."
    )
    return {
        "session": session.to_dict(), "df": df_train, "df_test": df_test,
        "checkpoints": ck, "messages": msgs,
    }


def node_eda(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state["df"]
    calls0 = LLMClient.call_count
    run_eda(df, session)
    ck = dict(state.get("checkpoints") or {})
    ck["after_eda"] = {"train": df.copy(), "test": state.get("df_test").copy()}
    msgs = list(state.get("messages") or [])
    msgs.append("### Phase 3 — EDA narrative\n\n" + (session.user_summary_eda or ""))
    _record_phase_calls(session, "eda", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 3 EDA", calls0))
    return {
        "session": session.to_dict(), "df": df, "df_test": state.get("df_test"),
        "checkpoints": ck, "messages": msgs,
    }


def node_feature_selection(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df_in = state["df"]
    df = select_features(df_in, session)
    ck = dict(state.get("checkpoints") or {})
    msgs = list(state.get("messages") or [])

    # Apply the identical drop decision to df_test — select_features only ever
    # drops columns, so the dropped set is exactly what's missing from `df`.
    df_test_in = state.get("df_test")
    df_test = None
    if df_test_in is not None:
        dropped = [c for c in df_in.columns if c not in df.columns]
        df_test = df_test_in.drop(columns=[c for c in dropped if c in df_test_in.columns])

    # Scenario: too few features remain → pipeline stop
    if (session.task_config or {}).get("_pipeline_stop") == "too_few_features":
        stop_msg = (session.preprocessing_meta or {}).get(
            "feature_selection_stop_message",
            "Feature selection left fewer than 2 feature columns. "
            "Cannot continue model training."
        )
        msgs.append(f"### Phase 3 — Feature selection\n\n> **Pipeline stopped:** {stop_msg}")
        emit = list(state.get("events_to_emit") or [])
        emit.append({"type": "phase_done", "phase": "feature_selection"})
        emit.append({"type": "pipeline_stopped", "reason": "too_few_features"})
        return {
            "session": session.to_dict(), "df": df, "df_test": df_test, "checkpoints": ck,
            "messages": msgs, "pipeline_complete": True, "events_to_emit": emit,
        }

    ck["after_feature_selection"] = {"train": df.copy(), "test": df_test.copy() if df_test is not None else None}
    lines = ["### Phase 3 — Feature selection\n"]
    high_drop = (session.preprocessing_meta or {}).get("feature_selection_high_drop_warning")
    if high_drop:
        lines.append(f"\n> **Warning:** {high_drop}\n")
    for entry in session.feature_selection_log or []:
        col = entry.get("column")
        why = entry.get("rationale") or ""
        tier = entry.get("tier", "")
        tier_badge = f" `[{tier}]`" if tier else ""
        lines.append(f"- **Dropped `{col}`**{tier_badge}: {why}")
    if len(lines) == 1 or (len(lines) == 2 and high_drop):
        lines.append("- No columns dropped at this step.")
    msgs.append("\n".join(lines))
    _record_phase_calls(session, "feature_selection", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Feature selection", calls0))
    return {
        "session": session.to_dict(), "df": df, "df_test": df_test,
        "checkpoints": ck, "messages": msgs,
    }


def node_deep_cleaning(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df_test_in = state.get("df_test")
    result = deep_clean(state["df"], session, df_test=df_test_in)
    df, df_test = result if df_test_in is not None else (result, None)
    ck = dict(state.get("checkpoints") or {})
    ck["after_deep_cleaning"] = {"train": df.copy(), "test": df_test.copy() if df_test is not None else None}
    msgs = list(state.get("messages") or [])
    lines = ["### Phase 3 — Deep cleaning & encoding\n"]
    assign = session.deep_cleaning_assignments or {}
    if isinstance(assign, dict):
        for col, strat in assign.items():
            if not isinstance(strat, dict):
                continue
            imp = strat.get("imputation", "?")
            enc = strat.get("encoding", "?")
            sc = strat.get("scaling", "?")
            lines.append(
                f"- **{col}:** imputation `{imp}`, encoding `{enc}`, scaling `{sc}`"
            )
    if len(lines) == 1:
        lines.append("- Assignments applied; see session log for details.")
    # Dimensionality warning after encoding
    dim_warn = (session.preprocessing_meta or {}).get("dimensionality_warning")
    if dim_warn:
        lines.append(f"\n> **Dimensionality warning:** {dim_warn}")
    msgs.append("\n".join(lines))
    _record_phase_calls(session, "deep_cleaning", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Deep cleaning", calls0))
    return {
        "session": session.to_dict(), "df": df, "df_test": df_test,
        "checkpoints": ck, "messages": msgs,
    }


def node_feature_engineering(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df_test_in = state.get("df_test")
    # Original-scale rows for CV-gated FE: the split after feature selection
    # (which only drops columns), i.e. the same rows before deep cleaning.
    split_ck = (state.get("checkpoints") or {}).get("after_split") or {}
    keep = list((session.preprocessing_meta or {}).get("feature_selection_columns") or [])
    def _kept(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        cols = [c for c in keep if c in frame.columns]
        return frame[cols].copy() if cols else frame.copy()

    raw_train, raw_test = _kept(split_ck.get("train")), _kept(split_ck.get("test"))
    result = engineer_features(
        state["df"], session, df_test=df_test_in, raw_train=raw_train, raw_test=raw_test
    )
    df, df_test = result if df_test_in is not None else (result, None)
    ck = dict(state.get("checkpoints") or {})
    ck["after_feature_engineering"] = {"train": df.copy(), "test": df_test.copy() if df_test is not None else None}
    msgs = list(state.get("messages") or [])
    lines = ["### Phase 3 — Feature engineering\n"]
    fe_log = session.feature_engineering_log or []
    # Scenario: dataset too small → skipped
    if fe_log and isinstance(fe_log[0], dict) and fe_log[0].get("status") == "skipped":
        reason = fe_log[0].get("reason", "Dataset too small.")
        lines.append(f"- *Skipped:* {reason}")
    else:
        for item in fe_log:
            if isinstance(item, dict):
                name = item.get("feature") or item.get("name") or "feature"
                st = item.get("status") or "proposed"
                why = item.get("rationale") or item.get("reason") or ""
                lines.append(f"- **{name}** ({st}): {why}")
        if len(lines) == 1:
            lines.append("- No engineered features added.")
    msgs.append("\n".join(lines))
    _record_phase_calls(session, "feature_engineering", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Feature engineering", calls0))
    return {
        "session": session.to_dict(), "df": df, "df_test": df_test,
        "checkpoints": ck, "messages": msgs,
    }


def node_verification(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state["df"]
    df_test = state.get("df_test")
    ck = dict(state.get("checkpoints") or {})
    ok, raw_msg = verify_preprocessing(df, session, df_test=df_test)
    user_msg, fail_step = parse_verification_route(raw_msg)
    msgs = list(state.get("messages") or [])
    msgs.append(user_msg if not ok else raw_msg)

    out: OrchestratorState = {
        "session": session.to_dict(),
        "df": df,
        "df_test": df_test,
        "checkpoints": ck,
        "messages": msgs,
        "verification_ok": ok,
        "retry_target": None,
        "retry_exceeded": False,
    }

    if ok:
        lv = dict(session.last_verification or {})
        lv["ok"] = True
        lv["message"] = raw_msg
        session.last_verification = lv
        out["session"] = session.to_dict()
        # Verification passed — the retry-rewind checkpoints (full dataframe copies at
        # every preprocessing stage) are never read again for the rest of the run.
        # Drop them now instead of carrying several dataframe copies through Phase 5-7.
        # Keep only the raw train/test split: Phase 7 runs the final script against it.
        out["checkpoints"] = {k: v for k, v in ck.items() if k == "after_split"}
        return out

    if fail_step:
        session.retry_counts[fail_step] = session.retry_counts.get(fail_step, 0) + 1
        if session.retry_counts[fail_step] > config.MAX_STEP_RETRIES:
            out["retry_exceeded"] = True
            out["retry_target"] = None
            msgs.append(
                f"Verification check for **{fail_step}** could not be resolved after "
                f"{config.MAX_STEP_RETRIES} attempts. Proceeding with caution."
            )
        else:
            out["retry_target"] = fail_step
            fix = propose_verification_fix(session, user_msg)
            if fix:
                msgs.append(f"Suggested fix: {fix}")
            key = _REWIND_CHECKPOINT.get(fail_step)
            if key and key in ck:
                snap = ck[key]
                if isinstance(snap, dict) and "train" in snap:
                    out["df"] = snap["train"].copy()
                    out["df_test"] = snap["test"].copy() if snap.get("test") is not None else None
                else:
                    out["df"] = snap.copy()
                msgs.append(f"Retry {session.retry_counts[fail_step]}/{config.MAX_STEP_RETRIES}: rewinding to {fail_step}.")
            else:
                logger.warning("Missing checkpoint %s for retry %s", key, fail_step)
    elif not ok:
        out["retry_exceeded"] = True
        msgs.append("Verification failed with no routable step.")
    out["messages"] = msgs
    out["session"] = session.to_dict()
    return out


def node_phase4_planning(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state.get("df")
    calls0 = LLMClient.call_count
    if df is not None:
        # Refresh profile from final cleaned frame before execution planning.
        session.dataset_profile = compute_dataset_profile(
            df, target_column=str((session.task_config or {}).get("target_column") or "") or None
        )
    plan = make_execution_plan(session, df=df)
    msgs = list(state.get("messages") or [])

    block = "### Phase 4 — Planning (Mode 2: execution)\n\n"

    # Dataset difficulty score (computed by code, not LLM)
    difficulty = plan.get("difficulty") or (session.dataset_profile or {}).get("difficulty") or {}
    if difficulty:
        diff_label = str(difficulty.get("label") or "unknown")
        diff_score = float(difficulty.get("overall") or 0.0)
        diff_colors = {"easy": "🟢", "moderate": "🟡", "hard": "🟠", "very_hard": "🔴"}
        diff_icon = diff_colors.get(diff_label, "⚪")
        block += f"**Dataset difficulty:** {diff_icon} `{diff_label}` (score {diff_score:.2f})"
        comps = difficulty.get("components") or {}
        if comps:
            top = sorted(comps.items(), key=lambda x: x[1], reverse=True)[:2]
            block += " · drivers: " + ", ".join(f"`{k}={v:.2f}`" for k, v in top if v > 0.1)
        block += "\n\n"

    # Dataset context assembled by code (not LLM)
    ctx = plan.get("dataset_context") or {}
    if ctx:
        block += (
            f"**Dataset context (post-preprocessing):** "
            f"{ctx.get('n_rows')} rows · {ctx.get('n_features')} features "
            f"({ctx.get('n_numeric')} numeric, {ctx.get('n_categorical')} categorical)"
        )
        if ctx.get("imbalance_present"):
            block += f" · imbalance: `{ctx.get('imbalance_severity')}`"
        if ctx.get("scaling_applied"):
            block += " · scaling applied"
        block += "\n\n"

    # Model shortlist (LLM Call 1)
    block += "**Model shortlist (ranked):**\n"
    for m in plan.get("models") or []:
        if isinstance(m, dict):
            fam = m.get("family") or "?"
            why = m.get("justification") or m.get("rationale") or "Selected for this data regime."
            pg = m.get("param_grid")
            pg_note = f" · grid: `{list(pg.keys())[:3]}`" if isinstance(pg, dict) and pg else ""
            block += f"- **{fam}**{pg_note}: {why}\n"

    # Search + CV (code-enforced, but note reasoning)
    search = plan.get("search_strategy") or {}
    cv = plan.get("cv_design") or {}
    block += (
        f"\n**Search:** `{search.get('name', '?')}`"
    )
    if search.get("n_iter"):
        block += f" (n_iter={search['n_iter']})"
    if search.get("n_trials"):
        block += f" (trials={search['n_trials']})"
    block += (
        f" | **CV:** {cv.get('type', '?')} × {cv.get('folds', '?')} folds"
        f" | **Metric:** `{plan.get('metric', '?')}`\n"
    )

    # Threshold (LLM Call 2 reasoning)
    thr = plan.get("threshold")
    thr_note = plan.get("threshold_rationale") or ""
    if isinstance(thr, (int, float)):
        plan_metric = str(plan.get("metric") or "")
        is_neg_metric = plan_metric.startswith("neg_")
        disp_thr = -float(thr) if is_neg_metric else float(thr)
        cmp = "≤" if is_neg_metric else "≥"
        block += f"\n**Performance threshold:** `{cmp} {disp_thr:.4f}` — {thr_note}\n"

    msgs.append(block)
    _record_phase_calls(session, "execution_planning", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 4 Execution planning", calls0))
    return {"messages": msgs, "session": session.to_dict(), "df": df}



def node_train_evaluate(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state.get("df")
    df_test = state.get("df_test")
    if df is None:
        raise ValueError("train_evaluate requires state['df']")
    msgs = list(state.get("messages") or [])
    results: dict[str, Any] = {}
    calls0 = LLMClient.call_count
    try:
        results = run_pipeline(df, session, df_test=df_test)
        tbl = results.get("comparison_table") or []
        split_meta = results.get("split_meta") or {}
        n_train = int(split_meta.get("n_train") or 0)
        n_test = int(split_meta.get("n_test") or 0)
        total = n_train + n_test
        if total > 0:
            test_pct = round(n_test / total * 100)
        else:
            test_pct = int(float(split_meta.get("test_size") or 0.20) * 100)
        train_pct = 100 - test_pct
        msgs.append(
            f"### Phase 5 — Model training & evaluation\n\n"
            f"**Split:** {train_pct}/{test_pct} "
            f"(train={split_meta.get('n_train')}, test={split_meta.get('n_test')}, "
            f"type={split_meta.get('split_type')})\n"
        )
        # Show progressive elimination notice if applicable
        eliminated = next(
            (r.get("model_name") for r in tbl if r.get("status") == "eliminated_round1"), None
        )
        if eliminated:
            survivors = [r.get("model_name") for r in tbl if r.get("status") != "eliminated_round1"]
            msgs.append(
                f"**Progressive evaluation:** `{eliminated}` was eliminated after Round 1 "
                f"(20% subsample screening). Full training ran on: {', '.join(f'`{s}`' for s in survivors)}."
            )
        # Timed-out models
        timed_out = [r.get("model_name") for r in tbl if r.get("status") == "timed_out"]
        if timed_out:
            msgs.append(
                f"> **Timed out** (>5 min): {', '.join(f'`{m}`' for m in timed_out)} "
                "— marked in table, loop continued to next model."
            )
        # Leakage warning
        leakage_models = results.get("leakage_warning_models") or []
        if leakage_models:
            msgs.append(
                f"\n> ⚠️ **Leakage warning:** {', '.join(f'`{m}`' for m in leakage_models)} "
                "scored ≥0.999 on the test set — check for target-encoded features before deployment."
            )
        if results.get("status") == "no_models_succeeded":
            msgs.append(
                "**No models produced a valid test score.** Routing to failure diagnosis (model_issue)."
            )
    except MissingTargetError as e:
        text = str(e)
        results = {
            "comparison_table": [],
            "best_model_name": None,
            "status": "missing_target",
            "reason": text,
            "best_test_score": None,
        }
        session.execution_results = results
        session.execution_interpretation = (
            "Execution cannot run until Phase 1 resolves a valid target_column in session.task_config."
        )
        msgs.append(
            f"### Phase 5 — Execution blocked\n\n**Missing target:** {text}\n"
            "Resolve the target via the task classifier (Phase 1), then rerun."
        )
    _record_phase_calls(session, "model_training", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 5 Execution", calls0))
    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": df,
        "execution_results": results,
    }



def node_threshold_check(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    results = dict(state.get("execution_results") or session.execution_results or {})
    plan = session.execution_plan or {}
    threshold = plan.get("threshold")
    best = results.get("best_test_score")
    msgs = list(state.get("messages") or [])
    session.all_results = list(session.all_results or [])
    if results:
        session.all_results.append(results)

    ok = False
    msgs.append("### Phase 6 — Threshold check\n")
    if results.get("status") == "no_models_succeeded":
        msgs.append("**No successful models** — threshold cannot be evaluated; routing to failure diagnosis.")
        ok = False
    elif isinstance(threshold, (int, float)) and isinstance(best, (int, float)):
        ok = float(best) >= float(threshold)
        gap = float(best) - float(threshold)
        status_str = "**✅ passed**" if ok else "**❌ not met**"
        metric_key = str(plan.get("metric") or "")
        is_neg = metric_key.startswith("neg_")
        # Display human-readable values: negate neg_* scores so users see positive RMSE/MAE
        disp_thr = -float(threshold) if is_neg else float(threshold)
        disp_best = -float(best) if is_neg else float(best)
        disp_gap = disp_best - disp_thr if is_neg else gap
        cmp_sym = "≤" if is_neg else "≥"
        msgs.append(
            f"- Target threshold: `{cmp_sym} {disp_thr:.4f}` | Best score: `{disp_best:.4f}` "
            f"| Gap: `{disp_gap:+.4f}` → {status_str}"
        )
    else:
        msgs.append("Threshold check skipped: threshold or best score unavailable.")

    user_satisfied = bool((session.task_config or {}).get("user_satisfied", False))
    maxed = int(session.iteration_count) >= int(config.MAX_ITERATIONS)

    # Convergence check: no meaningful improvement from last iteration
    all_results_so_far = list(session.all_results or [])
    converged = False
    if not ok and not maxed and len(all_results_so_far) >= 2:
        prev_score = all_results_so_far[-2].get("best_test_score")
        curr_score = results.get("best_test_score")
        if isinstance(prev_score, (int, float)) and isinstance(curr_score, (int, float)):
            if float(curr_score) <= float(prev_score) + 0.001:
                converged = True

    # ── Early-exit conditions (prevent wasteful reiteration) ────────────────
    # Reiteration is only valuable when:
    #   (a) the diagnosis agent can identify something SPECIFIC to fix (preprocessing
    #       issue, leaky feature, wrong imputation), OR
    #   (b) there are model families that haven't been tried yet.
    # These conditions detect cases where neither is true — so we exit early.
    #
    # Note: Condition A (beats competitors) only applies in benchmark mode where
    # internal FLAML/AutoGluon runs happened — it won't fire in user mode because
    # best_competitor_score will be None.
    early_exit_reason: str | None = None
    preprocessing_has_issue = bool(
        session.preprocessing_meta.get("verification_failed")
        or session.preprocessing_meta.get("verification_null_cols")
    )

    if not ok and not maxed and not converged and isinstance(best, (int, float)):
        best_competitor = results.get("best_competitor_score")
        models_trained: list = results.get("models_trained") or []

        # Exit condition A (benchmark mode only): score already beats internal competitors.
        # Only fires when competitor scores exist — silent in user mode.
        if isinstance(best_competitor, (int, float)) and float(best) >= float(best_competitor):
            early_exit_reason = (
                f"Score {best:.4f} already beats or matches best competitor "
                f"({best_competitor:.4f}). Threshold may be aspirational — "
                "reiteration cannot improve competitive standing."
            )

        # Exit condition B: threshold gap is too large to close with model selection.
        # A gap > 0.10 means the data ceiling is below the threshold — not a fixable
        # modelling problem. Applies equally in user mode and benchmark mode.
        elif isinstance(threshold, (int, float)):
            gap = float(threshold) - float(best)
            if gap > 0.10:
                early_exit_reason = (
                    f"Gap to threshold is {gap:.4f} (threshold={threshold:.4f}, "
                    f"best={best:.4f}). A gap > 0.10 reflects a data ceiling — "
                    "model selection alone cannot close it. "
                    "Consider collecting more data or revisiting the target definition."
                )

        # Exit condition C: broad model family set already tried AND preprocessing is clean.
        # If preprocessing had a specific issue (null imputation failure, leaky feature),
        # we allow reiteration — diagnosis can route back to fix it.
        # But if preprocessing passed verification and we've tried 5+ families,
        # reiteration would just reshuffle the same candidates.
        elif len(models_trained) >= 5 and not preprocessing_has_issue:
            early_exit_reason = (
                f"Already trained {len(models_trained)} model families "
                f"({', '.join(models_trained[:5])}{'...' if len(models_trained) > 5 else ''}) "
                "and preprocessing passed all verification checks. "
                "Reiteration would reshuffle the same candidates with no new signal."
            )

    if early_exit_reason:
        msgs.append(f"→ **Early exit:** {early_exit_reason}")
        converged = True  # reuse the converged path to reach Phase 7

    # — Three explicit exit conditions —
    if ok:
        # Exit condition 1: threshold met → skip iterative loop, go to Phase 7
        msgs.append("→ **Exit condition 1:** Threshold met — proceeding to Phase 7.")
    elif maxed:
        # Exit condition 2: iteration budget exhausted
        msgs.append(
            f"→ **Exit condition 2:** Maximum iterations ({config.MAX_ITERATIONS}) reached without "
            "meeting threshold — proceeding to Phase 7 with best result available."
        )
    elif converged:
        if early_exit_reason:
            # early_exit_reason message already appended above; just signal Phase 7.
            msgs.append("→ Proceeding to Phase 7 with best result available.")
        elif len(all_results_so_far) >= 2:
            _prev = all_results_so_far[-2].get('best_test_score')
            _curr = results.get('best_test_score')
            msgs.append(
                "→ **Exit condition 4:** No meaningful score improvement from last iteration "
                f"(prev={_prev:.4f}, curr={_curr:.4f}) — further iterations unlikely to help. "
                "Proceeding to Phase 7 with best result available."
            )
        else:
            msgs.append("→ **Exit condition 4:** Proceeding to Phase 7 with best result available.")
    elif user_satisfied:
        msgs.append("→ User satisfaction flag set — proceeding to Phase 7.")
    # Exit condition 3 (fundamental_data_insufficiency) is detected in node_diagnose_failure
    # and routed to Phase 7 immediately via route_after_diagnosis

    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": results,
        "threshold_ok": ok or converged or maxed or user_satisfied,
    }


def node_diagnose_failure(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    results = dict(state.get("execution_results") or session.execution_results or {})
    msgs = list(state.get("messages") or [])

    # Max-iterations guard (degenerate path — node_threshold_check should catch this first)
    if int(session.iteration_count or 0) >= int(config.MAX_ITERATIONS):
        diagnosis = {
            "failure_type": "fundamental_data_insufficiency",
            "responsible_phase": "phase4_planning",
            "proposed_fix": "Maximum iteration budget reached without meeting threshold.",
            "should_exit": True,
        }
        session.last_failure_diagnosis = diagnosis
        msgs.append(
            f"Maximum iterations ({config.MAX_ITERATIONS}) exhausted before threshold. "
            "Proceeding to Phase 7 with best available result."
        )
        return {
            "messages": msgs, "session": session.to_dict(),
            "df": state.get("df"), "execution_results": results, "diagnosis": diagnosis,
        }

    session.iteration_count = int(session.iteration_count or 0) + 1

    # Scenario: all models failed → immediate model_issue diagnosis (no LLM call needed)
    if results.get("status") == "no_models_succeeded":
        diagnosis = {
            "failure_type": "model_issue",
            "responsible_phase": "phase4_planning",
            "proposed_fix": (
                "All models failed, timed out, or were skipped. "
                "Simplify the model list, verify the metric matches the task type, "
                "check for NaN-only columns, and review per-model tracebacks in the decisions log."
            ),
            "should_exit": False,
        }
        session.last_failure_diagnosis = diagnosis
        msgs.append(
            "**Failure diagnosis (all models failed — model_issue):** "
            + diagnosis["proposed_fix"]
        )
        return {
            "messages": msgs, "session": session.to_dict(),
            "df": state.get("df"), "execution_results": results, "diagnosis": diagnosis,
        }

    # General path: controlled experiments first, then the LLM diagnoses with that evidence
    probe_report = None
    if config.DIAG_PROBES and isinstance(state.get("df"), pd.DataFrame):
        try:
            split_ck = (state.get("checkpoints") or {}).get("after_split") or {}
            probe_report = run_probes(state["df"], split_ck.get("train"), session, results)
            msgs.append(
                f"### Phase 6 — Diagnosis experiments ({probe_report.get('seconds', 0):.0f}s)\n\n"
                + format_evidence(probe_report).replace("\n- ", "\n\n- ")
            )
        except Exception as exc:
            logger.warning("Diagnosis probes failed (continuing with LLM-only diagnosis): %s", exc)
    diagnosis = diagnose_failure(session, results, probe_report)
    note = " _(chosen from the experiment evidence)_" if diagnosis.get("overridden_by_evidence") else ""
    msgs.append(
        "**Failure diagnosis:** "
        f"{diagnosis.get('failure_type')} → {diagnosis.get('responsible_phase')}{note}. "
        f"Proposed fix: {diagnosis.get('proposed_fix')}"
    )

    # Exit condition 3: fundamental_data_insufficiency → skip targeted revision, go to Phase 7
    if diagnosis.get("failure_type") == "fundamental_data_insufficiency" or diagnosis.get("should_exit"):
        diagnosis["should_exit"] = True
        msgs.append(
            "→ **Exit condition 3:** Failure type is `fundamental_data_insufficiency` — "
            "further iterations cannot improve the result. Proceeding to Phase 7 immediately."
        )

    return {
        "messages": msgs, "session": session.to_dict(),
        "df": state.get("df"), "execution_results": results, "diagnosis": diagnosis,
    }


def node_targeted_revision(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    diagnosis = dict(state.get("diagnosis") or session.last_failure_diagnosis or {})
    failure_type = str(diagnosis.get("failure_type") or "model_issue")
    phase = str(diagnosis.get("responsible_phase") or "phase4_planning")
    fix = str(diagnosis.get("proposed_fix") or "")
    iteration = int(session.iteration_count or 0)

    # Resolve loop target: map responsible_phase to the earliest affected graph node.
    # data_quality_issue re-enters at the very beginning of that branch.
    # feature_issue re-enters at the selection or engineering step.
    # model_issue goes straight to Phase 4.
    _DATA_QUALITY_PHASES = {"basic_cleaning", "deep_cleaning"}
    _FEATURE_PHASES = {"feature_selection", "feature_engineering"}
    if failure_type == "data_quality_issue" and phase in _DATA_QUALITY_PHASES:
        loop_target = phase  # basic_cleaning or deep_cleaning exactly as diagnosed
    elif failure_type == "data_quality_issue":
        loop_target = "basic_cleaning"  # default for data quality
    elif failure_type == "feature_issue" and phase in _FEATURE_PHASES:
        loop_target = phase  # feature_selection or feature_engineering exactly as diagnosed
    elif failure_type == "feature_issue":
        loop_target = "feature_selection"  # default for feature issue
    else:
        loop_target = "phase4_planning"  # model_issue always → Phase 4

    # Build the revision context string that downstream LLM calls will prepend.
    revision_ctx = (
        f"[TARGETED REVISION — iteration {iteration}] "
        f"Previous run failed: failure_type={failure_type}, responsible_phase={phase}. "
        f"Fix instruction: {fix} "
        f"Apply this revision now. Do not repeat the same strategy that failed."
    )

    # Persist in preprocessing_meta so all downstream agents can see it.
    notes = list(session.preprocessing_meta.get("iteration_fix_notes") or [])
    if fix:
        notes.append({"iteration": iteration, "phase": phase, "fix": fix})
    session.preprocessing_meta["iteration_fix_notes"] = notes
    session.preprocessing_meta["targeted_revision_context"] = revision_ctx

    msgs = list(state.get("messages") or [])
    msgs.append(
        f"### Phase 6 — Targeted revision (iteration {iteration})\n\n"
        f"- **Failure type:** `{failure_type}`\n"
        f"- **Responsible phase:** `{phase}`\n"
        f"- **Re-entry point:** `{loop_target}`\n"
        f"- **Fix applied:** {fix or '(none specified)'}\n\n"
        "_Only the affected phase(s) and all phases downstream of it will re-run._"
    )

    # When re-entering preprocessing, clear the stale execution plan so Phase 4
    # re-plans from scratch with the corrected data — avoids using a plan built on
    # features that no longer exist after re-cleaning.
    if loop_target in {"basic_cleaning", "deep_cleaning", "feature_selection", "feature_engineering"}:
        session.execution_plan = {}

    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": dict(state.get("execution_results") or {}),
        "loop_target": loop_target,
    }


def node_phase7_final_output(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count

    # Unsupervised runs never populate all_results/execution_results (no
    # split, no CV, no test_score) — the supervised scoring/script logic below
    # doesn't apply and would misreport an "ok" clustering run as failed.
    task_type = str((session.task_config or {}).get("task_type") or "").lower()
    if task_type in UNSUPERVISED_TASK_TYPES:
        report_md = generate_report(session, [])
        session.final_report_markdown = report_md
        session.final_pipeline_script = ""
        session.final_comparison_table = []
        session.is_complete = True
        msgs = list(state.get("messages") or [])
        msgs.append("### Phase 7 — Final report\n\n" + (report_md or "_Report unavailable._"))
        _record_phase_calls(session, "report_generation", calls0)
        logger.info(LLMClient.phase_call_stats_message("Phase 7 Final report", calls0))
        token_tbl = dict(session.token_cost_table or {})
        token_tbl["_total"] = sum(v for k, v in token_tbl.items() if not k.startswith("_"))
        token_tbl["_cache_hits"] = LLMClient.cache_hits
        session.token_cost_table = token_tbl
        return {
            "messages": msgs,
            "session": session.to_dict(),
            "df": state.get("df"),
            "execution_results": {"token_cost_table": token_tbl},
            "phase7_done": True,
        }

    all_results = list(session.all_results or [])

    # ── Step 1: Explanation Report ─────────────────────────────────────────────
    # Scenario: pipeline failed completely (no successful models)
    failed_completely = not any(
        isinstance(r.get("best_test_score"), (int, float)) for r in all_results
    )
    # ── Step 3 (started early): Consolidated Code Output + sandbox validation ──
    # The script is generated first, then executed/repaired in a background thread
    # while the report is written, so validation adds little wall-clock time.
    df_final = state.get("df")
    target_col = str((session.task_config or {}).get("target_column") or "")
    feature_columns = (
        [c for c in df_final.columns if c != target_col] if isinstance(df_final, pd.DataFrame) else None
    )
    split_ck = (state.get("checkpoints") or {}).get("after_split") or {}
    script, script_source = generate_pipeline_script(
        session, feature_columns=feature_columns, df_raw_train=split_ck.get("train")
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        validation_future = pool.submit(
            validate_and_repair_script, session, script,
            split_ck.get("train"), split_ck.get("test"), script_source,
        )
        report_md = generate_report(session, all_results)
        script, validation = validation_future.result()

    # ── Step 2: Final Results Presentation ────────────────────────────────────
    # Compile all rows from all iterations, tagged with iteration number
    merged: list[dict[str, Any]] = []
    for i, res in enumerate(all_results, start=1):
        for row in res.get("comparison_table") or []:
            item = dict(row)
            item["iteration"] = i
            merged.append(item)

    # Tie-breaking: highest test_score → lowest training_time → lowest fold_variance
    # Also respect explainability preference from user constraints.
    user_constraints = str((session.task_config or {}).get("constraints") or "").lower()
    prefers_explainable = "explainable" in user_constraints or "interpretable" in user_constraints
    _EXPLAINABLE_FAMILIES = {"LogisticRegression", "DecisionTree", "Ridge", "Lasso",
                              "LinearRegression", "SGDClassifier", "SGDRegressor"}

    valid_rows = [r for r in merged if isinstance(r.get("test_score"), (int, float))]

    best_idx: int | None = None
    if valid_rows:
        # Sort: best test_score desc, then training_time asc, then fold_variance asc
        def _sort_key(r: dict[str, Any]) -> tuple[float, float, float]:
            ts = float(r.get("test_score") or 0.0)
            tt = float(r.get("training_time") or 9999.0)
            fv = float(r.get("fold_variance") or 9999.0)
            return (-ts, tt, fv)

        sorted_rows = sorted(valid_rows, key=_sort_key)
        best_row = sorted_rows[0]

        # Explainability override: if user prefers explainable AND a complex model is #1,
        # check if an explainable model is within 0.01 of the best score
        if prefers_explainable and str(best_row.get("model_name") or "") not in _EXPLAINABLE_FAMILIES:
            best_score = float(best_row.get("test_score") or 0.0)
            for r in sorted_rows[1:]:
                if str(r.get("model_name") or "") in _EXPLAINABLE_FAMILIES:
                    if best_score - float(r.get("test_score") or 0.0) <= 0.01:
                        best_row = r
                        break

        # Mark best row in the merged list
        for row in merged:
            row_matches = (
                row.get("model_name") == best_row.get("model_name")
                and row.get("iteration") == best_row.get("iteration")
            )
            row["is_best"] = row_matches
            if row_matches and row.get("model_name"):
                row["model_name"] = f"*BEST* {row['model_name']}"
        best_idx = next(
            (i for i, r in enumerate(merged) if r.get("is_best")), None
        )

    session.final_report_markdown = report_md
    session.final_pipeline_script = script
    session.script_validation = validation
    log_decision(
        session,
        step="Phase 7 - Script validation",
        action=f"script_validation:{validation.get('status')}",
        rationale=str(validation.get("message") or ""),
        attempts=validation.get("attempts"),
        backend=validation.get("backend"),
        reproduced_score=validation.get("reproduced_score"),
        reported_score=validation.get("reported_score"),
    )
    session.final_comparison_table = merged

    # ── Mark session complete ─────────────────────────────────────────────────
    session.is_complete = True

    msgs = list(state.get("messages") or [])

    # Failure scenario: all models failed — report explains what went wrong
    if failed_completely:
        msgs.append(
            "### Phase 7 — Final report\n\n"
            "> ⚠️ **Pipeline completed without successful models.** "
            "The report below explains what went wrong and suggests actionable next steps.\n\n"
            + (report_md or "_Report unavailable._")
        )
    else:
        msgs.append(
            "### Phase 7 — Final report\n\n"
            + (report_md or "_Report unavailable._")
        )

    # Final comparison table is displayed by ModelTable in the frontend; no markdown needed here.

    v_status = validation.get("status")
    v_icon = "✅" if v_status == STATUS_VERIFIED else "⏭️" if v_status == "skipped" else "⚠️"
    v_where = f" in the `{validation.get('backend')}` sandbox" if validation.get("backend") else ""
    v_tries = f" ({validation.get('attempts')} attempt(s))" if validation.get("attempts") else ""
    msgs.append(
        f"### Phase 7 — Script verification\n\n{v_icon} {validation.get('message') or 'Not validated.'}"
        f"{v_where}{v_tries}"
    )

    # Recommendation
    if best_idx is not None and valid_rows:
        best_name = str(best_row.get("model_name") or "").replace("*BEST* ", "")
        best_score_val = best_row.get("test_score")
        explainability_note = (
            f" (selected as best explainable model within 0.01 of the top score — "
            "user requested interpretability)"
        ) if prefers_explainable else ""
        msgs.append(
            f"\n**Recommendation:** deploy `{best_name}` as the final model "
            f"(test score `{best_score_val:.4f}` on metric `{(session.execution_plan or {}).get('metric', '?')}`)"
            f"{explainability_note}. "
            "Validate on a held-out or production-like slice before deployment."
        )

    _record_phase_calls(session, "report_generation", calls0)
    logger.info(LLMClient.phase_call_stats_message("Phase 7 Final report", calls0))

    # ── Token cost summary ────────────────────────────────────────────────────
    token_tbl = dict(session.token_cost_table or {})
    total_calls = sum(token_tbl.values())
    cache_hits = LLMClient.cache_hits
    token_tbl["_total"] = total_calls
    token_tbl["_cache_hits"] = cache_hits
    session.token_cost_table = token_tbl

    final_exec = dict(state.get("execution_results") or {})
    final_exec["token_cost_table"] = token_tbl

    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": final_exec,
        "phase7_done": True,
    }


def route_after_threshold_check(state: OrchestratorState) -> str:
    session = _session(state)
    results = dict(state.get("execution_results") or session.execution_results or {})
    if results.get("status") == "no_models_succeeded":
        return "diagnose_failure"
    if results.get("status") == "missing_target":
        return "diagnose_failure"
    if state.get("threshold_ok"):
        return "phase7_final_output"
    if int(session.iteration_count or 0) >= int(config.MAX_ITERATIONS):
        return "phase7_final_output"
    if bool((session.task_config or {}).get("user_satisfied", False)):
        return "phase7_final_output"
    # Convergence check: if the best score didn't improve from the previous iteration,
    # there's no point re-running the same models — go straight to Phase 7.
    all_results = list(session.all_results or [])
    if len(all_results) >= 2:
        prev_score = all_results[-2].get("best_test_score")
        curr_score = results.get("best_test_score")
        if isinstance(prev_score, (int, float)) and isinstance(curr_score, (int, float)):
            if float(curr_score) <= float(prev_score) + 0.001:
                return "phase7_final_output"
    return "diagnose_failure"


def route_after_diagnosis(state: OrchestratorState) -> str:
    diagnosis = dict(state.get("diagnosis") or {})
    if diagnosis.get("failure_type") == "fundamental_data_insufficiency" or diagnosis.get("should_exit"):
        return "phase7_final_output"
    return "targeted_revision"


def route_after_targeted_revision(state: OrchestratorState) -> str:
    target = str(state.get("loop_target") or "phase4_planning")
    if target in {
        "basic_cleaning",
        "deep_cleaning",
        "feature_selection",
        "feature_engineering",
        "phase4_planning",
    }:
        return target
    return "phase4_planning"


def route_after_feature_selection(state: OrchestratorState) -> str:
    session = _session(state)
    if (session.task_config or {}).get("_pipeline_stop") == "too_few_features":
        return "stop"
    return "deep_cleaning"


def route_after_verification(state: OrchestratorState) -> str:
    if state.get("verification_ok"):
        return "phase4_planning"
    if state.get("retry_exceeded"):
        return "phase4_planning"
    return state.get("retry_target") or "phase4_planning"


def _timed(name: str, fn):
    """Wrap a graph node: accumulate its wall-clock seconds in session.phase_timings."""

    def wrapper(state: OrchestratorState) -> OrchestratorState:
        t0 = time.perf_counter()
        out = fn(state)
        elapsed = time.perf_counter() - t0
        sess = (out or {}).get("session") or state.get("session")
        if isinstance(sess, dict):
            timings = dict(sess.get("phase_timings") or {})
            timings[name] = round(timings.get(name, 0.0) + elapsed, 2)
            out = {**(out or {}), "session": {**sess, "phase_timings": timings}}
            if name == "phase7_final_output":
                ranked = sorted(timings.items(), key=lambda kv: -kv[1])
                logger.info("[TIMING] total %.0fs — %s", sum(timings.values()),
                            ", ".join(f"{k}={v:.0f}s" for k, v in ranked))
        logger.info("[TIMING] %s: %.1fs", name, elapsed)
        return out

    return wrapper


def build_graph():
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("planning", _timed("planning", node_planning))
    workflow.add_node("basic_cleaning", _timed("basic_cleaning", node_basic_cleaning))
    workflow.add_node("split_data", _timed("split_data", node_split_data))
    workflow.add_node("unsupervised_train", _timed("unsupervised_train", node_unsupervised_train))
    workflow.add_node("eda", _timed("eda", node_eda))
    workflow.add_node("feature_selection", _timed("feature_selection", node_feature_selection))
    workflow.add_node("deep_cleaning", _timed("deep_cleaning", node_deep_cleaning))
    workflow.add_node("feature_engineering", _timed("feature_engineering", node_feature_engineering))
    workflow.add_node("verification", _timed("verification", node_verification))
    workflow.add_node("phase4_planning", _timed("phase4_planning", node_phase4_planning))
    workflow.add_node("train_evaluate", _timed("train_evaluate", node_train_evaluate))
    workflow.add_node("threshold_check", _timed("threshold_check", node_threshold_check))
    workflow.add_node("diagnose_failure", _timed("diagnose_failure", node_diagnose_failure))
    workflow.add_node("targeted_revision", _timed("targeted_revision", node_targeted_revision))
    workflow.add_node("phase7_final_output", _timed("phase7_final_output", node_phase7_final_output))

    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "basic_cleaning")
    workflow.add_conditional_edges(
        "basic_cleaning",
        route_after_basic_cleaning,
        {"split_data": "split_data", "unsupervised_train": "unsupervised_train"},
    )
    workflow.add_edge("unsupervised_train", "phase7_final_output")
    workflow.add_edge("split_data", "eda")
    workflow.add_edge("eda", "feature_selection")
    workflow.add_conditional_edges(
        "feature_selection",
        route_after_feature_selection,
        {"deep_cleaning": "deep_cleaning", "stop": END},
    )
    workflow.add_edge("deep_cleaning", "feature_engineering")
    workflow.add_edge("feature_engineering", "verification")

    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "phase4_planning": "phase4_planning",
            "basic_cleaning": "basic_cleaning",
            "eda": "eda",
            "feature_selection": "feature_selection",
            "deep_cleaning": "deep_cleaning",
            "feature_engineering": "feature_engineering",
        },
    )
    workflow.add_edge("phase4_planning", "train_evaluate")
    workflow.add_edge("train_evaluate", "threshold_check")
    workflow.add_conditional_edges(
        "threshold_check",
        route_after_threshold_check,
        {
            "diagnose_failure": "diagnose_failure",
            "phase7_final_output": "phase7_final_output",
        },
    )
    workflow.add_conditional_edges(
        "diagnose_failure",
        route_after_diagnosis,
        {
            "targeted_revision": "targeted_revision",
            "phase7_final_output": "phase7_final_output",
        },
    )
    workflow.add_conditional_edges(
        "targeted_revision",
        route_after_targeted_revision,
        {
            "basic_cleaning": "basic_cleaning",
            "deep_cleaning": "deep_cleaning",
            "feature_selection": "feature_selection",
            "feature_engineering": "feature_engineering",
            "phase4_planning": "phase4_planning",
        },
    )
    workflow.add_edge("phase7_final_output", END)
    return workflow.compile()


def run_preprocessing_pipeline(initial_session: SessionState, df: pd.DataFrame) -> OrchestratorState:
    """Convenience: compile graph and invoke with initial session + dataframe."""
    graph = build_graph()
    return graph.invoke(
        {
            "session": initial_session.to_dict(),
            "df": df,
            "messages": [],
            "checkpoints": {},
        }
    )
