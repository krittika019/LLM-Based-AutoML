"""
LangGraph workflow: Phase 2 planning + Phase 3 preprocessing + Phase 4/5 execution.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from automl_system import config
from automl_system.agents.failure_diagnosis import diagnose_failure
from automl_system.llm.client import LLMClient
from automl_system.agents.pipeline_planner import make_execution_plan
from automl_system.agents.preprocessing_planner import make_preprocessing_plan
from automl_system.agents.report_generator import generate_report, generate_stakeholder_report_text
from automl_system.pipeline.stakeholder_pdf import build_stakeholder_pdf
from automl_system.pipeline.cleaning import basic_clean
from automl_system.pipeline.code_generator import generate_pipeline_script
from automl_system.pipeline.deep_cleaning import deep_clean
from automl_system.pipeline.eda import run_eda
from automl_system.pipeline.execution import MissingTargetError, run_pipeline
from automl_system.pipeline.feature_engineering import engineer_features
from automl_system.pipeline.feature_selection import select_features
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
from automl_system.utils.dataset_profile import build_semantic_profile_vector, compute_dataset_profile
from automl_system.memory import semantic as semantic_memory

logger = logging.getLogger(__name__)

_REWIND_CHECKPOINT: dict[str, str] = {
    STEP_BASIC_CLEAN: "initial",
    STEP_EDA: "after_basic_cleaning",
    STEP_FEATURE_SELECTION: "after_eda",
    STEP_DEEP_CLEANING: "after_feature_selection",
    STEP_FEATURE_ENGINEERING: "after_deep_cleaning",
}


class OrchestratorState(TypedDict, total=False):
    """Graph state: session dict, dataframe, checkpoints for retry, routing flags."""

    session: dict[str, Any]
    df: Any
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
    fe = bool(plan.get("feature_engineering_recommended") or plan.get("feature_engineering_likely"))
    challenges.append(
        f"**Feature engineering:** {'will be attempted' if fe else 'is deferred'} "
        f"({'dataset size supports derived features' if fe else 'small or simple dataset — start with baseline features'})."
    )
    block = "### Phase 2 — Planning (Mode 1)\n\n"
    block += "\n".join(f"- {c}" for c in challenges) if challenges else "- Baseline preprocessing plan generated from profile.\n"
    block += f"\n\n{session.user_summary_planning or 'Planning complete.'}\n"
    msgs.append(block)
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


def node_eda(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state["df"]
    calls0 = LLMClient.call_count
    run_eda(df, session)
    ck = dict(state.get("checkpoints") or {})
    ck["after_eda"] = df.copy()
    msgs = list(state.get("messages") or [])
    msgs.append("### Phase 3 — EDA narrative\n\n" + (session.user_summary_eda or ""))
    for m in session.eda_chat_messages or []:
        if isinstance(m, dict) and m.get("content"):
            msgs.append(m)
        elif isinstance(m, str):
            msgs.append({"role": "assistant", "content": m})
    logger.info(LLMClient.phase_call_stats_message("Phase 3 EDA", calls0))
    return {"session": session.to_dict(), "df": df, "checkpoints": ck, "messages": msgs}


def node_feature_selection(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df = select_features(state["df"], session)
    ck = dict(state.get("checkpoints") or {})
    msgs = list(state.get("messages") or [])

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
            "session": session.to_dict(), "df": df, "checkpoints": ck,
            "messages": msgs, "pipeline_complete": True, "events_to_emit": emit,
        }

    ck["after_feature_selection"] = df.copy()
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
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Feature selection", calls0))
    return {"session": session.to_dict(), "df": df, "checkpoints": ck, "messages": msgs}


def node_deep_cleaning(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df = deep_clean(state["df"], session)
    ck = dict(state.get("checkpoints") or {})
    ck["after_deep_cleaning"] = df.copy()
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
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Deep cleaning", calls0))
    return {"session": session.to_dict(), "df": df, "checkpoints": ck, "messages": msgs}


def node_feature_engineering(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    calls0 = LLMClient.call_count
    df = engineer_features(state["df"], session)
    ck = dict(state.get("checkpoints") or {})
    ck["after_feature_engineering"] = df.copy()
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
    logger.info(LLMClient.phase_call_stats_message("Phase 3 Feature engineering", calls0))
    return {"session": session.to_dict(), "df": df, "checkpoints": ck, "messages": msgs}


def node_verification(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state["df"]
    ck = dict(state.get("checkpoints") or {})
    ok, raw_msg = verify_preprocessing(df, session)
    user_msg, fail_step = parse_verification_route(raw_msg)
    msgs = list(state.get("messages") or [])
    msgs.append(user_msg if not ok else raw_msg)

    out: OrchestratorState = {
        "session": session.to_dict(),
        "df": df,
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
                out["df"] = ck[key].copy()
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
    plan = make_execution_plan(session)
    msgs = list(state.get("messages") or [])

    block = "### Phase 4 — Planning (Mode 2: execution)\n\n"

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
        block += f"\n**Performance threshold:** `{float(thr):.4f}` — {thr_note}\n"

    msgs.append(block)
    logger.info(LLMClient.phase_call_stats_message("Phase 4 Execution planning", calls0))
    return {"messages": msgs, "session": session.to_dict(), "df": df}


def node_split_data(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    msgs = list(state.get("messages") or [])
    task_type = str((session.execution_plan or {}).get("task_type") or "tabular")
    msgs.append(f"Execution split_data: preparing a shared 80/20 {'stratified' if task_type == 'classification' else 'standard'} split.")
    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": dict(state.get("execution_results") or {}),
    }


def node_train_evaluate(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    df = state.get("df")
    if df is None:
        raise ValueError("train_evaluate requires state['df']")
    msgs = list(state.get("messages") or [])
    results: dict[str, Any] = {}
    calls0 = LLMClient.call_count
    try:
        results = run_pipeline(df, session)
        tbl = results.get("comparison_table") or []
        msgs.append("### Phase 5 — Model training & evaluation\n")
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
    logger.info(LLMClient.phase_call_stats_message("Phase 5 Execution", calls0))
    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": df,
        "execution_results": results,
    }


def node_interpret_results(state: OrchestratorState) -> OrchestratorState:
    # Interpretation is displayed by ModelTable in the frontend; no message needed here.
    session = _session(state)
    results = dict(state.get("execution_results") or session.execution_results or {})
    return {
        "messages": list(state.get("messages") or []),
        "session": session.to_dict(),
        "df": state.get("df"),
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
        msgs.append(
            f"- Target threshold: `{float(threshold):.4f}` | Best test score: `{float(best):.4f}` "
            f"| Gap: `{gap:+.4f}` → {status_str}"
        )
    else:
        msgs.append("Threshold check skipped: threshold or best score unavailable.")

    user_satisfied = bool((session.task_config or {}).get("user_satisfied", False))
    maxed = int(session.iteration_count) >= int(config.MAX_ITERATIONS)

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
    elif user_satisfied:
        msgs.append("→ User satisfaction flag set — proceeding to Phase 7.")
    # Exit condition 3 (fundamental_data_insufficiency) is detected in node_diagnose_failure
    # and routed to Phase 7 immediately via route_after_diagnosis

    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": results,
        "threshold_ok": ok,
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

    # General path: LLM-based failure diagnosis
    diagnosis = diagnose_failure(session, results)
    msgs.append(
        "**Failure diagnosis:** "
        f"{diagnosis.get('failure_type')} → {diagnosis.get('responsible_phase')}. "
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

    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": dict(state.get("execution_results") or {}),
        "loop_target": loop_target,
    }


def node_phase7_final_output(state: OrchestratorState) -> OrchestratorState:
    session = _session(state)
    all_results = list(session.all_results or [])
    calls0 = LLMClient.call_count

    # ── Step 1: Explanation Report ─────────────────────────────────────────────
    # Scenario: pipeline failed completely (no successful models)
    failed_completely = not any(
        isinstance(r.get("best_test_score"), (int, float)) for r in all_results
    )
    report_md = generate_report(session, all_results)

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

    # ── Step 3: Consolidated Code Output ──────────────────────────────────────
    script = generate_pipeline_script(session)

    session.final_report_markdown = report_md
    session.final_pipeline_script = script
    session.final_comparison_table = merged

    # ── Semantic memory update ────────────────────────────────────────────────
    try:
        profile = session.dataset_profile or {}
        pv = build_semantic_profile_vector(profile)
        best_res = all_results[-1] if all_results else {}
        outcome: dict[str, Any] = {
            "task": str((session.task_config or {}).get("goal") or ""),
            "target": str((session.task_config or {}).get("target_column") or ""),
            "best_model": best_res.get("best_model_name"),
            "best_test_score": best_res.get("best_test_score"),
            "metric": best_res.get("metric"),
            "iterations": int(session.iteration_count or 0),
            "last_failure_type": str(
                (session.last_failure_diagnosis or {}).get("failure_type") or ""
            ),
        }
        semantic_memory.store_run(pv, outcome)
        logger.info("Phase 7: semantic memory updated for this run.")
    except Exception as _sm_err:
        logger.warning("Phase 7: semantic memory update failed: %s", _sm_err)

    # ── Mark session complete ─────────────────────────────────────────────────
    session.is_complete = True

    # ── Stakeholder PDF (non-technical 1-page report) ─────────────────────────
    try:
        stakeholder_data = generate_stakeholder_report_text(session, all_results)
        pdf_bytes = build_stakeholder_pdf(
            stakeholder_data,
            all_results,
            goal=str((session.task_config or {}).get("goal") or ""),
        )
        pdf_path = Path(tempfile.gettempdir()) / "automl_stakeholder_report.pdf"
        pdf_path.write_bytes(pdf_bytes)
        session.stakeholder_pdf_path = str(pdf_path)
    except Exception as _sh_err:
        logger.warning("Stakeholder PDF generation skipped: %s", _sh_err)
        session.stakeholder_pdf_path = ""

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

    # Recommendation
    if best_idx is not None and valid_rows:
        best_name = str(valid_rows[0].get("model_name") or "").replace("*BEST* ", "")
        best_score_val = valid_rows[0].get("test_score")
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

    logger.info(LLMClient.phase_call_stats_message("Phase 7 Final report", calls0))
    return {
        "messages": msgs,
        "session": session.to_dict(),
        "df": state.get("df"),
        "execution_results": dict(state.get("execution_results") or {}),
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


def route_after_verification(state: OrchestratorState) -> str:
    if state.get("verification_ok"):
        return "phase4_planning"
    if state.get("retry_exceeded"):
        return "phase4_planning"
    return state.get("retry_target") or "phase4_planning"


def build_graph():
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("planning", node_planning)
    workflow.add_node("basic_cleaning", node_basic_cleaning)
    workflow.add_node("eda", node_eda)
    workflow.add_node("feature_selection", node_feature_selection)
    workflow.add_node("deep_cleaning", node_deep_cleaning)
    workflow.add_node("feature_engineering", node_feature_engineering)
    workflow.add_node("verification", node_verification)
    workflow.add_node("phase4_planning", node_phase4_planning)
    workflow.add_node("split_data", node_split_data)
    workflow.add_node("train_evaluate", node_train_evaluate)
    workflow.add_node("interpret_results", node_interpret_results)
    workflow.add_node("threshold_check", node_threshold_check)
    workflow.add_node("diagnose_failure", node_diagnose_failure)
    workflow.add_node("targeted_revision", node_targeted_revision)
    workflow.add_node("phase7_final_output", node_phase7_final_output)

    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "basic_cleaning")
    workflow.add_edge("basic_cleaning", "eda")
    workflow.add_edge("eda", "feature_selection")
    workflow.add_edge("feature_selection", "deep_cleaning")
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
    workflow.add_edge("phase4_planning", "split_data")
    workflow.add_edge("split_data", "train_evaluate")
    workflow.add_edge("train_evaluate", "interpret_results")
    workflow.add_edge("interpret_results", "threshold_check")
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
