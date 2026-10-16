"""Structured, per-phase content for the run page inspector.

`build_phase_details(session)` turns what a run recorded in the session into one dict per
phase (1-7): headline stats, the decisions made (and whether a rule, a statistic or the LLM
made them), visual sections and a short takeaway. Everything is read from recorded data —
nothing here calls the LLM — and every phase is built defensively: a phase that has not run
yet, or a run mode that skips it (unsupervised), comes back with `available: False` instead
of raising.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

PHASE_TITLES = {
    1: "Understand the data",
    2: "Plan preprocessing",
    3: "Prepare the data",
    4: "Plan the models",
    5: "Train & evaluate",
    6: "Check & revise",
    7: "Deliver",
}

# graph node -> phase (mirrors server._PHASE_NODE_META)
_NODE_PHASE = {
    "planning": 2, "basic_cleaning": 3, "split_data": 3, "eda": 3, "feature_selection": 3,
    "deep_cleaning": 3, "feature_engineering": 3, "verification": 3, "phase4_planning": 4,
    "train_evaluate": 5, "unsupervised_train": 5, "interpret_results": 5,
    "threshold_check": 6, "diagnose_failure": 6, "targeted_revision": 6,
    "phase7_final_output": 7,
}
# token_cost_table key -> phase
_LLM_PHASE = {
    "planning": 2, "eda": 3, "feature_selection": 3, "deep_cleaning": 3,
    "feature_engineering": 3, "execution_planning": 4, "model_training": 5,
    "unsupervised_train": 5, "report_generation": 7,
}
_STEP_PHASE_PREFIX = {"Verification": 3}

_SOURCE = {"rule": "rule", "stat": "stat", "llm": "llm", "llm_flagged": "llm",
           "llm_domain": "llm", "fallback": "rule", "router": "rule"}


# ── small helpers ─────────────────────────────────────────────────────────────

def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fmt(v: Any, digits: int = 4) -> str:
    f = _num(v)
    if f is None:
        return "—"
    if float(f).is_integer() and abs(f) >= 1:
        return f"{int(f):,}"
    return f"{f:.{digits}f}"


def _pct(v: Any) -> str:
    f = _num(v)
    return "—" if f is None else f"{f:.1f}%"


def _stat(label: str, value: Any, note: str = "") -> dict[str, str]:
    return {"label": label, "value": str(value), "note": note}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _humanize(name: str) -> str:
    return str(name or "").replace("_", " ").strip()


def _phase_of_step(step: str) -> int | None:
    step = str(step or "")
    if step.startswith("Phase "):
        head = step[6:7]
        return int(head) if head.isdigit() else None
    for prefix, phase in _STEP_PHASE_PREFIX.items():
        if step.startswith(prefix):
            return phase
    return None


def _log_entries(session: Any, phase: int) -> list[tuple[int, dict[str, Any]]]:
    """(1-based log number, entry) for decision-log entries belonging to `phase`."""
    out = []
    for i, e in enumerate(_get(session, "local_decisions_log") or [], start=1):
        if isinstance(e, dict) and _phase_of_step(e.get("step")) == phase:
            out.append((i, e))
    return out


def _entry_for(entries: list[tuple[int, dict]], action: str) -> int | None:
    for i, e in entries:
        if e.get("action") == action:
            return i
    return None


_LOG_SKIP = {"step", "action", "rationale", "timestamp", "phase"}


def _log_view(session: Any, phase: int, limit: int = 40) -> list[dict[str, Any]]:
    """Numbered decision-log entries for one phase, compacted for display."""
    out = []
    for n, e in _log_entries(session, phase)[-limit:]:
        note = str(e.get("rationale") or "").strip()
        if not note:
            extras = [f"{k}={v}" for k, v in e.items()
                      if k not in _LOG_SKIP and isinstance(v, (str, int, float, bool)) and v not in ("", None)]
            note = ", ".join(extras)
        out.append({"n": n, "step": str(e.get("step") or ""), "action": _humanize(str(e.get("action") or "")),
                    "note": note[:240]})
    return out


def _seconds(session: Any, phase: int) -> float | None:
    timings = _get(session, "phase_timings") or {}
    total = sum(float(v) for k, v in timings.items() if _NODE_PHASE.get(k) == phase and _num(v) is not None)
    return round(total, 1) if total else None


def _llm_calls(session: Any, phase: int) -> int:
    table = _get(session, "token_cost_table") or {}
    return int(sum(int(v or 0) for k, v in table.items() if _LLM_PHASE.get(k) == phase))


def _metric_label(session: Any) -> str:
    res = _get(session, "execution_results") or {}
    return str(res.get("metric_display") or (_get(session, "problem_summary") or {}).get("primary_metric_label")
               or res.get("metric") or "score")


def _display_score(v: Any, lower_better: bool) -> float | None:
    f = _num(v)
    if f is None:
        return None
    return -f if lower_better and f < 0 else f


# ── phases ────────────────────────────────────────────────────────────────────

def _phase1(s: Any) -> dict[str, Any]:
    ps = _get(s, "problem_summary") or {}
    tc = _get(s, "task_config") or {}
    if not ps and not tc.get("target_column"):
        return {"available": False}
    entries = _log_entries(s, 1)
    task = str(ps.get("task_type") or tc.get("task_type") or "")
    target = ps.get("target") or tc.get("target_column")
    rows, cols = ps.get("rows"), ps.get("columns")
    missing, dupes = ps.get("missing_values"), ps.get("duplicate_rows")

    stats = [
        _stat("Rows", _fmt(rows)),
        _stat("Columns", _fmt(cols), f"{ps.get('numerical_count', '—')} numeric · {ps.get('categorical_count', '—')} categorical"),
        _stat("Task", _humanize(task) or "—", f"target: {target}" if target else ""),
        _stat("Missing cells", _fmt(missing), f"{_fmt(dupes)} duplicate rows" if dupes is not None else ""),
    ]

    decisions = []
    conf = ps.get("target_confidence")
    target_entry = _entry_for(entries, "resolve_target_hint") or _entry_for(entries, "resolve_target_llm")
    by_llm = _entry_for(entries, "resolve_target_llm") is not None
    if target:
        decisions.append({
            "what": f"Target column: {target}",
            "why": ("Read from your prompt and checked against the columns." if not by_llm
                    else "Inferred from the prompt and column names, then checked against the data.")
                   + (f" Confidence: {conf}." if conf not in (None, "") else ""),
            "by": "llm" if by_llm else "rule", "entry": target_entry,
        })
    if task:
        decisions.append({
            "what": f"Task type: {_humanize(task)}",
            "why": "Decided from the target's type and number of distinct values.",
            "by": "stat", "entry": None,
        })
    metric = ps.get("primary_metric_label")
    if metric:
        decisions.append({
            "what": f"Primary metric: {metric}",
            "why": str(ps.get("objective") or "Chosen to fit the task and the class balance."),
            "by": "rule", "entry": _entry_for(entries, "metric_selected"),
        })

    sections = []
    table = [r for r in (ps.get("table") or []) if isinstance(r, dict)]
    if table:
        sections.append({
            "type": "table", "title": "Columns and their roles",
            "columns": ["Column", "Type", "Role"],
            "rows": [[str(r.get("name")), str(r.get("dtype") or ""), str(r.get("role") or "")] for r in table[:20]],
            "more": max(0, len(table) - 20),
        })
    profile_cols = (_get(s, "initial_dataset_profile") or _get(s, "dataset_profile") or {}).get("columns") or {}
    nulls = sorted(((c, _num(i.get("null_pct"))) for c, i in profile_cols.items() if isinstance(i, dict)),
                   key=lambda t: -(t[1] or 0))
    nulls = [(c, v) for c, v in nulls if v and v > 0][:8]
    if nulls:
        sections.append({"type": "bars", "title": "Missing values by column", "unit": "%",
                         "items": [{"label": c, "value": round(v, 1)} for c, v in nulls]})

    takeaway = (f"{_fmt(rows)} rows × {_fmt(cols)} columns, predicting {target} as a {_humanize(task)} problem."
                if target else "")
    if nulls:
        takeaway += f" {len(nulls)} column(s) have missing values; Phase 3 decides how to fill them."
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway.strip()}


def _phase2(s: Any) -> dict[str, Any]:
    plan = _get(s, "preprocessing_plan") or {}
    if not plan:
        return {"available": False}
    enc = plan.get("encoding_candidates") or []
    skew = plan.get("skewed_columns") or []
    risks = plan.get("leakage_risks") or []
    notes = [str(n) for n in (plan.get("notes") or []) if n]

    stats = [
        _stat("Columns to encode", len(enc) if isinstance(enc, list) else "—"),
        _stat("Skewed columns", len(skew) if isinstance(skew, list) else "—"),
        _stat("Leakage risks", len(risks) if isinstance(risks, list) else "—"),
        _stat("Scaling", "needed" if plan.get("scaling_needed") else "not needed"),
    ]
    decisions = [{
        "what": "Feature engineering " + ("recommended" if plan.get("feature_engineering_recommended") else "deferred"),
        "why": "Based on dataset size and how much structure the columns show.",
        "by": "llm", "entry": None,
    }]
    if isinstance(risks, list):
        for r in risks[:4]:
            decisions.append({"what": f"Watch for leakage: {r if isinstance(r, str) else r.get('column', r)}",
                              "why": "Flagged so later phases verify it does not encode the target.",
                              "by": "llm", "entry": None})
    sections = []
    if isinstance(enc, list) and enc:
        sections.append({"type": "chips", "title": "Encoding candidates", "items": [str(c) for c in enc[:24]]})
    if isinstance(skew, list) and skew:
        sections.append({"type": "chips", "title": "Skewed numeric columns", "items": [str(c) for c in skew[:24]]})
    null_notes = plan.get("null_strategy_notes")
    if null_notes:
        items = null_notes if isinstance(null_notes, list) else [null_notes]
        sections.append({"type": "list", "title": "Missing-value strategy", "items": [str(i) for i in items[:8]]})
    if notes:
        sections.append({"type": "list", "title": "Planner notes", "items": notes[:6]})
    takeaway = str(_get(s, "user_summary_planning") or "").strip()
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


def _phase3(s: Any) -> dict[str, Any]:
    cleaning = _get(s, "cleaning_report") or {}
    eda = _get(s, "eda_results") or {}
    selection = [e for e in (_get(s, "feature_selection_log") or []) if isinstance(e, dict)]
    assignments = _get(s, "deep_cleaning_assignments") or {}
    fe_log = [e for e in (_get(s, "feature_engineering_log") or []) if isinstance(e, dict)]
    meta = _get(s, "preprocessing_meta") or {}
    verified = _get(s, "last_verification") or {}
    if not (cleaning or eda or selection or assignments or fe_log or verified):
        return {"available": False}
    entries = _log_entries(s, 3)

    added = [e for e in fe_log if e.get("status") == "added"]
    tried = [e for e in fe_log if e.get("feature")]
    imputed = sum(1 for a in assignments.values()
                  if isinstance(a, dict) and a.get("imputation") not in (None, "none", "leave"))
    shape = verified.get("shape") or []
    stats = [
        _stat("Duplicates removed", _fmt(cleaning.get("duplicate_rows_removed") or 0)),
        _stat("Columns dropped", len(selection)),
        _stat("Features added", len(added), f"of {len(tried)} proposed" if tried else ""),
        _stat("Final shape", f"{shape[0]:,} × {shape[1]}" if len(shape) == 2 else "—",
              "checks passed" if verified.get("ok") else ""),
    ]

    decisions = []
    for e in selection[:8]:
        decisions.append({"what": f"Dropped {e.get('column')}", "why": str(e.get("rationale") or ""),
                          "by": _SOURCE.get(str(e.get("tier")), "llm"), "entry": None})
    if assignments:
        by_source: dict[str, int] = {}
        for a in assignments.values():
            if isinstance(a, dict):
                src = _SOURCE.get(str(a.get("source")), "rule")
                by_source[src] = by_source.get(src, 0) + 1
        parts = [f"{n} by {'a rule' if k == 'rule' else 'a statistic' if k == 'stat' else 'the LLM'}"
                 for k, n in sorted(by_source.items(), key=lambda kv: -kv[1])]
        decisions.append({
            "what": f"Cleaning recipe chosen for {len(assignments)} columns",
            "why": "Imputation, encoding, scaling and outlier handling per column — " + ", ".join(parts) + ".",
            "by": max(by_source, key=by_source.get) if by_source else "rule",
            "entry": _entry_for(entries, "all_assignments_logged_before_execution"),
        })
    for e in added[:6]:
        gain = _num(e.get("cv_gain"))
        decisions.append({
            "what": f"Added feature {e.get('feature')}",
            "why": (str(e.get("describe") or e.get("rationale") or "")
                    + (f" Cross-validated gain: +{gain:.4f}." if gain is not None else "")).strip(),
            "by": "stat" if gain is not None else "llm", "entry": None,
        })
    skipped = next((e for e in fe_log if e.get("status") == "skipped"), None)
    if skipped:
        decisions.append({"what": "Feature engineering skipped", "why": _humanize(str(skipped.get("reason") or "")),
                          "by": "rule", "entry": _entry_for(entries, "skipped_by_router")})

    sections: list[dict[str, Any]] = []
    steps = [
        {"name": "Basic cleaning", "detail": f"{_fmt(cleaning.get('duplicate_rows_removed') or 0)} duplicates removed, "
                                             f"{len(cleaning.get('coercion_log') or [])} type fixes"},
        {"name": "Train/test split", "detail": _split_detail(s)},
        {"name": "Exploration (EDA)", "detail": f"{len(eda.get('target_correlations') or {})} feature–target links measured"},
        {"name": "Feature selection", "detail": f"{len(selection)} column(s) dropped"},
        {"name": "Deep cleaning", "detail": f"{len(assignments)} columns, {imputed} imputed"},
        {"name": "Feature engineering", "detail": (f"{len(added)} added, {len(tried) - len(added)} rejected"
                                                   if tried else "skipped" if skipped else "none proposed")},
        {"name": "Verification", "detail": "all checks passed" if verified.get("ok") else "see log"},
    ]
    sections.append({"type": "steps", "title": "What happened, in order", "items": steps})

    corr = eda.get("target_correlations") or {}
    if isinstance(corr, dict) and corr:
        top = sorted(((k, _num(v)) for k, v in corr.items() if _num(v) is not None), key=lambda t: -abs(t[1]))[:8]
        if top:
            sections.append({"type": "bars", "title": "Strongest links to the target", "signed": True,
                             "items": [{"label": k, "value": round(v, 3)} for k, v in top]})
    imb = eda.get("target_imbalance") or {}
    if isinstance(imb, dict) and imb.get("minority_pct") is not None:
        sections.append({"type": "kv", "title": "Target balance", "items": [
            {"label": "Minority class", "value": _pct(imb.get("minority_pct"))},
            {"label": "Classes", "value": str(imb.get("n_classes") or "—")},
            {"label": "Severity", "value": _humanize(str(imb.get("severity") or "—"))},
        ]})
    if assignments:
        rows = []
        for col, a in list(assignments.items())[:16]:
            if isinstance(a, dict):
                rows.append([str(col), _humanize(str(a.get("imputation") or "")), _humanize(str(a.get("encoding") or "")),
                             _humanize(str(a.get("scaling") or "none")), _humanize(str(a.get("outlier") or "")),
                             _SOURCE.get(str(a.get("source")), "rule")])
        sections.append({"type": "table", "title": "Per-column cleaning recipe",
                         "columns": ["Column", "Missing values", "Encoding", "Scaling", "Outliers", "By"],
                         "rows": rows, "more": max(0, len(assignments) - 16), "source_col": 5})
    if tried:
        items = []
        for e in tried[:12]:
            items.append({"label": str(e.get("feature")), "status": _humanize(str(e.get("status") or "")),
                          "value": _num(e.get("cv_gain")), "accepted": e.get("status") == "added"})
        sections.append({"type": "features", "title": "Engineered features (cross-validated)", "items": items})
    preview = meta.get("row_preview")
    if isinstance(preview, dict) and preview.get("before"):
        sections.append({"type": "diff", "title": "One row, before and after", **preview})

    takeaway = _phase3_takeaway(len(selection), len(added), imputed, verified.get("ok"))
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


def _split_detail(s: Any) -> str:
    sm = (_get(s, "execution_results") or {}).get("split_meta") or {}
    if sm.get("n_train"):
        return f"{_fmt(sm.get('n_train'))} train / {_fmt(sm.get('n_test'))} test, {sm.get('split_type', '')}".strip(", ")
    return "held-out test set kept aside before any learning"


def _phase3_takeaway(dropped: int, added: int, imputed: int, ok: Any) -> str:
    parts = []
    if dropped:
        parts.append(f"{dropped} column(s) removed")
    if imputed:
        parts.append(f"{imputed} column(s) had missing values filled")
    if added:
        parts.append(f"{added} new feature(s) passed the cross-validation gate")
    head = ("; ".join(parts) + ".") if parts else "The data needed little change."
    return head + (" Every verification check passed, so training starts from clean data." if ok else "")


def _phase4(s: Any) -> dict[str, Any]:
    plan = _get(s, "execution_plan") or {}
    if not plan:
        return {"available": False}
    models = [m for m in (plan.get("models") or []) if isinstance(m, dict)]
    cv = plan.get("cv_design") or {}
    threshold = _num(plan.get("threshold"))
    metric = str(plan.get("metric") or "")
    lower = metric.startswith("neg_")
    stats = [
        _stat("Model families", len(models)),
        _stat("Validation", f"{cv.get('n_splits') or cv.get('folds') or '—'}-fold" if isinstance(cv, dict) else str(cv),
              _humanize(str(cv.get("type") or cv.get("strategy") or "")) if isinstance(cv, dict) else ""),
        _stat("Search", _humanize(str(plan.get("search_strategy") or "—"))),
        _stat("Target score", _fmt(_display_score(threshold, lower)) if threshold is not None else "—",
              _humanize(metric.replace("neg_", ""))),
    ]
    decisions = [{"what": f"Try {_humanize(m.get('family'))}", "why": str(m.get("rationale") or ""),
                  "by": "llm", "entry": None} for m in models[:8]]
    if threshold is not None:
        decisions.append({"what": f"Success threshold: {_fmt(_display_score(threshold, lower))}",
                          "why": str(plan.get("threshold_rationale") or "Score the best model has to reach before Phase 7."),
                          "by": "llm", "entry": None})
    sections = []
    if models:
        sections.append({"type": "table", "title": "Candidate models", "columns": ["Family", "Why it was picked"],
                         "rows": [[_humanize(m.get("family")), str(m.get("rationale") or "")] for m in models]})
    if isinstance(cv, dict) and cv:
        sections.append({"type": "kv", "title": "Validation design",
                         "items": [{"label": _humanize(k), "value": str(v)} for k, v in list(cv.items())[:6]
                                   if not isinstance(v, (dict, list))]})
    takeaway = (f"{len(models)} model families will be tuned with {_humanize(str(plan.get('search_strategy') or 'a search'))}"
                + (f" and must reach {_fmt(_display_score(threshold, lower))} to pass." if threshold is not None else "."))
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


def _phase5(s: Any) -> dict[str, Any]:
    res = _get(s, "execution_results") or {}
    unsup = _get(s, "unsupervised_results") or {}
    if not res and unsup:
        return _phase5_unsupervised(unsup)
    table = [r for r in (res.get("comparison_table") or []) if isinstance(r, dict)]
    if not res or not table:
        return {"available": False}
    entries = _log_entries(s, 5)
    lower = bool(res.get("metric_is_lower_better"))
    metric = _metric_label(s)
    ok_rows = [r for r in table if _num(r.get("test_score")) is not None and r.get("status", "ok") == "ok"]
    ok_rows.sort(key=lambda r: -float(r["test_score"]))
    best = ok_rows[0] if ok_rows else {}
    runner = ok_rows[1] if len(ok_rows) > 1 else {}
    best_score = _display_score(best.get("test_score"), lower)
    margin = None
    if runner:
        margin = abs(float(best["test_score"]) - float(runner["test_score"]))
    cv_gap = None
    if _num(best.get("cv_score")) is not None and _num(best.get("test_score")) is not None:
        cv_gap = abs(float(best["cv_score"]) - float(best["test_score"]))
    sm = res.get("split_meta") or {}

    stats = [
        _stat("Best model", str(res.get("best_model_name") or best.get("model_name") or "—")),
        _stat(f"Test {metric}", _fmt(best_score), f"+{margin:.4f} over next" if margin is not None else ""),
        _stat("Models compared", len(ok_rows), f"{len(table) - len(ok_rows)} failed or skipped" if len(table) > len(ok_rows) else ""),
        _stat("CV → test gap", _fmt(cv_gap) if cv_gap is not None else "—",
              ("stable" if cv_gap < 0.03 else "check for overfitting") if cv_gap is not None else ""),
    ]

    decisions = [{
        "what": f"Picked {res.get('best_model_name') or best.get('model_name')}",
        "why": f"Highest held-out {metric}" + (f", ahead of {runner.get('model_name')} by {margin:.4f}." if runner else "."),
        "by": "stat", "entry": _entry_for(entries, "train_evaluate_complete"),
    }]
    for action, what in (("seed_bagging", "Seed bagging"), ("voting_ensemble", "Voting ensemble"),
                         ("stacking_ensemble", "Stacking ensemble")):
        i = _entry_for(entries, action)
        if i is not None:
            e = next(e for n, e in entries if n == i)
            decisions.append({"what": what + " tried", "why": str(e.get("rationale") or "Kept only if it beat the single best model."),
                              "by": "stat", "entry": i})
    audit = res.get("elimination_audit") or {}
    if audit.get("eliminated_family"):
        decisions.append({"what": f"Dropped {audit.get('eliminated_family')} early",
                          "why": "Scored worst on a quick subsample, so it was not fully tuned"
                                 + (" — later confirmed correct." if audit.get("elimination_correct") else "."),
                          "by": "stat", "entry": None})
    leaks = res.get("leakage_warning_models") or []
    if leaks:
        decisions.append({"what": "Near-perfect score flagged", "why": f"{', '.join(leaks)} scored ≥ 0.999 — treat as possible leakage.",
                          "by": "rule", "entry": None})

    sections: list[dict[str, Any]] = []
    sections.append({
        "type": "leaderboard", "title": f"Leaderboard — test {metric}", "lower_better": lower,
        "items": [{"label": str(r.get("model_name")), "value": _display_score(r.get("test_score"), lower),
                   "cv": _display_score(r.get("cv_score"), lower), "best": r is best,
                   "time": _num(r.get("training_time"))} for r in ok_rows[:10]],
    })
    shap = res.get("shap_summary") or {}
    fi = res.get("feature_importances") or {}
    if shap.get("available") and shap.get("features"):
        sections.append({"type": "bars", "title": "What drives predictions (SHAP)",
                         "items": [{"label": f.get("name"), "value": round(float(f.get("mean_abs_shap") or 0), 4)}
                                   for f in shap["features"][:8]]})
    elif fi.get("top_features"):
        sections.append({"type": "bars", "title": "Most important features", "unit": "%",
                         "items": [{"label": str(n), "value": round(float(v), 1)} for n, v in fi["top_features"][:8]]})
    bm = res.get("best_model_metrics") or {}
    kv = [{"label": _humanize(k), "value": _fmt(v)} for k, v in bm.items() if _num(v) is not None][:8]
    if kv:
        sections.append({"type": "kv", "title": "Best model — full metrics", "items": kv})
    if sm:
        sections.append({"type": "split", "title": "Data used", "train": sm.get("n_train"), "test": sm.get("n_test"),
                         "kind": sm.get("split_type")})

    takeaway = (f"{res.get('best_model_name')} reached {_fmt(best_score)} {metric} on data it never saw during training"
                + (f", {margin:.4f} ahead of {runner.get('model_name')}." if runner else "."))
    if cv_gap is not None and cv_gap >= 0.03:
        takeaway += " The gap between cross-validation and test is noticeable, so expect some variance on new data."
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


def _phase5_unsupervised(u: dict[str, Any]) -> dict[str, Any]:
    stats = []
    for key, label in (("best_algorithm", "Algorithm"), ("n_clusters", "Clusters"),
                       ("silhouette", "Silhouette"), ("n_anomalies", "Anomalies")):
        if u.get(key) is not None:
            stats.append(_stat(label, _fmt(u[key]) if _num(u[key]) is not None else str(u[key])))
    sections = []
    sizes = u.get("cluster_sizes")
    if isinstance(sizes, dict) and sizes:
        sections.append({"type": "bars", "title": "Cluster sizes",
                         "items": [{"label": f"Cluster {k}", "value": _num(v) or 0} for k, v in list(sizes.items())[:12]]})
    return {"stats": stats, "decisions": [], "sections": sections,
            "takeaway": str(u.get("interpretation") or u.get("summary") or "")[:600]}


_OUTCOME_TEXT = {
    "met": "Threshold met — moved on to Phase 7.",
    "max_iterations": "Iteration budget used up — kept the best result.",
    "early_exit": "Stopped early: another round would not help.",
    "no_improvement": "No improvement over the last round — kept the best result.",
    "user_satisfied": "You marked the result as good enough.",
    "revise": "Below the threshold — diagnosed and sent back for a fix.",
}


def _phase6(s: Any) -> dict[str, Any]:
    meta = _get(s, "preprocessing_meta") or {}
    checks = [c for c in (meta.get("phase6_checks") or []) if isinstance(c, dict)]
    rounds = [r for r in (_get(s, "all_results") or []) if isinstance(r, dict)]
    diag = _get(s, "last_failure_diagnosis") or {}
    if not checks and not rounds:
        return {"available": False}
    plan = _get(s, "execution_plan") or {}
    lower = str(plan.get("metric") or "").startswith("neg_")
    last = checks[-1] if checks else {}
    thr = _display_score(last.get("threshold", plan.get("threshold")), lower)
    best = _display_score(last.get("best"), lower)
    gap = (best - thr) if best is not None and thr is not None else None
    if lower and gap is not None:
        gap = -gap
    stats = [
        _stat("Threshold", _fmt(thr)),
        _stat("Best score", _fmt(best)),
        _stat("Gap", f"{gap:+.4f}" if gap is not None else "—", "met" if gap is not None and gap >= 0 else "short"),
        _stat("Rounds", max(len(checks), len(rounds), 1)),
    ]
    decisions = []
    for c in checks:
        decisions.append({"what": f"Round {int(c.get('iteration') or 0) + 1}: {_OUTCOME_TEXT.get(c.get('outcome'), c.get('outcome'))}",
                          "why": str(c.get("reason") or f"Best: {c.get('best_model') or '—'} at {_fmt(_display_score(c.get('best'), lower))}."),
                          "by": "rule", "entry": None})
    if diag.get("failure_type"):
        decisions.append({"what": f"Diagnosis: {_humanize(diag.get('failure_type'))}",
                          "why": str(diag.get("proposed_fix") or ""),
                          "by": "stat" if diag.get("overridden_by_evidence") else "llm", "entry": None})
    sections: list[dict[str, Any]] = []
    scores = [{"label": f"Round {i}", "value": _display_score(r.get("best_test_score"), lower),
               "model": r.get("best_model_name")} for i, r in enumerate(rounds, start=1)]
    if scores:
        sections.append({"type": "rounds", "title": "Best score per round", "threshold": thr, "items": scores})
    if diag:
        kv = [{"label": "Failure type", "value": _humanize(str(diag.get("failure_type") or "—"))},
              {"label": "Sent back to", "value": _humanize(str(diag.get("responsible_phase") or "—"))},
              {"label": "Stop here", "value": "yes" if diag.get("should_exit") else "no"}]
        sections.append({"type": "kv", "title": "Failure diagnosis", "items": kv})
        ev = diag.get("evidence") or {}
        probes = [p for p in (ev.get("probes") or []) if isinstance(p, dict)] if isinstance(ev, dict) else []
        if probes:
            sections.append({"type": "probes", "title": "Probe experiments (cross-validated)",
                             "baseline": _num(ev.get("baseline")),
                             "items": [{"label": _humanize(str(p.get("probe"))), "value": _num(p.get("gain")),
                                        "se": _num(p.get("se")), "significant": bool(p.get("significant"))}
                                       for p in probes[:8]]})
        if diag.get("overridden_by_evidence"):
            llm = diag.get("llm_diagnosis") or {}
            sections.append({"type": "list", "title": "Evidence overrode the LLM", "items": [
                f"The LLM suggested {_humanize(str(llm.get('failure_type') or '?'))} in "
                f"{_humanize(str(llm.get('responsible_phase') or '?'))}, but the probes measured a real gain elsewhere."]})
    outcome = last.get("outcome")
    takeaway = _OUTCOME_TEXT.get(outcome, "") if outcome else ""
    if gap is not None:
        takeaway = (f"Best score {_fmt(best)} vs threshold {_fmt(thr)}. " + takeaway).strip()
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


def _phase7(s: Any) -> dict[str, Any]:
    v = _get(s, "script_validation") or {}
    script = str(_get(s, "final_pipeline_script") or "")
    report = str(_get(s, "final_report_markdown") or "")
    if not v and not script and not report:
        return {"available": False}
    entries = _log_entries(s, 7)
    status = str(v.get("status") or ("generated" if script else "—"))
    attempts = v.get("attempts")
    stats = [
        _stat("Script", _humanize(status), _humanize(str(v.get("source") or ""))),
        _stat("Reproduced score", _fmt(v.get("reproduced_score")), f"reported {_fmt(v.get('reported_score'))}"
              if v.get("reported_score") is not None else ""),
        _stat("Gap", _fmt(v.get("gap")) if v.get("gap") is not None else "—", "within tolerance" if status == "verified" else ""),
        _stat("Attempts", attempts if attempts is not None else "—",
              f"{v.get('backend')} sandbox" if v.get("backend") else ""),
    ]
    decisions = []
    if v:
        decisions.append({"what": "Script " + ("verified in the sandbox" if status == "verified" else _humanize(status)),
                          "why": str(v.get("message") or "The script was re-run on the raw split and its score compared."),
                          "by": "stat" if status == "verified" else "rule",
                          "entry": entries[-1][0] if entries else None})
    if v.get("source") == "assembled":
        decisions.append({"what": "Script assembled from the recorded run",
                          "why": "Built from the exact transforms and model the run used, so it reproduces the score.",
                          "by": "rule", "entry": None})
    sections = []
    hist = [h for h in (v.get("history") or []) if isinstance(h, dict)]
    if hist:
        sections.append({"type": "steps", "title": "Validation attempts", "items": [
            {"name": f"Attempt {h.get('attempt')} — {_humanize(str(h.get('stage') or ''))}",
             "detail": (str(h.get("problem") or "")[:160]
                        or (f"reproduced {_fmt(h.get('reproduced_score'))}" if h.get("reproduced_score") is not None else "passed")),
             "ok": not h.get("problem")} for h in hist]})
    lines = script.count("\n") + 1 if script else 0
    deliver = []
    if script:
        deliver.append({"label": "Pipeline script", "value": f"{lines} lines of Python"})
    if report:
        deliver.append({"label": "Report", "value": f"{len(report.split())} words"})
    deliver.append({"label": "Train / test CSVs", "value": "the exact split the script expects"})
    sections.append({"type": "kv", "title": "What you get", "items": deliver})
    takeaway = ("The script was run in an isolated sandbox and reproduced the reported score, so it can be trusted to rebuild the model."
                if status == "verified" else
                "The script was generated; check the validation attempts before relying on it." if script else "")
    return {"stats": stats, "decisions": decisions, "sections": sections, "takeaway": takeaway}


_BUILDERS = {1: _phase1, 2: _phase2, 3: _phase3, 4: _phase4, 5: _phase5, 6: _phase6, 7: _phase7}


def build_phase_details(session: Any) -> dict[str, dict[str, Any]]:
    """Return {"1": {...}, ..., "7": {...}}. Never raises; a failing phase is marked unavailable."""
    out: dict[str, dict[str, Any]] = {}
    for phase, build in _BUILDERS.items():
        try:
            d = build(session) or {}
        except Exception:  # never break the run page over a display detail
            logger.warning("phase_details: phase %s failed", phase, exc_info=True)
            d = {"available": False, "error": True}
        d.setdefault("available", True)
        d.setdefault("stats", [])
        d.setdefault("decisions", [])
        d.setdefault("sections", [])
        d.setdefault("takeaway", "")
        d["title"] = PHASE_TITLES[phase]
        d["seconds"] = _seconds(session, phase)
        d["llm_calls"] = _llm_calls(session, phase)
        try:
            d["log"] = _log_view(session, phase)
        except Exception:
            d["log"] = []
        out[str(phase)] = d
    return out
