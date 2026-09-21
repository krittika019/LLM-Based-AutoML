"""Phase 1: Task Classification — target resolution + metric selection."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_TARGET_NAME_HINTS = (
    "target", "label", "outcome", "result", "churn", "survived",
    "price", "sales", "class", "y", "output", "response", "dependent",
)

_METRIC_ALIASES: dict[str, str] = {
    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "auc_roc": "roc_auc",
    "accuracy": "accuracy",
    "rmse": "r2",
    "mse": "r2",
    "mae": "r2",
    "r2": "r2",
    "neg_root_mean_squared_error": "r2",
    "neg_mean_absolute_error": "r2",
    "neg_mean_squared_error": "r2",
}

_VALID_METRICS_BY_TYPE: dict[str, set[str]] = {
    "classification": {"f1", "f1_macro", "f1_weighted", "roc_auc", "accuracy"},
    "regression": {"r2"},
}


def _select_metric_from_profile(
    task_type: str,
    target_col: str,
    profile: dict,
    metric_hint: str | None,
    df: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """Select evaluation metric. Returns (sklearn_metric_name, rationale)."""
    task = task_type.lower()
    target_col_profile = (profile.get("columns") or {}).get(target_col) or {}
    target_dist = profile.get("target_distribution") or {}
    nunique = int(target_col_profile.get("nunique") or 0)

    # Honour user hint if valid for task type
    hint_override: str | None = None
    hint_conflict_note = ""
    if metric_hint:
        normalised = _METRIC_ALIASES.get(metric_hint.lower().replace("-", "_"), metric_hint)
        valid = _VALID_METRICS_BY_TYPE.get(task, set())
        if normalised in valid:
            hint_override = normalised
        else:
            hint_conflict_note = (
                f" (Note: your metric preference '{metric_hint}' is not valid for {task_type}; "
                f"using automatic selection instead.)"
            )

    # Classification
    if "classif" in task:
        if hint_override:
            return hint_override, f"Using user-specified metric '{metric_hint}'."

        if nunique > 2:
            note = " Multiclass problem may be complex." if nunique > 10 else ""
            return "f1_macro", f"Multiclass ({nunique} classes): F1-macro.{note}{hint_conflict_note}"

        counts = target_dist.get("counts") or {}
        if counts and isinstance(counts, dict) and len(counts) >= 2:
            sorted_vals = sorted(counts.values())
            total = sum(sorted_vals)
            minority_pct = (sorted_vals[0] / total * 100) if total > 0 else 50.0
            if minority_pct < 30.0:
                return "roc_auc", (
                    f"Binary classification, imbalanced (minority {minority_pct:.1f}%): "
                    f"ROC-AUC.{hint_conflict_note}"
                )

        return "f1", f"Binary classification, balanced classes: F1.{hint_conflict_note}"

    # Regression — always r2 (scale-invariant, threshold of 0.60 is universally meaningful)
    if "regress" in task:
        if hint_override:
            return hint_override, f"Using user-specified metric '{metric_hint}'."
        return "r2", (
            f"Regression: R² (scale-invariant; threshold 0.60 = model explains ≥60% of "
            f"target variance).{hint_conflict_note}"
        )

    return "f1", f"Could not determine task type; defaulting to F1.{hint_conflict_note}"


def run_task_classification(
    session: SessionState,
    df: pd.DataFrame,
) -> tuple[str | None, str, float, list[str]]:
    """
    Phase 1: Validate target column and select metric.

    Path A — target already in task_config: validate + select metric.
    Path B — target deferred: LLM infers target from column profiles.

    Returns (resolved_target | None, message, confidence, [])
    """
    task = session.task_config or {}
    current_target = task.get("target_column")
    task_type = str(task.get("task_type") or "classification").lower()
    metric_hint = str(task.get("metric_hint") or "").strip() or None
    profile = session.dataset_profile or {}
    col_names = list(df.columns)
    col_lower_map = {c.lower(): c for c in col_names}

    # User-specified column not in dataset
    if current_target and str(current_target) not in col_names:
        # Try case-insensitive match before failing
        matched = col_lower_map.get(str(current_target).lower())
        if matched:
            session.task_config["target_column"] = matched
            current_target = matched
        else:
            col_preview = ", ".join(f"`{c}`" for c in col_names[:20])
            msg = (
                f"Column **`{current_target}`** not found. "
                f"Available columns: {col_preview}."
            )
            log_decision(session, step="Phase 1", action="target_not_found",
                         target_column=current_target)
            return None, msg, 0.0, []

    # Path A: target confirmed
    if current_target and str(current_target) in col_names:
        resolved = str(current_target)

        target_nunique = int(df[resolved].nunique(dropna=True))
        if target_nunique <= 1:
            msg = (
                f"Target **`{resolved}`** has only {target_nunique} unique value — "
                "a model needs at least two distinct outcomes."
            )
            session.task_config["_pipeline_stop"] = "single_value_target"
            return None, msg, 0.0, []

        metric, rationale = _select_metric_from_profile(
            task_type, resolved, profile, metric_hint, df=df
        )
        session.task_config["evaluation_metric"] = metric
        session.task_config["metric_rationale"] = rationale
        log_decision(session, step="Phase 1", action="metric_selected",
                     target_column=resolved, metric=metric)
        return resolved, f"Task: **{task_type}** | metric: **{metric}**. {rationale}", 1.0, []

    # Path B: deferred — try name hints first before spending an LLM call
    profile_cols = profile.get("columns") or {}

    hint_candidates = [
        col_lower_map[hint]
        for hint in _TARGET_NAME_HINTS
        if hint in col_lower_map
    ]
    # Filter out ID-like or fully-null columns
    hint_candidates = [
        c for c in hint_candidates
        if not (profile_cols.get(c) or {}).get("cardinality_equals_row_count")
        and not (profile_cols.get(c) or {}).get("fully_null")
    ]
    if len(hint_candidates) == 1:
        resolved = hint_candidates[0]
        target_nunique = int(df[resolved].nunique(dropna=True))
        if target_nunique > 1:
            session.task_config["target_column"] = resolved
            metric, rationale = _select_metric_from_profile(
                task_type, resolved, profile, metric_hint, df=df
            )
            session.task_config["evaluation_metric"] = metric
            session.task_config["metric_rationale"] = rationale
            log_decision(session, step="Phase 1", action="resolve_target_hint",
                         target_column=resolved, metric=metric)
            return (
                resolved,
                f"Identified **`{resolved}`** as target via name heuristic. "
                f"Task: **{task_type}** | Metric: **{metric}**.",
                0.95,
                [],
            )

    col_info = [
        {
            "name": c,
            "nunique": m.get("nunique"),
            "inferred_dtype": m.get("inferred_dtype"),
            "null_pct": m.get("null_pct"),
            "is_id_like": m.get("cardinality_equals_row_count"),
            "fully_null": m.get("fully_null"),
        }
        for c, m in profile_cols.items()
    ]

    prompt = f"""You are an ML expert. Identify the most likely TARGET column for supervised learning.

GOAL: {task.get('goal', '(not specified)')}
TASK_TYPE: {task_type}
DATASET_COLUMNS: {col_names}
COLUMN_PROFILES: {col_info[:80]}

RULES:
1. Prefer columns named: target, label, outcome, result, churn, survived, price, sales, class, y
2. Classification: prefer low-cardinality columns (nunique 2-20)
3. Regression: prefer continuous numeric columns (high nunique, not ID-like)
4. NEVER pick is_id_like=true or fully_null columns

Return JSON only:
{{"target_column": "<exact column name or null>", "confidence": <float 0.0-1.0>, "task_type": "classification"|"regression", "reasoning": "<one sentence>"}}"""

    raw = LLMClient().call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed = extract_json_object(raw) if raw else None

    llm_col: str | None = None
    llm_conf: float = 0.0
    llm_task_type: str = task_type

    if parsed:
        llm_col = parsed.get("target_column")
        raw_conf = parsed.get("confidence")
        if isinstance(raw_conf, (int, float)):
            llm_conf = max(0.0, min(1.0, float(raw_conf)))
        llm_task_type = str(parsed.get("task_type") or task_type).strip().lower()
        if llm_task_type not in ("classification", "regression"):
            llm_task_type = task_type

    # Case-insensitive match
    if llm_col:
        matched = col_lower_map.get(str(llm_col).lower())
        llm_col = matched  # None if not found

    if not llm_col:
        col_list = ", ".join(f"`{c}`" for c in col_names[:25])
        msg = (
            "Could not identify the target column automatically. "
            f"Please include the target column name in your prompt. Columns: {col_list}."
        )
        return None, msg, 0.0, []

    resolved = str(llm_col)
    target_nunique = int(df[resolved].nunique(dropna=True))
    if target_nunique <= 1:
        session.task_config["_pipeline_stop"] = "single_value_target"
        return None, f"Inferred target **`{resolved}`** has only {target_nunique} unique value.", 0.0, []

    if llm_task_type != task_type:
        session.task_config["task_type"] = llm_task_type
        task_type = llm_task_type

    session.task_config["target_column"] = resolved
    metric, rationale = _select_metric_from_profile(
        task_type, resolved, profile, metric_hint, df=df
    )
    session.task_config["evaluation_metric"] = metric
    session.task_config["metric_rationale"] = rationale

    log_decision(session, step="Phase 1", action="resolve_target_llm",
                 target_column=resolved, confidence=llm_conf, metric=metric, task_type=task_type)

    msg = (
        f"Identified **`{resolved}`** as target (confidence {llm_conf:.0%}). "
        f"Task: **{task_type}** | Metric: **{metric}**."
    )
    return resolved, msg, llm_conf, []


# Alias used by server.py
resolve_target_from_data = run_task_classification
