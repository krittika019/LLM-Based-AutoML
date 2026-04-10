"""Phase 1 Step 5: Task Classification Agent — target resolution + metric selection."""

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

# Confidence threshold below which user confirmation is requested
_CONFIDENCE_THRESHOLD = 0.7

# Name hints for target column inference
_TARGET_NAME_HINTS = (
    "target", "label", "outcome", "result", "churn", "survived",
    "price", "sales", "class", "y", "output", "response", "dependent",
)

# Sklearn metric name aliases
_METRIC_ALIASES: dict[str, str] = {
    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "auc_roc": "roc_auc",
    "accuracy": "accuracy",
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
}

_VALID_METRICS_BY_TYPE: dict[str, set[str]] = {
    "classification": {"f1", "f1_macro", "f1_weighted", "roc_auc", "accuracy"},
    "regression": {"neg_root_mean_squared_error", "neg_mean_absolute_error",
                   "neg_mean_squared_error", "r2"},
}


def _select_metric_from_profile(
    task_type: str,
    target_col: str,
    profile: dict,
    metric_hint: str | None,
    df: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """
    Select evaluation metric per spec logic. Returns (sklearn_metric_name, rationale).

    Binary classification: minority class < 30% → roc_auc; else f1
    Multiclass (>2 classes): f1_macro; note complexity if >10 classes
    Regression: IQR outlier check (skew >1.5 or outlier_pct >5%) → mae; else rmse
    metric_hint overrides automatic selection unless technically invalid for the task type.
    """
    task = (task_type or "").lower()

    # ── Resolve target distribution from profile or df fallback ──────────────
    target_dist = (profile.get("target_distribution") or {})
    target_col_profile = (profile.get("columns") or {}).get(target_col) or {}
    if not target_dist and df is not None and target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        target_dist = {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "nunique": int(df[target_col].nunique(dropna=True)),
        }
    nunique = (target_dist.get("nunique") or target_col_profile.get("nunique") or
               (df[target_col].nunique(dropna=True) if df is not None and target_col in (df.columns if df is not None else []) else 0))

    # ── Validate metric_hint against task type ────────────────────────────────
    hint_override: str | None = None
    hint_conflict_note: str = ""
    if metric_hint:
        normalized = _METRIC_ALIASES.get(str(metric_hint).lower().strip().replace("-", "_").replace(" ", "_"))
        if normalized:
            valid_for_type = _VALID_METRICS_BY_TYPE.get(
                "classification" if "classif" in task else "regression" if "regress" in task else "", set()
            )
            if not valid_for_type or normalized in valid_for_type:
                hint_override = normalized
            else:
                # Hint is technically wrong — note it and use automatic selection
                hint_conflict_note = (
                    f" (Note: your metric preference '{metric_hint}' is not valid for {task_type}; "
                    f"using automatic selection instead.)"
                )
                logger.info(
                    "metric_hint %r invalid for %s; using automatic selection.", metric_hint, task_type
                )

    # ── Classification ────────────────────────────────────────────────────────
    if "classif" in task:
        if hint_override:
            return hint_override, f"Using user-specified metric '{metric_hint}' for {task_type}."

        if nunique > 2:
            complexity_note = " This multiclass problem may be complex." if nunique > 10 else ""
            rationale = f"Multiclass classification ({nunique} classes): using macro F1.{complexity_note}{hint_conflict_note}"
            return "f1_macro", rationale

        # Binary: check class balance via minority-class percentage
        counts = target_dist.get("counts") or {}
        if counts:
            sorted_vals = sorted(counts.values())
            total = sum(sorted_vals)
            minority_pct = (sorted_vals[0] / total * 100) if total > 0 else 50.0
            if minority_pct < 30.0:
                rationale = (
                    f"Binary classification with imbalanced classes "
                    f"(minority {minority_pct:.1f}% < 30%): using AUC-ROC.{hint_conflict_note}"
                )
                return "roc_auc", rationale

        rationale = f"Binary classification with balanced classes: using F1.{hint_conflict_note}"
        return "f1", rationale

    # ── Regression ────────────────────────────────────────────────────────────
    if "regress" in task:
        if hint_override:
            return hint_override, f"Using user-specified metric '{metric_hint}' for {task_type}."

        skew = float(target_col_profile.get("skew") or 0.0)
        outlier_count = int(target_col_profile.get("outlier_count_iqr") or 0)
        n_rows = int(profile.get("n_rows") or 1)
        outlier_pct = outlier_count / n_rows * 100 if n_rows > 0 else 0.0

        if abs(skew) > 1.5 or outlier_pct > 5.0:
            rationale = (
                f"Regression with significant outliers "
                f"(skew={skew:.2f}, outlier_pct={outlier_pct:.1f}%): using MAE.{hint_conflict_note}"
            )
            return "neg_mean_absolute_error", rationale

        rationale = f"Regression without significant outliers: using RMSE.{hint_conflict_note}"
        return "neg_root_mean_squared_error", rationale

    # Fallback
    return "f1", f"Could not determine task type precisely; defaulting to F1.{hint_conflict_note}"


def run_task_classification(
    session: SessionState,
    df: pd.DataFrame,
) -> tuple[str | None, str, float, list[str]]:
    """
    Phase 1 Step 5: Task Classification Agent.

    Path A — target already set in task_config:
        Validates single-value check, then selects metric from profile.

    Path B — target is deferred:
        LLM receives full profile + task_type; infers target column.
        confidence >= 0.7 → set automatically, inform user.
        confidence < 0.7 → yield confirmation request, wait.
        No candidate → ask user to specify (blocking wait).

    Scenarios handled:
        - Target column not in dataset → immediate error with column list.
        - Target has only 1 unique value → pipeline stop error.

    Returns: (resolved_target | None, message_for_user, confidence_0_1, extra_msgs)
    task_config is frozen after this call — no downstream phase may modify it.
    """
    task = session.task_config or {}
    current_target = task.get("target_column")
    task_type = str(task.get("task_type") or "classification").lower()
    metric_hint = str(task.get("metric_hint") or "").strip() or None
    profile = session.dataset_profile or {}
    col_names = list(df.columns)

    # ── Scenario: user-specified column not in dataset ─────────────────────────
    if current_target and str(current_target) not in col_names:
        col_preview = ", ".join(f"`{c}`" for c in col_names[:20])
        if len(col_names) > 20:
            col_preview += f" … and {len(col_names) - 20} more"
        msg = (
            f"The column **`{current_target}`** was not found in your dataset. "
            f"Your dataset has these columns: {col_preview}. "
            "Please specify the correct target column name."
        )
        session.pending_target_confirmation = True
        session.proposed_target_column = None
        log_decision(session, step="Phase 1 - Task classification",
                     action="target_not_found", target_column=current_target)
        return None, msg, 0.0, []

    # ── Path A: target already confirmed ──────────────────────────────────────
    if current_target and str(current_target) in col_names:
        resolved = str(current_target)

        # Scenario: target has only one unique value
        target_nunique = int(df[resolved].nunique(dropna=True))
        if target_nunique <= 1:
            msg = (
                f"The target column **`{resolved}`** has only **{target_nunique}** unique value "
                "throughout the dataset. A predictive model requires at least two distinct outcomes."
            )
            log_decision(session, step="Phase 1 - Task classification",
                         action="single_value_target_error",
                         target_column=resolved, nunique=target_nunique)
            session.task_config["_pipeline_stop"] = "single_value_target"
            return None, msg, 0.0, []

        # Metric selection (only if not already done)
        if not session.task_config.get("evaluation_metric"):
            metric, rationale = _select_metric_from_profile(
                task_type, resolved, profile, metric_hint, df=df
            )
            session.task_config["evaluation_metric"] = metric
            session.task_config["metric_rationale"] = rationale
            log_decision(session, step="Phase 1 - Task classification",
                         action="metric_selected", target_column=resolved,
                         metric=metric, rationale=rationale)
            msg = f"Task: **{task_type}** | metric: **{metric}**. {rationale}"
        else:
            msg = (
                f"Target: **`{resolved}`** | Task: **{task_type}** | "
                f"Metric: **{session.task_config['evaluation_metric']}**."
            )

        # Freeze task_config
        session.task_config["_frozen"] = True
        return resolved, msg, 1.0, []

    # ── Path B: target deferred — LLM inference ───────────────────────────────
    profile_cols = profile.get("columns") or {}
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
COLUMN_PROFILES (nunique, dtype, null_pct, is_id_like, fully_null):
{col_info[:80]}

TARGET SELECTION RULES:
1. Prefer columns whose names contain: target, label, outcome, result, churn, survived, price, sales, class, y, output, response
2. For classification: prefer low-cardinality columns (nunique 2-20) with binary or multiclass pattern
3. For regression: prefer continuous numeric columns (high nunique, not ID-like)
4. NEVER pick columns where is_id_like=true (these are IDs, not labels)
5. NEVER pick fully_null columns
6. Return confidence as a float 0.0 to 1.0

Return JSON only — no markdown:
{{"target_column": "<exact column name or null>", "confidence": <float 0.0-1.0>, "task_type": "classification"|"regression", "reasoning": "<one sentence>"}}"""

    raw = LLMClient().call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed = extract_json_object(raw) if raw else None

    llm_col: str | None = None
    llm_conf: float = 0.0
    llm_task_type: str = task_type
    llm_reasoning: str = ""

    if parsed:
        llm_col = parsed.get("target_column")
        raw_conf = parsed.get("confidence")
        if isinstance(raw_conf, (int, float)):
            llm_conf = max(0.0, min(1.0, float(raw_conf)))
        elif isinstance(raw_conf, str):
            llm_conf = {"high": 0.9, "medium": 0.6, "low": 0.35}.get(raw_conf.lower(), 0.5)
        llm_task_type = str(parsed.get("task_type") or task_type)
        llm_reasoning = str(parsed.get("reasoning") or "")

    # Discard if column not in dataset
    if llm_col and str(llm_col) not in col_names:
        logger.warning("Task classifier proposed unknown column %r; discarding.", llm_col)
        llm_col = None
        llm_conf = 0.0

    # ── Scenario: no candidate found ─────────────────────────────────────────
    if not llm_col:
        session.pending_target_confirmation = True
        session.proposed_target_column = None
        col_list = ", ".join(f"`{c}`" for c in col_names[:25])
        if len(col_names) > 25:
            col_list += f" … and {len(col_names) - 25} more"
        msg = (
            "I could not identify the target column automatically. "
            "Please reply with the exact name of the column you want to predict. "
            f"Available columns: {col_list}."
        )
        return None, msg, 0.0, []

    resolved = str(llm_col)

    # ── Scenario: only one unique value ──────────────────────────────────────
    target_nunique = int(df[resolved].nunique(dropna=True))
    if target_nunique <= 1:
        msg = (
            f"The inferred target column **`{resolved}`** has only **{target_nunique}** unique value. "
            "A predictive model requires at least two distinct outcomes."
        )
        log_decision(session, step="Phase 1 - Task classification",
                     action="single_value_target_error",
                     target_column=resolved, nunique=target_nunique)
        session.task_config["_pipeline_stop"] = "single_value_target"
        return None, msg, 0.0, []

    # Update task_type if LLM refined it
    if llm_task_type and llm_task_type != task_type:
        session.task_config["task_type"] = llm_task_type
        task_type = llm_task_type

    session.task_config["target_column"] = resolved
    metric, rationale = _select_metric_from_profile(
        task_type, resolved, profile, metric_hint, df=df
    )
    session.task_config["evaluation_metric"] = metric
    session.task_config["metric_rationale"] = rationale

    log_decision(session, step="Phase 1 - Task classification", action="resolve_target_llm",
                 target_column=resolved, confidence=llm_conf, metric=metric,
                 task_type=task_type, reasoning=llm_reasoning)

    if llm_conf < _CONFIDENCE_THRESHOLD:
        session.pending_target_confirmation = True
        session.proposed_target_column = resolved
        msg = (
            f"I think you want to predict **`{resolved}`** — is that correct? "
            "Reply **yes** to confirm, or type the exact column name."
        )
        return resolved, msg, llm_conf, []

    # High confidence → set automatically and freeze
    session.task_config["_frozen"] = True
    msg = (
        f"I identified **`{resolved}`** as your target variable based on its name and distribution. "
        f"Task: **{task_type}** | Metric: **{metric}**."
    )
    return resolved, msg, llm_conf, []


# Backward-compatible alias
resolve_target_from_data = run_task_classification


def apply_target_confirmation_reply(
    user_reply: str, session: SessionState, df: pd.DataFrame
) -> tuple[bool, str]:
    """Handle yes/no or alternate column name after low-confidence proposal."""
    prop = session.proposed_target_column
    reply = (user_reply or "").strip().lower()
    task_type = str(session.task_config.get("task_type") or "classification")
    metric_hint = str(session.task_config.get("metric_hint") or "").strip() or None
    profile = session.dataset_profile or {}

    def _confirm(col: str) -> tuple[bool, str]:
        session.task_config["target_column"] = col
        session.pending_target_confirmation = False
        session.proposed_target_column = None
        # Select metric on confirmed target
        if not session.task_config.get("evaluation_metric"):
            metric, rationale = _select_metric_from_profile(
                task_type, col, profile, metric_hint, df=df
            )
            session.task_config["evaluation_metric"] = metric
            session.task_config["metric_rationale"] = rationale
        else:
            metric = session.task_config["evaluation_metric"]
        session.task_config["_frozen"] = True
        log_decision(session, step="Phase 1 - Task classification",
                     action="confirm_target", target_column=col, metric=metric)
        return True, f"Confirmed target **`{col}`**. Metric: **{metric}**."

    if reply in ("yes", "y", "ok", "confirm", "confirmed"):
        if prop and prop in df.columns:
            return _confirm(prop)

    # Check if reply is a valid column name
    reply_raw = (user_reply or "").strip()
    for c in df.columns:
        if c.lower() == reply_raw.lower() or c == reply_raw:
            return _confirm(c)

    return False, "Could not understand confirmation. Reply **yes** or type the exact column name."
