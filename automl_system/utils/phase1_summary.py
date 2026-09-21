"""Phase 1 — Task & Data Understanding: structured problem/dataset summary.

Builds a single structured object from facts already computed elsewhere in
Phase 1 (intent parsing, target resolution, dataset profiling) — no new LLM
calls. This is deliberately fact-only: it identifies and validates the ML
problem and dataset structure, and must NOT make or suggest preprocessing
decisions (scaling, encoding, imputation, transforms, feature engineering,
model selection) — those belong to Phase 2 onward.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from automl_system.state.session import SessionState
from automl_system.utils.metric_labels import humanize_metric

# Below this row count, modeling (even a simple CV split) isn't meaningful.
_MIN_ROWS_FOR_MODELING = 30

# Target-resolution confidence below this is treated as too uncertain to
# proceed silently (enforced as a hard stop in server.py before this module
# is ever called — kept here too so the summary can explain why).
TARGET_CONFIDENCE_THRESHOLD = 0.7

_SECONDARY_METRICS_REGRESSION = ["mae", "rmse"]
_SECONDARY_METRICS_CLASSIFICATION: dict[str, list[str]] = {
    "f1": ["accuracy", "precision"],
    "f1_macro": ["accuracy", "precision"],
    "f1_weighted": ["accuracy", "precision"],
    "roc_auc": ["accuracy", "f1"],
    "accuracy": ["precision", "recall"],
}


def _secondary_metrics(task_type: str, primary_metric: str) -> list[dict[str, str]]:
    if "regress" in task_type:
        codes = _SECONDARY_METRICS_REGRESSION
    else:
        codes = _SECONDARY_METRICS_CLASSIFICATION.get(primary_metric, ["accuracy"])
    return [{"code": c, "label": humanize_metric(c)} for c in codes]


def _is_boolean_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    uniq = set(non_null.unique().tolist())
    if len(uniq) > 2:
        return False
    bool_like_sets = [
        {0, 1}, {0.0, 1.0}, {True, False},
        {"true", "false"}, {"yes", "no"},
    ]
    uniq_norm = {str(v).strip().lower() if isinstance(v, str) else v for v in uniq}
    return any(uniq_norm <= s or uniq <= s for s in bool_like_sets)


def build_phase1_summary(
    session: SessionState,
    df: pd.DataFrame,
    target_confidence: float,
    target_resolution_note: str | None,
    source_filename: str = "",
) -> dict[str, Any]:
    """Assemble the structured Phase 1 state consumed by the UI and by Phase 2+."""
    task = session.task_config or {}
    profile = session.dataset_profile or {}
    profile_cols: dict[str, dict] = profile.get("columns") or {}

    task_type = str(task.get("task_type") or "unknown")
    target_col = str(task.get("target_column") or "")
    primary_metric = str(task.get("evaluation_metric") or "")

    # ── Problem ──────────────────────────────────────────────────────────────
    problem = {
        "task_type": task_type,
        "target": target_col,
        "objective": str(task.get("goal") or "").strip(),
        "primary_metric": primary_metric,
        "primary_metric_label": humanize_metric(primary_metric),
        "secondary_metrics": _secondary_metrics(task_type, primary_metric),
        "target_confidence": round(float(target_confidence), 4),
    }

    # ── Column roles + dtype buckets ────────────────────────────────────────
    numerical: list[str] = []
    categorical: list[str] = []
    datetime_cols: list[str] = []
    boolean_cols: list[str] = []
    identifiers: list[str] = []
    unsupported: list[str] = []
    column_table: list[dict[str, str]] = []

    n_rows = int(profile.get("n_rows") or len(df))

    for col in df.columns:
        meta = profile_cols.get(col) or {}
        inferred = meta.get("inferred_dtype")

        if col == target_col:
            role = "target"
        elif meta.get("fully_null"):
            role = "unsupported"
        elif meta.get("cardinality_equals_row_count") and n_rows > 1:
            role = "identifier"
        else:
            role = "feature"

        if inferred == "datetime":
            dtype_label = "datetime"
        elif inferred == "fully_null":
            dtype_label = "unsupported"
        elif inferred == "numeric":
            dtype_label = "boolean" if _is_boolean_series(df[col]) else "numerical"
        elif inferred == "categorical":
            dtype_label = "categorical"
        else:
            dtype_label = inferred or "unknown"

        column_table.append({"name": col, "dtype": dtype_label, "role": role})

        if role == "identifier":
            identifiers.append(col)
        elif role == "unsupported":
            unsupported.append(col)
        elif role == "feature":
            if dtype_label == "numerical":
                numerical.append(col)
            elif dtype_label == "categorical":
                categorical.append(col)
            elif dtype_label == "datetime":
                datetime_cols.append(col)
            elif dtype_label == "boolean":
                boolean_cols.append(col)

    feature_count = len(numerical) + len(categorical) + len(datetime_cols) + len(boolean_cols)

    # ── Dataset ──────────────────────────────────────────────────────────────
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    dataset = {
        "source_filename": source_filename,
        "rows": n_rows,
        "columns": int(profile.get("n_columns") or len(df.columns)),
        "feature_count": feature_count,
        "numerical_count": len(numerical),
        "categorical_count": len(categorical),
        "datetime_count": len(datetime_cols),
        "boolean_count": len(boolean_cols),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }

    columns = {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "identifiers": identifiers,
        "unsupported": unsupported,
        "target": target_col,
        "table": column_table,
    }

    # ── Validation (facts only — no remediation advice) ────────────────────
    target_meta = profile_cols.get(target_col) or {}
    target_missing = int(target_meta.get("null_count") or 0)
    target_valid = bool(target_col) and (
        target_meta.get("inferred_dtype") == "numeric" if "regress" in task_type
        else int(target_meta.get("nunique") or 0) >= 2
    )
    constant_features = [
        c for c in (numerical + categorical + boolean_cols)
        if int((profile_cols.get(c) or {}).get("nunique") or 0) <= 1
    ]
    ambiguous_target = target_confidence < TARGET_CONFIDENCE_THRESHOLD

    checks = [
        {"id": "target_identified", "label": f"Target identified: `{target_col}`", "passed": bool(target_col)},
        {
            "id": "target_compatible",
            "label": f"Target compatible with {task_type}",
            "passed": target_valid,
        },
        {
            "id": "usable_rows",
            "label": f"{n_rows:,} usable records detected",
            "passed": n_rows >= _MIN_ROWS_FOR_MODELING,
        },
        {
            "id": "predictor_features",
            "label": "Predictor features detected",
            "passed": feature_count >= 1,
        },
        {
            "id": "target_missing",
            "label": "Target has no missing values" if target_missing == 0 else f"Target has {target_missing} missing value(s)",
            "passed": target_missing == 0,
        },
        {
            "id": "missing_values",
            "label": "No missing values" if missing_values == 0 else f"{missing_values} missing value(s) detected",
            "passed": missing_values == 0,
        },
        {
            "id": "duplicate_rows",
            "label": "No duplicate rows" if duplicate_rows == 0 else f"{duplicate_rows} duplicate row(s) detected",
            "passed": duplicate_rows == 0,
        },
    ]

    validation = {
        "target_valid": target_valid,
        "dataset_usable": target_valid and n_rows >= _MIN_ROWS_FOR_MODELING and feature_count >= 1,
        "has_missing_values": missing_values > 0,
        "has_duplicates": duplicate_rows > 0,
        "checks": checks,
    }

    # ── Assumptions & warnings (problem/data understanding only) ───────────
    assumptions: list[dict[str, str]] = []
    if target_resolution_note:
        assumptions.append({"level": "info", "text": target_resolution_note})
    if ambiguous_target:
        assumptions.append({
            "level": "warning",
            "text": (
                f"Target confidence was {target_confidence:.0%} — below the threshold normally "
                "required to proceed automatically."
            ),
        })
    for c in identifiers:
        assumptions.append({"level": "warning", "text": f"`{c}` looks like an identifier column (unique per row)."})
    for c in constant_features:
        assumptions.append({"level": "warning", "text": f"`{c}` has only one distinct value across the dataset."})
    if unsupported:
        assumptions.append({
            "level": "warning",
            "text": f"{len(unsupported)} column(s) are entirely empty and unusable: {', '.join(unsupported)}.",
        })
    if "regress" in task_type and target_valid:
        assumptions.append({
            "level": "info",
            "text": f"Treated as regression because `{target_col}` is a continuous numeric value.",
        })
    elif "classif" in task_type and target_valid:
        n_classes = int(target_meta.get("nunique") or 0)
        assumptions.append({
            "level": "info",
            "text": f"Treated as classification because `{target_col}` has {n_classes} distinct classes.",
        })

    # ── What I understood (deterministic natural-language summary) ─────────
    objective_text = problem["objective"] or f"predict `{target_col}`"
    what_i_understood = (
        f"You want to {objective_text[0].lower() + objective_text[1:] if objective_text else objective_text}. "
        f"Problem type: {task_type.capitalize()}. Target: `{target_col}`. "
        f"Dataset: {n_rows:,} records. Predictor features: {feature_count}."
    )

    return {
        "problem": problem,
        "dataset": dataset,
        "columns": columns,
        "validation": validation,
        "assumptions": assumptions,
        "what_i_understood": what_i_understood,
    }
