"""Human-readable display labels for sklearn-style metric names.

Shared by Phase 1 (problem summary) and Phase 5 (model comparison table) so
both surfaces describe the same metric the same way.
"""

from __future__ import annotations

NEG_METRIC_DISPLAY: dict[str, str] = {
    "neg_mean_absolute_error": "MAE",
    "neg_mean_squared_error": "MSE",
    "neg_root_mean_squared_error": "RMSE",
    "neg_median_absolute_error": "Median AE",
}

_DISPLAY_NAMES: dict[str, str] = {
    "r2": "R² Score",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "f1_weighted": "F1 Score (weighted)",
    "f1_macro": "F1 Score (macro)",
    "f1_micro": "F1 Score (micro)",
    "roc_auc": "ROC-AUC",
    "roc_auc_ovr_weighted": "ROC-AUC (weighted OVR)",
    "precision": "Precision",
    "precision_weighted": "Precision (weighted)",
    "recall": "Recall",
    "recall_weighted": "Recall (weighted)",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
}


def is_neg_metric(metric: str) -> bool:
    return str(metric).startswith("neg_")


def humanize_metric(metric: str) -> str:
    """Convert an internal scorer name to a human-readable display label."""
    if metric in NEG_METRIC_DISPLAY:
        return f"{NEG_METRIC_DISPLAY[metric]} (lower is better)"
    return _DISPLAY_NAMES.get(metric, metric)
