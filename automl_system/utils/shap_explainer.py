"""SHAP-based feature impact summary for the Analytics page.

Computes a ranked, JSON-serializable feature-impact summary for the winning
model of Phase 5. Three explainer tiers keep the added runtime bounded:

  TreeExplainer   — RandomForest, ExtraTrees, GradientBoosting, HistGB,
                     XGBoost, LightGBM, CatBoost. Exact, near-instant.
  LinearExplainer — LogisticRegression, Ridge, Lasso, ElasticNet,
                     LinearRegression. Closed-form, instant.
  KernelExplainer — MLP, SVM, KNN, and anything else. Model-agnostic but
                     slow, so background/explain samples are capped to keep
                     it bounded (~10-30s) rather than growing with dataset size.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TREE_FAMILIES = (
    "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
    "XGBoost", "LightGBM", "CatBoost",
)
_LINEAR_FAMILIES = ("LogisticRegression", "Ridge", "Lasso", "ElasticNet", "LinearRegression")

# KernelExplainer cost ≈ explain_rows × background_rows × nsamples model predictions.
# With ensembles (each prediction runs every member) the shap defaults reached ~10M
# predictions and dominated Phase 5; these caps keep the ranking stable at ~1-3% of that.
_KERNEL_BACKGROUND_ROWS = 25
_KERNEL_EXPLAIN_ROWS = 50
_KERNEL_MAX_NSAMPLES = 400
_TREE_EXPLAIN_ROWS = 500  # generous — TreeExplainer is fast even at this size


def compute_shap_summary(
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    task_type: str,
    model_family: str,
    top_n: int = 15,
) -> dict[str, Any]:
    """Return a ranked feature-impact summary, or {"available": False, ...} on failure.

    Shape: {
      "available": bool,
      "method": "tree" | "linear" | "kernel",
      "n_samples_explained": int,
      "features": [{"name": str, "mean_abs_shap": float, "mean_signed_shap": float}, ...],
      "note": str,   # human-readable caveat (e.g. sampled explainer used)
    }
    """
    if estimator is None or x_test is None or len(x_test) == 0:
        return {"available": False, "reason": "no_fitted_estimator_or_test_data"}

    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "shap_not_installed"}

    feature_names = list(x_test.columns)

    try:
        if model_family in _TREE_FAMILIES:
            method = "tree"
            sample = x_test.iloc[: min(_TREE_EXPLAIN_ROWS, len(x_test))]
            explainer = shap.TreeExplainer(estimator)
            raw = explainer.shap_values(sample)
            note = ""
        elif model_family in _LINEAR_FAMILIES:
            method = "linear"
            sample = x_test.iloc[: min(_TREE_EXPLAIN_ROWS, len(x_test))]
            bg = shap.maskers.Independent(x_train, max_samples=min(100, len(x_train)))
            explainer = shap.LinearExplainer(estimator, bg)
            raw = explainer.shap_values(sample)
            note = ""
        else:
            method = "kernel"
            n_bg = min(_KERNEL_BACKGROUND_ROWS, len(x_train))
            n_ex = min(_KERNEL_EXPLAIN_ROWS, len(x_test))
            background = shap.kmeans(x_train, n_bg) if len(x_train) > n_bg else x_train
            sample = x_test.iloc[:n_ex]
            predict_fn = estimator.predict_proba if (task_type == "classification" and hasattr(estimator, "predict_proba")) else estimator.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            nsamples = int(min(_KERNEL_MAX_NSAMPLES, max(100, 10 * len(feature_names))))
            raw = explainer.shap_values(sample, nsamples=nsamples, silent=True)
            note = (f"Model-agnostic KernelExplainer on a {n_ex}-row sample "
                    f"({nsamples} coalitions per row; capped for runtime).")

        values = _reduce_to_2d(raw, task_type)
        if values is None or values.shape[1] != len(feature_names):
            return {"available": False, "reason": "shap_output_shape_mismatch"}

        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)

        order = np.argsort(mean_abs)[::-1][:top_n]
        features = [
            {
                "name": feature_names[i],
                "mean_abs_shap": round(float(mean_abs[i]), 6),
                "mean_signed_shap": round(float(mean_signed[i]), 6),
            }
            for i in order
        ]

        return {
            "available": True,
            "method": method,
            "n_samples_explained": int(len(sample)),
            "features": features,
            "note": note,
        }
    except Exception as exc:  # pragma: no cover — SHAP must never break the pipeline
        logger.warning("SHAP computation failed for family=%s: %s", model_family, exc)
        return {"available": False, "reason": f"error: {exc}"}


def _reduce_to_2d(raw: Any, task_type: str) -> "np.ndarray | None":
    """Normalize the various shapes shap.*_values() can return to (n_samples, n_features)."""
    if isinstance(raw, list):
        # Multi-class classification: list of (n_samples, n_features) per class.
        # Average |impact| across classes to get one ranking.
        arrs = [np.array(a) for a in raw]
        if not arrs:
            return None
        return np.mean(np.abs(np.stack(arrs, axis=0)), axis=0)
    arr = np.array(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) — newer shap TreeExplainer output for classification.
        return np.mean(np.abs(arr), axis=2)
    if arr.ndim == 2:
        return arr
    return None
