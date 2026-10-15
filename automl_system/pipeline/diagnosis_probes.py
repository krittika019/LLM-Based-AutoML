"""Phase 6: experiment-based failure diagnosis.

Instead of asking the LLM to guess why a run missed its threshold, run a small set
of controlled experiments on the TRAIN rows (fold-paired cross-validation with a
fast LightGBM proxy) and hand the measured results to the diagnosis step. Each probe
changes exactly one thing relative to the current pipeline output:

  feature_selection   add back the columns feature selection dropped
  feature_engineering remove the engineered features
  preprocessing       simple encoding of the original columns instead of deep cleaning
  class_weight        balanced class weights, scored on the run's own metric
  data_quantity       50% vs 100% of the rows (is the model still data-limited?)

The test set is never used. Probes are our own code, so they run in-process.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from automl_system import config
from automl_system.pipeline.feature_engineering import _proxy_model, _significant_gain
from automl_system.pipeline.replay import safe_numeric_features

logger = logging.getLogger(__name__)

_MAX_ROWS = 5_000


def _encode_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Minimal encoding of original columns: numeric as-is, categories → integer codes."""
    out = pd.DataFrame(index=frame.index)
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.factorize(s.astype(str), sort=True)[0].astype(float)
    return out


class _Experiment:
    """Fold-paired proxy CV on a fixed row subsample, scored with the run's metric."""

    def __init__(self, y: pd.Series, task_type: str, metric: str):
        self.task_type = "classification" if task_type == "classification" else "regression"
        n = len(y)
        self.idx = (np.sort(np.random.default_rng(0).choice(n, size=_MAX_ROWS, replace=False))
                    if n > _MAX_ROWS else np.arange(n))
        yv = y.iloc[self.idx]
        if self.task_type == "classification":
            self.y = pd.factorize(yv.astype(str), sort=True)[0]
            min_class = int(np.bincount(self.y).min())
        else:
            self.y = pd.to_numeric(yv, errors="coerce").to_numpy(dtype=float)
            min_class = 0
        repeats = 3 if len(self.idx) < 2_000 else 2
        if self.task_type == "classification" and min_class >= 3:
            splitter = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeats, random_state=1)
            self.splits = list(splitter.split(np.zeros(len(self.idx)), self.y))
        else:
            splitter = RepeatedKFold(n_splits=3, n_repeats=repeats, random_state=1)
            self.splits = list(splitter.split(np.zeros(len(self.idx))))
        try:
            self.scorer = get_scorer(metric)
            self.metric = metric
        except Exception:
            self.metric = "roc_auc_ovr" if self.task_type == "classification" else "r2"
            self.scorer = get_scorer(self.metric)

    def scores(self, x: pd.DataFrame, *, rows_frac: float = 1.0, balanced: bool = False) -> np.ndarray:
        xm = x.iloc[self.idx].to_numpy(dtype=float)
        rng = np.random.default_rng(2)
        out = []
        for tr, va in self.splits:
            if rows_frac < 1.0:
                tr = np.sort(rng.choice(tr, size=max(20, int(len(tr) * rows_frac)), replace=False))
            model = _proxy_model(self.task_type)
            if balanced:
                model.set_params(class_weight="balanced")
            model.fit(xm[tr], self.y[tr])
            try:
                out.append(float(self.scorer(model, xm[va], self.y[va])))
            except ValueError:  # e.g. a class absent from this fold — identical for every variant
                out.append(float(np.mean(model.predict(xm[va]) == self.y[va])))
        return np.asarray(out)


def _result(name: str, diff: np.ndarray, *, points_to: tuple[str, str] | None, finding: str,
            fix: str) -> dict[str, Any]:
    gain = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    significant = _significant_gain(diff, config.DIAG_MIN_GAIN)
    return {
        "probe": name, "gain": round(gain, 5), "se": round(se, 5), "significant": significant,
        "failure_type": points_to[0] if points_to else None,
        "responsible_phase": points_to[1] if points_to else None,
        "finding": finding, "fix": fix if significant else "",
    }


def run_probes(
    df_train: pd.DataFrame,
    raw_split_train: pd.DataFrame | None,
    session: Any,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Run all applicable probes. `df_train` is the pipeline's processed training frame
    (with target); `raw_split_train` the same rows before preprocessing."""
    t0 = time.perf_counter()
    task = session.task_config or {}
    target = str(task.get("target_column") or "")
    task_type = str(results.get("task_type") or task.get("task_type") or "classification")
    metric = str(results.get("metric") or "")
    if not target or target not in df_train.columns:
        return {"probes": [], "error": "target not available"}

    y = df_train[target].reset_index(drop=True)
    x_cur = safe_numeric_features(df_train.drop(columns=[target])).reset_index(drop=True)
    exp = _Experiment(y, task_type, metric)
    base = exp.scores(x_cur)
    probes: list[dict[str, Any]] = []

    keep = list((session.preprocessing_meta or {}).get("feature_selection_columns") or [])
    raw = raw_split_train.reset_index(drop=True) if raw_split_train is not None else None
    if raw is not None and len(raw) != len(x_cur):
        raw = None

    # 1. Feature selection: add back the dropped columns
    if raw is not None and keep:
        dropped = [c for c in raw.columns if c not in keep and c != target]
        if dropped:
            x = pd.concat([x_cur, _encode_raw(raw[dropped]).add_prefix("__readd__")], axis=1)
            probes.append(_result(
                "feature_selection", exp.scores(x) - base,
                points_to=("feature_issue", "feature_selection"),
                finding=f"Re-adding the {len(dropped)} column(s) dropped by feature selection {dropped[:8]}",
                fix=f"Keep these columns during feature selection: {dropped[:8]}.",
            ))

    # 2. Engineered features: remove them
    engineered = [e.get("feature") for e in (session.feature_engineering_log or [])
                  if e.get("status") == "added" and e.get("feature") in x_cur.columns]
    if engineered:
        x = x_cur.drop(columns=engineered)
        probes.append(_result(
            "feature_engineering", exp.scores(x) - base,
            points_to=("feature_issue", "feature_engineering"),
            finding=f"Removing the {len(engineered)} engineered feature(s) {engineered[:8]}",
            fix=f"Drop the engineered features {engineered[:8]}; they do not generalize.",
        ))

    # 3. Preprocessing: minimal encoding of the original columns instead of deep cleaning
    if raw is not None:
        cols = [c for c in (keep or list(raw.columns)) if c in raw.columns and c != target]
        if cols:
            probes.append(_result(
                "preprocessing", exp.scores(_encode_raw(raw[cols])) - base,
                points_to=("data_quality_issue", "deep_cleaning"),
                finding="Using the original columns with simple encoding instead of the deep-cleaned ones",
                fix="Deep cleaning is hurting: use lighter imputation/outlier handling and avoid "
                    "aggressive transforms (e.g. log or IQR capping) on informative columns.",
            ))

    # 4. Class imbalance: balanced class weights, scored on the run's own metric
    if exp.task_type == "classification":
        counts = np.bincount(exp.y)
        if counts.min() / counts.sum() < 0.35:
            probes.append(_result(
                "class_weight", exp.scores(x_cur, balanced=True) - base,
                points_to=("model_issue", "phase4_planning"),
                finding=f"Training with class_weight='balanced' (minority class {counts.min() / counts.sum():.0%})",
                fix="Use class_weight='balanced' (or scale_pos_weight) for all supporting models.",
            ))

    # 5. Data quantity: is the model still improving with more rows? (evidence only)
    half = exp.scores(x_cur, rows_frac=0.5)
    probes.append(_result(
        "data_quantity", base - half, points_to=None,
        finding="Going from 50% to 100% of the training rows",
        fix="",
    ))

    actionable = [p for p in probes if p["significant"] and p["failure_type"]]
    best = max(actionable, key=lambda p: p["gain"]) if actionable else None
    return {
        "metric": exp.metric,
        "baseline": round(float(base.mean()), 5),
        "probes": probes,
        "best": best,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def format_evidence(report: dict[str, Any]) -> str:
    """Compact text for the diagnosis prompt and the user-facing message."""
    if not report.get("probes"):
        return "No experiments could be run."
    lines = [f"Baseline proxy CV ({report['metric']}) on training rows: {report['baseline']}"]
    for p in report["probes"]:
        verdict = "SIGNIFICANT" if p["significant"] else "no reliable effect"
        lines.append(f"- {p['finding']}: change {p['gain']:+.4f} (±{p['se']:.4f}) → {verdict}")
    return "\n".join(lines)
