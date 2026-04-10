"""Phase 5: model execution, comparison, and LLM interpretation."""

from __future__ import annotations

import concurrent.futures
import logging
import time
import traceback
import importlib
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from automl_system import config
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import get_scorer_names
except ImportError:  # pragma: no cover
    get_scorer_names = None  # type: ignore[misc, assignment]
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR

from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision

logger = logging.getLogger(__name__)


class MissingTargetError(RuntimeError):
    """Supervised training requires session.task_config['target_column'] present in the dataframe."""

    pass

def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover - optional dependency
        return None


mlflow = _optional_import("mlflow")
optuna = _optional_import("optuna")
_xgb = _optional_import("xgboost")
_lgbm = _optional_import("lightgbm")
XGBClassifier = getattr(_xgb, "XGBClassifier", None)
XGBRegressor = getattr(_xgb, "XGBRegressor", None)
LGBMClassifier = getattr(_lgbm, "LGBMClassifier", None)
LGBMRegressor = getattr(_lgbm, "LGBMRegressor", None)

_MLFLOW_READY = False
_TRAINING_TIMEOUT_SECONDS = 300  # 5-minute hard ceiling per model


def _ensure_mlflow_local_store() -> None:
    global _MLFLOW_READY
    if _MLFLOW_READY or mlflow is None:
        return
    try:
        root = Path(__file__).resolve().parents[2]
        mlruns_dir = root / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())
        mlflow.set_experiment("automl_local")
        _MLFLOW_READY = True
    except Exception as e:
        logger.warning("MLflow local store initialization failed: %s", e)


def _infer_task_type(df: pd.DataFrame, session: SessionState) -> str:
    planned = str((session.execution_plan or {}).get("task_type") or "").strip().lower()
    if planned in {"classification", "regression"}:
        return planned
    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        return "classification"
    s = df[target]
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    if int(s.nunique(dropna=True)) <= 20:
        return "classification"
    return "regression"


def _safe_numeric_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.factorize(out[c].astype(str), sort=True)[0]
    return out


def _split_data(df: pd.DataFrame, session: SessionState) -> dict[str, Any]:
    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        raise MissingTargetError(
            "run_pipeline requires a valid target_column in session.task_config that exists in the dataframe. "
            "Resolve the target in Phase 1 (task_classifier) before execution."
        )

    task_type = _infer_task_type(df, session)
    x = df.drop(columns=[target])
    y = df[target]
    x = _safe_numeric_features(x)

    stratify_used = False
    stratify_arg = None
    if task_type == "classification":
        vc = y.value_counts(dropna=False)
        if len(vc) > 1 and int(vc.min()) >= 2:
            stratify_arg = y
            stratify_used = True

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)

    split_meta = {
        "test_size": 0.2,
        "random_state": 42,
        "split_type": "stratified" if stratify_used else "standard",
        "task_type": task_type,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    log_decision(
        session,
        step="Phase 5 - Execution",
        action="split_data",
        rationale=f"{split_meta['split_type']} split; train={split_meta['n_train']} test={split_meta['n_test']}",
        split_meta=split_meta,
    )

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_meta": split_meta,
        "task_type": task_type,
        "label_encoder": label_encoder,
    }


def _build_estimator(family: str, task_type: str):
    if family == "LogisticRegression":
        return LogisticRegression(max_iter=2000)
    if family == "RandomForest":
        return RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
    if family == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42) if task_type == "classification" else HistGradientBoostingRegressor(random_state=42)
    if family == "XGBoost":
        if task_type == "classification":
            if XGBClassifier is None:
                return None
            return XGBClassifier(random_state=42, eval_metric="logloss")
        if XGBRegressor is None:
            return None
        return XGBRegressor(random_state=42)
    if family == "LightGBM":
        if task_type == "classification":
            if LGBMClassifier is None:
                return None
            return LGBMClassifier(random_state=42)
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(random_state=42)
    if family == "LinearRegression":
        return LinearRegression()
    if family == "Ridge":
        return Ridge(random_state=42)
    if family == "SVR":
        return SVR()
    return None


def _param_space(family: str, strategy_name: str, n_rows: int = 5000) -> dict[str, list[Any]]:
    """Return hyperparameter grid scaled to dataset size.

    tiny  (<1 500 rows): 2-4 combinations per model → ~8-12 fits × 3-fold CV
    medium(<10 000 rows): moderate grid
    large (≥10 000 rows): full grid
    """
    tiny = n_rows < 1500
    medium = n_rows < 10000

    if family == "LogisticRegression":
        return {"C": [0.1, 1.0] if tiny else [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]}

    if family == "RandomForest":
        if tiny:
            return {"n_estimators": [100, 200], "max_depth": [None, 8]}
        if medium:
            return {"n_estimators": [100, 200], "max_depth": [None, 8, 15], "min_samples_split": [2, 5]}
        return {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}

    if family == "HistGradientBoosting":
        if tiny:
            return {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}
        if medium:
            return {"learning_rate": [0.03, 0.1], "max_depth": [None, 8], "max_leaf_nodes": [15, 31]}
        return {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [None, 5, 10], "max_leaf_nodes": [15, 31, 63]}

    if family == "XGBoost":
        if tiny:
            return {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]}
        if medium:
            return {
                "n_estimators": [150, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [4, 6, 8],
            }
        return {
            "n_estimators": [150, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        }

    if family == "LightGBM":
        if tiny:
            return {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]}
        if medium:
            return {"n_estimators": [150, 300], "learning_rate": [0.03, 0.1], "num_leaves": [31, 63]}
        return {
            "n_estimators": [150, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "subsample": [0.7, 0.9, 1.0],
        }

    if family == "LinearRegression":
        return {}

    if family == "Ridge":
        return {"alpha": [0.1, 1.0, 10.0] if tiny else [0.01, 0.1, 1.0, 10.0, 100.0]}

    if family == "SVR":
        if tiny:
            return {"C": [0.1, 1.0, 10.0], "kernel": ["rbf"]}
        return {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.3], "kernel": ["rbf", "linear"]}

    return {}


def _valid_sklearn_scorer_names() -> frozenset[str] | None:
    if get_scorer_names is None:
        return None
    try:
        return frozenset(get_scorer_names())
    except Exception:  # pragma: no cover
        return None


def _normalize_sklearn_scorer(metric: str, task_type: str, y_train: pd.Series) -> str:
    """
    Map LLM / user-friendly metric names to strings accepted by GridSearchCV scoring=.
    e.g. f1_score, f1-score -> f1 (binary) or f1_weighted (multiclass).
    """
    raw = str(metric or "").strip()
    if not raw:
        raw = "accuracy" if task_type == "classification" else "r2"
    key = raw.lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")

    if task_type == "regression":
        reg_aliases = {
            "mae": "neg_mean_absolute_error",
            "mean_absolute_error": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "mean_squared_error": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "root_mean_squared_error": "neg_root_mean_squared_error",
            "r_squared": "r2",
            "rsquared": "r2",
        }
        out = reg_aliases.get(key, raw)
    else:
        n_classes = int(pd.Series(y_train).nunique(dropna=True))
        cls_aliases = {
            "f1_score": None,
            "f1score": None,
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "f1_weighted": "f1_weighted",
            "precision_score": None,
            "precisionscore": None,
            "recall_score": None,
            "recallscore": None,
            "auc": None,
            "roc_auc_score": None,
            "rocauc": None,
            "accuracy_score": "accuracy",
            "acc": "accuracy",
        }
        if key in ("f1_score", "f1score"):
            out = "f1" if n_classes <= 2 else "f1_weighted"
        elif key == "f1" and n_classes > 2:
            out = "f1_weighted"
        elif key in ("precision_score", "precisionscore"):
            out = "precision" if n_classes <= 2 else "precision_weighted"
        elif key in ("recall_score", "recallscore"):
            out = "recall" if n_classes <= 2 else "recall_weighted"
        elif key in ("auc", "roc_auc_score", "rocauc"):
            out = "roc_auc" if n_classes <= 2 else "roc_auc_ovr_weighted"
        elif key in cls_aliases and cls_aliases[key] is not None:
            out = cls_aliases[key]
        elif key in cls_aliases:
            out = "accuracy"
        else:
            out = raw

    valid = _valid_sklearn_scorer_names()
    if valid is not None and out not in valid:
        logger.warning(
            "Unknown or invalid sklearn scorer %r (task=%s); falling back to %s",
            out,
            task_type,
            "accuracy" if task_type == "classification" else "r2",
        )
        out = "accuracy" if task_type == "classification" else "r2"
    return out


def _cv_from_plan(session: SessionState, y_train: pd.Series):
    plan = session.execution_plan or {}
    cv_design = plan.get("cv_design") or {}
    folds = int(cv_design.get("folds") or 5)
    # Cap folds for small datasets to avoid slow cross-validation
    n_rows = int((session.dataset_profile or {}).get("n_rows") or len(y_train))
    if n_rows < 2000:
        folds = min(folds, 3)
    task_type = str((plan.get("task_type") or "")).lower()
    if not task_type:
        task_type = "classification" if int(pd.Series(y_train).nunique()) <= 20 else "regression"
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def _compute_full_metrics(
    task_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, Any]:
    """Compute the full evaluation metric suite appropriate for the task type."""
    out: dict[str, Any] = {}
    if task_type == "classification":
        n_cls = int(pd.Series(y_true).nunique(dropna=True))
        avg = "binary" if n_cls <= 2 else "weighted"
        out["accuracy"]  = round(float(accuracy_score(y_true, y_pred)), 6)
        out["precision"] = round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 6)
        out["recall"]    = round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 6)
        out["f1"]        = round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 6)
        try:
            out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        except Exception:
            pass
        if y_proba is not None:
            try:
                if n_cls <= 2:
                    p1 = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] >= 2) else y_proba.ravel()
                    out["roc_auc"]     = round(float(roc_auc_score(y_true, p1)), 6)
                    out["pr_auc"]      = round(float(average_precision_score(y_true, p1)), 6)
                    out["log_loss"]    = round(float(log_loss(y_true, y_proba)), 6)
                    out["brier_score"] = round(float(brier_score_loss(y_true, p1)), 6)
                else:
                    out["roc_auc"]  = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 6)
                    out["log_loss"] = round(float(log_loss(y_true, y_proba)), 6)
            except Exception as exc:
                logger.warning("Probabilistic metrics computation failed: %s", exc)
    else:  # regression
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        out["mae"]       = round(float(mean_absolute_error(y_t, y_p)), 6)
        out["mse"]       = round(float(mean_squared_error(y_t, y_p)), 6)
        out["rmse"]      = round(float(np.sqrt(mean_squared_error(y_t, y_p))), 6)
        out["r2"]        = round(float(r2_score(y_t, y_p)), 6)
        out["median_ae"] = round(float(median_absolute_error(y_t, y_p)), 6)
        try:
            nonzero = y_t != 0
            if nonzero.sum() > 0:
                out["mape_pct"] = round(
                    float(np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero])) * 100), 4
                )
        except Exception:
            pass
    return out


def _test_score(metric: str, task_type: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> float:
    metric = str(metric or "").strip()
    if task_type == "classification":
        n_classes = int(pd.Series(y_true).nunique(dropna=True))
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "f1":
            avg = "binary" if n_classes <= 2 else "weighted"
            return float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        if metric in ("f1_macro", "f1_micro", "f1_weighted", "f1_samples"):
            avg = metric.split("_", 1)[1]
            return float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        if metric in ("precision_macro", "precision_micro", "precision_weighted", "precision_samples"):
            avg = metric.split("_", 1)[1]
            return float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        if metric == "precision":
            avg = "binary" if n_classes <= 2 else "weighted"
            return float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        if metric in ("recall_macro", "recall_micro", "recall_weighted", "recall_samples"):
            avg = metric.split("_", 1)[1]
            return float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        if metric == "recall":
            avg = "binary" if n_classes <= 2 else "weighted"
            return float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        if metric == "roc_auc":
            if y_proba is None:
                return float(accuracy_score(y_true, y_pred))
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                return float(roc_auc_score(y_true, y_proba))
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        if metric in ("roc_auc_ovr_weighted", "roc_auc_ovo_weighted", "roc_auc_ovr", "roc_auc_ovo"):
            if y_proba is None:
                return float(accuracy_score(y_true, y_pred))
            parts = metric.split("_")
            multi = "ovr" if "ovr" in parts else "ovo"
            average = "weighted" if metric.endswith("_weighted") else "macro"
            return float(roc_auc_score(y_true, y_proba, multi_class=multi, average=average))
        return float(accuracy_score(y_true, y_pred))

    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "neg_mean_absolute_error":
        return -float(mean_absolute_error(y_true, y_pred))
    if metric == "neg_mean_squared_error":
        return -float(mean_squared_error(y_true, y_pred))
    if metric == "neg_root_mean_squared_error":
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(r2_score(y_true, y_pred))


def _mlflow_ctx(family: str):
    if mlflow is None:
        return nullcontext()
    _ensure_mlflow_local_store()
    return mlflow.start_run(run_name=family, nested=True)


def _tune_with_optuna(
    family: str,
    base_estimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    metric: str,
    n_trials: int,
    task_type: str,
    timeout_seconds: int = _TRAINING_TIMEOUT_SECONDS,
) -> tuple[Any, dict[str, Any], float, float]:
    if optuna is None:
        raise RuntimeError("Optuna search requested but optuna is not installed")

    def objective(trial):
        est = clone(base_estimator)
        params = {}

        if family in {"RandomForest", "XGBoost", "LightGBM"}:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=100)
        if family in {"RandomForest", "HistGradientBoosting", "XGBoost"}:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        if family in {"HistGradientBoosting", "XGBoost", "LightGBM"}:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        if family == "Ridge":
            params["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        if family == "SVR":
            params["C"] = trial.suggest_float("C", 0.1, 20.0, log=True)
            params["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.5)
        if family == "LogisticRegression":
            params["C"] = trial.suggest_float("C", 0.01, 20.0, log=True)

        if params:
            est.set_params(**params)

        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(est, x_train, y_train, scoring=metric, cv=cv, n_jobs=-1)
        trial.set_user_attr("fold_variance", float(np.var(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    tuned = clone(base_estimator)
    if study.best_params:
        tuned.set_params(**study.best_params)
    tuned.fit(x_train, y_train)
    best_var = float(study.best_trial.user_attrs.get("fold_variance", 0.0))
    return tuned, study.best_params, float(study.best_value), best_var


# ---------------------------------------------------------------------------
# Progressive Model Evaluation (successive-halving inspired)
# ---------------------------------------------------------------------------

def _subsample_score(
    family: str,
    task_type: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    metric: str,
    subsample_frac: float = 0.20,
) -> float:
    """Train a default estimator on a random subsample and return its mean CV score.

    Uses no hyperparameter search — the goal is a fast, relative ranking.
    """
    from sklearn.model_selection import cross_val_score

    est = _build_estimator(family, task_type)
    if est is None:
        return float("-inf")

    n_sub = max(int(len(x_train) * subsample_frac), 50)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_train), size=min(n_sub, len(x_train)), replace=False)
    x_sub = x_train.iloc[idx]
    y_sub = y_train.iloc[idx]

    try:
        scores = cross_val_score(est, x_sub, y_sub, scoring=metric, cv=cv, n_jobs=-1)
        return float(np.mean(scores))
    except Exception as exc:
        logger.warning("Subsample score failed for %s: %s", family, exc)
        return float("-inf")


def _llm_eliminate_model(
    subsample_results: list[dict[str, Any]],
    task_type: str,
    metric: str,
    profile_summary: dict[str, Any],
) -> str:
    """Ask the LLM which candidate to eliminate after the subsample round.

    The LLM considers both the subsample score and each model's theoretical fit
    for the dataset characteristics — not just raw performance.

    Returns the family name of the model to eliminate.
    """
    client = LLMClient()
    candidates_text = "\n".join(
        f"- {r['family']}: subsample_score={r['subsample_score']:.4f}, rationale={r.get('rationale', 'N/A')}"
        for r in subsample_results
    )
    prompt = f"""You are an AutoML system deciding which model to eliminate after a quick subsample screening round.

Task type: {task_type}
Evaluation metric: {metric}
Dataset rows: {profile_summary.get('n_rows', 'unknown')}, columns: {profile_summary.get('n_columns', 'unknown')}
Dataset characteristics: numeric_ratio={profile_summary.get('numeric_ratio', 'unknown')}, high_cardinality_categoricals={profile_summary.get('high_cardinality_categoricals', False)}

Subsample screening results (20% of training data, no hyperparameter tuning):
{candidates_text}

Decide which ONE model to eliminate. Consider:
1. Subsample score (lower is worse but not the only factor)
2. Theoretical fit: does the model suit this task type, dataset size, and feature types?
3. Redundancy: if two models are similar (e.g., both linear), eliminate the weaker one

Respond with ONLY the model family name to eliminate. No explanation. Choose from: {[r['family'] for r in subsample_results]}"""

    response = client.call(prompt, max_tokens=32, temperature=0.0).strip()
    # Parse: find the first family name that appears in the response
    for r in sorted(subsample_results, key=lambda x: x["subsample_score"]):
        if r["family"].lower() in response.lower():
            return r["family"]
    # Fallback: eliminate the worst subsample scorer
    worst = min(subsample_results, key=lambda x: x["subsample_score"])
    logger.warning(
        "LLM elimination response %r not parseable; falling back to worst scorer %s",
        response,
        worst["family"],
    )
    return worst["family"]


# ---------------------------------------------------------------------------
# Optional AutoML Framework Benchmarks
# Each runner attempts to import the framework, trains within `time_limit`
# seconds, and returns a comparison row compatible with the main table.
# Returns a "skipped_unavailable_dependency" row if the library is absent.
# ---------------------------------------------------------------------------

_FRAMEWORK_SOURCE = "automl_framework"


def _benchmark_autogluon(
    task_type: str,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    metric: str, time_limit: int,
    label_col: str = "__target__",
) -> dict[str, Any]:
    """AutoGluon benchmark (cross-platform; gracefully skipped if absent)."""
    try:
        from autogluon.tabular import TabularPredictor  # noqa: F401
    except ImportError:
        return {
            "model_name": "AutoGluon", "status": "skipped_unavailable_dependency",
            "cv_score": None, "test_score": None, "training_time": 0.0,
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }

    start = time.perf_counter()
    try:
        from autogluon.tabular import TabularPredictor
        train_df = x_train.copy()
        train_df[label_col] = y_train.values
        test_df = x_test.copy()

        ag_metric = (
            "accuracy" if task_type == "classification" and "accuracy" in metric
            else "roc_auc" if "roc_auc" in metric
            else "f1" if "f1" in metric
            else "r2" if metric == "r2"
            else "root_mean_squared_error" if "rmse" in metric or "root_mean" in metric
            else "mean_absolute_error" if "mae" in metric or "mean_abs" in metric
            else ("accuracy" if task_type == "classification" else "root_mean_squared_error")
        )

        predictor = TabularPredictor(
            label=label_col, eval_metric=ag_metric, verbosity=0,
        ).fit(train_df, time_limit=time_limit, presets="medium_quality")

        y_pred_raw = predictor.predict(test_df)
        # AutoGluon may return string labels; convert to int if le was applied
        try:
            y_pred = y_pred_raw.astype(y_test.dtype).values
        except Exception:
            y_pred = y_pred_raw.values

        y_proba = None
        if task_type == "classification":
            try:
                prob_df = predictor.predict_proba(test_df)
                y_proba = prob_df.values
            except Exception:
                pass

        test_score = _test_score(metric, task_type, y_test, y_pred, y_proba)
        full_metrics = _compute_full_metrics(task_type, y_test, y_pred, y_proba)
        elapsed = time.perf_counter() - start
        return {
            "model_name": "AutoGluon", "cv_score": None,
            "test_score": round(test_score, 6), "training_time": round(elapsed, 2),
            "fold_variance": None, "status": "ok",
            "detailed_metrics": full_metrics, "source": _FRAMEWORK_SOURCE,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("AutoGluon benchmark failed: %s", exc)
        return {
            "model_name": "AutoGluon", "status": f"failed: {exc}",
            "cv_score": None, "test_score": None, "training_time": round(elapsed, 2),
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }


def _benchmark_h2o(
    task_type: str,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    metric: str, time_limit: int,
    label_col: str = "__target__",
) -> dict[str, Any]:
    """H2O AutoML benchmark (requires Java ≥ 8; gracefully skipped if absent)."""
    try:
        import h2o  # noqa: F401
        from h2o.automl import H2OAutoML  # noqa: F401
    except ImportError:
        return {
            "model_name": "H2O AutoML", "status": "skipped_unavailable_dependency",
            "cv_score": None, "test_score": None, "training_time": 0.0,
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }

    start = time.perf_counter()
    try:
        import h2o
        from h2o.automl import H2OAutoML

        h2o.init(nthreads=-1, max_mem_size="2G", verbose=False)

        train_df = x_train.copy()
        train_df[label_col] = y_train.values
        test_df = x_test.copy()
        test_df[label_col] = y_test.values

        h2o_train = h2o.H2OFrame(train_df)
        h2o_test = h2o.H2OFrame(test_df)

        if task_type == "classification":
            h2o_train[label_col] = h2o_train[label_col].asfactor()
            h2o_test[label_col] = h2o_test[label_col].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=time_limit,
            seed=42,
            verbosity=None,
            nfolds=3,
        )
        feature_cols = list(x_train.columns)
        aml.train(x=feature_cols, y=label_col, training_frame=h2o_train)

        preds_frame = aml.leader.predict(h2o_test)
        preds_df = preds_frame.as_data_frame()

        if task_type == "classification":
            y_pred = preds_df["predict"].values
            try:
                y_pred = y_pred.astype(y_test.dtype)
            except Exception:
                pass
            y_proba = None
            proba_cols = [c for c in preds_df.columns if c != "predict"]
            if proba_cols:
                y_proba = preds_df[proba_cols].values
        else:
            y_pred = preds_df["predict"].values.astype(float)
            y_proba = None

        test_score = _test_score(metric, task_type, y_test, y_pred, y_proba)
        full_metrics = _compute_full_metrics(task_type, y_test, y_pred, y_proba)
        elapsed = time.perf_counter() - start

        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass

        return {
            "model_name": "H2O AutoML", "cv_score": None,
            "test_score": round(test_score, 6), "training_time": round(elapsed, 2),
            "fold_variance": None, "status": "ok",
            "detailed_metrics": full_metrics, "source": _FRAMEWORK_SOURCE,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("H2O AutoML benchmark failed: %s", exc)
        try:
            import h2o
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass
        return {
            "model_name": "H2O AutoML", "status": f"failed: {exc}",
            "cv_score": None, "test_score": None, "training_time": round(elapsed, 2),
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }


def run_pipeline(df: pd.DataFrame, session: SessionState) -> dict[str, Any]:
    """
    Execute split, train/evaluate planned models, then ask LLM for interpretation.

    Returns dict including comparison_table and best_model_name.
    """
    if not session.execution_plan:
        raise ValueError("session.execution_plan is empty; call make_execution_plan first")

    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        raise MissingTargetError(
            "target_column is missing or not in the dataframe at execution time; run task_classifier (Phase 1) first."
        )

    split = _split_data(df, session)
    x_train = split["x_train"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    task_type = split["task_type"]

    feature_cols = list(x_train.columns)
    print(
        f"[EXECUTION] target_column={target!r} | n_features={len(feature_cols)} | "
        f"train_shape={x_train.shape} | test_shape={x_test.shape} | task_type={task_type}"
    )

    plan = session.execution_plan or {}
    models = list(plan.get("models") or [])
    raw_metric = plan.get("metric")
    metric = _normalize_sklearn_scorer(
        str(raw_metric or ("accuracy" if task_type == "classification" else "r2")),
        task_type,
        y_train,
    )
    if str(raw_metric or "").strip() and str(raw_metric).strip().lower().replace("-", "_") != metric.lower():
        logger.info("execution metric normalized: %r -> %r", raw_metric, metric)
    cv = _cv_from_plan(session, y_train)
    search = plan.get("search_strategy") or {"name": "GridSearchCV"}
    search_name = str(search.get("name") or "GridSearchCV")
    n_rows = int((session.dataset_profile or {}).get("n_rows") or len(x_train))

    # Index param grids from the plan (LLM Call 2 provides these); fall back to _param_space if absent
    _plan_param_grids: dict[str, dict] = {
        str(m.get("family") or ""): m["param_grid"]
        for m in models
        if isinstance(m, dict) and isinstance(m.get("param_grid"), dict) and m.get("param_grid")
    }

    comparison_table: list[dict[str, Any]] = []

    def _refresh_session_execution_partial(best_name: str | None, best_score: float | None) -> None:
        session.execution_results = {
            "comparison_table": [dict(r) for r in comparison_table],
            "metric": metric,
            "search_strategy": search_name,
            "best_model_name": best_name,
            "best_test_score": best_score,
            "status": "in_progress",
        }

    # -----------------------------------------------------------------------
    # Helper: fully train one model on x_train / y_train and score on x_test
    # -----------------------------------------------------------------------
    def _train_and_evaluate(family: str) -> dict[str, Any]:
        est = _build_estimator(family, task_type)
        if est is None:
            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"model_skipped_dependency:{family}",
                rationale="Optional dependency not installed",
            )
            return {
                "model_name": family,
                "cv_score": None,
                "test_score": None,
                "training_time": 0.0,
                "fold_variance": None,
                "status": "skipped_unavailable_dependency",
            }

        params = _plan_param_grids.get(family) or _param_space(family, search_name, n_rows)
        start = time.perf_counter()
        best_estimator = None
        best_params: dict[str, Any] = {}
        cv_score = float("-inf")
        fold_variance = 0.0

        try:
            if search_name == "GridSearchCV":
                if params:
                    searcher = GridSearchCV(est, param_grid=params, scoring=metric, cv=cv, n_jobs=-1)
                    searcher.fit(x_train, y_train)
                    best_estimator = searcher.best_estimator_
                    best_params = dict(searcher.best_params_)
                    cv_score = float(searcher.best_score_)
                    fold_variance = float(searcher.cv_results_["std_test_score"][searcher.best_index_] ** 2)
                    print(f"[EXECUTION] {family} GridSearchCV best CV score (primary metric): {cv_score:.6f}")
                else:
                    est.fit(x_train, y_train)
                    best_estimator = est
                    best_params = {}
                    cv_score = float(est.score(x_train, y_train))
                    print(f"[EXECUTION] {family} fit score (train proxy): {cv_score:.6f}")
            elif search_name == "RandomizedSearchCV":
                if params:
                    n_iter = min(12, max(4, len(params) * 2))
                    searcher = RandomizedSearchCV(
                        est,
                        param_distributions=params,
                        n_iter=n_iter,
                        scoring=metric,
                        cv=cv,
                        n_jobs=-1,
                        random_state=42,
                    )
                    searcher.fit(x_train, y_train)
                    best_estimator = searcher.best_estimator_
                    best_params = dict(searcher.best_params_)
                    cv_score = float(searcher.best_score_)
                    fold_variance = float(searcher.cv_results_["std_test_score"][searcher.best_index_] ** 2)
                    print(f"[EXECUTION] {family} RandomizedSearchCV best CV score: {cv_score:.6f}")
                else:
                    est.fit(x_train, y_train)
                    best_estimator = est
                    best_params = {}
                    cv_score = float(est.score(x_train, y_train))
                    print(f"[EXECUTION] {family} fit score (train proxy): {cv_score:.6f}")
            else:
                best_estimator, best_params, cv_score, fold_variance = _tune_with_optuna(
                    family=family,
                    base_estimator=est,
                    x_train=x_train,
                    y_train=y_train,
                    cv=cv,
                    metric=metric,
                    n_trials=int(search.get("n_trials") or 50),
                    task_type=task_type,
                    timeout_seconds=_TRAINING_TIMEOUT_SECONDS,
                )
                print(f"[EXECUTION] {family} Optuna best CV score: {cv_score:.6f}")

            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"hyperparameter_search_complete:{family}",
                rationale=f"Best CV score {cv_score:.6f} using {search_name}; fold_variance={fold_variance:.6f}",
                metric=metric,
                cv_score=cv_score,
            )

            if best_estimator is None:
                raise RuntimeError("No estimator produced after search")

            y_pred = best_estimator.predict(x_test)
            y_proba = None
            if task_type == "classification" and hasattr(best_estimator, "predict_proba"):
                try:
                    y_proba = best_estimator.predict_proba(x_test)
                except Exception:
                    y_proba = None
            test_score = _test_score(metric, task_type, y_test, y_pred, y_proba)
            full_metrics = _compute_full_metrics(task_type, y_test, y_pred, y_proba)

            # Scenario: near-perfect test score → likely data leakage
            leakage_suspected = isinstance(test_score, (int, float)) and float(test_score) >= 0.999
            if leakage_suspected:
                logger.warning(
                    "Model %s scored %.6f — near-perfect score is suspicious, possible data leakage.",
                    family, test_score,
                )
                log_decision(
                    session,
                    step="Phase 5 - Execution",
                    action=f"leakage_suspected:{family}",
                    rationale=(
                        f"test_score={test_score:.6f} >= 0.999. "
                        "Verification agent should have caught this — re-check for features "
                        "that are transformations of the target."
                    ),
                )

            elapsed = float(time.perf_counter() - start)

            with _mlflow_ctx(family):
                if mlflow is not None:
                    mlflow.log_param("model_name", family)
                    mlflow.log_param("search_strategy", search_name)
                    mlflow.log_param("metric", metric)
                    for k, v in best_params.items():
                        mlflow.log_param(k, v)
                    mlflow.log_metric("cv_score", cv_score)
                    mlflow.log_metric("test_score", test_score)
                    mlflow.log_metric("training_time", elapsed)

            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"model_trained:{family}",
                rationale=f"test_score={test_score:.6f}, cv_score={cv_score:.6f}, time_s={elapsed:.2f}",
                test_score=test_score,
                cv_score=cv_score,
                training_time_s=elapsed,
            )
            print(f"[EXECUTION] {family} test_score={test_score:.6f} (metric={metric})")
            return {
                "model_name": family,
                "cv_score": cv_score,
                "test_score": test_score,
                "training_time": elapsed,
                "fold_variance": fold_variance,
                "status": "ok",
                "best_params": best_params,
                "leakage_suspected": leakage_suspected,
                "detailed_metrics": full_metrics,
            }

        except Exception as e:
            elapsed = float(time.perf_counter() - start)
            tb = traceback.format_exc()
            logger.error("Model %s failed during training/evaluation:\n%s", family, tb)
            print(f"[EXECUTION] ERROR {family}: {e}\n{tb}")
            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"model_failed:{family}",
                rationale=str(e),
                traceback_excerpt=tb[:2000],
            )
            return {
                "model_name": family,
                "cv_score": None,
                "test_score": None,
                "training_time": elapsed,
                "fold_variance": None,
                "status": f"failed: {e}",
                "error": str(e),
            }

    # -----------------------------------------------------------------------
    # Resolve family names and separate available vs unavailable models
    # -----------------------------------------------------------------------
    available_families: list[str] = []
    for item in models:
        family = str(item.get("family") if isinstance(item, dict) else item)
        if _build_estimator(family, task_type) is None:
            comparison_table.append({
                "model_name": family,
                "cv_score": None,
                "test_score": None,
                "training_time": 0.0,
                "fold_variance": None,
                "status": "skipped_unavailable_dependency",
            })
            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"model_skipped_dependency:{family}",
                rationale="Optional dependency not installed",
            )
        else:
            available_families.append(family)
    _refresh_session_execution_partial(None, None)

    # -----------------------------------------------------------------------
    # Progressive Model Evaluation (successive-halving inspired)
    # When 3+ models are available, run a fast subsample round first to
    # eliminate the weakest candidate before full hyperparameter search.
    # Round 1 (20% data, no tuning) → LLM eliminates worst → Round 2 (100%)
    # Work ratio: 20%×N + 100%×(N-1) vs 100%×N  (saves ~40-50% for N=3)
    # -----------------------------------------------------------------------
    eliminated_family: str | None = None
    subsample_results: list[dict[str, Any]] = []

    if len(available_families) >= 3:
        print(f"[EXECUTION] Progressive evaluation: Round 1 — subsample screening ({len(available_families)} candidates)")
        model_rationales: dict[str, str] = {
            str(item.get("family") if isinstance(item, dict) else item): str(item.get("rationale") or "")
            for item in models
            if isinstance(item, dict)
        }
        profile_summary = {
            "n_rows": n_rows,
            "n_columns": int((session.dataset_profile or {}).get("n_columns") or 0),
            "numeric_ratio": (session.dataset_profile or {}).get("numeric_ratio"),
            "high_cardinality_categoricals": (session.dataset_profile or {}).get("high_cardinality_categoricals", False),
        }
        for family in available_families:
            t0 = time.perf_counter()
            score = _subsample_score(family, task_type, x_train, y_train, cv, metric, subsample_frac=0.20)
            elapsed_sub = time.perf_counter() - t0
            subsample_results.append({
                "family": family,
                "subsample_score": score,
                "rationale": model_rationales.get(family, ""),
            })
            print(f"[EXECUTION] Round 1 subsample | {family}: score={score:.4f} ({elapsed_sub:.1f}s)")

        try:
            eliminated_family = _llm_eliminate_model(subsample_results, task_type, metric, profile_summary)
            print(f"[EXECUTION] Progressive evaluation: LLM eliminated {eliminated_family!r} after Round 1")
            log_decision(
                session,
                step="Phase 5 - Execution",
                action="progressive_elimination",
                rationale=(
                    f"LLM eliminated {eliminated_family!r} after subsample screening. "
                    f"Subsample scores: { {r['family']: round(r['subsample_score'], 4) for r in subsample_results} }"
                ),
                subsample_scores={r["family"]: r["subsample_score"] for r in subsample_results},
                eliminated=eliminated_family,
            )
            # Record the eliminated model in comparison table with its subsample score
            comparison_table.append({
                "model_name": eliminated_family,
                "cv_score": next(r["subsample_score"] for r in subsample_results if r["family"] == eliminated_family),
                "test_score": None,
                "training_time": 0.0,
                "fold_variance": None,
                "status": "eliminated_round1",
                "note": "Eliminated after subsample screening — not trained on full data",
            })
        except Exception as exc:
            logger.warning("Progressive elimination failed (%s); training all models", exc)
            eliminated_family = None

        _refresh_session_execution_partial(None, None)

    # Round 2 (or plain training when <3 models): full hyperparameter search on survivors
    survivors = [f for f in available_families if f != eliminated_family]
    if eliminated_family:
        print(f"[EXECUTION] Round 2 — full training on {len(survivors)} survivors: {survivors}")
    for family in survivors:
        # Scenario: training exceeds 5-minute ceiling → mark timed_out, continue loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
            _future = _executor.submit(_train_and_evaluate, family)
            try:
                row = _future.result(timeout=_TRAINING_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                row = {
                    "model_name": family,
                    "cv_score": None,
                    "test_score": None,
                    "training_time": float(_TRAINING_TIMEOUT_SECONDS),
                    "fold_variance": None,
                    "status": "timed_out",
                    "note": f"Training exceeded {_TRAINING_TIMEOUT_SECONDS}s hard limit.",
                }
                log_decision(
                    session,
                    step="Phase 5 - Execution",
                    action=f"model_timed_out:{family}",
                    rationale=f"Exceeded {_TRAINING_TIMEOUT_SECONDS}s timeout; marked timed_out and skipped.",
                )
                logger.warning("Model %s timed out after %ds; continuing to next model.", family, _TRAINING_TIMEOUT_SECONDS)
        comparison_table.append(row)
        valid_now = [r for r in comparison_table if isinstance(r.get("test_score"), (int, float))]
        best_row = max(valid_now, key=lambda r: float(r["test_score"])) if valid_now else None
        _refresh_session_execution_partial(
            str(best_row["model_name"]) if best_row else None,
            float(best_row["test_score"]) if best_row else None,
        )

    # ── Scenario: all models fail ─────────────────────────────────────────────
    valid_rows = [r for r in comparison_table if isinstance(r.get("test_score"), (int, float))]
    if not valid_rows:
        logger.error("No model completed successfully in execution phase; routing to failure diagnosis.")
        print("[EXECUTION] FAILURE: no model produced a valid test score.")
        log_decision(
            session,
            step="Phase 5 - Execution",
            action="no_models_succeeded",
            rationale=(
                "Every model either failed with an exception, timed out, or was skipped. "
                "Failure diagnosis is triggered with category model_issue."
            ),
        )
        results = {
            "comparison_table": comparison_table,
            "best_model_name": None,
            "metric": metric,
            "search_strategy": search_name,
            "cv_design": plan.get("cv_design") or {},
            "split_meta": split["split_meta"],
            "interpretation": (
                "No model produced a valid score. Possible causes: incompatible metric, "
                "all-NaN features, mismatched label encoding, or dependency errors. "
                "See the decisions log for per-model tracebacks."
            ),
            "best_test_score": None,
            "status": "no_models_succeeded",
            "failure_hint": "model_issue",
        }
        session.execution_results = results
        session.execution_interpretation = results["interpretation"]
        if session.task_config is not None:
            session.task_config["best_model"] = None
        return results

    best_row = max(valid_rows, key=lambda r: float(r["test_score"]))
    best_model_name = str(best_row["model_name"])
    if session.task_config is not None:
        session.task_config["best_model"] = best_model_name

    # ── Optional AutoML framework benchmarks ──────────────────────────────────
    benchmark_time = int(config.AUTOML_BENCHMARK_TIME_LIMIT)
    if benchmark_time > 0:
        logger.info("Running AutoML framework benchmarks (time_limit=%ds each)…", benchmark_time)
        print(f"[EXECUTION] AutoML benchmarks: time_limit={benchmark_time}s each")
        _benchmark_runners = [
            (_benchmark_autogluon, {"label_col": f"__target_{target}__"}),
            (_benchmark_h2o,       {"label_col": f"__target_{target}__"}),
        ]
        for _runner, _kwargs in _benchmark_runners:
            _brow = _runner(
                task_type=task_type,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                metric=metric, time_limit=benchmark_time,
                **_kwargs,
            )
            comparison_table.append(_brow)
            _status = _brow.get("status", "")
            if _status == "ok":
                print(f"[EXECUTION] {_brow['model_name']} benchmark → test_score={_brow['test_score']}")
            elif _status.startswith("skipped"):
                print(f"[EXECUTION] {_brow['model_name']} not installed — skipped")
            else:
                print(f"[EXECUTION] {_brow['model_name']} benchmark failed: {_status}")
            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"automl_benchmark:{_brow['model_name']}",
                rationale=f"status={_status} test_score={_brow.get('test_score')}",
            )

    # Drop AutoML benchmark rows that were merely skipped (library not installed) — they add noise, not signal
    comparison_table = [
        r for r in comparison_table
        if not (r.get("source") == _FRAMEWORK_SOURCE and str(r.get("status", "")).startswith("skipped"))
    ]

    # Collect any leakage-suspected models for surfacing in interpretation
    leakage_models = [r["model_name"] for r in valid_rows if r.get("leakage_suspected")]

    # Best model's full evaluation metrics (for LLM context + frontend display)
    best_full_metrics: dict[str, Any] = best_row.get("detailed_metrics") or {}

    # ── Step 3: Result Interpretation — one LLM call ─────────────────────────
    table_summary = [
        {
            "model": r.get("model_name"),
            "cv_score": r.get("cv_score"),
            "test_score": r.get("test_score"),
            "fold_variance": r.get("fold_variance"),
            "status": r.get("status"),
            "leakage_suspected": r.get("leakage_suspected", False),
            "source": r.get("source", "custom"),
        }
        for r in comparison_table
    ]
    client = LLMClient()
    interpretation_prompt = f"""Phase 5 Step 3 — Result Interpretation.
You received these model evaluation results:

Metric (primary, used for ranking): {metric}
Task type: {task_type}
Performance threshold: {plan.get("threshold")} ({plan.get("threshold_rationale", "")})

Results table:
{table_summary}

Best model ({best_model_name}) full evaluation metrics:
{best_full_metrics}

Write a focused interpretation covering ALL five points below. 4-6 sentences total.

1. Which model performed best and by how much over the next-best (use exact score numbers).
2. Overfitting signals: check for large CV-to-test score gap (>0.05) or high fold_variance (>0.01) in any model.
3. Simpler vs complex: if a simpler model (LogisticRegression/Ridge) matched a complex one (RandomForest/XGBoost) within 0.01, recommend the simpler one.
4. Data leakage: note any model with near-perfect score (≥0.999) or leakage_suspected=True; state this is suspicious.
5. Overall quality: did the best score meet the threshold of {plan.get("threshold")}? Comment on the full metric suite (e.g. precision/recall trade-off for classification, or MAE/RMSE for regression).

Plain text only. No JSON. No bullet points. Flowing sentences."""

    interpretation = client.call(
        interpretation_prompt,
        max_tokens=config.LLM_MAX_TOKENS_NARRATIVE,
        temperature=config.LLM_TEMPERATURE_NARRATIVE,
    ).strip()
    if not interpretation:
        gap_notes = []
        for r in valid_rows:
            cv_s = r.get("cv_score")
            ts = r.get("test_score")
            if isinstance(cv_s, float) and isinstance(ts, float) and abs(cv_s - ts) > 0.05:
                gap_notes.append(f"{r['model_name']} (gap {abs(cv_s - ts):.3f})")
        interpretation = (
            f"Best model: {best_model_name} with test score {best_row['test_score']:.4f}. "
            + (f"Overfitting suspected in: {', '.join(gap_notes)}. " if gap_notes else "")
            + (f"Leakage warning: {', '.join(leakage_models)} scored near-perfect. " if leakage_models else "")
            + "Inspect fold variance for instability and CV-test gap for potential overfitting."
        )

    # Build leakage warning string if any model hit the threshold
    leakage_note = ""
    if leakage_models:
        leakage_note = (
            f"\n\n> ⚠️ **Leakage warning:** {', '.join(leakage_models)} produced a near-perfect score "
            "(≥ 0.999). This is highly suspicious and may indicate a feature encoding the target. "
            "The verification agent should have caught this — inspect feature–target correlations before deployment."
        )

    results = {
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "metric": metric,
        "task_type": task_type,
        "search_strategy": search_name,
        "cv_design": plan.get("cv_design") or {},
        "split_meta": split["split_meta"],
        "interpretation": interpretation + leakage_note,
        "best_test_score": float(best_row["test_score"]),
        "best_model_metrics": best_full_metrics,
        "status": "ok",
        "leakage_warning_models": leakage_models,
    }

    session.execution_results = results
    session.execution_interpretation = interpretation

    log_decision(
        session,
        step="Phase 5 - Execution",
        action="train_evaluate_complete",
        rationale=f"Best model {best_model_name} with test score {best_row['test_score']}",
        best_model_name=best_model_name,
        metric=metric,
    )
    log_decision(
        session,
        step="Phase 5 - Execution",
        action="interpret_results",
        rationale="LLM interpretation of comparison table",
        interpretation_excerpt=interpretation[:4000],
    )

    return results
