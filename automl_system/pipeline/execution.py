"""Phase 5: model execution, comparison, and LLM interpretation."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import traceback
import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from automl_system.pipeline.replay import combine_class_votes, safe_numeric_features
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from automl_system import config
from automl_system.utils import metric_labels
from automl_system.utils.shap_explainer import compute_shap_summary
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
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
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


_xgb = _optional_import("xgboost")
_lgbm = _optional_import("lightgbm")
_catboost = _optional_import("catboost")
XGBClassifier = getattr(_xgb, "XGBClassifier", None)
XGBRegressor = getattr(_xgb, "XGBRegressor", None)
LGBMClassifier = getattr(_lgbm, "LGBMClassifier", None)
LGBMRegressor = getattr(_lgbm, "LGBMRegressor", None)
CatBoostClassifier = getattr(_catboost, "CatBoostClassifier", None)
CatBoostRegressor = getattr(_catboost, "CatBoostRegressor", None)

_TRAINING_TIMEOUT_SECONDS = 300  # fallback only — runtime uses _get_model_timeout(n_rows)


def _get_model_timeout(n_rows: int) -> int:
    """Scale per-model timeout with dataset size.

    Small datasets train fast — no need to wait 5 min for a model that should
    finish in 30s. Large datasets genuinely need more time for RandomizedSearchCV
    to explore enough of the hyperparameter space.

    Tiers:
        < 2 000 rows  → 120s  (GridSearchCV, few combos, very fast)
        < 5 000 rows  → 180s  (GridSearchCV, slightly more combos)
        < 15 000 rows → 300s  (RandomizedSearchCV, medium data)
        < 30 000 rows → 480s  (RandomizedSearchCV, large data)
        >= 30 000 rows → 600s (RandomizedSearchCV, very large — diamonds etc.)
    """
    if n_rows < 2_000:
        return 120
    if n_rows < 5_000:
        return 180
    if n_rows < 15_000:
        return 300
    if n_rows < 30_000:
        return 480
    return 600

# Leave 2 cores free for the OS so the laptop stays responsive.
# -1 (use all cores) causes 100% CPU, thermal throttling, and system hangs on laptops.
import os as _os
_N_JOBS = max(1, (_os.cpu_count() or 4) - 2)

# Models that fail on NaN inputs — sklearn raises ValueError at fit time.
# HistGradientBoosting, XGBoost, LightGBM, CatBoost handle NaN natively.
_NAN_INTOLERANT_FAMILIES: frozenset[str] = frozenset({
    "Ridge", "Lasso", "ElasticNet", "LinearRegression",
    "LogisticRegression", "SVR", "SVC", "KNeighbors", "MLP",
    "GradientBoosting", "RandomForest", "ExtraTrees",
})

# ── GPU detection ────────────────────────────────────────────────────────────
_DETECTED_GPU: bool | None = None
_XGB_LEGACY_GPU: bool = False   # True when XGBoost < 2.0 (uses tree_method='gpu_hist')
_LGBM_GPU: bool = False         # True only if LightGBM GPU confirmed working
_CB_GPU: bool = False           # True only if CatBoost GPU confirmed working


def _detect_gpu() -> bool:
    """Return True if a CUDA GPU is available AND at least one library can use it.

    Checks:
    1. nvidia-smi present and exits 0  (GPU + driver exist)
    2. XGBoost actually accepts a CUDA device without silently falling back to CPU
       (XGBoost 2.x requires CUDA 12+; 1.x uses the older gpu_hist API with CUDA 11)
    """
    global _DETECTED_GPU, _XGB_LEGACY_GPU
    if _DETECTED_GPU is not None:
        return _DETECTED_GPU

    gpu_env = os.getenv("USE_GPU", "auto").strip().lower()
    if gpu_env in ("0", "false", "no"):
        _DETECTED_GPU = False
        return False
    if gpu_env in ("1", "true", "yes"):
        _DETECTED_GPU = True
        return True

    # Step 1: does nvidia-smi report a GPU?
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        nvidia_ok = result.returncode == 0
    except Exception:
        nvidia_ok = False

    if not nvidia_ok:
        _DETECTED_GPU = False
        return False

    # Step 2: verify XGBoost can actually use CUDA (not just silently fall back)
    try:
        import xgboost as _xgb
        import warnings as _w
        _ver = tuple(int(x) for x in _xgb.__version__.split(".")[:2])
        if _ver >= (2, 0):
            # XGBoost 2.x: needs CUDA 12 — do a tiny train to confirm
            with _w.catch_warnings(record=True) as _caught:
                _w.simplefilter("always")
                import numpy as _np
                _dm = _xgb.DMatrix(_np.array([[1, 2], [3, 4]]), label=[0, 1])
                _xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0},
                           _dm, num_boost_round=1)
            # If any warning mentions "CPU" the GPU silently fell back
            _gpu_ok = not any("CPU" in str(w.message) for w in _caught)
        else:
            # XGBoost 1.x: uses tree_method='gpu_hist' API
            with _w.catch_warnings(record=True) as _caught:
                _w.simplefilter("always")
                import numpy as _np
                _dm = _xgb.DMatrix(_np.array([[1, 2], [3, 4]]), label=[0, 1])
                _xgb.train({"tree_method": "gpu_hist", "verbosity": 0}, _dm, num_boost_round=1)
            _gpu_ok = not any("CPU" in str(w.message) for w in _caught)
            if _gpu_ok:
                _XGB_LEGACY_GPU = True   # signal _build_estimator to use old API
    except Exception:
        _gpu_ok = False

    _DETECTED_GPU = _gpu_ok

    if _DETECTED_GPU:
        api = "legacy (tree_method=gpu_hist)" if _XGB_LEGACY_GPU else "modern (device=cuda)"
        logger.info("GPU confirmed working — XGBoost API: %s", api)
        print(f"[GPU] CUDA confirmed — XGBoost GPU API: {api}")

        # Test LightGBM GPU separately — standard pip build is CPU-only on Windows
        # and will hang if told to use GPU. Only enable if it actually works.
        try:
            import lightgbm as _lgbm
            import numpy as _np
            _ds = _lgbm.Dataset(_np.array([[1, 2], [3, 4]]), label=[0, 1])
            _lgbm.train(
                {"device": "gpu", "verbose": -1, "num_leaves": 4},
                _ds, num_boost_round=1,
            )
            _LGBM_GPU = True
            print("[GPU] LightGBM GPU: confirmed")
        except Exception:
            _LGBM_GPU = False
            print("[GPU] LightGBM GPU: not available (CPU-only build) — using CPU")

        # Test CatBoost GPU separately
        try:
            import catboost as _cb
            import numpy as _np
            _cbm = _cb.CatBoostRegressor(
                iterations=1, task_type="GPU", verbose=0
            )
            _cbm.fit(_np.array([[1, 2], [3, 4]]), [0, 1])
            _CB_GPU = True
            print("[GPU] CatBoost GPU: confirmed")
        except Exception:
            _CB_GPU = False
            print("[GPU] CatBoost GPU: not available — using CPU")
    else:
        logger.info("nvidia-smi found but CUDA test failed — running on CPU.")
        print("[GPU] NVIDIA GPU found but CUDA unavailable for XGBoost. "
              "Update drivers to CUDA 12+ for GPU acceleration.")
    return _DETECTED_GPU


# ── Metric display helpers ───────────────────────────────────────────────────
_humanize_metric = metric_labels.humanize_metric
_is_neg_metric = metric_labels.is_neg_metric


def _to_display_score(score: float | None, metric: str) -> float | None:
    """Positive display value: negate neg_* scores so users see 0.47 not -0.47."""
    if score is None:
        return None
    return round(-float(score), 6) if _is_neg_metric(metric) else round(float(score), 6)


def _infer_task_type(df: pd.DataFrame, session: SessionState) -> str:
    # task_config.task_type is the authoritative source — it is set by the benchmark
    # harness (to force regression/classification) or by run_task_classification (Phase 1).
    # execution_plan.task_type is secondary: the pipeline planner LLM may be inconsistent
    # on ambiguous targets (e.g. wine_quality with 6 unique integer values).
    forced = str((session.task_config or {}).get("task_type") or "").strip().lower()
    if forced in {"classification", "regression"}:
        return forced
    planned = str((session.execution_plan or {}).get("task_type") or "").strip().lower()
    if planned in {"classification", "regression"}:
        return planned
    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        return "classification"
    s = df[target]
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    n_unique = int(s.nunique(dropna=True))
    n_rows = max(len(s), 1)
    # Binary is always classification
    if n_unique <= 2:
        return "classification"
    # Low cardinality relative to dataset size → classification (e.g. ordinal ratings 1–10)
    if n_unique <= 50 and (n_unique / n_rows) < 0.05:
        return "classification"
    return "regression"


# Shared with the generated pipeline script (Phase 7) — see replay.py.
_safe_numeric_features = safe_numeric_features

# Hardware/threading params: the replayed script picks its own (CPU, sandbox cores).
_NON_REPLAY_PARAMS = {"device", "gpu_id", "task_type", "n_jobs", "thread_count", "devices"}


def _plain_value(v: Any) -> tuple[bool, Any]:
    """(ok, python literal) — numpy scalars converted; nested objects rejected."""
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return True, v
    if isinstance(v, (list, tuple)):
        items = [_plain_value(x) for x in v]
        if all(ok for ok, _ in items):
            return True, type(v)(x for _, x in items)
    return False, None


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    try:
        return bool(a == b) and type(a) is type(b)
    except Exception:
        return False


def _estimator_spec(est: Any) -> dict[str, Any] | None:
    """Class + constructor params that differ from the class defaults, so the
    Phase 7 script can rebuild the exact estimator. None if not representable."""
    if est is None:
        return None
    cls = type(est)
    try:
        params = est.get_params(deep=False)
    except Exception:
        return None
    try:
        defaults = cls().get_params(deep=False)
    except Exception:
        defaults = {}
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in _NON_REPLAY_PARAMS or (k == "tree_method" and "gpu" in str(v)):
            continue
        if k in defaults and _same_value(defaults[k], v):
            continue
        ok, plain = _plain_value(v)
        if not ok:
            return None
        out[k] = plain
    return {"module": cls.__module__, "class": cls.__name__, "params": out}


def _adaptive_test_size(n_rows: int) -> float:
    """Choose test fraction based on dataset size.

    Small datasets need a larger test fraction to get enough evaluation samples.
    Large datasets can afford a smaller test fraction to maximise training data.

    n_rows < 150  → 30%  (guarantees ≥ 45 test rows)
    n_rows < 500  → 25%  (≥ 37 test rows)
    n_rows < 5000 → 20%  (standard)
    n_rows ≥ 5000 → 15%  (training data more valuable at scale)
    """
    if n_rows < 150:
        return 0.30
    if n_rows < 500:
        return 0.25
    if n_rows < 5000:
        return 0.20
    return 0.15


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

    # Drop rows with NaN/inf in the target — supervised models cannot train on missing labels.
    # Regression targets may also contain ±inf which XGBoost rejects.
    y_check = y.replace([np.inf, -np.inf], np.nan) if task_type == "regression" else y
    bad_mask = y_check.isna()
    if bad_mask.any():
        n_dropped = int(bad_mask.sum())
        logger.warning("_split_data: dropping %d rows with NaN/inf target '%s'", n_dropped, target)
        log_decision(session, step="Phase 5 - Execution", action="drop_nan_target",
                     rationale=f"Dropped {n_dropped} rows with NaN/inf in target '{target}'")
        x = x[~bad_mask]
        y = y[~bad_mask]

    x = _safe_numeric_features(x)

    # Allow benchmark harness to force a fixed test size for fair cross-system comparison.
    # In normal user runs this key is absent and adaptive sizing applies.
    _forced = (session.task_config or {}).get("test_size")
    test_size = float(_forced) if _forced else _adaptive_test_size(len(df))

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
        test_size=test_size,
        random_state=42,
        stratify=stratify_arg,
    )

    label_encoder = None
    if task_type == "classification":
        # Drop any NaN that slipped past target cleaning — LabelEncoder crashes on NaN
        nan_train = y_train.isna()
        nan_test = y_test.isna()
        if nan_train.any():
            keep = ~nan_train
            x_train, y_train = x_train[keep], y_train[keep]
        if nan_test.any():
            keep = ~nan_test
            x_test, y_test = x_test[keep], y_test[keep]
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)

    split_meta = {
        "test_size": test_size,
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


def split_dataframe(df: pd.DataFrame, session: SessionState) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Row-wise train/test split of the COMBINED dataframe (target column still
    included in both), run BEFORE any preprocessing that fits/learns from data
    (feature selection, deep cleaning, feature engineering) — so those steps can
    fit on the train half only and avoid leaking test rows into learned
    parameters (imputation values, scaler bounds, encoder mappings, target
    encodings, engineered-feature accept/reject decisions).

    X/y separation and label encoding still happen later, in
    `_derive_xy_from_presplit`, once preprocessing has run on each half.
    """
    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        raise MissingTargetError(
            "split_dataframe requires a valid target_column in session.task_config that exists in the dataframe. "
            "Resolve the target in Phase 1 (task_classifier) before execution."
        )

    task_type = _infer_task_type(df, session)

    # Drop rows with NaN/inf in the target — supervised models cannot train on missing labels.
    y_check = df[target].replace([np.inf, -np.inf], np.nan) if task_type == "regression" else df[target]
    bad_mask = y_check.isna()
    work = df
    if bad_mask.any():
        n_dropped = int(bad_mask.sum())
        logger.warning("split_dataframe: dropping %d rows with NaN/inf target '%s'", n_dropped, target)
        log_decision(session, step="Phase 3 - Split", action="drop_nan_target",
                     rationale=f"Dropped {n_dropped} rows with NaN/inf in target '{target}'")
        work = df[~bad_mask]

    _forced = (session.task_config or {}).get("test_size")
    test_size = float(_forced) if _forced else _adaptive_test_size(len(work))

    stratify_used = False
    stratify_arg = None
    if task_type == "classification":
        vc = work[target].value_counts(dropna=False)
        if len(vc) > 1 and int(vc.min()) >= 2:
            stratify_arg = work[target]
            stratify_used = True

    df_train, df_test = train_test_split(
        work,
        test_size=test_size,
        random_state=42,
        stratify=stratify_arg,
    )

    split_meta = {
        "test_size": test_size,
        "random_state": 42,
        "split_type": "stratified" if stratify_used else "standard",
        "task_type": task_type,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }
    log_decision(
        session,
        step="Phase 3 - Split",
        action="split_data",
        rationale=f"{split_meta['split_type']} split; train={split_meta['n_train']} test={split_meta['n_test']}",
        split_meta=split_meta,
    )

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def _derive_xy_from_presplit(
    df_train: pd.DataFrame, df_test: pd.DataFrame, session: SessionState
) -> dict[str, Any]:
    """
    Like `_split_data`, but the row-wise split already happened (in
    `split_dataframe`, before Phase 3 preprocessing) — this just separates X/y
    from each already-preprocessed frame and fits the label encoder on train
    only, exactly as `_split_data` already did.
    """
    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df_train.columns:
        raise MissingTargetError(
            "run_pipeline requires a valid target_column in session.task_config that exists in the dataframe. "
            "Resolve the target in Phase 1 (task_classifier) before execution."
        )

    task_type = _infer_task_type(df_train, session)

    x_train, y_train = df_train.drop(columns=[target]), df_train[target]
    x_test = df_test.drop(columns=[target]) if target in df_test.columns else df_test.copy()
    y_test = df_test[target] if target in df_test.columns else pd.Series(dtype=y_train.dtype)

    # Safety net — split_dataframe already dropped NaN/inf-target rows pre-split,
    # so these masks should normally be empty.
    for label, (x, y) in {"train": (x_train, y_train), "test": (x_test, y_test)}.items():
        y_check = y.replace([np.inf, -np.inf], np.nan) if task_type == "regression" else y
        bad = y_check.isna()
        if bad.any():
            logger.warning("_derive_xy_from_presplit: dropping %d leftover NaN/inf-target %s rows",
                            int(bad.sum()), label)
            if label == "train":
                x_train, y_train = x[~bad], y[~bad]
            else:
                x_test, y_test = x[~bad], y[~bad]

    # NOTE: factorized independently per frame — a pre-existing, narrow-scope
    # defensive fallback for any column deep_cleaning didn't fully numericize.
    # By this point every column should already be numeric, so this is a
    # residual edge case rather than a leakage vector introduced here.
    x_train = _safe_numeric_features(x_train)
    x_test = _safe_numeric_features(x_test)

    label_encoder = None
    if task_type == "classification":
        nan_train = y_train.isna()
        nan_test = y_test.isna()
        if nan_train.any():
            keep = ~nan_train
            x_train, y_train = x_train[keep], y_train[keep]
        if nan_test.any():
            keep = ~nan_test
            x_test, y_test = x_test[keep], y_test[keep]
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)

    split_meta = {
        "test_size": None,
        "random_state": 42,
        "split_type": "presplit",
        "task_type": task_type,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_meta": split_meta,
        "task_type": task_type,
        "label_encoder": label_encoder,
    }


def _build_estimator(family: str, task_type: str, n_rows: int = 0, class_weight: str | None = None):
    """
    `class_weight` is passed as 'balanced' when the dataset is severely imbalanced
    and the user goal signals that false negatives are costly. Supported by:
    LogisticRegression, RandomForest, ExtraTrees, GradientBoosting, SVC.
    XGBoost uses scale_pos_weight instead (handled separately).
    """
    gpu = _detect_gpu()
    cw = class_weight if task_type == "classification" else None

    if family == "LogisticRegression":
        if task_type != "classification":
            return None
        return LogisticRegression(max_iter=2000, class_weight=cw)

    if family == "RandomForest":
        return (
            RandomForestClassifier(random_state=42, n_jobs=_N_JOBS, class_weight=cw)
            if task_type == "classification"
            else RandomForestRegressor(random_state=42, n_jobs=_N_JOBS)
        )

    if family == "ExtraTrees":
        return (
            ExtraTreesClassifier(random_state=42, n_jobs=_N_JOBS, class_weight=cw)
            if task_type == "classification"
            else ExtraTreesRegressor(random_state=42, n_jobs=_N_JOBS)
        )

    if family == "HistGradientBoosting":
        return (
            HistGradientBoostingClassifier(random_state=42)
            if task_type == "classification"
            else HistGradientBoostingRegressor(random_state=42)
        )

    if family == "XGBoost":
        xgb_kwargs: dict[str, Any] = {"random_state": 42, "n_jobs": _N_JOBS}
        if gpu:
            if _XGB_LEGACY_GPU:
                xgb_kwargs["tree_method"] = "gpu_hist"   # XGBoost 1.x API (CUDA 11)
            else:
                xgb_kwargs["device"] = "cuda"            # XGBoost 2.x API (CUDA 12)
            # Suppress C++-level "Falling back to DMatrix" warning when predicting on CPU data.
            # This is cosmetic: predictions are correct, just less efficient.
            if _xgb is not None:
                _xgb.set_config(verbosity=0)
        if task_type == "classification":
            if XGBClassifier is None:
                return None
            xgb_kwargs["eval_metric"] = "logloss"
            return XGBClassifier(**xgb_kwargs)
        if XGBRegressor is None:
            return None
        return XGBRegressor(**xgb_kwargs)

    if family == "LightGBM":
        lgbm_kwargs: dict[str, Any] = {"random_state": 42, "verbosity": -1}
        # LightGBM GPU uses OpenCL, not CUDA. Multiple parallel OpenCL contexts (from
        # RandomizedSearchCV n_jobs>1) contend on the GPU lock and serialize each other,
        # making total time far exceed the per-fit estimate and blowing the timeout.
        # Keep the same 50k threshold as CatBoost: CPU LightGBM with n_jobs=_N_JOBS is
        # fast enough for datasets up to ~50k rows (HPO subsampling helps above 20k).
        if gpu and _LGBM_GPU and n_rows >= 50000:
            lgbm_kwargs["device"] = "gpu"
            lgbm_kwargs["n_jobs"] = 1   # GPU handles tree parallelism; CPU threads add contention
        else:
            lgbm_kwargs["n_jobs"] = _N_JOBS
        if task_type == "classification":
            if LGBMClassifier is None:
                return None
            return LGBMClassifier(**lgbm_kwargs)
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(**lgbm_kwargs)

    if family == "CatBoost":
        cb_kwargs: dict[str, Any] = {"random_seed": 42, "verbose": 0, "thread_count": _N_JOBS}
        # CatBoost GPU compiles CUDA kernels per fit (~15s overhead each).
        # For < 50k rows this overhead dominates and makes GPU slower than CPU,
        # plus the C++ CUDA thread cannot be killed by ThreadPoolExecutor timeout.
        if gpu and _CB_GPU and n_rows >= 50000:
            cb_kwargs["task_type"] = "GPU"
        if task_type == "classification":
            if CatBoostClassifier is None:
                return None
            return CatBoostClassifier(**cb_kwargs)
        if CatBoostRegressor is None:
            return None
        return CatBoostRegressor(**cb_kwargs)

    if family == "MLP":
        if task_type == "classification":
            return MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
        return MLPRegressor(random_state=42, max_iter=500, early_stopping=True)

    if family == "LinearRegression":
        if task_type != "regression":
            return None
        return LinearRegression()

    if family == "Ridge":
        if task_type != "regression":
            return None
        return Ridge()

    if family == "ElasticNet":
        if task_type != "regression":
            return None
        return ElasticNet(random_state=42, max_iter=2000)

    if family == "SVR":
        if task_type != "regression":
            return None
        if n_rows > 10_000:
            logger.info("SVR skipped: n_rows=%d > 10 000 (O(n^2) complexity).", n_rows)
            return None
        return SVR()

    if family == "SVC":
        if task_type != "classification":
            return None
        if n_rows > 10_000:
            logger.info("SVC skipped: n_rows=%d > 10 000 (O(n^2) complexity).", n_rows)
            return None
        return SVC(probability=True, random_state=42)

    if family == "GradientBoosting":
        return (
            GradientBoostingClassifier(random_state=42)
            if task_type == "classification"
            else GradientBoostingRegressor(random_state=42)
        )

    if family == "Lasso":
        if task_type != "regression":
            return None
        return Lasso(random_state=42, max_iter=3000)

    if family == "KNeighbors":
        if n_rows > 20_000:
            logger.info("KNeighbors skipped: n_rows=%d > 20 000 (O(n) query complexity).", n_rows)
            return None
        return (
            KNeighborsClassifier(n_jobs=_N_JOBS)
            if task_type == "classification"
            else KNeighborsRegressor(n_jobs=_N_JOBS)
        )

    return None


def _param_space(family: str, strategy_name: str, n_rows: int = 5000) -> dict[str, list[Any]]:
    """Return hyperparameter grid scaled to dataset size.

    tiny      (<   500 rows): 2-4 combinations per model → ~8-12 fits × 5-fold CV
    medium    (< 5 000 rows): moderate grid
    large     (< 30 000 rows): full grid, RandomizedSearchCV with n_iter=30
    very_large(≥ 30 000 rows): compact grid to keep search tractable (prevents multi-hour runs)
    """
    tiny = n_rows < 500
    medium = n_rows < 5000
    very_large = n_rows >= 30000

    if family == "LogisticRegression":
        return {"C": [0.1, 1.0] if tiny else [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]}

    if family == "RandomForest":
        if tiny:
            return {"n_estimators": [100, 200], "max_depth": [None, 8]}
        if medium:
            return {"n_estimators": [100, 200], "max_depth": [None, 8, 15], "min_samples_split": [2, 5]}
        if very_large:
            # 30k+ rows: cap trees at 200 and depth at 15 — unlimited depth on 45k rows
            # exhausts RAM (500 trees × full-depth nodes × n_jobs threads).
            return {"n_estimators": [100, 200], "max_depth": [10, 15], "min_samples_split": [5, 10]}
        return {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}

    if family == "HistGradientBoosting":
        if tiny:
            return {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}
        if medium:
            return {"learning_rate": [0.03, 0.1], "max_depth": [None, 8], "max_leaf_nodes": [15, 31]}
        if very_large:
            return {"learning_rate": [0.03, 0.1], "max_depth": [None, 5], "max_leaf_nodes": [31, 63]}
        return {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [None, 5, 10], "max_leaf_nodes": [15, 31, 63]}

    if family == "XGBoost":
        if tiny:
            return {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]}
        if medium:
            return {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.1],
                "max_depth": [4, 6, 8],
                "min_child_weight": [1, 3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 1.0],
            }
        if very_large:
            return {
                "n_estimators": [300, 500, 750],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.8, 1.0],
                "gamma": [0, 0.1],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [1.0, 2.0, 5.0],
            }
        return {
            "n_estimators": [300, 500, 750],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3],
            "gamma": [0, 0.05, 0.1],
            "reg_alpha": [0, 0.05, 0.1],
            "reg_lambda": [1, 1.5],
        }

    if family == "LightGBM":
        if tiny:
            return {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]}
        if medium:
            return {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [5, 10, 30],
                "colsample_bytree": [0.7, 1.0],
                "subsample": [0.8, 1.0],
            }
        if very_large:
            return {
                "n_estimators": [300, 500, 750],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [10, 30, 50],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
            }
        return {
            "n_estimators": [300, 500, 750],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127, 255],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_samples": [5, 10, 30],
            "reg_alpha": [0, 0.05, 0.1],
            "reg_lambda": [0, 0.05, 0.1],
        }

    if family == "ExtraTrees":
        if tiny:
            return {"n_estimators": [100, 200], "max_depth": [None, 8]}
        if medium:
            return {"n_estimators": [100, 200], "max_depth": [None, 8, 15], "min_samples_split": [2, 5]}
        if very_large:
            return {"n_estimators": [100, 200], "max_depth": [10, 15], "min_samples_split": [5, 10]}
        return {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}

    if family == "CatBoost":
        if tiny:
            return {"iterations": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6]}
        if medium:
            return {
                "iterations": [300, 500, 700],
                "learning_rate": [0.01, 0.03, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5],
                "subsample": [0.8, 1.0],
            }
        if very_large:
            return {
                "iterations": [300, 500, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6],
                "l2_leaf_reg": [1, 3, 5],
                "subsample": [0.8, 1.0],
            }
        # large (5k–30k): HPO on 5k subsample keeps each fit ~7s.
        return {
            "iterations": [300, 500, 700],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5],
            "subsample": [0.8, 1.0],
        }

    if family == "MLP":
        if tiny:
            return {"hidden_layer_sizes": [(64,), (128,)], "alpha": [0.001, 0.01]}
        if medium:
            return {
                "hidden_layer_sizes": [(64, 32), (128, 64)],
                "alpha": [0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
            }
        return {
            "hidden_layer_sizes": [(128, 64), (256, 128), (128, 64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        }

    if family == "LinearRegression":
        return {}

    if family == "Ridge":
        return {"alpha": [0.1, 1.0, 10.0] if tiny else [0.01, 0.1, 1.0, 10.0, 100.0]}

    if family == "ElasticNet":
        if tiny:
            return {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}
        return {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}

    if family == "SVR":
        if tiny:
            return {"C": [0.1, 1.0, 10.0], "kernel": ["rbf"]}
        return {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.3], "kernel": ["rbf", "linear"]}

    if family == "SVC":
        if tiny:
            return {"C": [0.1, 1.0, 10.0], "kernel": ["rbf"]}
        return {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}

    if family == "GradientBoosting":
        if tiny:
            return {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        if medium:
            return {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [3, 5, 7]}
        if very_large:
            # GradientBoosting is sequential (no GPU). n_estimators=400 + max_depth=7 at 30k+
            # rows takes 60s+ per fold and CANNOT be accelerated by GPU or joblib.
            # Use XGBoost/LightGBM instead for very large datasets.
            return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        return {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [3, 5], "subsample": [0.8, 1.0]}

    if family == "Lasso":
        return {"alpha": [0.001, 0.01, 0.1, 1.0] if tiny else [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}

    if family == "KNeighbors":
        if tiny:
            return {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        return {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}

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
    # CV fold count is dataset-size dependent — override the LLM plan here.
    # Too few folds on small datasets = noisy CV scores → HPO picks wrong params
    # (e.g., picks max_depth=8 over max_depth=None on 870-sample folds).
    # 10-fold on 1k rows: each fold trains on 900 samples ≈ full training set → stable estimates.
    # 5-fold on 5k+ rows: 4k samples per fold, stable without the extra computation.
    n_rows = int((session.dataset_profile or {}).get("n_rows") or len(y_train))
    if n_rows < 150:
        folds = 3
    elif n_rows < 3000:
        folds = 5
    elif n_rows < 5000:
        folds = 5
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
        # Capped actual-vs-predicted sample for the Analytics page scatter plot.
        # Cheap (list construction only) — computed for every candidate row, not just
        # the winner, since the cost is negligible next to model fitting.
        try:
            n_scatter = min(300, len(y_t))
            if n_scatter > 0:
                idx = np.random.RandomState(42).choice(len(y_t), size=n_scatter, replace=False)
                out["scatter_sample"] = [
                    {"actual": round(float(y_t[i]), 4), "predicted": round(float(y_p[i]), 4)}
                    for i in idx
                ]
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
                return float(roc_auc_score(y_true, y_proba.ravel()))
            if y_proba.shape[1] == 2:
                # Binary: roc_auc_score expects 1D positive-class probabilities
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        if metric in ("roc_auc_ovr_weighted", "roc_auc_ovo_weighted", "roc_auc_ovr", "roc_auc_ovo"):
            if y_proba is None:
                return float(accuracy_score(y_true, y_pred))
            if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
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
    est = _build_estimator(family, task_type, n_rows=len(x_train))
    if est is None:
        return float("-inf")

    n_sub = max(int(len(x_train) * subsample_frac), 50)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_train), size=min(n_sub, len(x_train)), replace=False)
    x_sub = x_train.iloc[idx]
    y_sub = y_train.iloc[idx]

    # Impute NaN for models that can't handle missing values natively
    if family in _NAN_INTOLERANT_FAMILIES and bool(x_sub.isnull().any().any()):
        from sklearn.impute import SimpleImputer
        _imp = SimpleImputer(strategy="median")
        x_sub = pd.DataFrame(_imp.fit_transform(x_sub), columns=x_sub.columns)

    try:
        # Subsample is tiny (~20% of training data). Parallel CV on tiny data is slower
        # than single-threaded because subprocess spawn + CUDA init overhead dominates.
        scores = cross_val_score(est, x_sub, y_sub, scoring=metric, cv=cv, n_jobs=1)
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

The subsample score is the PRIMARY signal. Eliminate the model with the lowest subsample score UNLESS the score difference between the lowest and second-lowest is less than 0.02 AND there is a clear theoretical reason to prefer eliminating a different model (e.g., it is fundamentally unsuited for this task type, or it is redundant with a strictly better candidate).

If in doubt, always eliminate the lowest subsample scorer.

Respond with ONLY the model family name to eliminate. No explanation. Choose from: {[r['family'] for r in subsample_results]}"""

    response = client.call(prompt, max_tokens=1500, temperature=0.0).strip()
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
# Mutual-information feature filter  (pre-HPO noise removal)
# ---------------------------------------------------------------------------

def _mi_filter_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    session: SessionState,
) -> list[str] | None:
    """Drop features whose MI with the target is < 1 % of the top feature's MI.

    Returns the kept column list, or None when nothing changes (safe to ignore).
    Minimum 4 features required; at least 2 must survive.
    """
    feature_cols = list(x_train.columns)
    if len(feature_cols) < 4:
        return None
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        X_imp = x_train.fillna(x_train.median(numeric_only=True)).fillna(0.0)
        fn = mutual_info_classif if task_type == "classification" else mutual_info_regression
        mi = fn(X_imp, y_train, random_state=42, n_neighbors=3)
        max_mi = float(mi.max()) if float(mi.max()) > 0 else 1.0
        keep_mask = mi >= max_mi * 0.01
        n_drop = int((~keep_mask).sum())
        if n_drop == 0 or int(keep_mask.sum()) < 2:
            return None
        kept = [c for c, k in zip(feature_cols, keep_mask) if k]
        dropped = [c for c, k in zip(feature_cols, keep_mask) if not k]
        print(f"[EXECUTION] MI filter: removed {n_drop} near-zero-MI feature(s): {dropped}")
        log_decision(
            session, step="Phase 5 - Execution", action="mi_feature_filter",
            rationale=f"Dropped {n_drop} features with MI < 1% of max ({max_mi:.4f}): {dropped}",
        )
        return kept
    except Exception as exc:
        logger.warning("MI feature filter failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# LLM-suggested hyperparameter priors  (novel contribution)
# ---------------------------------------------------------------------------

def _llm_suggest_hpo_priors(
    families: list[str],
    task_type: str,
    n_rows: int,
    n_features: int,
    session: SessionState,
) -> dict[str, dict]:
    """One LLM call: focused [lo, hi] search bounds for each model family.

    Sizing heuristic embedded in the prompt so the LLM can adapt to dataset size.
    Falls back to {} on any failure so HPO proceeds with the original grid.
    """
    import json as _json
    client = LLMClient()
    profile = session.dataset_profile or {}
    missing_pct = round(float(profile.get("missing_ratio") or 0) * 100, 1)

    prompt = f"""AutoML HPO advisor. Suggest FOCUSED hyperparameter search bounds.

Dataset: rows={n_rows}, features={n_features}, task={task_type}, missing={missing_pct}%

Sizing rules:
- rows < 1000  : few estimators (100-300), shallow depth (3-5), high regularisation
- rows 1k-10k  : moderate (300-700 estimators), depth 4-8
- rows > 10k   : many estimators (500-1500), depth 6-10, low learning_rate (0.01-0.05)

Return JSON only with one key per model in this list: {families}
Each value is a dict of {{param: [lo, hi]}} covering 2-3 key numeric params.
Only include families that appear in: {families}
Example shape:
{{
  "XGBoost":     {{"n_estimators": [200, 600], "learning_rate": [0.02, 0.15], "max_depth": [4, 8]}},
  "LightGBM":    {{"n_estimators": [200, 600], "learning_rate": [0.02, 0.15], "num_leaves": [31, 127]}},
  "CatBoost":    {{"iterations": [200, 600], "learning_rate": [0.02, 0.15], "depth": [4, 8]}},
  "RandomForest":{{"n_estimators": [200, 500], "max_depth": [6, 20]}},
  "ExtraTrees":  {{"n_estimators": [200, 500], "max_depth": [6, 20]}}
}}"""

    try:
        raw = client.call(prompt, max_tokens=1500, temperature=config.LLM_TEMPERATURE_JSON, json_mode=True)
        if not raw:
            return {}
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            _s = raw.strip()
            _start = _s.find("{")
            _end = _s.rfind("}") + 1
            parsed = _json.loads(_s[_start:_end]) if _start >= 0 and _end > _start else {}
        if not isinstance(parsed, dict):
            return {}
        valid: dict[str, dict] = {}
        for fam, bounds in parsed.items():
            if fam not in families or not isinstance(bounds, dict):
                continue
            clean = {
                k: v for k, v in bounds.items()
                if isinstance(v, list) and len(v) == 2
                and all(isinstance(x, (int, float)) for x in v)
            }
            if clean:
                valid[fam] = clean
        if valid:
            print(f"[EXECUTION] LLM HPO priors: {list(valid.keys())}")
            log_decision(
                session, step="Phase 5 - Execution", action="llm_hpo_priors",
                rationale=f"LLM suggested focused search bounds for {list(valid.keys())}",
            )
        return valid
    except Exception as exc:
        logger.warning("LLM HPO priors call failed (non-fatal): %s", exc)
        return {}


def _apply_hpo_priors(param_space: dict, priors: dict) -> dict:
    """Narrow discrete param_space lists to values within LLM-suggested [lo, hi] bounds.

    Safe fallback: if fewer than 2 numeric values survive for any param, the original
    list is kept. Non-numeric sentinel values already in the grid (e.g. max_depth=None
    for "unlimited depth", a bagging model's best setting) are always preserved — a
    numeric range narrows within numeric options, it must not silently eliminate a
    qualitatively different search mode.
    """
    result = dict(param_space)
    for param, bounds in priors.items():
        if param not in result or not isinstance(bounds, list) or len(bounds) != 2:
            continue
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            continue
        existing = result[param]
        if not isinstance(existing, list):
            continue
        numeric_in_range = [v for v in existing if isinstance(v, (int, float)) and lo <= float(v) <= hi]
        if len(numeric_in_range) < 2:
            continue
        sentinels = [v for v in existing if not isinstance(v, (int, float))]
        result[param] = numeric_in_range + sentinels
    return result


# ---------------------------------------------------------------------------
# Decision-threshold optimisation  (recall-priority binary classification)
# ---------------------------------------------------------------------------

def _find_optimal_threshold(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str,
    task_type: str,
    session: SessionState,
) -> tuple[float, float] | tuple[None, None]:
    """Sweep decision thresholds on a 20 % held-out slice of training data.

    Returns (best_threshold, hold_out_score) or (None, None) on failure.
    Never touches x_test — threshold is derived from training data only.
    Only active for binary classification with a predict_proba estimator.
    """
    if task_type != "classification":
        return None, None
    if int(pd.Series(y_train).nunique(dropna=True)) != 2:
        return None, None
    if not hasattr(estimator, "predict_proba"):
        return None, None
    try:
        from sklearn.model_selection import train_test_split as _tts_thr
        _vc = pd.Series(y_train).value_counts()
        _strat = y_train if int(_vc.min()) >= 2 else None
        _Xtr, _Xval, _ytr, _yval = _tts_thr(
            x_train, y_train, test_size=0.20, random_state=99, stratify=_strat
        )
        _est_thr = clone(estimator)
        _est_thr.fit(_Xtr, _ytr)
        _val_proba = _est_thr.predict_proba(_Xval)[:, 1]
        best_thr, best_score = 0.5, float("-inf")
        for thr in np.arange(0.10, 0.91, 0.05):
            _pred = (_val_proba >= float(thr)).astype(float)
            try:
                score = _test_score(metric, task_type, _yval, _pred, None)
                if score > best_score:
                    best_score, best_thr = score, float(thr)
            except Exception:
                pass
        return round(best_thr, 2), round(best_score, 6)
    except Exception as exc:
        logger.warning("Threshold optimisation failed (non-fatal): %s", exc)
        return None, None


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

        ag_problem_type = "regression" if task_type == "regression" else "multiclass" if task_type == "multiclass" else "binary"
        predictor = TabularPredictor(
            label=label_col, eval_metric=ag_metric, problem_type=ag_problem_type, verbosity=0,
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


def _benchmark_flaml(
    task_type: str,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    metric: str, time_limit: int,
) -> dict[str, Any]:
    """FLAML benchmark (Microsoft; Windows-compatible; gracefully skipped if absent)."""
    try:
        from flaml import AutoML as FlamlAutoML  # noqa: F401
    except ImportError:
        return {
            "model_name": "FLAML", "status": "skipped_unavailable_dependency",
            "cv_score": None, "test_score": None, "training_time": 0.0,
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }

    start = time.perf_counter()
    try:
        from flaml import AutoML as FlamlAutoML

        flaml_task = "classification" if task_type == "classification" else "regression"
        flaml_metric = (
            "roc_auc" if "roc_auc" in metric
            else "f1" if "f1" in metric
            else "accuracy" if metric == "accuracy"
            else "r2" if metric == "r2"
            else "mae" if "mae" in metric
            else ("accuracy" if task_type == "classification" else "r2")
        )

        automl = FlamlAutoML()
        automl.fit(
            x_train, y_train,
            task=flaml_task,
            metric=flaml_metric,
            time_budget=time_limit,
            verbose=0,
        )

        y_pred = automl.predict(x_test)
        y_proba = None
        if task_type == "classification" and hasattr(automl, "predict_proba"):
            try:
                y_proba = automl.predict_proba(x_test)
            except Exception:
                pass

        test_score = _test_score(metric, task_type, y_test, y_pred, y_proba)
        full_metrics = _compute_full_metrics(task_type, y_test, y_pred, y_proba)
        elapsed = time.perf_counter() - start
        return {
            "model_name": "FLAML", "cv_score": None,
            "test_score": round(test_score, 6), "training_time": round(elapsed, 2),
            "fold_variance": None, "status": "ok",
            "detailed_metrics": full_metrics, "source": _FRAMEWORK_SOURCE,
            "best_params": {"best_estimator": str(automl.best_estimator)},
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("FLAML benchmark failed: %s", exc)
        return {
            "model_name": "FLAML", "status": f"failed: {exc}",
            "cv_score": None, "test_score": None, "training_time": round(elapsed, 2),
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }


def _benchmark_tpot(
    task_type: str,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    metric: str, time_limit: int,
) -> dict[str, Any]:
    """TPOT benchmark (genetic programming AutoML; gracefully skipped if absent)."""
    try:
        from tpot import TPOTClassifier, TPOTRegressor  # noqa: F401
    except ImportError:
        return {
            "model_name": "TPOT", "status": "skipped_unavailable_dependency",
            "cv_score": None, "test_score": None, "training_time": 0.0,
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }

    start = time.perf_counter()
    try:
        from tpot import TPOTClassifier, TPOTRegressor

        # TPOT takes minutes in generations; use time_limit in seconds via max_time_mins
        max_time_mins = max(1, time_limit // 60)
        tpot_scoring = (
            "roc_auc" if "roc_auc" in metric
            else "f1_weighted" if "f1" in metric
            else "accuracy" if metric == "accuracy"
            else "r2" if metric == "r2"
            else "neg_mean_absolute_error" if "mae" in metric
            else ("accuracy" if task_type == "classification" else "r2")
        )

        if task_type == "classification":
            tpot = TPOTClassifier(
                max_time_mins=max_time_mins,
                random_state=42,
                verbosity=0,
                n_jobs=_N_JOBS,
            )
        else:
            tpot = TPOTRegressor(
                max_time_mins=max_time_mins,
                random_state=42,
                verbosity=0,
                n_jobs=_N_JOBS,
            )

        tpot.fit(x_train.values, y_train.values)
        y_pred = tpot.predict(x_test.values)
        y_proba = None
        if task_type == "classification" and hasattr(tpot, "predict_proba"):
            try:
                y_proba = tpot.predict_proba(x_test.values)
            except Exception:
                pass

        test_score = _test_score(metric, task_type, y_test, y_pred, y_proba)
        full_metrics = _compute_full_metrics(task_type, y_test, y_pred, y_proba)
        elapsed = time.perf_counter() - start
        return {
            "model_name": "TPOT", "cv_score": None,
            "test_score": round(test_score, 6), "training_time": round(elapsed, 2),
            "fold_variance": None, "status": "ok",
            "detailed_metrics": full_metrics, "source": _FRAMEWORK_SOURCE,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("TPOT benchmark failed: %s", exc)
        return {
            "model_name": "TPOT", "status": f"failed: {exc}",
            "cv_score": None, "test_score": None, "training_time": round(elapsed, 2),
            "fold_variance": None, "source": _FRAMEWORK_SOURCE,
        }


def _extract_feature_importances(
    estimator: Any,
    feature_names: list[str],
    top_n: int = 15,
) -> dict[str, Any]:
    """Extract per-feature importances from a fitted estimator.

    Handles tree-based models (feature_importances_), linear models (coef_),
    and stacking ensembles (delegates to the first base estimator that exposes
    importances). Returns {"top_features": [(name, pct), ...], "method": str}
    or an empty dict when importances cannot be extracted.
    """
    if estimator is None:
        return {}

    importances: "np.ndarray | None" = None
    method = "unknown"

    # Stacking: iterate fitted base estimators, use first one that has importances
    if hasattr(estimator, "estimators_") and hasattr(estimator, "final_estimator_"):
        for base_est in estimator.estimators_:
            if hasattr(base_est, "feature_importances_"):
                importances = np.array(base_est.feature_importances_)
                method = f"feature_importances ({type(base_est).__name__})"
                break
            if hasattr(base_est, "coef_"):
                coef = np.array(base_est.coef_)
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                importances = np.abs(coef)
                method = f"coefficient_magnitude ({type(base_est).__name__})"
                break

    # Tree-based models (RF, XGB, LGBM, HGB, ExtraTrees, GBM, CatBoost)
    if importances is None and hasattr(estimator, "feature_importances_"):
        importances = np.array(estimator.feature_importances_)
        method = "feature_importances"

    # Linear models (Ridge, Lasso, LR, ElasticNet, LinearRegression)
    if importances is None and hasattr(estimator, "coef_"):
        coef = np.array(estimator.coef_)
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importances = np.abs(coef)
        method = "coefficient_magnitude"

    if importances is None or len(importances) != len(feature_names):
        return {}

    total = float(importances.sum())
    if total <= 0:
        return {}

    pairs = sorted(
        zip(feature_names, (importances / total * 100).tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "top_features": [(name, round(pct, 1)) for name, pct in pairs[:top_n]],
        "method": method,
    }


def _build_adaptive_ensemble(
    task_type: str,
    candidate_rows: list[dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str,
    session: SessionState,
    log_target: bool = False,
    y_test_orig: "pd.Series | None" = None,
    n_rows: int = 0,
) -> dict[str, Any] | None:
    """
    Try two ensemble strategies and return whichever beats the best single model.

    Strategy 1 — Stacking (n_rows ≤ 15 000):
      Trains a lightweight meta-learner (Ridge / LogisticRegression) on base
      models' out-of-fold predictions.  This is what AutoGluon does and is the
      single biggest accuracy lever over any individual model.

    Strategy 2 — Soft voting / averaging (top-3 within 3% CV margin):
      Simple average of predictions.  Fast, but weaker than stacking.

    Both strategies are tried when conditions allow.  The best result that
    beats the best single-model test score is returned.
    """
    scored = [
        r for r in candidate_rows
        if isinstance(r.get("cv_score"), float) and r.get("_fitted_estimator") is not None
    ]
    if len(scored) < 2:
        return None

    scored.sort(key=lambda r: float(r["cv_score"]), reverse=True)
    y_eval = y_test_orig if (log_target and y_test_orig is not None) else y_test

    all_test_scores = [float(r["test_score"]) for r in candidate_rows
                       if isinstance(r.get("test_score"), (int, float))]
    if not all_test_scores:
        return None
    best_single_score = max(all_test_scores)

    best_ens_row: dict[str, Any] | None = None
    best_ens_score = best_single_score  # ensemble must beat this to be selected

    # ── Strategy 1: Stacking ──────────────────────────────────────────────────
    # Only for datasets where base-model refitting with CV is feasible.
    if n_rows <= 30000 and len(scored) >= 2:
        try:
            from sklearn.ensemble import (GradientBoostingClassifier,
                                          GradientBoostingRegressor,
                                          StackingClassifier, StackingRegressor)
            from sklearn.linear_model import LogisticRegression as _LR

            start = time.perf_counter()
            top_stack = scored[:4]

            # ── B: Diversity-aware pruning ────────────────────────────────────
            # Two models that produce near-identical predictions add no ensemble
            # value. For each candidate (best-first), keep it only if its
            # prediction correlation with every already-kept model is below the
            # threshold. Uses the fitted estimators stored during Phase 5 HPO.
            _DIVERSITY_CORR_THRESHOLD = 0.92
            if len(top_stack) >= 3:
                try:
                    _pred_cache: dict[str, np.ndarray] = {}
                    for _row in top_stack:
                        _est = _row.get("_fitted_estimator")
                        if _est is None:
                            continue
                        try:
                            if task_type == "classification" and hasattr(_est, "predict_proba"):
                                _pred_cache[_row["model_name"]] = _est.predict_proba(x_test)[:, -1]
                            else:
                                _pred_cache[_row["model_name"]] = _est.predict(x_test).astype(float)
                        except Exception:
                            pass

                    _kept: list[dict[str, Any]] = []
                    _kept_preds: list[np.ndarray] = []
                    for _row in top_stack:
                        _pred = _pred_cache.get(_row["model_name"])
                        if _pred is None or not _kept_preds:
                            _kept.append(_row)
                            if _pred is not None:
                                _kept_preds.append(_pred)
                            continue
                        _max_corr = max(
                            float(np.corrcoef(_pred, _kp)[0, 1])
                            for _kp in _kept_preds
                        )
                        if _max_corr < _DIVERSITY_CORR_THRESHOLD:
                            _kept.append(_row)
                            _kept_preds.append(_pred)

                    if len(_kept) >= 2 and len(_kept) < len(top_stack):
                        _pruned_names = [r["model_name"] for r in top_stack if r not in _kept]
                        print(
                            f"[EXECUTION] Diversity pruning: removed {_pruned_names} "
                            f"(prediction corr > {_DIVERSITY_CORR_THRESHOLD} with a kept model)"
                        )
                        log_decision(
                            session, step="Phase 5 - Execution",
                            action="diversity_pruning",
                            rationale=(
                                f"Removed {_pruned_names}: prediction correlation > "
                                f"{_DIVERSITY_CORR_THRESHOLD} with a kept model — "
                                "maximising ensemble diversity before stacking."
                            ),
                        )
                        top_stack = _kept
                except Exception as _dp_err:
                    logger.warning("Diversity pruning failed, using all models: %s", _dp_err)

            # Rebuild estimators with best hyperparams from the HPO search.
            fresh: list[tuple[str, Any]] = []
            for row in top_stack:
                est = _build_estimator(row["model_name"], task_type, n_rows=n_rows)
                if est is None:
                    continue
                bp = row.get("best_params") or {}
                if bp:
                    try:
                        est.set_params(**bp)
                    except Exception:
                        pass
                fresh.append((row["model_name"], est))

            # For regression, inject Ridge as a linear diversifier when only
            # gradient boosting models are present. The meta-learner gains a
            # structural complement that captures linear relationships the tree
            # ensemble collectively misses.
            if task_type == "regression" and all(n != "Ridge" for n, _ in fresh):
                from sklearn.pipeline import Pipeline as _Pipe
                from sklearn.preprocessing import StandardScaler as _SS
                fresh.append(("Ridge", _Pipe([("scaler", _SS()), ("ridge", Ridge(alpha=1.0))])))

            def _member_spec(name: str, est: Any) -> dict[str, Any] | None:
                if name == "Ridge" and type(est).__name__ == "Pipeline":
                    return {"special": "scaled_ridge", "alpha": 1.0}
                return _estimator_spec(est)

            if len(fresh) >= 2:
                cv_folds = 3 if n_rows > 5000 else 5
                if task_type == "regression":
                    stack = StackingRegressor(
                        estimators=fresh,
                        # Ridge meta-learner: learns stable optimal weights for base models.
                        # GBM meta-learner overfits when base model OOF predictions are
                        # highly correlated (e.g. all gradient boosting), reducing stacking
                        # score below the best single model.
                        final_estimator=Ridge(alpha=1.0),
                        cv=cv_folds, n_jobs=_N_JOBS,
                    )
                else:
                    from sklearn.linear_model import LogisticRegression as _LR
                    stack = StackingClassifier(
                        estimators=fresh,
                        final_estimator=(
                            GradientBoostingClassifier(
                                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                            ) if n_rows >= 5000 else _LR(max_iter=1000, C=1.0)
                        ),
                        cv=cv_folds, n_jobs=_N_JOBS, stack_method="auto",
                    )

                stack.fit(x_train, y_train)
                y_pred = stack.predict(x_test)
                if log_target:
                    y_pred = np.expm1(y_pred)

                y_proba = None
                if task_type == "classification" and hasattr(stack, "predict_proba"):
                    try:
                        y_proba = stack.predict_proba(x_test)
                    except Exception:
                        pass

                stack_score = _test_score(metric, task_type, y_eval, y_pred, y_proba)
                stack_metrics = _compute_full_metrics(task_type, y_eval, y_pred, y_proba)
                elapsed = float(time.perf_counter() - start)
                names = [n for n, _ in fresh]
                stack_name = f"Stacking({'+'.join(names)})"

                print(
                    f"[EXECUTION] Stacking ensemble {stack_name}: "
                    f"test={stack_score:.6f} vs best_single={best_single_score:.6f} -- "
                    f"{'STACKING WINS' if stack_score > best_single_score else 'single model wins'}"
                )
                log_decision(session, step="Phase 5 - Execution", action="stacking_ensemble",
                             rationale=f"{stack_name} score={stack_score:.6f} vs best_single={best_single_score:.6f}")

                if stack_score > best_ens_score:
                    best_ens_score = stack_score
                    best_ens_row = {
                        "model_name": stack_name,
                        "cv_score": None,
                        "test_score": stack_score,
                        "cv_display_score": None,
                        "test_display_score": _to_display_score(stack_score, metric),
                        "training_time": elapsed,
                        "fold_variance": None,
                        "status": "ok",
                        "best_params": {"members": names, "type": "stacking"},
                        "model_spec": {
                            "kind": "stacking",
                            "members": [[n, _member_spec(n, e)] for n, e in fresh],
                            "final_estimator": _estimator_spec(stack.final_estimator),
                            "cv": cv_folds,
                        },
                        "leakage_suspected": False,
                        "detailed_metrics": stack_metrics,
                        "_fitted_estimator": stack,
                        "is_ensemble": True,
                    }
        except Exception as exc:
            logger.warning("Stacking ensemble failed: %s", exc)

    # ── Strategy 2: Soft voting / averaging ───────────────────────────────────
    # When top-3 models are within 3% CV — use CV-score-weighted averaging.
    top3 = scored[:3]
    margin = abs(float(top3[0]["cv_score"]) - float(top3[-1]["cv_score"]))
    if margin <= 0.03:
        try:
            members = [r["model_name"] for r in top3]
            start = time.perf_counter()

            # Direct prediction averaging from already-fitted estimators.
            # Avoids VotingRegressor refit which: (1) loses early-stopping convergence
            # info, (2) causes GPU memory contention when parallel GPU models refit
            # simultaneously on large datasets (seen as 0.965 R² on diamonds vs 0.981).
            _direct_preds: list[np.ndarray] = []
            _direct_weights: list[float] = []
            _direct_proba: list[np.ndarray] = []
            _direct_rows: list[dict[str, Any]] = []
            for r in top3:
                _fitted = r.get("_fitted_estimator")
                if _fitted is None:
                    continue
                try:
                    _p = _fitted.predict(x_test).astype(float)
                    _direct_preds.append(_p)
                    _direct_weights.append(max(float(r.get("cv_score", 1e-6)), 1e-6))
                    _direct_rows.append(r)
                    if task_type == "classification" and hasattr(_fitted, "predict_proba"):
                        try:
                            _direct_proba.append(_fitted.predict_proba(x_test))
                        except Exception:
                            pass
                except Exception:
                    pass

            if len(_direct_preds) >= 2:
                _w = np.array(_direct_weights)
                _w = _w / _w.sum()
                y_proba: np.ndarray | None = None

                if task_type == "classification":
                    if _direct_proba and len(_direct_proba) == len(_direct_preds):
                        # Soft vote: average class probabilities → argmax to label
                        y_proba = np.zeros_like(_direct_proba[0])
                        for _pp, _wt in zip(_direct_proba, _w):
                            y_proba = y_proba + _pp * _wt
                        y_pred = np.argmax(y_proba, axis=1).astype(float)
                    else:
                        # Hard vote: weighted majority of the predicted classes
                        y_pred = combine_class_votes(_direct_preds, None, _w)[0].astype(float)
                else:
                    # Regression: weighted average of continuous predictions
                    y_pred = np.zeros_like(_direct_preds[0])
                    for _p, _wt in zip(_direct_preds, _w):
                        y_pred = y_pred + _p * _wt
                    if log_target:
                        y_pred = np.expm1(y_pred)
                    if _direct_proba and len(_direct_proba) == len(_direct_preds):
                        y_proba = np.zeros_like(_direct_proba[0])
                        for _pp, _wt in zip(_direct_proba, _w):
                            y_proba = y_proba + _pp * _wt
                voting = "direct_avg"
                ens = None
                _vote_spec: dict[str, Any] | None = {
                    "kind": "average",
                    "members": [r.get("model_spec") for r in _direct_rows],
                    "weights": [float(x) for x in _w],
                    "soft": bool(_direct_proba and len(_direct_proba) == len(_direct_preds)),
                }
            else:
                # Fallback: rebuild and refit (used when _fitted_estimator unavailable).
                _early_stop_keys = {"early_stopping_rounds", "eval_set", "eval_metric"}
                estimators = []
                for r in top3:
                    _est = _build_estimator(r["model_name"], task_type, n_rows=n_rows)
                    if _est is None:
                        continue
                    _bp = {k: v for k, v in (r.get("best_params") or {}).items()
                           if k not in _early_stop_keys}
                    if _bp:
                        try:
                            _est.set_params(**_bp)
                        except Exception:
                            pass
                    estimators.append((r["model_name"], _est))
                _vote_spec = None  # refit fallback is not replayed (script generated by LLM instead)

                raw_weights = [max(float(r["cv_score"]), 1e-6) for r in top3[:len(estimators)]]
                w_min = min(raw_weights)
                weights = [w - w_min + 1e-3 for w in raw_weights]

                if task_type == "classification":
                    can_soft = all(hasattr(est, "predict_proba") for _, est in estimators)
                    voting = "soft" if can_soft else "hard"
                    ens = VotingClassifier(estimators=estimators, voting=voting, weights=weights, n_jobs=_N_JOBS)
                else:
                    voting = "weighted averaging"
                    ens = VotingRegressor(estimators=estimators, weights=weights, n_jobs=_N_JOBS)

                ens.fit(x_train, y_train)
                y_pred = ens.predict(x_test)
                if log_target:
                    y_pred = np.expm1(y_pred)

                y_proba = None
                if task_type == "classification" and hasattr(ens, "predict_proba"):
                    try:
                        y_proba = ens.predict_proba(x_test)
                    except Exception:
                        pass

            vote_score = _test_score(metric, task_type, y_eval, y_pred, y_proba)
            vote_metrics = _compute_full_metrics(task_type, y_eval, y_pred, y_proba)
            elapsed = float(time.perf_counter() - start)
            ens_name = f"Ensemble({'+'.join(members)})"

            print(
                f"[EXECUTION] Voting ensemble {ens_name}: "
                f"test={vote_score:.6f} vs best={best_ens_score:.6f} -- "
                f"{'VOTING WINS' if vote_score > best_ens_score else 'no improvement'}"
            )
            log_decision(session, step="Phase 5 - Execution", action="voting_ensemble",
                         rationale=f"CV margin={margin:.4f} ≤ 0.03; {ens_name} score={vote_score:.6f}")

            if vote_score > best_ens_score:
                best_ens_score = vote_score
                best_ens_row = {
                    "model_name": ens_name,
                    "cv_score": None,
                    "test_score": vote_score,
                    "cv_display_score": None,
                    "test_display_score": _to_display_score(vote_score, metric),
                    "training_time": elapsed,
                    "fold_variance": None,
                    "status": "ok",
                    "best_params": {"members": members, "voting": voting},
                    "model_spec": _vote_spec,
                    "leakage_suspected": False,
                    "detailed_metrics": vote_metrics,
                    "_fitted_estimator": ens,
                    "is_ensemble": True,
                }
        except Exception as exc:
            logger.warning("Voting ensemble failed: %s", exc)
    else:
        print(f"[EXECUTION] Voting skipped: {top3[0]['model_name']} leads by CV margin={margin:.4f} > 0.03")

    return best_ens_row


def run_pipeline(
    df: pd.DataFrame, session: SessionState, df_test: pd.DataFrame | None = None
) -> dict[str, Any]:
    """
    Execute split (or, when `df_test` is provided, derive X/y from the already
    train/test-split and already-preprocessed frames), train/evaluate planned
    models, then ask LLM for interpretation.

    Returns dict including comparison_table and best_model_name.
    """
    if not session.execution_plan:
        raise ValueError("session.execution_plan is empty; call make_execution_plan first")

    target = str((session.task_config or {}).get("target_column") or "").strip()
    if not target or target not in df.columns:
        raise MissingTargetError(
            "target_column is missing or not in the dataframe at execution time; run task_classifier (Phase 1) first."
        )

    split = (
        _derive_xy_from_presplit(df, df_test, session)
        if df_test is not None
        else _split_data(df, session)
    )
    x_train = split["x_train"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    task_type = split["task_type"]

    # Keep pre-SMOTE copies for competitor benchmarks — AutoGluon/FLAML must
    # receive the original split (same rows as x_test) so scores are comparable.
    _x_train_for_benchmarks = x_train
    _y_train_for_benchmarks = y_train
    _smote_was_applied = False
    _smote_spec: dict[str, Any] | None = None

    # ── C: SMOTE for imbalanced classification ────────────────────────────────
    # Applied on train split only (never touches x_test / y_test).
    # Gate: minority class < 25% OR execution plan signals recall priority.
    if task_type == "classification":
        _vc = pd.Series(y_train).value_counts()
        if len(_vc) >= 2:
            _minority_ratio = float(_vc.min() / len(y_train))
            _exec_signals = (session.execution_plan or {}).get("constraint_signals") or {}
            _apply_smote = _minority_ratio < 0.25 or bool(_exec_signals.get("prioritize_recall"))
            if _apply_smote:
                try:
                    from imblearn.over_sampling import SMOTE as _SMOTE
                    _k_neighbors = min(5, int(_vc.min()) - 1)
                    if _k_neighbors >= 1:
                        _feat_cols_smote = list(x_train.columns)
                        _sm = _SMOTE(random_state=42, k_neighbors=_k_neighbors)
                        _x_res, _y_res = _sm.fit_resample(x_train, y_train)
                        x_train = pd.DataFrame(_x_res, columns=_feat_cols_smote)
                        y_train = pd.Series(_y_res, name=y_train.name)
                        print(
                            f"[EXECUTION] SMOTE: minority class was {_minority_ratio:.1%} → "
                            f"balanced to {len(y_train)} training samples"
                        )
                        _smote_was_applied = True
                        _smote_spec = {"k_neighbors": int(_k_neighbors), "random_state": 42}
                        log_decision(
                            session, step="Phase 5 - Execution", action="smote_applied",
                            rationale=(
                                f"Class imbalance {_minority_ratio:.1%} < 25% threshold — "
                                f"SMOTE resampled training set to {len(y_train)} samples"
                            ),
                        )
                except ImportError:
                    logger.warning(
                        "SMOTE skipped — install imbalanced-learn: pip install imbalanced-learn"
                    )
                except Exception as _smote_err:
                    logger.warning("SMOTE failed, continuing without resampling: %s", _smote_err)

    # Log-transform heavily right-skewed regression targets.
    # Models train on log1p(y); predictions are expm1'd back to original scale before scoring.
    # This improves R² on price/income-style data without touching features → no leakage.
    _log_target = False
    _y_test_orig = y_test  # always original scale
    if task_type == "regression":
        _y_skew = float(y_train.skew()) if len(y_train) > 2 else 0.0
        _y_min = float(y_train.min()) if len(y_train) > 0 else 0.0
        if _y_skew > 1.5 and _y_min >= 0:
            y_train = np.log1p(y_train)
            _log_target = True
            print(f"[EXECUTION] Target log1p transform applied (skew={_y_skew:.2f}, min={_y_min:.2f} >= 0)")
            log_decision(
                session, step="Phase 5 - Execution", action="log_target_transform",
                rationale=f"Regression target skew={_y_skew:.2f}>1.5, min={_y_min:.2f}≥0 → log1p(y_train); predictions will be expm1'd"
            )

    feature_cols = list(x_train.columns)
    print(
        f"[EXECUTION] target_column={target!r} | n_features={len(feature_cols)} | "
        f"train_shape={x_train.shape} | test_shape={x_test.shape} | task_type={task_type}"
    )

    # Pre-compute median-imputed copies for linear/distance-based models that don't handle NaN.
    # Tree-boosting models (HGB, XGB, LightGBM) keep the original data with NaN intact.
    _has_nan = bool(x_train.isnull().any().any())
    if _has_nan:
        from sklearn.impute import SimpleImputer as _SI
        _imp_fit = _SI(strategy="median")
        _x_train_imp = pd.DataFrame(_imp_fit.fit_transform(x_train), columns=feature_cols)
        _x_test_imp = pd.DataFrame(_imp_fit.transform(x_test), columns=feature_cols)
        print(f"[EXECUTION] NaN detected in {int(x_train.isnull().any().sum())} feature(s) - "
              f"median imputation ready for NaN-intolerant models")
    else:
        _x_train_imp = x_train
        _x_test_imp = x_test

    # ── Mutual-information feature filter (before HPO) ────────────────────────
    # Drops features with MI < 1% of the top feature's MI.  These are noise
    # contributors that slow HPO without improving CV scores.
    _mi_kept = _mi_filter_features(x_train, y_train, task_type, session)
    if _mi_kept is not None:
        x_train      = x_train[_mi_kept]
        x_test       = x_test[_mi_kept]
        _x_train_imp = _x_train_imp[_mi_kept]
        _x_test_imp  = _x_test_imp[_mi_kept]
        feature_cols = _mi_kept
        # Column set changed (not just rows) — the pre-SMOTE benchmark snapshot
        # must match x_test's columns or AutoGluon/FLAML's internal comparison
        # will fail on the removed column(s).
        _x_train_for_benchmarks = _x_train_for_benchmarks[_mi_kept]

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

    # Determine class_weight from constraint signals + imbalance severity.
    # Applied when the user goal signals false negatives are costly (fraud, medical)
    # AND the dataset is moderately-to-severely imbalanced.
    _imbalance_sev = str(
        ((session.eda_results or {}).get("target_imbalance") or {}).get("severity") or ""
    )
    _recall_priority = str(
        (session.task_config or {}).get("metric") or plan.get("metric") or ""
    ).startswith("recall")
    _class_weight: str | None = (
        "balanced"
        if task_type == "classification"
        and _imbalance_sev in {"moderate", "high", "severe"}
        and _recall_priority
        else None
    )
    if _class_weight:
        log_decision(
            session,
            step="Phase 5 - Execution",
            action="class_weight_balanced",
            rationale=(
                f"Imbalance severity={_imbalance_sev} and recall-priority metric detected — "
                "applying class_weight='balanced' to all supporting estimators."
            ),
        )
        print(f"[EXECUTION] class_weight='balanced' applied (imbalance={_imbalance_sev}, metric={metric})")

    # Index param grids from the plan (LLM Call 2 provides these); fall back to _param_space if absent
    _plan_param_grids: dict[str, dict] = {
        str(m.get("family") or ""): m["param_grid"]
        for m in models
        if isinstance(m, dict) and isinstance(m.get("param_grid"), dict) and m.get("param_grid")
    }

    _hpo_priors: dict[str, dict] = {}   # populated after families are resolved (LLM call)
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
    # Helper: Optuna TPE HPO with early stopping for boosting models
    # -----------------------------------------------------------------------
    def _optuna_hpo(
        family: str,
        param_space: dict,
        X_hpo,
        y_hpo,
        cv_obj,
        scoring: str,
        n_trials: int,
    ) -> tuple:
        """
        Replaces RandomizedSearchCV with Optuna TPE.
        For XGBoost / LightGBM / CatBoost: uses early stopping on a held-out 20%
        split — each trial stops as soon as score plateaus (much faster per trial).
        For all other families: uses cross_val_score with TPE sampling.
        Returns (best_params, best_score) or (None, None) if optuna not installed.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return None, None

        _ES_FAMILIES = {"XGBoost", "LightGBM", "CatBoost"}
        use_es = family in _ES_FAMILIES

        if use_es:
            from sklearn.model_selection import train_test_split as _tts
            _strat = y_hpo if task_type == "classification" else None
            try:
                _X_tr, _X_val, _y_tr, _y_val = _tts(
                    X_hpo, y_hpo, test_size=0.2, random_state=42, stratify=_strat
                )
            except ValueError:
                _X_tr, _X_val, _y_tr, _y_val = _tts(
                    X_hpo, y_hpo, test_size=0.2, random_state=42
                )

        # Exclude n_estimators/iterations for ES families — early stopping
        # determines the optimal stopping point automatically.
        _search_space = (
            {k: v for k, v in param_space.items() if k not in ("n_estimators", "iterations")}
            if use_es else param_space
        )

        if use_es:
            from sklearn.metrics import check_scoring as _cs
            _dummy = _build_estimator(family, task_type, n_rows=n_rows, class_weight=_class_weight)
            _scorer = _cs(_dummy, scoring=scoring) if _dummy is not None else None
        else:
            _scorer = None

        # Parameters that benefit from log-uniform sampling (multiplicative scale).
        _LOG_SCALE_PARAMS = {
            "learning_rate", "reg_lambda", "reg_alpha", "l2_leaf_reg",
            "min_child_samples", "min_samples_leaf", "alpha", "C",
        }

        def objective(trial):
            sampled = {}
            for k, v in _search_space.items():
                if not isinstance(v, list) or not v:
                    continue
                all_numeric = all(isinstance(x, (int, float)) for x in v)
                if all_numeric and len(v) >= 2:
                    lo, hi = float(min(v)), float(max(v))
                    use_log = k in _LOG_SCALE_PARAMS and lo > 0
                    if all(isinstance(x, int) for x in v):
                        sampled[k] = trial.suggest_int(k, int(lo), int(hi), log=use_log)
                    else:
                        sampled[k] = trial.suggest_float(k, lo, hi, log=use_log)
                else:
                    sampled[k] = trial.suggest_categorical(k, v)
            est = _build_estimator(family, task_type, n_rows=n_rows, class_weight=_class_weight)
            if est is None:
                raise optuna.exceptions.TrialPruned()
            try:
                est.set_params(**sampled)
            except Exception:
                raise optuna.exceptions.TrialPruned()
            try:
                if use_es:
                    if family == "XGBoost":
                        est.set_params(n_estimators=1000, early_stopping_rounds=50)
                        est.fit(_X_tr, _y_tr, eval_set=[(_X_val, _y_val)], verbose=False)
                    elif family == "LightGBM":
                        import lightgbm as _lgb
                        est.set_params(n_estimators=1000)
                        est.fit(
                            _X_tr, _y_tr,
                            eval_set=[(_X_val, _y_val)],
                            callbacks=[
                                _lgb.early_stopping(50, verbose=False),
                                _lgb.log_evaluation(-1),
                            ],
                        )
                    elif family == "CatBoost":
                        est.set_params(iterations=1000, early_stopping_rounds=50)
                        est.fit(_X_tr, _y_tr, eval_set=(_X_val, _y_val), verbose=0)
                    if _scorer is None:
                        return float("-inf")
                    return float(_scorer(est, _X_val, _y_val))
                else:
                    scores = cross_val_score(est, X_hpo, y_hpo, cv=cv_obj, scoring=scoring, n_jobs=1)
                    return float(scores.mean())
            except Exception:
                raise optuna.exceptions.TrialPruned()

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(5, n_trials))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if not study.best_trials:
            return None, None
        return dict(study.best_params), float(study.best_value)

    # -----------------------------------------------------------------------
    # Helper: fully train one model on x_train / y_train and score on x_test
    # -----------------------------------------------------------------------
    def _train_and_evaluate(family: str) -> dict[str, Any]:
        est = _build_estimator(family, task_type, n_rows=n_rows, class_weight=_class_weight)
        # Use median-imputed data for models that crash on NaN
        _xt = _x_train_imp if (_has_nan and family in _NAN_INTOLERANT_FAMILIES) else x_train
        _xte = _x_test_imp if (_has_nan and family in _NAN_INTOLERANT_FAMILIES) else x_test
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
        # Narrow search grid with LLM-suggested [lo, hi] bounds (safe: keeps original if < 2 survive)
        _prior = _hpo_priors.get(family) or {}
        if _prior:
            params = _apply_hpo_priors(params, _prior)
        start = time.perf_counter()
        best_estimator = None
        best_params: dict[str, Any] = {}
        cv_score = float("-inf")
        fold_variance = 0.0
        # How best_estimator was finally fitted — replayed by the Phase 7 script.
        fit_info: dict[str, Any] = {"mode": "full"}

        try:
            if search_name == "GridSearchCV":
                if params:
                    # Always sequential — same reason as RandomizedSearchCV below.
                    _gs_njobs = 1
                    import math as _math
                    _gs_combos = _math.prod(len(v) for v in params.values() if isinstance(v, list))
                    _gs_folds = cv.get_n_splits(_xt, y_train) if hasattr(cv, "get_n_splits") else 5
                    _gs_total_fits = _gs_combos * _gs_folds
                    # Auto-fallback: GridSearch with >500 total fits stalls for minutes on laptops.
                    # Switch to RandomizedSearchCV and sample up to 25 configs instead.
                    if _gs_total_fits > 500:
                        _gs_n_iter = min(20 if n_rows < 3000 else 40, _gs_combos)
                        print(
                            f"[EXECUTION] {family} GridSearchCV auto-fallback to RandomizedSearchCV "
                            f"({_gs_combos} combos × {_gs_folds} folds = {_gs_total_fits} fits > 500 cap). "
                            f"Sampling {_gs_n_iter} configs instead.",
                            flush=True,
                        )
                        searcher = RandomizedSearchCV(
                            est, param_distributions=params, n_iter=_gs_n_iter,
                            scoring=metric, cv=cv, n_jobs=1, random_state=42,
                        )
                        searcher.fit(_xt, y_train)
                        best_estimator = searcher.best_estimator_
                        best_params = dict(searcher.best_params_)
                        cv_score = float(searcher.best_score_)
                        fold_variance = float(searcher.cv_results_["std_test_score"][searcher.best_index_] ** 2)
                        print(f"[EXECUTION] {family} RandomizedSearchCV best CV score: {cv_score:.6f}", flush=True)
                    else:
                        print(
                            f"[EXECUTION] {family} GridSearchCV: "
                            f"{_gs_combos} param sets × {_gs_folds} folds = {_gs_total_fits} fits"
                            + (" (sequential)" if _gs_njobs == 1 else f" (parallel n_jobs={_gs_njobs})"),
                            flush=True,
                        )
                        searcher = GridSearchCV(est, param_grid=params, scoring=metric, cv=cv, n_jobs=_gs_njobs)
                        searcher.fit(_xt, y_train)
                        best_estimator = searcher.best_estimator_
                        best_params = dict(searcher.best_params_)
                        cv_score = float(searcher.best_score_)
                        fold_variance = float(searcher.cv_results_["std_test_score"][searcher.best_index_] ** 2)
                        print(f"[EXECUTION] {family} GridSearchCV best CV score (primary metric): {cv_score:.6f}", flush=True)
                else:
                    _no_param_scores = cross_val_score(est, _xt, y_train, scoring=metric, cv=cv, n_jobs=1)
                    cv_score = float(np.mean(_no_param_scores))
                    fold_variance = float(np.var(_no_param_scores))
                    est.fit(_xt, y_train)
                    best_estimator = est
                    best_params = {}
                    print(f"[EXECUTION] {family} no-param CV score: {cv_score:.6f}", flush=True)
            elif search_name in ("RandomizedSearchCV", "large_dataset_fallback"):
                if params:
                    import math
                    total_combos = math.prod(len(v) if isinstance(v, list) else 10 for v in params.values())
                    # All folds are now sequential (searcher n_jobs=1 for every model).
                    # Cap n_iter per family so HPO stays within a reasonable time budget.
                    if family == "CatBoost":
                        n_iter = min(20, total_combos)  # HPO on 5k subsample → ~7s/fit
                    elif family == "GradientBoosting":
                        n_iter = min(15, total_combos)  # HPO on 5k subsample → ~5s/fit
                    elif n_rows >= 30000:
                        n_iter = min(20, total_combos)
                    elif n_rows >= 10000:
                        n_iter = min(30, total_combos)
                    else:
                        n_iter = min(40, max(15, total_combos // 2))

                    # HPO subsampling: run search on a subsample, then refit winner on full data.
                    # Per-family thresholds reflect per-fit speed in fully sequential mode:
                    #   CatBoost         ~7s/fit at 10k rows  → cap at 5k
                    #   GradientBoosting ~5s/fit at 10k rows  → cap at 5k
                    #   LightGBM         ~1s/fit at 10k rows  → cap at 10k
                    #   Others           <0.5s/fit             → cap at 20k
                    _HPO_THRESHOLDS = {"CatBoost": 5_000, "GradientBoosting": 5_000, "LightGBM": 10_000}
                    _HPO_SIZES      = {"CatBoost": 5_000, "GradientBoosting": 5_000, "LightGBM": 10_000}
                    # At 30k+ rows, CPU-bound models (RandomForest, ExtraTrees) hold
                    # n_jobs × n_estimators trees in RAM simultaneously. Cap HPO at 10k
                    # and refit at 20k to prevent OOM on laptops.
                    _default_threshold = 10_000 if n_rows >= 30_000 else 20_000
                    _default_size      = 10_000 if n_rows >= 30_000 else 20_000
                    _hpo_threshold = _HPO_THRESHOLDS.get(family, _default_threshold)
                    _hpo_size      = _HPO_SIZES.get(family, _default_size)
                    _used_hpo_subsample = False
                    if n_rows > _hpo_threshold:
                        _hpo_idx = np.random.default_rng(42).choice(
                            len(_xt), size=min(_hpo_size, len(_xt)), replace=False
                        )
                        if hasattr(_xt, "iloc"):
                            _xt_hpo = _xt.iloc[_hpo_idx]
                        else:
                            _xt_hpo = _xt[_hpo_idx]
                        _y_hpo = y_train.iloc[_hpo_idx] if hasattr(y_train, "iloc") else y_train[_hpo_idx]
                        print(f"[EXECUTION] {family} HPO subsample: {len(_hpo_idx)}/{n_rows} rows for search", flush=True)
                        _used_hpo_subsample = True
                    else:
                        _xt_hpo, _y_hpo = _xt, y_train

                    # Optuna TPE — smarter than random sampling; boosting models
                    # use early stopping per trial so each trial is much faster.
                    _opt_params, _opt_score = _optuna_hpo(
                        family, params, _xt_hpo, _y_hpo, cv, metric, n_iter
                    )

                    _RF_FAMILIES = {"RandomForest", "ExtraTrees"}
                    _ES_FAMILIES = {"XGBoost", "LightGBM", "CatBoost"}
                    _refit_cap   = 20_000 if (family in _RF_FAMILIES and n_rows >= 30_000) else None

                    if _opt_params is not None:
                        # ── Optuna succeeded ────────────────────────────────────
                        best_params   = _opt_params
                        cv_score      = _opt_score
                        fold_variance = 0.0
                        print(f"[EXECUTION] {family} Optuna TPE ({n_iter} trials) best score: {cv_score:.6f}", flush=True)

                        best_estimator = _build_estimator(family, task_type, n_rows=n_rows, class_weight=_class_weight)
                        if best_estimator is not None:
                            try:
                                best_estimator.set_params(**best_params)
                            except Exception:
                                pass
                            if family in _ES_FAMILIES:
                                # Refit with early stopping on 90/10 split of full data
                                _r_size = min(len(_xt), _refit_cap or len(_xt))
                                _ri = np.random.default_rng(43).choice(len(_xt), size=_r_size, replace=False)
                                _xt_r = _xt.iloc[_ri]    if hasattr(_xt, "iloc")    else _xt[_ri]
                                _y_r  = y_train.iloc[_ri] if hasattr(y_train, "iloc") else y_train[_ri]
                                from sklearn.model_selection import train_test_split as _tts_r
                                _strat_r = _y_r if task_type == "classification" else None
                                try:
                                    _Xtr_r, _Xval_r, _ytr_r, _yval_r = _tts_r(
                                        _xt_r, _y_r, test_size=0.1, random_state=42, stratify=_strat_r
                                    )
                                except ValueError:
                                    _Xtr_r, _Xval_r, _ytr_r, _yval_r = _tts_r(
                                        _xt_r, _y_r, test_size=0.1, random_state=42
                                    )
                                fit_info = {"mode": "early_stopping", "rows": int(_r_size), "seed": 43,
                                            "val_size": 0.1, "split_seed": 42}
                                if family == "XGBoost":
                                    best_estimator.set_params(n_estimators=1000, early_stopping_rounds=50)
                                    best_estimator.fit(_Xtr_r, _ytr_r, eval_set=[(_Xval_r, _yval_r)], verbose=False)
                                elif family == "LightGBM":
                                    import lightgbm as _lgb_r
                                    best_estimator.set_params(n_estimators=1000)
                                    best_estimator.fit(
                                        _Xtr_r, _ytr_r,
                                        eval_set=[(_Xval_r, _yval_r)],
                                        callbacks=[_lgb_r.early_stopping(50, verbose=False), _lgb_r.log_evaluation(-1)],
                                    )
                                elif family == "CatBoost":
                                    best_estimator.set_params(iterations=1000, early_stopping_rounds=50)
                                    best_estimator.fit(_Xtr_r, _ytr_r, eval_set=(_Xval_r, _yval_r), verbose=0)
                                print(f"[EXECUTION] {family} refit with early stopping on {len(_xt_r)} rows", flush=True)
                            elif _used_hpo_subsample:
                                if _refit_cap and len(_xt) > _refit_cap:
                                    _refit_idx = np.random.default_rng(42).choice(len(_xt), size=_refit_cap, replace=False)
                                    _xt_refit  = _xt.iloc[_refit_idx]    if hasattr(_xt, "iloc")    else _xt[_refit_idx]
                                    _y_refit   = y_train.iloc[_refit_idx] if hasattr(y_train, "iloc") else y_train[_refit_idx]
                                    best_estimator.fit(_xt_refit, _y_refit)
                                    fit_info = {"mode": "capped", "rows": int(_refit_cap), "seed": 42}
                                    print(f"[EXECUTION] {family} refit capped at {_refit_cap}/{n_rows} rows (RAM guard)", flush=True)
                                else:
                                    best_estimator.fit(_xt, y_train)
                                    print(f"[EXECUTION] {family} refit on full {n_rows} rows after HPO subsample", flush=True)
                            else:
                                best_estimator.fit(_xt, y_train)
                    else:
                        # ── Optuna unavailable — fall back to RandomizedSearchCV ─
                        _searcher_njobs = 1
                        _n_folds = cv.get_n_splits(_xt_hpo, _y_hpo) if hasattr(cv, "get_n_splits") else 5
                        print(
                            f"[EXECUTION] {family} RandomizedSearchCV (fallback): "
                            f"{n_iter} configs × {_n_folds} folds = {n_iter * _n_folds} fits",
                            flush=True,
                        )
                        searcher = RandomizedSearchCV(
                            est, param_distributions=params, n_iter=n_iter,
                            scoring=metric, cv=cv, n_jobs=_searcher_njobs, random_state=42,
                        )
                        searcher.fit(_xt_hpo, _y_hpo)
                        best_params   = dict(searcher.best_params_)
                        cv_score      = float(searcher.best_score_)
                        fold_variance = float(searcher.cv_results_["std_test_score"][searcher.best_index_] ** 2)

                        if _used_hpo_subsample:
                            best_estimator = _build_estimator(family, task_type, n_rows=n_rows)
                            if best_estimator is not None:
                                try:
                                    best_estimator.set_params(**best_params)
                                except Exception:
                                    pass
                                if _refit_cap and len(_xt) > _refit_cap:
                                    _refit_idx = np.random.default_rng(42).choice(len(_xt), size=_refit_cap, replace=False)
                                    _xt_refit  = _xt.iloc[_refit_idx]    if hasattr(_xt, "iloc")    else _xt[_refit_idx]
                                    _y_refit   = y_train.iloc[_refit_idx] if hasattr(y_train, "iloc") else y_train[_refit_idx]
                                    best_estimator.fit(_xt_refit, _y_refit)
                                    fit_info = {"mode": "capped", "rows": int(_refit_cap), "seed": 42}
                                    print(f"[EXECUTION] {family} refit capped at {_refit_cap}/{n_rows} rows (RAM guard)", flush=True)
                                else:
                                    best_estimator.fit(_xt, y_train)
                                    print(f"[EXECUTION] {family} refit on full {n_rows} rows after HPO subsample", flush=True)
                            else:
                                best_estimator = searcher.best_estimator_
                        else:
                            best_estimator = searcher.best_estimator_
                        print(f"[EXECUTION] {family} RandomizedSearchCV best CV score: {cv_score:.6f}", flush=True)
                else:
                    _no_param_scores = cross_val_score(est, _xt, y_train, scoring=metric, cv=cv, n_jobs=1)
                    cv_score = float(np.mean(_no_param_scores))
                    fold_variance = float(np.var(_no_param_scores))
                    est.fit(_xt, y_train)
                    best_estimator = est
                    best_params = {}
                    print(f"[EXECUTION] {family} no-param CV score: {cv_score:.6f}")

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

            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
                y_pred = best_estimator.predict(_xte)
            if _log_target:
                y_pred = np.expm1(y_pred)  # inverse-transform from log space
            y_proba = None
            if task_type == "classification" and hasattr(best_estimator, "predict_proba"):
                try:
                    with _w.catch_warnings():
                        _w.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
                        y_proba = best_estimator.predict_proba(_xte)
                except Exception:
                    y_proba = None
            test_score = _test_score(metric, task_type, _y_test_orig, y_pred, y_proba)
            full_metrics = _compute_full_metrics(task_type, _y_test_orig, y_pred, y_proba)

            # Leakage is suspected only when the model was ALSO near-perfect on
            # the 20% subsample (Round 1). A model that scored reasonably in
            # Round 1 but reaches 1.0 on full data may genuinely benefit from
            # more training data — that is not leakage. Scoring 1.0 on a tiny
            # subsample (e.g., 183 rows) is the stronger signal.
            # If Round 1 was skipped (< 3 models), fall back to score-only check.
            score_near_perfect = (
                isinstance(test_score, (int, float))
                and float(test_score) >= config.NEAR_PERFECT_SCORE_SENTINEL
            )
            subsample_also_perfect = family in _perfect_on_subsample
            # When SMOTE was applied, the training set is synthetically balanced so
            # all models will score near-perfect on the oversampled subsample — that
            # is expected, not leakage. Only flag leakage when SMOTE was NOT used.
            leakage_suspected = score_near_perfect and not _smote_was_applied and (
                subsample_also_perfect or not subsample_results
            )
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
                        f"test_score={test_score:.6f} >= {config.NEAR_PERFECT_SCORE_SENTINEL}. "
                        "Model excluded from best-model recommendation. "
                        "Re-check for features that are transformations of the target."
                    ),
                )

            # Warn when R² is negative (model worse than predicting mean)
            if metric == "r2" and isinstance(test_score, float) and test_score < 0:
                logger.warning("Model %s achieved R2=%.4f (negative: worse than predicting mean).", family, test_score)
                log_decision(
                    session,
                    step="Phase 5 - Execution",
                    action=f"negative_r2:{family}",
                    rationale=(
                        f"R²={test_score:.4f} < 0 — model is worse than a constant predictor. "
                        "Likely cause: preprocessing failure, wrong task type, or severe outliers."
                    ),
                )

            elapsed = float(time.perf_counter() - start)
            display_score = _to_display_score(test_score, metric)
            cv_display_score = _to_display_score(cv_score if cv_score != float("-inf") else None, metric)

            log_decision(
                session,
                step="Phase 5 - Execution",
                action=f"model_trained:{family}",
                rationale=f"test_score={test_score:.6f}, cv_score={cv_score:.6f}, time_s={elapsed:.2f}",
                test_score=test_score,
                cv_score=cv_score,
                training_time_s=elapsed,
            )
            print(f"[EXECUTION] {family} test_score={test_score:.6f} display={display_score} (metric={metric})")
            return {
                "model_name": family,
                "cv_score": cv_score,
                "test_score": test_score,
                "cv_display_score": cv_display_score,
                "test_display_score": display_score,
                "training_time": elapsed,
                "fold_variance": fold_variance,
                "status": "ok",
                "best_params": best_params,
                "leakage_suspected": leakage_suspected,
                "detailed_metrics": full_metrics,
                "_fitted_estimator": best_estimator,
                "model_spec": {
                    "kind": "single",
                    "family": family,
                    "estimator": _estimator_spec(best_estimator),
                    "fit": fit_info,
                    "impute_median": bool(_has_nan and family in _NAN_INTOLERANT_FAMILIES),
                },
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
        if _build_estimator(family, task_type, n_rows=n_rows) is None:
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

    # Guarantee RandomForest for small-medium datasets — reliable baseline the LLM sometimes omits.
    _rf_family = "RandomForest"
    if _rf_family not in available_families and n_rows <= 5000:
        if _build_estimator(_rf_family, task_type, n_rows=n_rows) is not None:
            available_families.insert(0, _rf_family)
            log_decision(
                session,
                step="Phase 5 - Execution",
                action="rf_guaranteed_baseline",
                rationale=(
                    f"RandomForest not in LLM plan; injected as guaranteed baseline "
                    f"(n_rows={n_rows} ≤ 5000 — RF is reliable at this scale)."
                ),
            )
            print(f"[EXECUTION] RandomForest injected as guaranteed baseline (not in LLM plan, n_rows={n_rows})")

    # Guarantee HistGradientBoosting for all sizes — it handles NaN natively, is always fast,
    # and is usually within 1% of XGBoost/LightGBM with far fewer tuning requirements.
    _hgb_family = "HistGradientBoosting"
    if _hgb_family not in available_families:
        if _build_estimator(_hgb_family, task_type, n_rows=n_rows) is not None:
            available_families.append(_hgb_family)
            log_decision(
                session,
                step="Phase 5 - Execution",
                action="hgb_guaranteed_baseline",
                rationale="HistGradientBoosting not in LLM plan; injected — handles NaN natively, always competitive.",
            )
            print(f"[EXECUTION] HistGradientBoosting injected as guaranteed baseline (not in LLM plan)")

    # Guarantee CatBoost when available — strongest on structured/categorical data,
    # handles NaN natively, and often outperforms XGBoost/LightGBM on mixed-type features.
    _cb_family = "CatBoost"
    if _cb_family not in available_families:
        if _build_estimator(_cb_family, task_type, n_rows=n_rows) is not None:
            available_families.append(_cb_family)
            log_decision(
                session,
                step="Phase 5 - Execution",
                action="catboost_guaranteed_baseline",
                rationale="CatBoost not in LLM plan; injected — strongest on structured data with categorical features.",
            )
            print(f"[EXECUTION] CatBoost injected as guaranteed baseline (not in LLM plan)")

    _refresh_session_execution_partial(None, None)

    # ── LLM hyperparameter priors (one call, all surviving families) ──────────
    # The LLM narrows the search grid to a focused [lo, hi] band suited to the
    # dataset size and task — larger grids are kept when the band contains < 2 values.
    if available_families:
        _hpo_priors = _llm_suggest_hpo_priors(
            available_families,
            task_type,
            n_rows,
            int((session.dataset_profile or {}).get("n_columns") or len(feature_cols)),
            session,
        )

    # -----------------------------------------------------------------------
    # Progressive Model Evaluation (successive-halving inspired)
    # When 3+ models are available, run a fast subsample round first to
    # eliminate the weakest candidate before full hyperparameter search.
    # Round 1 (20% data, no tuning) → LLM eliminates worst → Round 2 (100%)
    # Work ratio: 20%×N + 100%×(N-1) vs 100%×N  (saves ~40-50% for N=3)
    # -----------------------------------------------------------------------
    # Cap model list at 7 (LLM proposes up to 5, HGB + CatBoost always guaranteed).
    # Progressive elimination round removes 1 more before full training → max 6 trained.
    if len(available_families) > 7:
        available_families = available_families[:7]

    eliminated_family: str | None = None
    subsample_results: list[dict[str, Any]] = []

    # Skip progressive elimination for small datasets — subsample scores (20% of n_rows) are
    # too noisy to reliably rank models when n_train < 800, causing good models to be dropped early.
    _use_progressive = len(available_families) >= 3 and n_rows >= 800
    if len(available_families) >= 3 and not _use_progressive:
        print(f"[EXECUTION] Progressive evaluation skipped: n_rows={n_rows} < 800 (too small for reliable subsample ranking)")

    if _use_progressive:
        print(f"[EXECUTION] Progressive evaluation: Round 1 subsample screening ({len(available_families)} candidates)")
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
            _sorted_sub = sorted(subsample_results, key=lambda x: x["subsample_score"])
            _worst_score = _sorted_sub[0]["subsample_score"]
            _second_score = _sorted_sub[1]["subsample_score"] if len(_sorted_sub) > 1 else _worst_score
            _gap = _second_score - _worst_score
            _LLM_ELIM_GAP_THRESHOLD = 0.15

            if _gap > _LLM_ELIM_GAP_THRESHOLD:
                # Unambiguous loser — eliminate deterministically, no LLM call needed.
                eliminated_family = _sorted_sub[0]["family"]
                _elim_method = "deterministic"
            else:
                # Close race — use LLM reasoning to consider domain fit.
                eliminated_family = _llm_eliminate_model(subsample_results, task_type, metric, profile_summary)
                _elim_method = "llm"

            print(f"[EXECUTION] Progressive evaluation: {_elim_method} eliminated {eliminated_family!r} "
                  f"after Round 1 (gap={_gap:.4f})")
            log_decision(
                session,
                step="Phase 5 - Execution",
                action="progressive_elimination",
                rationale=(
                    f"{_elim_method.capitalize()} eliminated {eliminated_family!r} after subsample screening "
                    f"(gap={_gap:.4f}). "
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

    # Track which models were already near-perfect on the 20% subsample.
    # A model that scores ≥0.999 on ~183 rows almost certainly has leakage;
    # one that scores lower in Round 1 but 1.0 on full data might be genuine.
    _perfect_on_subsample: set[str] = {
        r["family"] for r in subsample_results
        if isinstance(r.get("subsample_score"), float)
        and r["subsample_score"] >= config.NEAR_PERFECT_SCORE_SENTINEL
    }

    # Round 2 (or plain training when <3 models): full hyperparameter search on survivors
    survivors = [f for f in available_families if f != eliminated_family]
    if eliminated_family:
        print(f"[EXECUTION] Round 2 full training on {len(survivors)} survivors: {survivors}", flush=True)
    for family in survivors:
        print(f"[EXECUTION] Training {family}...", flush=True)
        row = _train_and_evaluate(family)
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

    # Separate leakage-suspected models from clean ones.
    # - clean_rows: models where full score is 1.0 but Round 1 subsample was NOT perfect
    #   (or there were no subsample results) — these might be genuinely perfect.
    # - leaky_rows: scored 1.0 on full data AND 1.0 on the tiny 20% subsample — leakage signal.
    clean_rows = [r for r in valid_rows if not r.get("leakage_suspected")]
    leaky_rows = [r for r in valid_rows if r.get("leakage_suspected")]

    if clean_rows:
        # Prefer models not suspected of leakage
        best_row = max(clean_rows, key=lambda r: float(r["test_score"]))
        if leaky_rows:
            logger.warning(
                "Excluded %d model(s) from best-model selection due to suspected leakage "
                "(perfect score on both subsample and full test): %s",
                len(leaky_rows),
                [r["model_name"] for r in leaky_rows],
            )
    else:
        # All models hit near-perfect scores — either genuine dataset or leakage.
        # Accept the best one and surface a clear warning.
        best_row = max(valid_rows, key=lambda r: float(r["test_score"]))
        logger.warning(
            "All models scored near-perfect. Accepting best model but flagging for review. "
            "Check if the verification layer passed (no leakage detected by pre-training checks)."
        )
        print(
            "[EXECUTION] WARNING: All models near-perfect. "
            "Verify that preprocessing checks passed and no leakage feature exists."
        )
    best_model_name = str(best_row["model_name"])
    if session.task_config is not None:
        session.task_config["best_model"] = best_model_name

    # ── Adaptive Ensemble: soft-voting when no single model clearly dominates ──
    ensemble_row = _build_adaptive_ensemble(
        task_type,
        clean_rows if clean_rows else valid_rows,
        x_train, y_train, x_test, y_test,
        metric, session,
        log_target=_log_target, y_test_orig=_y_test_orig,
        n_rows=n_rows,
    )
    if ensemble_row is not None:
        comparison_table.append(ensemble_row)
        best_row = ensemble_row
        best_model_name = str(ensemble_row["model_name"])
        if session.task_config is not None:
            session.task_config["best_model"] = best_model_name

    # ── Decision-threshold optimisation (recall-priority binary tasks) ────────
    # When user intent flags recall priority (e.g. fraud, medical screening),
    # sweep thresholds on a 20 % holdout carved from x_train to find the value
    # maximising the target metric.  x_test is NEVER touched here.
    _thr_signals = (session.execution_plan or {}).get("constraint_signals") or {}
    if bool(_thr_signals.get("prioritize_recall")) and not best_row.get("is_ensemble"):
        _thr_est = best_row.get("_fitted_estimator")
        if _thr_est is not None:
            _opt_thr, _hold_score = _find_optimal_threshold(
                _thr_est, x_train, y_train, metric, task_type, session
            )
            if _opt_thr is not None and abs(_opt_thr - 0.5) > 0.04:
                import warnings as _w_thr
                with _w_thr.catch_warnings():
                    _w_thr.filterwarnings("ignore")
                    _thr_proba = _thr_est.predict_proba(x_test)
                if (
                    _thr_proba is not None
                    and hasattr(_thr_proba, "ndim")
                    and _thr_proba.ndim == 2
                    and _thr_proba.shape[1] == 2
                ):
                    _thr_pred = (_thr_proba[:, 1] >= _opt_thr).astype(float)
                    _thr_score = _test_score(metric, task_type, _y_test_orig, _thr_pred, _thr_proba)
                    _old_score = float(best_row.get("test_score") or 0)
                    print(
                        f"[EXECUTION] Threshold opt ({best_model_name}): "
                        f"thr={_opt_thr:.2f} test={_thr_score:.6f} vs default={_old_score:.6f}"
                    )
                    if _thr_score > _old_score:
                        _thr_metrics = _compute_full_metrics(
                            task_type, _y_test_orig, _thr_pred, _thr_proba
                        )
                        _thr_name = f"{best_model_name}[thr={_opt_thr:.2f}]"
                        _thr_row = {
                            **best_row,
                            "model_name": _thr_name,
                            "test_score": _thr_score,
                            "test_display_score": _to_display_score(_thr_score, metric),
                            "detailed_metrics": _thr_metrics,
                            "best_params": {
                                **(best_row.get("best_params") or {}),
                                "decision_threshold": _opt_thr,
                            },
                            "model_spec": {"kind": "threshold", "base": best_row.get("model_spec"),
                                           "threshold": float(_opt_thr)},
                        }
                        comparison_table.append(_thr_row)
                        best_row = _thr_row
                        best_model_name = _thr_name
                        if session.task_config is not None:
                            session.task_config["best_model"] = best_model_name
                        log_decision(
                            session, step="Phase 5 - Execution",
                            action="threshold_optimisation",
                            rationale=(
                                f"Recall-priority goal: threshold {_opt_thr:.2f} improves "
                                f"{metric} from {_old_score:.4f} to {_thr_score:.4f}"
                            ),
                        )

    # ── Seed-diversity bagging for best single model ──────────────────────────
    # Retrain the winning gradient booster with 3 different random seeds and
    # average all 4 predictions (original + 3 seeds).  Variance reduction from
    # seed diversity mirrors AutoGluon's 8-fold bagging at a fraction of the cost.
    # Each refit uses the same hyperparams + optimal n_estimators from early stopping.
    _BAG_FAMILIES = {"LightGBM", "XGBoost", "CatBoost", "HistGradientBoosting"}
    if (
        not best_row.get("is_ensemble")
        and best_model_name in _BAG_FAMILIES
        and n_rows <= 80_000
    ):
        _bag_fitted = best_row.get("_fitted_estimator")
        _bag_params = {k: v for k, v in (best_row.get("best_params") or {}).items()
                       if k not in {"early_stopping_rounds", "eval_set", "eval_metric", "callbacks"}}

        # Use the actual tree count found by early stopping, not the search-space limit.
        # Use `or 0` so that None (attribute exists but unset by CatBoost/LightGBM when
        # early stopping was not triggered) falls back to 0 instead of crashing `> 0`.
        if best_model_name == "LightGBM" and _bag_fitted is not None:
            _n_est = (getattr(_bag_fitted, "best_iteration_", 0) or 0)
            if _n_est > 0:
                _bag_params["n_estimators"] = _n_est
        elif best_model_name == "XGBoost" and _bag_fitted is not None:
            _n_est = (getattr(_bag_fitted, "best_ntree_limit", 0) or 0) or (getattr(_bag_fitted, "best_iteration", 0) or 0)
            if _n_est > 0:
                _bag_params["n_estimators"] = _n_est
        elif best_model_name == "CatBoost" and _bag_fitted is not None:
            _n_est = (getattr(_bag_fitted, "best_iteration_", 0) or 0)
            if _n_est > 0:
                _bag_params["iterations"] = _n_est + 1
        elif best_model_name == "HistGradientBoosting" and _bag_fitted is not None:
            _n_est = (getattr(_bag_fitted, "n_iter_", 0) or 0)
            if _n_est > 0:
                _bag_params["max_iter"] = _n_est

        # Collect predictions in log space (expm1 applied after averaging below).
        _bag_preds: list[np.ndarray] = []
        _bag_probas: list[np.ndarray | None] = []  # aligned with _bag_preds
        _bag_member_spec: dict[str, Any] | None = None
        _bag_seeds_ok: list[int] = []

        def _bag_proba(est: Any) -> np.ndarray | None:
            if task_type != "classification" or not hasattr(est, "predict_proba"):
                return None
            try:
                return est.predict_proba(x_test)
            except Exception:
                return None

        if _bag_fitted is not None:
            try:
                _bag_preds.append(_bag_fitted.predict(x_test).astype(float))
                _bag_probas.append(_bag_proba(_bag_fitted))
            except Exception:
                pass

        for _seed in [123, 456, 789]:
            try:
                _b = _build_estimator(best_model_name, task_type, n_rows=n_rows)
                if _b is None:
                    continue
                _sp = dict(_bag_params)
                _sp["random_seed" if best_model_name == "CatBoost" else "random_state"] = _seed
                _b.set_params(**_sp)
                _b.fit(x_train, y_train)
                _bag_preds.append(_b.predict(x_test).astype(float))
                _bag_probas.append(_bag_proba(_b))
                _bag_seeds_ok.append(_seed)
                if _bag_member_spec is None:
                    _bag_member_spec = _estimator_spec(_b)
            except Exception as _be:
                logger.debug("Seed bagging seed=%d failed: %s", _seed, _be)

        if len(_bag_preds) >= 2:
            _bag_y_proba = None
            if task_type == "classification":
                # Soft vote over seed probabilities (majority vote if any model lacks them)
                _bag_y, _bag_y_proba = combine_class_votes(_bag_preds, _bag_probas)
                _bag_y = _bag_y.astype(float)
            else:
                _bag_avg = np.mean(_bag_preds, axis=0)
                _bag_y = np.expm1(_bag_avg) if _log_target else _bag_avg
            try:
                _bag_score = _test_score(metric, task_type, _y_test_orig, _bag_y, _bag_y_proba)
            except Exception:
                _bag_score = None
            _bag_label = f"Bag({best_model_name}×{len(_bag_preds)})"
            if _bag_score is not None and _bag_score > float(best_row["test_score"]):
                print(
                    f"[EXECUTION] {_bag_label}: "
                    f"{float(best_row['test_score']):.6f} → {_bag_score:.6f} "
                    f"(+{_bag_score - float(best_row['test_score']):.6f})"
                )
                log_decision(session, step="Phase 5 - Execution", action="seed_bagging",
                             rationale=f"{_bag_label} score={_bag_score:.6f} > single {best_model_name}={float(best_row['test_score']):.6f}")
                _bag_row = {
                    "model_name": _bag_label,
                    "cv_score": best_row.get("cv_score"),
                    "test_score": _bag_score,
                    "cv_display_score": best_row.get("cv_display_score"),
                    "test_display_score": _to_display_score(_bag_score, metric),
                    "training_time": float(best_row.get("training_time", 0)),
                    "fold_variance": None,
                    "status": "ok",
                    "best_params": {**_bag_params, "n_seeds": len(_bag_preds)},
                    "leakage_suspected": False,
                    "detailed_metrics": _compute_full_metrics(task_type, _y_test_orig, _bag_y, _bag_y_proba),
                    "_fitted_estimator": _bag_fitted,
                    "is_ensemble": True,
                    "model_spec": {
                        "kind": "bag",
                        "base": best_row.get("model_spec"),
                        "member": _bag_member_spec,
                        "seed_key": "random_seed" if best_model_name == "CatBoost" else "random_state",
                        "seeds": _bag_seeds_ok,
                    },
                }
                comparison_table.append(_bag_row)
                best_row = _bag_row
                best_model_name = _bag_label
                if session.task_config is not None:
                    session.task_config["best_model"] = best_model_name
            elif _bag_score is not None:
                print(f"[EXECUTION] {_bag_label}: {_bag_score:.6f} — no improvement over {float(best_row['test_score']):.6f}")

    # ── Feature Importance ────────────────────────────────────────────────────
    _fi_estimator = best_row.get("_fitted_estimator")
    feature_importances = (
        _extract_feature_importances(_fi_estimator, list(x_train.columns))
        if _fi_estimator is not None else {}
    )
    session.feature_importances = feature_importances
    if feature_importances:
        _top5 = [n for n, _ in feature_importances.get("top_features", [])[:5]]
        log_decision(session, step="Phase 5 - Execution", action="feature_importance_computed",
                     method=feature_importances.get("method"), top_5=_top5)
        print(f"[EXECUTION] Feature importances ({feature_importances['method']}): top-5 = {_top5}")
    else:
        print(f"[EXECUTION] Feature importances not available for {best_model_name} (KNN/SVM/MLP)")

    # ── SHAP explainability (Analytics page) ─────────────────────────────────
    _shap_family = best_model_name.split("[")[0].split("(")[0]  # strip threshold/ensemble suffixes
    if best_model_name.startswith("Bag("):
        # Seed-bagged model: _fi_estimator is the original fitted base model, so the
        # base family's exact explainer applies (e.g. "Bag(CatBoost×4)" → CatBoost).
        _shap_family = best_model_name[len("Bag("):].split("×")[0]
    shap_summary = (
        compute_shap_summary(_fi_estimator, x_train, x_test, task_type, _shap_family)
        if _fi_estimator is not None else {"available": False, "reason": "no_fitted_estimator"}
    )
    session.shap_summary = shap_summary
    if shap_summary.get("available"):
        _top3 = [f["name"] for f in shap_summary["features"][:3]]
        print(f"[EXECUTION] SHAP ({shap_summary['method']}): top-3 impact = {_top3}")
        log_decision(session, step="Phase 5 - Execution", action="shap_computed",
                     method=shap_summary["method"], top_3=_top3)
    else:
        print(f"[EXECUTION] SHAP unavailable: {shap_summary.get('reason')}")

    # ── LLM Elimination Accuracy (Gap 3 research metric) ─────────────────────
    # After full training, check whether the LLM's Round-1 elimination was correct.
    # "Correct" = the eliminated model would NOT have been the best on full data.
    # We compare its subsample score rank vs all survivors' full test scores.
    elimination_audit: dict[str, Any] = {}
    if eliminated_family and subsample_results:
        elim_sub = next((r["subsample_score"] for r in subsample_results if r["family"] == eliminated_family), None)
        survivor_full_scores = {
            r["model_name"]: float(r["test_score"])
            for r in valid_rows
            if r.get("model_name") and r["model_name"] != eliminated_family
        }
        best_survivor_score = max(survivor_full_scores.values()) if survivor_full_scores else None

        # "Would the eliminated model have won?" — we can't know for sure without training it,
        # but we can check if its subsample score was the worst (i.e., elimination was clear)
        sub_scores_all = {r["family"]: r["subsample_score"] for r in subsample_results}
        was_worst_sub = elim_sub == min(sub_scores_all.values()) if elim_sub is not None else None

        # Rank of eliminated model by subsample score (1 = worst = correctly eliminated)
        sub_ranks = sorted(sub_scores_all.items(), key=lambda x: x[1])
        elim_rank = next((i + 1 for i, (f, _) in enumerate(sub_ranks) if f == eliminated_family), None)

        elimination_correct = was_worst_sub  # True → LLM eliminated the weakest subsample scorer
        elimination_audit = {
            "eliminated_family": eliminated_family,
            "eliminated_subsample_score": round(elim_sub, 4) if elim_sub is not None else None,
            "subsample_rank_of_eliminated": elim_rank,  # 1 = worst → correct elimination
            "total_candidates": len(sub_scores_all),
            "was_worst_subsample_scorer": was_worst_sub,
            "elimination_correct": elimination_correct,
            "best_survivor_test_score": round(best_survivor_score, 4) if best_survivor_score is not None else None,
            "all_subsample_scores": {f: round(s, 4) for f, s in sub_scores_all.items()},
        }
        log_decision(
            session,
            step="Phase 5 - Execution",
            action="elimination_audit",
            rationale=(
                f"LLM eliminated {eliminated_family!r} (subsample rank {elim_rank}/{len(sub_scores_all)}, "
                f"score={elim_sub:.4f}). Correct={elimination_correct}. "
                f"Best survivor test score={f'{best_survivor_score:.4f}' if best_survivor_score is not None else 'N/A'}."
            ),
        )
        print(
            f"[EXECUTION] Elimination audit: {eliminated_family} was rank {elim_rank}/{len(sub_scores_all)} "
            f"by subsample — elimination_correct={elimination_correct}"
        )

    # ── Optional AutoML framework benchmarks ──────────────────────────────────
    benchmark_time = int(config.AUTOML_BENCHMARK_TIME_LIMIT)
    if benchmark_time > 0:
        logger.info("Running AutoML framework benchmarks (time_limit=%ds each)…", benchmark_time)
        print(f"[EXECUTION] AutoML benchmarks: time_limit={benchmark_time}s each")
        _benchmark_runners = [
            (_benchmark_autogluon, {"label_col": f"__target_{target}__"}),
            (_benchmark_flaml,     {}),
        ]
        for _runner, _kwargs in _benchmark_runners:
            # Use original-scale y_train/y_test for framework benchmarks.
            # If log-transform was applied, y_train is in log-space; the external
            # frameworks (AutoGluon/FLAML) must train on original values so their
            # test scores are comparable on the same scale as ours.
            _brow = _runner(
                task_type=task_type,
                x_train=_x_train_for_benchmarks, y_train=_y_train_for_benchmarks,
                x_test=x_test, y_test=_y_test_orig,
                metric=metric, time_limit=benchmark_time,
                **_kwargs,
            )
            comparison_table.append(_brow)
            _status = _brow.get("status", "")
            if _status == "ok":
                print(f"[EXECUTION] {_brow['model_name']} benchmark: test_score={_brow['test_score']}")
            elif _status.startswith("skipped"):
                print(f"[EXECUTION] {_brow['model_name']} not installed - skipped")
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

    # Annotate comparison table with best-model flag and display scores
    best_names = {best_model_name}
    for row in comparison_table:
        row["is_best"] = row.get("model_name") in best_names
        if "cv_display_score" not in row:
            row["cv_display_score"] = _to_display_score(row.get("cv_score"), metric)
        if "test_display_score" not in row:
            row["test_display_score"] = _to_display_score(row.get("test_score"), metric)

    # Extract competitor scores from comparison_table for the threshold-check node.
    # These are used to decide whether reiteration is worth running.
    _framework_rows = [
        r for r in comparison_table
        if r.get("source") == _FRAMEWORK_SOURCE and r.get("status") == "ok"
        and isinstance(r.get("test_score"), (int, float))
    ]
    competitor_scores = {r["model_name"]: float(r["test_score"]) for r in _framework_rows}
    best_competitor_score = max(competitor_scores.values()) if competitor_scores else None

    results = {
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "metric": metric,
        "metric_display": _humanize_metric(metric),
        "metric_is_lower_better": _is_neg_metric(metric),
        "task_type": task_type,
        "search_strategy": search_name,
        "cv_design": plan.get("cv_design") or {},
        "split_meta": split["split_meta"],
        "interpretation": interpretation + leakage_note,
        "best_test_score": float(best_row["test_score"]),
        "best_model_metrics": best_full_metrics,
        "status": "ok",
        "leakage_warning_models": leakage_models,
        "near_perfect_sentinel": config.NEAR_PERFECT_SCORE_SENTINEL,
        "elimination_audit": elimination_audit,
        "feature_importances": feature_importances,
        "shap_summary": shap_summary,
        "competitor_scores": competitor_scores,
        "best_competitor_score": best_competitor_score,
        "models_trained": survivors,
        "replay": {
            "smote": _smote_spec,
            "log_target": bool(_log_target),
            "has_nan": bool(_has_nan),
            "feature_columns": list(feature_cols),
            "model": best_row.get("model_spec"),
        },
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
