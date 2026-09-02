"""Compute dataset_profile dict from a DataFrame (Phase 1 Step 4)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from automl_system import config

logger = logging.getLogger(__name__)

# Columns beyond this threshold get summary stats only (not full stats)
_FULL_STATS_COL_LIMIT = 100
# Fraction of values that must successfully cast to float to reclassify object→numeric
_NUMERIC_CAST_THRESHOLD = 0.95
# Max cardinality for value counts to be collected on categoricals
_CATEGORICAL_CARD_LIMIT = 20


def _is_datetime_col(col_name: str, series: pd.Series) -> bool:
    """Heuristic: detect datetime-like columns by name or dtype."""
    try:
        if np.issubdtype(series.dtype, np.datetime64):
            return True
    except TypeError:
        pass  # pandas extension dtypes (StringDtype, etc.) are not numpy dtypes
    name_lower = col_name.lower()
    date_hints = ("date", "time", "timestamp", "datetime", "created", "updated", "year_month")
    if any(h in name_lower for h in date_hints):
        # Only flag if values look like dates when sampled
        sample = series.dropna().head(10)
        if len(sample):
            try:
                pd.to_datetime(sample, errors="raise")
                return True
            except Exception:
                pass
    return False


def _outlier_count_iqr(clean: pd.Series) -> int:
    """Count values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def _profile_column_full(col: str, series: pd.Series, n_rows: int) -> dict:
    """Full statistics for a single column."""
    null_count = int(series.isna().sum())
    null_pct = round(float(null_count / n_rows * 100), 4) if n_rows else 0.0
    nunique = int(series.nunique(dropna=True))
    fully_null = null_count == n_rows and n_rows > 0
    raw_dtype = str(series.dtype)

    if fully_null:
        return {
            "dtype": raw_dtype,
            "inferred_dtype": "fully_null",
            "null_count": null_count,
            "null_pct": null_pct,
            "nunique": nunique,
            "fully_null": True,
            "cardinality_equals_row_count": False,
        }

    # ── Datetime detection ────────────────────────────────────────────────────
    if _is_datetime_col(col, series):
        return {
            "dtype": raw_dtype,
            "inferred_dtype": "datetime",
            "null_count": null_count,
            "null_pct": null_pct,
            "nunique": nunique,
            "fully_null": False,
            "cardinality_equals_row_count": nunique == n_rows and n_rows > 0,
        }

    # ── Numeric column ────────────────────────────────────────────────────────
    is_numeric = pd.api.types.is_numeric_dtype(series)
    original_dtype: str | None = None
    reclassified = False

    if not is_numeric and series.dtype == object:
        # Attempt numeric cast for object columns
        cast = pd.to_numeric(series, errors="coerce")
        success_rate = float(cast.notna().sum()) / max(null_count + cast.notna().sum(), 1)
        non_null_total = int(series.notna().sum())
        if non_null_total > 0:
            success_rate = float(cast.notna().sum()) / non_null_total
        if success_rate >= _NUMERIC_CAST_THRESHOLD:
            original_dtype = raw_dtype
            series = cast
            is_numeric = True
            reclassified = True

    if is_numeric:
        clean = series.dropna()
        result: dict = {
            "dtype": raw_dtype,
            "inferred_dtype": "numeric",
            "null_count": null_count,
            "null_pct": null_pct,
            "nunique": nunique,
            "fully_null": False,
            "cardinality_equals_row_count": nunique == n_rows and n_rows > 0,
        }
        if original_dtype is not None:
            result["original_dtype"] = original_dtype
            result["reclassified_from_string"] = reclassified
        if len(clean) > 2:
            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            result.update({
                "mean": round(float(clean.mean()), 6),
                "median": round(float(clean.median()), 6),
                "std": round(float(clean.std()), 6),
                "min": round(float(clean.min()), 6),
                "max": round(float(clean.max()), 6),
                "skew": round(float(clean.skew()), 6),
                "kurtosis": round(float(clean.kurtosis()), 6),
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "iqr": round(q3 - q1, 6),
                "outlier_count_iqr": _outlier_count_iqr(clean),
            })
        return result

    # ── Categorical column ────────────────────────────────────────────────────
    result = {
        "dtype": raw_dtype,
        "inferred_dtype": "categorical",
        "null_count": null_count,
        "null_pct": null_pct,
        "nunique": nunique,
        "fully_null": False,
        "cardinality_equals_row_count": nunique == n_rows and n_rows > 0,
        "skew": None,
        "value_counts": None,
    }
    if nunique <= _CATEGORICAL_CARD_LIMIT and nunique > 0:
        non_null_count = int(series.notna().sum())
        vc = series.value_counts(dropna=False).head(_CATEGORICAL_CARD_LIMIT)
        result["value_counts"] = {
            str(k): {
                "count": int(v),
                "pct": round(float(v / n_rows * 100), 2) if n_rows else 0.0,
            }
            for k, v in vc.items()
        }
    return result


def _profile_column_summary(col: str, series: pd.Series, n_rows: int) -> dict:
    """Lightweight stats for columns beyond the full-stats threshold."""
    null_count = int(series.isna().sum())
    null_pct = round(float(null_count / n_rows * 100), 4) if n_rows else 0.0
    nunique = int(series.nunique(dropna=True))
    return {
        "dtype": str(series.dtype),
        "inferred_dtype": "summary_only",
        "null_count": null_count,
        "null_pct": null_pct,
        "nunique": nunique,
        "fully_null": null_count == n_rows and n_rows > 0,
        "cardinality_equals_row_count": nunique == n_rows and n_rows > 0,
    }


def compute_dataset_profile(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """
    Phase 1 Step 4: full dataset profiling snapshot.

    For every column computes inferred dtype, null stats, cardinality,
    numeric statistics (mean/median/std/min/max/skew/kurtosis/IQR/outliers),
    categorical value counts with percentages, and datetime flags.

    For >500 columns: full stats for first 100, summary only for the rest.
    Target column: class distribution + imbalance ratio appended.
    """
    n_rows, n_cols = len(df), len(df.columns)
    col_list = list(df.columns)

    if n_cols > 500:
        logger.warning(
            "Dataset has %d columns (>500). Full stats computed for first %d only.",
            n_cols, _FULL_STATS_COL_LIMIT,
        )

    columns: dict[str, dict] = {}
    for i, col in enumerate(col_list):
        series = df[col]
        if i < _FULL_STATS_COL_LIMIT or n_cols <= 500:
            columns[col] = _profile_column_full(col, series, n_rows)
        else:
            columns[col] = _profile_column_summary(col, series, n_rows)

    # ── Target distribution + imbalance ratio ─────────────────────────────────
    target_distribution: dict | None = None
    if target_column and target_column in df.columns:
        s = df[target_column]
        vc = s.value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in vc.items()}
        imbalance_ratio: float | None = None
        if len(vc) >= 2:
            imbalance_ratio = round(float(vc.iloc[0]) / float(vc.iloc[-1]), 4)
        target_distribution = {
            "counts": counts,
            "nunique": int(s.nunique(dropna=True)),
            "imbalance_ratio": imbalance_ratio,
        }

    profile: dict = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns,
        "target_distribution": target_distribution,
        "stub": False,
        "full_stats_columns": min(n_cols, _FULL_STATS_COL_LIMIT) if n_cols > 500 else n_cols,
    }
    profile["difficulty"] = compute_difficulty_score(profile)
    return profile


def compute_difficulty_score(profile: dict) -> dict:
    """
    Compute a dataset difficulty score (0.0 = easy, 1.0 = very hard).

    Components:
    - null_rate     : avg null % across all columns (high nulls → harder cleaning)
    - skew          : avg |skew| of numeric columns (heavy skew → needs transforms)
    - imbalance     : class imbalance ratio mapped to 0-1 (only for classification)
    - dimensionality: feature/row ratio (high dim → harder generalization)
    - cardinality   : fraction of categorical cols with >50 unique values (high card → encoding cost)

    Each component is normalized to [0, 1] and weighted into an overall score.
    Label: easy (<0.3), moderate (0.3–0.55), hard (0.55–0.75), very_hard (>0.75).
    """
    cols = profile.get("columns") or {}
    n_rows = max(int(profile.get("n_rows") or 1), 1)
    n_cols = max(int(profile.get("n_columns") or 1), 1)

    # ── null rate ─────────────────────────────────────────────────────────────
    null_pcts = [float(m.get("null_pct") or 0) for m in cols.values()]
    avg_null = sum(null_pcts) / max(len(null_pcts), 1) / 100.0  # 0-1

    # ── skewness ─────────────────────────────────────────────────────────────
    import math
    skews = [abs(float(m["skew"])) for m in cols.values()
             if m.get("skew") is not None and not math.isnan(float(m["skew"]))]
    avg_skew = min(sum(skews) / max(len(skews), 1) / 5.0, 1.0)  # cap at skew=5

    # ── class imbalance ───────────────────────────────────────────────────────
    td = profile.get("target_distribution") or {}
    imbalance_ratio = float(td.get("imbalance_ratio") or 1.0)
    # ratio=1 → balanced (0), ratio=20+ → very imbalanced (1)
    imbalance_score = min((imbalance_ratio - 1.0) / 19.0, 1.0) if imbalance_ratio >= 1 else 0.0

    # ── dimensionality ────────────────────────────────────────────────────────
    n_features = max(n_cols - 1, 1)
    dim_ratio = min(n_features / n_rows, 1.0)  # features/rows, capped at 1

    # ── high-cardinality categoricals ─────────────────────────────────────────
    cat_cols = [m for m in cols.values() if m.get("inferred_dtype") == "categorical"]
    high_card = sum(1 for m in cat_cols if int(m.get("nunique") or 0) > 50)
    cardinality_score = high_card / max(len(cat_cols), 1) if cat_cols else 0.0

    # ── weighted composite ────────────────────────────────────────────────────
    weights = {
        "null_rate": 0.25,
        "skew": 0.20,
        "imbalance": 0.20,
        "dimensionality": 0.20,
        "cardinality": 0.15,
    }
    components = {
        "null_rate": round(avg_null, 4),
        "skew": round(avg_skew, 4),
        "imbalance": round(imbalance_score, 4),
        "dimensionality": round(dim_ratio, 4),
        "cardinality": round(cardinality_score, 4),
    }
    overall = sum(components[k] * weights[k] for k in weights)
    overall = round(overall, 4)

    if overall < 0.30:
        label = "easy"
    elif overall < 0.55:
        label = "moderate"
    elif overall < 0.75:
        label = "hard"
    else:
        label = "very_hard"

    return {"overall": overall, "label": label, "components": components}
