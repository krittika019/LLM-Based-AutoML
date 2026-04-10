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

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns,
        "target_distribution": target_distribution,
        "stub": False,
        "full_stats_columns": min(n_cols, _FULL_STATS_COL_LIMIT) if n_cols > 500 else n_cols,
    }


def build_semantic_profile_vector(profile: dict) -> list[float]:
    """
    Fixed-length vector for semantic memory retrieval (Planning Mode 1 soft recommendations).
    Padded/truncated to match retrieve_similar expectations across runs.
    """
    dim = min(32, config.MAX_PROFILE_VECTOR_DIM)
    vec = [0.0] * dim
    vec[0] = float(profile.get("n_rows") or 0) / 1e6
    vec[1] = float(profile.get("n_columns") or 0) / 1e3
    cols = profile.get("columns") or {}
    nulls = [float(c.get("null_pct") or 0) for c in cols.values()]
    vec[2] = sum(nulls) / max(len(nulls), 1) / 100.0
    vec[3] = float(len([c for c, m in cols.items() if float(m.get("null_pct") or 0) > 30])) / max(
        len(cols), 1
    )
    vec[4] = float(len([c for c, m in cols.items() if m.get("cardinality_equals_row_count")])) / max(
        len(cols), 1
    )
    skewed = [m.get("skew") for m in cols.values() if m.get("skew") is not None]
    vec[5] = sum(abs(float(x)) for x in skewed) / max(len(skewed), 1) if skewed else 0.0
    return vec[:dim]
