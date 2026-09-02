"""Phase 3 Step 4: deep cleaning — Decision Router pre-fills deterministic assignments;
LLM only handles genuinely ambiguous columns. Single-pass: impute → outlier → encode → scale."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.decision_router import RouterStats, route_columns_batch, columns_needing_llm, build_partial_assignment
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_KNN_ROW_LIMIT = 5_000
_TARGET_ENC_MIN_ROWS = 500
_OHE_CARDINALITY_LIMIT = 50
_WIDE_COLUMNS_WARNING = 500
_LLM_COLUMN_BATCH_SIZE = 35


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "deep_cleaning", **entry})


def _default_assignment(col: str, s: pd.Series) -> dict[str, str]:
    if pd.api.types.is_numeric_dtype(s):
        return {"imputation": "median", "outlier": "leave", "encoding": "none", "scaling": "standard"}
    return {"imputation": "mode", "outlier": "leave", "encoding": "one_hot", "scaling": "none"}


def _llm_assignments(df: pd.DataFrame, session: SessionState, router_stats: RouterStats) -> dict[str, dict[str, str]]:
    """
    Tier 1/2: Decision Router handles deterministic columns.
    Tier 3: LLM only sees ambiguous columns.
    Post-merge: hard scenario checks enforce KNN/encoding size limits.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    n_rows = len(df)

    routes = route_columns_batch(df, target, router_stats)
    ambiguous_cols = columns_needing_llm(routes)

    out: dict[str, dict[str, str]] = {
        col: build_partial_assignment(route)
        for col, route in routes.items()
    }

    if ambiguous_cols:
        cols_meta = []
        for c in ambiguous_cols:
            s = df[c]
            cols_meta.append({
                "column": c,
                "dtype": str(s.dtype),
                "null_pct": float(s.isna().mean() * 100),
                "nunique": int(s.nunique(dropna=True)),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
                "skew": round(float(s.skew()), 3) if pd.api.types.is_numeric_dtype(s) else None,
            })
        client = LLMClient()
        prompt = f"""Assign preprocessing strategy for these AMBIGUOUS columns only.
(Deterministic columns were already handled without LLM.)

TASK: {task}
PLAN: {session.preprocessing_plan}
DATASET_ROWS: {n_rows}
TARGET: {target!r}

COLUMNS_NEEDING_JUDGMENT ({len(ambiguous_cols)} of {len(routes)} total):
{cols_meta}

Rules:
- KNN only for numeric with correlated neighbours; NOT if n_rows > {_KNN_ROW_LIMIT}
- Target encoding for high-cardinality categoricals (11–100 unique) only if n_rows >= {_TARGET_ENC_MIN_ROWS}
- Robust scaling for numeric columns with |skew| > 2
- Never assign any strategy to the target column {target!r}

Return JSON: {{"columns": {{"col_name": {{
  "imputation": "mean"|"median"|"mode"|"knn"|"flag_as_missing",
  "outlier": "iqr_cap"|"log"|"leave",
  "encoding": "ordinal"|"one_hot"|"target"|"binary"|"none",
  "scaling": "standard"|"minmax"|"robust"|"none"
}}}}}}"""

        raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
        parsed = extract_json_object(raw) if raw else None
        if parsed and isinstance(parsed.get("columns"), dict):
            for k, v in parsed["columns"].items():
                k = str(k)
                if k in ambiguous_cols and k in out and isinstance(v, dict):
                    route = routes.get(k)
                    merged = dict(out[k])
                    if route and route.imputation is None and v.get("imputation"):
                        merged["imputation"] = str(v["imputation"])
                    if route and route.outlier is None and v.get("outlier"):
                        merged["outlier"] = str(v["outlier"])
                    if route and route.encoding is None and v.get("encoding"):
                        merged["encoding"] = str(v["encoding"])
                    if v.get("scaling"):
                        merged["scaling"] = str(v["scaling"])
                    out[k] = merged

    # Ensure all non-target columns have a complete assignment
    for c in df.columns:
        if str(c) == str(target):
            continue
        if c not in out:
            out[c] = _default_assignment(c, df[c])

    # ── Hard scenario checks after merge ──────────────────────────────────────
    deviations: list[dict[str, Any]] = []
    for col, a in out.items():
        s = df[col] if col in df.columns else None

        if a.get("imputation") == "knn" and n_rows > _KNN_ROW_LIMIT:
            a["imputation"] = "median"
            deviations.append({"column": col, "deviation": "knn_downgrade",
                                "reason": f"n_rows={n_rows} > {_KNN_ROW_LIMIT}"})

        if a.get("encoding") == "target" and n_rows < _TARGET_ENC_MIN_ROWS:
            a["encoding"] = "one_hot"
            deviations.append({"column": col, "deviation": "target_enc_too_small",
                                "reason": f"n_rows={n_rows} < {_TARGET_ENC_MIN_ROWS}"})

        if a.get("encoding") == "one_hot" and s is not None and not pd.api.types.is_numeric_dtype(s):
            nu = int(s.nunique(dropna=True))
            if nu > _OHE_CARDINALITY_LIMIT:
                a["encoding"] = "target" if n_rows >= _TARGET_ENC_MIN_ROWS else "ordinal"
                deviations.append({"column": col, "deviation": "ohe_cardinality_override",
                                    "reason": f"nunique={nu} > {_OHE_CARDINALITY_LIMIT}"})

    if deviations:
        _log(session, {"action": "assignment_deviations", "deviations": deviations})
        logger.info("deep_clean: %d assignment deviation(s) applied", len(deviations))
        # Propagate to session so Phase 4 model selection can adapt
        existing = list(session.preprocessing_deviations or [])
        for d in deviations:
            existing.append({"type": d["deviation"], "column": d["column"], "reason": d["reason"]})
        session.preprocessing_deviations = existing

    return out


def _impute_numeric(s: pd.Series, imp: str) -> pd.Series:
    out = s.astype(float).replace([np.inf, -np.inf], np.nan)
    fill = float(out.mean()) if imp == "mean" else float(out.median())
    return out.fillna(fill)


def _fit_scaler(sc: str):
    if sc == "standard":
        return StandardScaler()
    if sc == "minmax":
        return MinMaxScaler()
    if sc == "robust":
        return RobustScaler()
    return None


def _apply_categorical(
    col: str,
    df: pd.DataFrame,
    assignment: dict[str, str],
    target: str | None,
    df_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fits the assigned categorical strategy on `df` (train) and, when
    `df_test` is given, applies the identical fitted transform (same mode
    fill value / encoder / one-hot vocabulary) to it — never refits on test,
    which would leak test-set category/target information into the encoding."""
    s = df[col].astype(str).replace("nan", np.nan)
    s_test = df_test[col].astype(str).replace("nan", np.nan) if df_test is not None and col in df_test.columns else None

    imp = assignment.get("imputation", "mode")
    if imp == "mode":
        mode = s.mode()
        fill = mode.iloc[0] if len(mode) else "missing"
    else:
        fill = "missing"
    s = s.fillna(fill)
    if s_test is not None:
        s_test = s_test.fillna(fill)

    enc = assignment.get("encoding", "one_hot")

    if enc == "binary":
        unique_vals = s.dropna().unique()
        if len(unique_vals) == 2:
            train_out = pd.DataFrame({col: (s == unique_vals[0]).astype(float)}, index=df.index)
            test_out = (
                pd.DataFrame({col: (s_test == unique_vals[0]).astype(float)}, index=s_test.index)
                if s_test is not None else None
            )
            return train_out, test_out
        # Fall through to one_hot if not truly binary

    if enc == "ordinal":
        try:
            oe = OrdinalEncoder(handle_unknown="value", handle_missing="value")
            v = oe.fit_transform(s.to_frame(name=col))
            train_out = pd.DataFrame({col: v[col].astype(float)}, index=df.index)
            test_out = None
            if s_test is not None:
                vt = oe.transform(s_test.to_frame(name=col))
                test_out = pd.DataFrame({col: vt[col].astype(float)}, index=s_test.index)
            return train_out, test_out
        except Exception:
            codes, uniques = pd.factorize(s, sort=True)
            train_out = pd.DataFrame({col: codes.astype(float)}, index=df.index)
            test_out = None
            if s_test is not None:
                mapping = {v: i for i, v in enumerate(uniques)}
                test_codes = s_test.map(mapping).astype(float)
                test_out = pd.DataFrame({col: test_codes.fillna(-1.0)}, index=s_test.index)
            return train_out, test_out

    if enc == "target" and target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        # LeaveOneOutEncoder: each row's encoding excludes its own target value,
        # avoiding the direct leakage that TargetEncoder.fit_transform causes on full data.
        # Fit on train only; test rows use .transform() (the library's standard,
        # non-leaking way to score held-out data — no y needed at transform time).
        te = LeaveOneOutEncoder(sigma=0.0)
        z = te.fit_transform(s.to_frame(), df[target])
        train_out = pd.DataFrame({f"{col}_te": z.values.ravel()}, index=df.index)
        test_out = None
        if s_test is not None:
            zt = te.transform(s_test.to_frame())
            test_out = pd.DataFrame({f"{col}_te": zt.values.ravel()}, index=s_test.index)
        return train_out, test_out

    train_dummies = pd.get_dummies(s, prefix=col, dtype=float)
    test_out = None
    if s_test is not None:
        test_dummies = pd.get_dummies(s_test, prefix=col, dtype=float)
        test_out = test_dummies.reindex(columns=train_dummies.columns, fill_value=0.0)
    return train_dummies, test_out


def deep_clean(
    df: pd.DataFrame, session: SessionState, df_test: pd.DataFrame | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decision Router pre-fills deterministic column assignments (rule/stat tier);
    LLM only sees genuinely ambiguous columns. Then execute imputation, outlier
    handling, encoding, and scaling in one pipeline pass.

    Every stateful transform (imputed fill values, IQR bounds, scaler
    parameters, encoder mappings) is fit on `df` (train) only. When `df_test`
    is given, the identical fitted transform — never refit — is applied to it
    too, and `(out_train, out_test)` is returned instead of a single frame.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if target_s and target_s not in df.columns:
        target_s = None

    router_stats = RouterStats()
    assignments = _llm_assignments(df, session, router_stats)
    session.deep_cleaning_assignments = assignments

    router_msg = router_stats.as_message()
    router_summary = router_stats.summary()
    _log(session, {"action": "assignments_recorded", "n_columns": len(assignments),
                   "router_summary": router_summary})
    meta = dict(session.preprocessing_meta or {})
    meta["deep_clean_router_msg"] = [router_msg]
    meta["router_stats"] = router_summary   # persisted for benchmark extraction
    session.preprocessing_meta = meta
    logger.info("deep_clean: %s", router_msg)

    log_decision(session, step="Phase 3 - Deep Cleaning",
                 action="all_assignments_logged_before_execution",
                 n_columns=len(assignments))

    work = df.copy()
    work_test = df_test.copy() if df_test is not None else None

    knn_cols = [
        c for c, a in assignments.items()
        if c in work.columns
        and a.get("imputation") == "knn"
        and pd.api.types.is_numeric_dtype(work[c])
        and len(work) <= _KNN_ROW_LIMIT
    ]
    if knn_cols:
        n_neighbors = max(2, min(5, len(work) // 10 or 2))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        work[knn_cols] = imputer.fit_transform(work[knn_cols])
        if work_test is not None:
            work_test[knn_cols] = imputer.transform(work_test[knn_cols])

    new_parts: list[pd.DataFrame] = []
    new_parts_test: list[pd.DataFrame] = []
    if target_s and target_s in work.columns:
        new_parts.append(work[[target_s]].copy())
    if work_test is not None and target_s and target_s in work_test.columns:
        new_parts_test.append(work_test[[target_s]].copy())

    for col in work.columns:
        if target_s and str(col) == target_s:
            continue
        a = assignments.get(col, _default_assignment(col, work[col]))
        s = work[col]
        s_test = work_test[col] if work_test is not None and col in work_test.columns else None

        if pd.api.types.is_numeric_dtype(s):
            imp = a.get("imputation", "median")
            num_test: pd.Series | None = None
            if col in knn_cols:
                num = s.astype(float)
                if s_test is not None:
                    num_test = s_test.astype(float)
            elif imp == "mean":
                imputer = SimpleImputer(strategy="mean")
                num = pd.Series(imputer.fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
                if s_test is not None:
                    num_test = pd.Series(
                        imputer.transform(s_test.values.reshape(-1, 1)).ravel(), index=s_test.index)
            elif imp == "flag_as_missing":
                fill = float(s.median())
                flag_col = f"{col}__was_null"
                work[flag_col] = s.isna().astype(float)
                num = pd.Series(s.fillna(fill), index=s.index).astype(float)
                new_parts.append(pd.DataFrame({flag_col: work[flag_col]}))
                if s_test is not None:
                    work_test[flag_col] = s_test.isna().astype(float)
                    num_test = pd.Series(s_test.fillna(fill), index=s_test.index).astype(float)
                    new_parts_test.append(pd.DataFrame({flag_col: work_test[flag_col]}))
            else:
                imputer = SimpleImputer(strategy="median")
                num = pd.Series(imputer.fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
                if s_test is not None:
                    num_test = pd.Series(
                        imputer.transform(s_test.values.reshape(-1, 1)).ravel(), index=s_test.index)

            if not isinstance(num, pd.Series):
                num = pd.Series(num.ravel(), index=s.index)
            num = num.astype(float)
            if num_test is not None:
                if not isinstance(num_test, pd.Series):
                    num_test = pd.Series(num_test.ravel(), index=s_test.index)
                num_test = num_test.astype(float)

            o = a.get("outlier", "leave")
            if o == "iqr_cap":
                q1, q3 = num.quantile(0.25), num.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                num = num.clip(lo, hi)
                if num_test is not None:
                    num_test = num_test.clip(lo, hi)
            elif o == "log":
                mn = float(num.min())
                shift = 1 - mn if mn <= 0 else 0
                num = pd.Series(np.log1p(num + shift), index=num.index)
                if num_test is not None:
                    num_test = pd.Series(np.log1p(num_test + shift), index=num_test.index)

            scaler = _fit_scaler(a.get("scaling", "none"))
            if scaler is not None:
                num = pd.Series(scaler.fit_transform(num.values.reshape(-1, 1)).ravel(), index=num.index)
                if num_test is not None:
                    num_test = pd.Series(
                        scaler.transform(num_test.values.reshape(-1, 1)).ravel(), index=num_test.index)

            new_parts.append(pd.DataFrame({col: num}))
            if num_test is not None:
                new_parts_test.append(pd.DataFrame({col: num_test}))
        else:
            cat_train, cat_test = _apply_categorical(col, work, a, target_s, work_test)
            new_parts.append(cat_train)
            if cat_test is not None:
                new_parts_test.append(cat_test)

    out = pd.concat(new_parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    out_test = None
    if work_test is not None:
        out_test = (
            pd.concat(new_parts_test, axis=1) if new_parts_test
            else pd.DataFrame(index=work_test.index)
        )
        out_test = out_test.loc[:, ~out_test.columns.duplicated()]
        # Defensive: guarantee identical column set/order to train even if some
        # edge case (e.g. a test-only-missing column) produced a gap.
        out_test = out_test.reindex(columns=out.columns, fill_value=0.0)

    if out.shape[1] > _WIDE_COLUMNS_WARNING:
        msg = (
            f"Dataset has {out.shape[1]} columns after encoding. "
            "This is very wide; feature selection in the next step will be critical."
        )
        _log(session, {"action": "dimensionality_warning", "n_columns": out.shape[1]})
        session.preprocessing_meta["dimensionality_warning"] = msg
        logger.warning(msg)

    _log(session, {"action": "deep_clean_complete", "shape": [int(out.shape[0]), int(out.shape[1])]})
    log_decision(session, step="Phase 3 - Deep Cleaning",
                 action="execution_complete",
                 shape=[int(out.shape[0]), int(out.shape[1])])
    return (out, out_test) if df_test is not None else out
