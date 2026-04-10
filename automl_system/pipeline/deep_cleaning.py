"""Phase 3 Step 4: deep cleaning — one LLM call for all assignments, single-pass execution."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, TargetEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.decision_router import RouterStats, route_columns_batch, columns_needing_llm, build_partial_assignment
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_KNN_ROW_LIMIT = 10_000    # rows above this → downgrade KNN → median
_TARGET_ENC_MIN_ROWS = 500  # below this → switch target encoding → one-hot
_OHE_CARDINALITY_LIMIT = 50  # above this with one_hot → override to target encoding
_WIDE_COLUMNS_WARNING = 500  # after encoding, warn if columns exceed this


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "deep_cleaning", **entry})


def _default_assignment(col: str, s: pd.Series) -> dict[str, str]:
    if pd.api.types.is_numeric_dtype(s):
        return {"imputation": "median", "outlier": "leave", "encoding": "none", "scaling": "standard",
                "imputation_rationale": "default", "outlier_rationale": "default",
                "encoding_rationale": "default", "scaling_rationale": "default"}
    return {"imputation": "mode", "outlier": "leave", "encoding": "one_hot", "scaling": "none",
            "imputation_rationale": "default", "outlier_rationale": "default",
            "encoding_rationale": "default", "scaling_rationale": "default"}


def _llm_assignments(df: pd.DataFrame, session: SessionState) -> dict[str, dict[str, str]]:
    """
    One LLM call for ALL column assignments. Returns per-column strategy dict.
    All assignments are written to the decisions log before any code runs.
    """
    task = session.task_config or {}
    target = str(task.get("target_column") or "")
    n_rows = len(df)

    cols_meta = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        cols_meta.append({
            "column": c,
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 2),
            "nunique": int(s.nunique(dropna=True)),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "skew": round(float(s.skew()), 3) if pd.api.types.is_numeric_dtype(s) else None,
        })

    plan_summary = session.preprocessing_plan or {}
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    prompt = f"""Phase 3 Step 4 — Deep Data Cleaning. Assign strategy for EVERY column in one call.{revision_block}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2500]}
PREPROCESSING_PLAN: {{"null_strategy_notes": {plan_summary.get("null_strategy_notes", {})}, "skewed_columns": {plan_summary.get("skewed_columns", [])}}}
DATASET_ROWS: {n_rows}
TARGET: {target!r}

ALL_FEATURE_COLUMNS (dtype, null_pct, nunique, is_numeric, skew):
{cols_meta}

For EVERY column above, provide:
- imputation: "mean"|"median"|"mode"|"knn"|"flag_as_missing"  (use "none" if null_pct == 0)
- imputation_rationale: one sentence
- outlier: "iqr_cap"|"log"|"leave"  (only for numeric; always "leave" for categorical)
- outlier_rationale: one sentence
- encoding: "one_hot"|"ordinal"|"target"|"binary"|"none"  (none for numeric)
- encoding_rationale: one sentence
- scaling: "standard"|"minmax"|"robust"|"none"  (none for categoricals; tree models rarely need scaling but distance/linear do)
- scaling_rationale: one sentence

Rules:
- Use KNN imputation only for numeric columns with correlated neighbours; NOT if n_rows > {_KNN_ROW_LIMIT}
- Use target encoding for high-cardinality categoricals (>15 unique) only if n_rows >= {_TARGET_ENC_MIN_ROWS}
- Use robust scaling for numeric columns with high skew (|skew| > 2)
- Never assign any strategy to the target column {target!r}

Return JSON only:
{{"columns": {{"col_name": {{"imputation": "...", "imputation_rationale": "...",
  "outlier": "...", "outlier_rationale": "...",
  "encoding": "...", "encoding_rationale": "...",
  "scaling": "...", "scaling_rationale": "..."}}}}}}"""

    client = LLMClient()
    raw = client.call(prompt)
    parsed = extract_json_object(raw) if raw else None

    assignments: dict[str, dict[str, str]] = {}
    if parsed and isinstance(parsed.get("columns"), dict):
        for k, v in parsed["columns"].items():
            if isinstance(v, dict):
                assignments[str(k)] = {
                    "imputation": str(v.get("imputation") or "median"),
                    "imputation_rationale": str(v.get("imputation_rationale") or ""),
                    "outlier": str(v.get("outlier") or "leave"),
                    "outlier_rationale": str(v.get("outlier_rationale") or ""),
                    "encoding": str(v.get("encoding") or "none"),
                    "encoding_rationale": str(v.get("encoding_rationale") or ""),
                    "scaling": str(v.get("scaling") or "none"),
                    "scaling_rationale": str(v.get("scaling_rationale") or ""),
                }

    # Fill any missing columns with defaults
    for c in df.columns:
        if str(c) == target:
            continue
        if c not in assignments:
            assignments[c] = _default_assignment(c, df[c])

    # ── Scenario checks before logging ─────────────────────────────────────────
    deviations: list[dict[str, Any]] = []
    for col, a in assignments.items():
        s = df[col] if col in df.columns else None

        # Scenario: KNN on large dataset → downgrade to median
        if a.get("imputation") == "knn" and n_rows > _KNN_ROW_LIMIT:
            old = a["imputation"]
            a["imputation"] = "median"
            a["imputation_rationale"] = f"KNN downgraded to median (dataset has {n_rows} rows > {_KNN_ROW_LIMIT} threshold)."
            deviations.append({"column": col, "deviation": "knn_downgrade", "original": old,
                                "override": "median", "reason": f"n_rows={n_rows}"})

        # Scenario: target encoding on small dataset → switch to one-hot
        if a.get("encoding") == "target" and n_rows < _TARGET_ENC_MIN_ROWS:
            old = a["encoding"]
            a["encoding"] = "one_hot"
            a["encoding_rationale"] = f"Target encoding switched to one_hot (n_rows={n_rows} < {_TARGET_ENC_MIN_ROWS} — leakage risk too high)."
            deviations.append({"column": col, "deviation": "target_enc_too_small", "original": old,
                                "override": "one_hot", "reason": f"n_rows={n_rows}"})

        # Scenario: one-hot on high-cardinality column → override to target encoding
        if a.get("encoding") == "one_hot" and s is not None and not pd.api.types.is_numeric_dtype(s):
            nu = int(s.nunique(dropna=True))
            if nu > _OHE_CARDINALITY_LIMIT:
                old = a["encoding"]
                a["encoding"] = "target"
                a["encoding_rationale"] = f"one_hot overridden to target encoding ({nu} unique values > {_OHE_CARDINALITY_LIMIT} limit)."
                deviations.append({"column": col, "deviation": "ohe_cardinality_override", "original": old,
                                    "override": "target", "reason": f"nunique={nu}"})

    if deviations:
        _log(session, {"action": "assignment_deviations", "deviations": deviations})
        logger.info("deep_clean: %d assignment deviation(s) applied", len(deviations))

    return assignments


def _iqr_cap(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s.clip(lo, hi)


def _scale_numeric(s: pd.Series, sc: str) -> pd.Series:
    if sc == "standard":
        return pd.Series(StandardScaler().fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
    if sc == "minmax":
        return pd.Series(MinMaxScaler().fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
    if sc == "robust":
        return pd.Series(RobustScaler().fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
    return s


def _apply_categorical(
    col: str, df: pd.DataFrame, assignment: dict[str, str], target: str | None, n_rows: int
) -> pd.DataFrame:
    s = df[col].astype(str).replace("nan", np.nan)
    imp = assignment.get("imputation", "mode")
    if imp == "mode":
        mode = s.mode()
        s = s.fillna(mode.iloc[0] if len(mode) else "missing")
    elif imp == "flag_as_missing":
        s = s.fillna("__missing__")
    else:
        s = s.fillna("missing")

    enc = assignment.get("encoding", "one_hot")

    if enc == "binary":
        unique_vals = s.dropna().unique()
        if len(unique_vals) == 2:
            s_encoded = (s == unique_vals[0]).astype(float)
            return pd.DataFrame({col: s_encoded}, index=df.index)

    if enc == "ordinal":
        try:
            oe = OrdinalEncoder(handle_unknown="value", handle_missing="value")
            v = oe.fit_transform(s.to_frame(name=col))
            return pd.DataFrame({col: v[col].astype(float)}, index=df.index)
        except Exception:
            codes = pd.Series(pd.factorize(s, sort=True)[0], index=df.index).astype(float)
            return pd.DataFrame({col: codes}, index=df.index)

    if enc == "target" and target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        te = TargetEncoder(handle_unknown="value")
        y = df[target]
        z = te.fit_transform(s.to_frame(), y)
        return pd.DataFrame({f"{col}_te": z.values.ravel()}, index=df.index)

    # Default: one_hot
    return pd.get_dummies(s, prefix=col, dtype=float)


def deep_clean(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Step 4: One LLM call for all assignments → log everything before execution →
    single-pass: impute → outlier cap → encode → scale.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if target_s and target_s not in df.columns:
        target_s = None

    n_rows = len(df)

    # ── ONE LLM call for all column assignments ─────────────────────────────
    assignments = _llm_assignments(df, session)
    session.deep_cleaning_assignments = assignments

    # ── Log ALL assignments BEFORE code runs ────────────────────────────────
    for col, a in assignments.items():
        _log(session, {
            "action": "assignment_logged",
            "column": col,
            "imputation": a.get("imputation"),
            "imputation_rationale": a.get("imputation_rationale"),
            "outlier": a.get("outlier"),
            "outlier_rationale": a.get("outlier_rationale"),
            "encoding": a.get("encoding"),
            "encoding_rationale": a.get("encoding_rationale"),
            "scaling": a.get("scaling"),
            "scaling_rationale": a.get("scaling_rationale"),
        })

    log_decision(session, step="Phase 3 - Deep Cleaning",
                 action="all_assignments_logged_before_execution",
                 n_columns=len(assignments))

    work = df.copy()

    # ── Pass 1: KNN imputation (bulk, requires complete block) ──────────────
    knn_cols = [
        c for c, a in assignments.items()
        if c in work.columns
        and a.get("imputation") == "knn"
        and pd.api.types.is_numeric_dtype(work[c])
        and n_rows <= _KNN_ROW_LIMIT  # safety guard (deviations should have cleared this)
    ]
    if knn_cols:
        n_neighbors = max(2, min(5, n_rows // 10 or 2))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        work[knn_cols] = imputer.fit_transform(work[knn_cols])

    # ── Single pass: impute → outlier → encode → scale ───────────────────────
    new_parts: list[pd.DataFrame] = []
    if target_s and target_s in work.columns:
        new_parts.append(work[[target_s]].copy())

    for col in work.columns:
        if target_s and str(col) == target_s:
            continue
        a = assignments.get(col, _default_assignment(col, work[col]))
        s = work[col]

        if pd.api.types.is_numeric_dtype(s):
            # Step 1: Impute
            imp = a.get("imputation", "median")
            if col in knn_cols:
                num = s.astype(float)
            elif imp == "mean":
                num = pd.Series(
                    SimpleImputer(strategy="mean").fit_transform(s.values.reshape(-1, 1)).ravel(),
                    index=s.index)
            elif imp == "flag_as_missing":
                fill = float(s.median())
                flag_col = f"{col}__was_null"
                work[flag_col] = s.isna().astype(float)
                num = pd.Series(s.fillna(fill), index=s.index).astype(float)
                new_parts.append(pd.DataFrame({flag_col: work[flag_col]}))
            else:  # median + knn fallback
                num = pd.Series(
                    SimpleImputer(strategy="median").fit_transform(s.values.reshape(-1, 1)).ravel(),
                    index=s.index)
            if not isinstance(num, pd.Series):
                num = pd.Series(num.ravel(), index=s.index)
            num = num.astype(float)

            # Step 2: Outlier treatment
            o = a.get("outlier", "leave")
            if o == "iqr_cap":
                num = _iqr_cap(num)
            elif o == "log":
                mn = float(num.min())
                shift = 1 - mn if mn <= 0 else 0
                num = pd.Series(np.log1p(num + shift), index=num.index)

            # Step 3: Encoding (no-op for numeric)
            # Step 4: Scale
            num = _scale_numeric(num, a.get("scaling", "none"))
            new_parts.append(pd.DataFrame({col: num}))
        else:
            new_parts.append(_apply_categorical(col, work, a, target_s, n_rows))

    out = pd.concat(new_parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    # ── Scenario: >500 columns after encoding → dimensionality warning ────────
    if out.shape[1] > _WIDE_COLUMNS_WARNING:
        msg = (
            f"Dataset has {out.shape[1]} columns after encoding. "
            "This is very wide; feature selection in the next step will be critical."
        )
        _log(session, {"action": "dimensionality_warning", "n_columns": out.shape[1]})
        session.preprocessing_meta["dimensionality_warning"] = msg
        logger.warning(msg)

    _log(session, {"action": "deep_clean_complete",
                   "shape": [int(out.shape[0]), int(out.shape[1])]})
    log_decision(session, step="Phase 3 - Deep Cleaning",
                 action="execution_complete",
                 shape=[int(out.shape[0]), int(out.shape[1])])
    return out


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "deep_cleaning", **entry})


def _default_assignment(col: str, s: pd.Series) -> dict[str, str]:
    if pd.api.types.is_numeric_dtype(s):
        return {
            "imputation": "median",
            "outlier": "leave",
            "encoding": "none",
            "scaling": "standard",
        }
    return {
        "imputation": "mode",
        "outlier": "leave",
        "encoding": "one_hot",
        "scaling": "none",
    }


def _llm_assignments(
    df: pd.DataFrame,
    session: SessionState,
    router_stats: RouterStats,
) -> dict[str, dict[str, str]]:
    """
    Assign preprocessing strategy per column using a tiered approach:
      1. Route each column through the Decision Router (rule/stat tiers).
      2. Collect only ambiguous columns for the LLM prompt — reducing token usage.
      3. Merge LLM responses with rule-based decisions.
    """
    task = session.task_config or {}
    target = task.get("target_column")

    # ── Tier 1/2: rule & stat routing ─────────────────────────────────────────
    routes = route_columns_batch(df, target, router_stats)
    ambiguous_cols = columns_needing_llm(routes)

    # Build base assignments from rule/stat decisions
    out: dict[str, dict[str, str]] = {
        col: build_partial_assignment(route)
        for col, route in routes.items()
    }

    # ── Tier 3: LLM for ambiguous columns only ─────────────────────────────────
    if ambiguous_cols:
        cols_meta = []
        for c in ambiguous_cols:
            s = df[c]
            cols_meta.append(
                {
                    "column": c,
                    "dtype": str(s.dtype),
                    "null_pct": float(s.isna().mean() * 100),
                    "nunique": int(s.nunique(dropna=True)),
                    "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
                }
            )
        client = LLMClient()
        prompt = f"""Assign preprocessing strategy for these AMBIGUOUS columns only.
(Simple/deterministic columns were already handled without LLM.)

TASK: {task}
PLAN: {session.preprocessing_plan}

COLUMNS_NEEDING_JUDGMENT ({len(ambiguous_cols)} of {len(routes)} total):
{cols_meta}

Return JSON: {{"columns": {{"col_name": {{
  "imputation": "mean"|"median"|"mode"|"knn"|"flag_as_missing",
  "outlier": "iqr_cap"|"log"|"leave",
  "encoding": "ordinal"|"one_hot"|"target"|"none",
  "scaling": "standard"|"minmax"|"none"
}}}}}}

Use knn for numeric with moderate missingness + correlated neighbors.
Use target encoding for high-cardinality categoricals (11–100 unique).
Never transform the target column {target!r}."""

        raw = client.call(prompt)
        parsed = extract_json_object(raw) if raw else None
        if parsed and isinstance(parsed.get("columns"), dict):
            for k, v in parsed["columns"].items():
                k = str(k)
                if k in out and isinstance(v, dict):
                    # Override only the fields the router left as None/default
                    route = routes.get(k)
                    merged = dict(out[k])
                    if route and route.imputation is None and v.get("imputation"):
                        merged["imputation"] = str(v["imputation"])
                    if route and route.outlier is None and v.get("outlier"):
                        merged["outlier"] = str(v["outlier"])
                    if route and route.encoding is None and v.get("encoding"):
                        merged["encoding"] = str(v["encoding"])
                    out[k] = merged

    # Ensure all non-target columns have a complete assignment
    for c in df.columns:
        if str(c) == str(target):
            continue
        if c not in out:
            out[c] = _default_assignment(c, df[c])
    return out


def _iqr_cap(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s.clip(lo, hi)


def _impute_numeric(s: pd.Series, imp: str) -> pd.Series:
    out = s.astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    if imp == "mean":
        fill = float(out.mean())
    elif imp in ("median", "knn", "flag_as_missing"):
        fill = float(out.median())
    else:
        fill = float(out.median())
    return out.fillna(fill)


def _scale_numeric(s: pd.Series, sc: str) -> pd.Series:
    if sc == "standard":
        return pd.Series(
            StandardScaler().fit_transform(s.values.reshape(-1, 1)).ravel(),
            index=s.index,
        )
    if sc == "minmax":
        return pd.Series(
            MinMaxScaler().fit_transform(s.values.reshape(-1, 1)).ravel(),
            index=s.index,
        )
    return s


def _apply_categorical(
    col: str,
    df: pd.DataFrame,
    assignment: dict[str, str],
    target: str | None,
) -> pd.DataFrame:
    s = df[col].astype(str).replace("nan", np.nan)
    imp = assignment.get("imputation", "mode")
    if imp == "mode":
        mode = s.mode()
        s = s.fillna(mode.iloc[0] if len(mode) else "missing")
    else:
        s = s.fillna("missing")
    enc = assignment.get("encoding", "one_hot")
    if enc == "ordinal":
        try:
            # category_encoders.OrdinalEncoder supports handle_unknown/handle_missing="value".
            oe = OrdinalEncoder(handle_unknown="value", handle_missing="value")
            v = oe.fit_transform(s.to_frame(name=col))
            return pd.DataFrame({col: v[col].astype(float)}, index=df.index)
        except Exception:
            # Fallback keeps pipeline running even if encoder behavior differs by version.
            codes = pd.Series(pd.factorize(s, sort=True)[0], index=df.index).astype(float)
            return pd.DataFrame({col: codes}, index=df.index)
    if enc == "target" and target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        te = TargetEncoder(handle_unknown="value")
        y = df[target]
        z = te.fit_transform(s.to_frame(), y)
        return pd.DataFrame({f"{col}_te": z.values.ravel()}, index=df.index)
    return pd.get_dummies(s, prefix=col, dtype=float)


def deep_clean(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Decision Router pre-fills deterministic column assignments (rule/stat tier);
    LLM only sees genuinely ambiguous columns. Then execute imputation, outlier
    handling, encoding, and scaling in one pipeline pass.
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
    _log(session, {
        "action": "assignments_recorded",
        "n_columns": len(assignments),
        "router_summary": router_stats.summary(),
    })
    # Store router message so orchestrator node can surface it in chat
    meta = dict(session.preprocessing_meta or {})
    meta["deep_clean_router_msg"] = [router_msg]
    session.preprocessing_meta = meta
    logger.info("deep_clean: %s", router_msg)

    work = df.copy()
    knn_cols = [
        c
        for c, a in assignments.items()
        if c in work.columns
        and a.get("imputation") == "knn"
        and pd.api.types.is_numeric_dtype(work[c])
    ]
    if knn_cols:
        n_neighbors = max(2, min(5, len(work) // 10 or 2))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        work[knn_cols] = imputer.fit_transform(work[knn_cols])

    new_parts: list[pd.DataFrame] = []
    if target_s:
        new_parts.append(work[[target_s]].copy())

    for col in work.columns:
        if target_s and str(col) == target_s:
            continue
        a = assignments.get(col, _default_assignment(col, work[col]))
        s = work[col]
        if pd.api.types.is_numeric_dtype(s):
            imp = a.get("imputation", "median")
            if col in knn_cols:
                num = s.astype(float)
            else:
                if imp == "mean":
                    num = SimpleImputer(strategy="mean").fit_transform(s.values.reshape(-1, 1))
                elif imp == "median":
                    num = SimpleImputer(strategy="median").fit_transform(s.values.reshape(-1, 1))
                else:
                    num = _impute_numeric(s, imp).values.reshape(-1, 1)
                num = pd.Series(num.ravel(), index=s.index)
            o = a.get("outlier", "leave")
            if o == "iqr_cap":
                num = _iqr_cap(num)
            elif o == "log":
                mn = float(num.min())
                shift = 1 - mn if mn <= 0 else 0
                num = pd.Series(np.log1p(num.astype(float) + shift), index=num.index)
            num = _scale_numeric(num, a.get("scaling", "none"))
            new_parts.append(pd.DataFrame({col: num}))
        else:
            new_parts.append(_apply_categorical(col, work, a, target_s))

    out = pd.concat(new_parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    _log(session, {"action": "deep_clean_complete", "shape": [int(out.shape[0]), int(out.shape[1])]})
    return out
