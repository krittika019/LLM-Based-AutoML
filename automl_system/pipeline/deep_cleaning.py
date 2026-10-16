"""Phase 3 Step 4: deep cleaning — Decision Router pre-fills deterministic assignments;
LLM only handles genuinely ambiguous columns. Single-pass: impute → outlier → encode → scale."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.pipeline.replay import KNN_ROW_LIMIT, apply_deep_clean, default_assignment
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.decision_router import RouterStats, route_columns_batch, columns_needing_llm, build_partial_assignment
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_KNN_ROW_LIMIT = KNN_ROW_LIMIT
_TARGET_ENC_MIN_ROWS = 500
_OHE_CARDINALITY_LIMIT = 50
_WIDE_COLUMNS_WARNING = 500
_LLM_COLUMN_BATCH_SIZE = 35


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "deep_cleaning", **entry})


_default_assignment = default_assignment


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
        col: {**build_partial_assignment(route), "source": route.tier if route.tier in ("rule", "stat") else "rule"}
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
                    merged["source"] = "llm"
                    out[k] = merged

    # Ensure all non-target columns have a complete assignment
    for c in df.columns:
        if str(c) == str(target):
            continue
        if c not in out:
            out[c] = {**_default_assignment(c, df[c]), "source": "rule"}

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

    # Shared with the generated pipeline script (Phase 7) — see replay.py.
    out, out_test = apply_deep_clean(df, df_test, assignments, target_s, knn_row_limit=_KNN_ROW_LIMIT)

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
