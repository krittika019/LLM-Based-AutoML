"""Phase 3: preprocessing verification with routing hints for LangGraph."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision

logger = logging.getLogger(__name__)


def propose_verification_fix(session: SessionState, failure_user_message: str) -> str:
    """LLM explains failure and proposes a fix; logged to episodic memory."""
    if not failure_user_message.strip():
        return ""
    client = LLMClient()
    prompt = f"""Preprocessing verification failed:
{failure_user_message}

Identify which preprocessing step is responsible and propose one concrete fix. 2-4 sentences. Plain English."""
    text = client.call(prompt).strip()
    if session.decisions_log_id and text:
        episodic.append_decision(
            session.decisions_log_id,
            {"phase": "verification", "action": "failure_diagnosis", "text": text[:2000]},
        )
    return text


# Steps the orchestrator can route back to (must match graph node names)
STEP_BASIC_CLEAN = "basic_cleaning"
STEP_EDA = "eda"
STEP_FEATURE_SELECTION = "feature_selection"
STEP_DEEP_CLEANING = "deep_cleaning"
STEP_FEATURE_ENGINEERING = "feature_engineering"

ROUTE_STEP_MARKER = "\n__ROUTE_STEP__:"


def parse_verification_route(message: str) -> tuple[str, str | None]:
    """Split user-facing text and routing step from verify_preprocessing output."""
    if ROUTE_STEP_MARKER in message:
        body, step = message.split(ROUTE_STEP_MARKER, 1)
        return body.strip(), step.strip() or None
    return message.strip(), None


def verify_preprocessing(df: pd.DataFrame, session: SessionState) -> tuple[bool, str]:
    """
    Pure code checks — LLM only called by the orchestrator when a check fails.
    Checks in order:
    1. No null values remain anywhere
    2. All features correctly typed (no object where numeric expected)
    3. Dataset shape consistent with expected (row count)
    4. No feature has correlation >= 0.98 with target (leakage)
    5. No feature has zero variance (constant column that survived)

    On pass: sets session.preprocessing_meta['verification_passed'] = True.
    On fail: returns message ending with ROUTE_STEP_MARKER + step name.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None

    # ── Check 1: No nulls ────────────────────────────────────────────────────
    null_mask = df.isna().any()
    if null_mask.any():
        cols_with_nulls = df.columns[null_mask].tolist()
        msg = (
            f"Null values remain in {len(cols_with_nulls)} column(s): "
            f"{cols_with_nulls[:10]}{'...' if len(cols_with_nulls) > 10 else ''}. "
            "Re-running imputation for these columns only."
            f"{ROUTE_STEP_MARKER}{STEP_DEEP_CLEANING}"
        )
        # Store which columns need re-imputation so deep_clean can target only them
        session.preprocessing_meta["verification_null_cols"] = cols_with_nulls
        log_decision(session, step="Verification", action="null_check_failed",
                     null_cols=cols_with_nulls)
        return False, msg

    # ── Check 2: All features correctly typed ────────────────────────────────
    for c in df.columns:
        if c == target_s:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if str(s.dtype) == "category":
            continue
        log_decision(session, step="Verification", action="type_check_failed", column=c, dtype=str(s.dtype))
        return False, (
            f"Feature {c!r} is still non-numeric ({s.dtype}) after preprocessing."
            f"{ROUTE_STEP_MARKER}{STEP_DEEP_CLEANING}"
        )

    # ── Check 3: Row count consistent with expected ───────────────────────────
    exp = session.expected_shape
    if exp is not None and len(df) != exp[0]:
        log_decision(session, step="Verification", action="shape_check_failed",
                     expected_rows=exp[0], actual_rows=len(df))
        return False, (
            f"Row count {len(df)} does not match expected {exp[0]} after preprocessing."
            f"{ROUTE_STEP_MARKER}{STEP_BASIC_CLEAN}"
        )

    # ── Check 4: No feature has |corr| >= 0.98 with target ───────────────────
    if target_s and target_s in df.columns:
        y = pd.to_numeric(df[target_s], errors="coerce")
        for c in df.columns:
            if c == target_s:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            sub = pd.DataFrame({"x": s, "y": y}).dropna()
            if len(sub) < 5 or sub["x"].nunique() <= 1:
                continue
            corr = float(abs(sub["x"].corr(sub["y"])))
            if corr >= 0.98:
                log_decision(session, step="Verification", action="leakage_check_failed",
                             column=c, corr=corr)
                return False, (
                    f"Feature {c!r} has correlation {corr:.4f} ≥ 0.98 with target — "
                    "likely leakage or a near-duplicate of the target variable."
                    f"{ROUTE_STEP_MARKER}{STEP_FEATURE_ENGINEERING}"
                )

    # ── Check 5: No zero-variance feature ────────────────────────────────────
    for c in df.columns:
        if c == target_s:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            v = float(df[c].var(skipna=True) or 0.0)
            if v == 0.0 or not np.isfinite(v):
                log_decision(session, step="Verification", action="zero_variance_check_failed", column=c)
                return False, (
                    f"Feature {c!r} has zero variance (constant column). "
                    "This column survived preprocessing but carries no signal."
                    f"{ROUTE_STEP_MARKER}{STEP_FEATURE_SELECTION}"
                )

    # ── All checks passed ────────────────────────────────────────────────────
    session.preprocessing_meta["verification_passed"] = True
    session.last_verification = {"ok": True, "shape": [int(df.shape[0]), int(df.shape[1])]}
    log_decision(session, step="Verification", action="all_checks_passed",
                 shape=[int(df.shape[0]), int(df.shape[1])])
    return True, "All preprocessing checks passed."
