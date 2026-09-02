"""Phase 3: preprocessing verification with routing hints for LangGraph."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
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


def verify_preprocessing(
    df: pd.DataFrame, session: SessionState, df_test: pd.DataFrame | None = None
) -> tuple[bool, str]:
    """
    Pure code checks — LLM only called by the orchestrator when a check fails.
    Checks in order:
    0. Target column is still present in the dataframe
    1. No null values remain anywhere
    2. No infinite values (np.inf / -np.inf) in numeric columns
    3. All features correctly typed (no object where numeric expected)
    4. Dataset shape consistent with expected (row count)
    5. No feature has correlation >= threshold with target (linear leakage)
       + No engineered feature is near-perfectly predictive alone (non-linear leakage)
    6. No feature has zero variance (constant column that survived)
    7. Feature/sample ratio warning — routes to feature_selection if n_features > n_rows / 5

    On pass: sets session.preprocessing_meta['verification_passed'] = True.
    On fail: returns message ending with ROUTE_STEP_MARKER + step name.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None

    # ── Check 0: Target column must still exist ──────────────────────────────
    if target_s and target_s not in df.columns:
        log_decision(session, step="Verification", action="target_missing",
                     target=target_s, columns=df.columns.tolist()[:20])
        session.preprocessing_meta["verification_failed"] = True
        return False, (
            f"Target column {target_s!r} is missing from the dataframe after preprocessing. "
            "Preprocessing may have accidentally dropped it."
            f"{ROUTE_STEP_MARKER}{STEP_BASIC_CLEAN}"
        )

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
        session.preprocessing_meta["verification_failed"] = True
        log_decision(session, step="Verification", action="null_check_failed",
                     null_cols=cols_with_nulls)
        return False, msg

    # ── Check 2: No infinite values in numeric columns ───────────────────────
    # np.inf / -np.inf are NOT caught by isna(). Models either crash or produce
    # NaN predictions silently when they encounter infinite values.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        log_decision(session, step="Verification", action="inf_check_failed",
                     columns=inf_cols)
        session.preprocessing_meta["verification_failed"] = True
        return False, (
            f"Infinite values (±inf) found in {len(inf_cols)} column(s): "
            f"{inf_cols[:10]}{'...' if len(inf_cols) > 10 else ''}. "
            "These are not null but will cause model training to fail or produce NaN predictions."
            f"{ROUTE_STEP_MARKER}{STEP_DEEP_CLEANING}"
        )

    # ── Check 3: All features correctly typed ────────────────────────────────
    for c in df.columns:
        if c == target_s:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if str(s.dtype) == "category":
            continue
        log_decision(session, step="Verification", action="type_check_failed", column=c, dtype=str(s.dtype))
        session.preprocessing_meta["verification_failed"] = True
        return False, (
            f"Feature {c!r} is still non-numeric ({s.dtype}) after preprocessing."
            f"{ROUTE_STEP_MARKER}{STEP_DEEP_CLEANING}"
        )

    # ── Check 4: Row count consistent with expected ───────────────────────────
    exp = session.expected_shape
    if exp is not None and len(df) != exp[0]:
        log_decision(session, step="Verification", action="shape_check_failed",
                     expected_rows=exp[0], actual_rows=len(df))
        session.preprocessing_meta["verification_failed"] = True
        return False, (
            f"Row count {len(df)} does not match expected {exp[0]} after preprocessing."
            f"{ROUTE_STEP_MARKER}{STEP_BASIC_CLEAN}"
        )

    # ── Check 5: Leakage detection (linear + non-linear) ────────────────────
    # 4a: Pearson correlation — catches linear leakage (e.g. feature = target * constant)
    # 4b: Single-feature predictive power — catches non-linear leakage that Pearson misses
    #     (e.g. a ratio feature that perfectly separates classes without linear correlation)
    if target_s and target_s in df.columns:
        task_type_v = str((session.task_config or {}).get("task_type") or "").lower()
        y_raw = df[target_s]
        y_num = pd.to_numeric(y_raw, errors="coerce")

        numeric_feature_cols = [
            c for c in df.columns
            if c != target_s and pd.api.types.is_numeric_dtype(df[c])
        ]

        # 4a: Linear correlation check
        for c in numeric_feature_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            sub = pd.DataFrame({"x": s, "y": y_num}).dropna()
            if len(sub) < 5 or sub["x"].nunique() <= 1:
                continue
            corr = float(abs(sub["x"].corr(sub["y"])))
            if corr >= config.LEAKAGE_CORR_THRESHOLD:
                log_decision(session, step="Verification", action="leakage_check_failed",
                             column=c, corr=corr, method="pearson")
                session.preprocessing_meta["verification_failed"] = True
                return False, (
                    f"Feature {c!r} has linear correlation {corr:.4f} with target — "
                    "likely leakage or near-duplicate of the target variable."
                    f"{ROUTE_STEP_MARKER}{STEP_FEATURE_ENGINEERING}"
                )

        # 4b: Non-linear single-feature power check using a shallow decision tree
        # A single feature achieving >95% accuracy/R² on its own is almost certainly leaky.
        # Only check engineered features (not original columns) to avoid false positives
        # on genuinely strong natural predictors.
        fe_log = session.feature_engineering_log or []
        engineered_names = {
            str(e.get("feature")) for e in fe_log
            if isinstance(e, dict) and e.get("status") == "added"
        }
        cols_to_check = [c for c in numeric_feature_cols if c in engineered_names]

        if cols_to_check:
            try:
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.model_selection import cross_val_score as _cvs

                _SINGLE_FEAT_LEAKAGE_THRESHOLD = 0.97
                y_check = y_raw if task_type_v == "classification" else y_num

                for c in cols_to_check:
                    x_feat = df[[c]].fillna(df[c].median())
                    valid = x_feat[c].notna() & y_check.notna()
                    if valid.sum() < 20:
                        continue
                    estimator = (
                        DecisionTreeClassifier(max_depth=3, random_state=42)
                        if task_type_v == "classification"
                        else DecisionTreeRegressor(max_depth=3, random_state=42)
                    )
                    scoring = "f1_weighted" if task_type_v == "classification" else "r2"
                    scores = _cvs(estimator, x_feat[valid], y_check[valid],
                                  cv=3, scoring=scoring)
                    mean_score = float(np.mean(scores))
                    if mean_score >= _SINGLE_FEAT_LEAKAGE_THRESHOLD:
                        log_decision(
                            session, step="Verification",
                            action="nonlinear_leakage_check_failed",
                            column=c, single_feature_score=mean_score,
                            method="decision_tree_cv",
                        )
                        session.preprocessing_meta["verification_failed"] = True
                        return False, (
                            f"Engineered feature {c!r} achieves {mean_score:.4f} alone "
                            f"(threshold {_SINGLE_FEAT_LEAKAGE_THRESHOLD}) — "
                            "this indicates data leakage. The feature likely encodes target information."
                            f"{ROUTE_STEP_MARKER}{STEP_FEATURE_ENGINEERING}"
                        )
            except Exception as e:
                logger.debug("Non-linear leakage check skipped: %s", e)

    # ── Check 6: No zero-variance feature (numeric or constant categorical) ───
    for c in df.columns:
        if c == target_s:
            continue
        col_s = df[c]
        if pd.api.types.is_numeric_dtype(col_s):
            v = float(col_s.var(skipna=True) or 0.0)
            if v == 0.0 or not np.isfinite(v):
                log_decision(session, step="Verification", action="zero_variance_check_failed", column=c)
                session.preprocessing_meta["verification_failed"] = True
                return False, (
                    f"Feature {c!r} has zero variance (constant numeric column). "
                    "This column survived preprocessing but carries no signal."
                    f"{ROUTE_STEP_MARKER}{STEP_FEATURE_SELECTION}"
                )
        else:
            # Categorical: flag if a single value covers all non-null rows
            nu = int(col_s.nunique(dropna=True))
            if nu <= 1:
                log_decision(session, step="Verification", action="zero_variance_check_failed",
                             column=c, dtype=str(col_s.dtype))
                session.preprocessing_meta["verification_failed"] = True
                return False, (
                    f"Feature {c!r} is a constant categorical column (all values identical). "
                    "This column carries no signal and should be dropped."
                    f"{ROUTE_STEP_MARKER}{STEP_FEATURE_SELECTION}"
                )

    # ── Check 7: Feature/sample ratio ────────────────────────────────────────
    # If n_features > n_rows / 5, the model is likely to overfit regardless of
    # regularisation. Route to feature selection to reduce dimensionality.
    # Target column is excluded from the feature count.
    n_rows = len(df)
    n_features = len([c for c in df.columns if c != target_s])
    if n_rows > 0 and n_features > n_rows / 5:
        log_decision(session, step="Verification", action="high_feature_ratio",
                     n_features=n_features, n_rows=n_rows,
                     ratio=round(n_features / n_rows, 3))
        session.preprocessing_meta["verification_failed"] = True
        return False, (
            f"Feature/sample ratio is too high: {n_features} features for {n_rows} rows "
            f"(ratio={n_features/n_rows:.2f}, threshold=0.20). "
            "High dimensionality relative to sample size causes overfitting. "
            "Reduce the feature count via feature selection."
            f"{ROUTE_STEP_MARKER}{STEP_FEATURE_SELECTION}"
        )

    # ── All checks passed ────────────────────────────────────────────────────
    session.preprocessing_meta["verification_passed"] = True
    session.preprocessing_meta["verification_failed"] = False
    session.preprocessing_meta.pop("verification_null_cols", None)
    session.last_verification = {"ok": True, "shape": [int(df.shape[0]), int(df.shape[1])]}
    log_decision(session, step="Verification", action="all_checks_passed",
                 shape=[int(df.shape[0]), int(df.shape[1])])

    # Lightweight sanity check (not a hard failure): a column mismatch here
    # would indicate a bug in one of the fit-train/apply-test preprocessing
    # steps, not a real user-facing preprocessing problem — surface it as a
    # log warning for debugging rather than a cryptic verification failure.
    if df_test is not None and list(df_test.columns) != list(df.columns):
        logger.warning(
            "verify_preprocessing: df_test columns %s differ from df (train) columns %s",
            list(df_test.columns), list(df.columns),
        )

    return True, "All preprocessing checks passed."
