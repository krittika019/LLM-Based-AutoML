"""Phase 3 Step 5: LLM-proposed feature engineering with leakage guard."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import compute_dataset_profile
from automl_system.utils.decision_log import log_decision
from automl_system.utils.decision_router import RouterStats, should_engineer_features
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_LEAKAGE_CORR_THRESHOLD = 0.95
_MIN_USEFUL_CORR = 0.03
_MIN_ROWS_FOR_FE = 200       # skip entirely if fewer rows than this


def _dt_reasoning_block(
    df: pd.DataFrame,
    numeric: list[str],
    y: pd.Series,
    task_type: str,
) -> str:
    """Train a shallow decision tree and return a block of split rules for the LLM prompt.

    Grounds LLM feature proposals in actual data patterns (NeurIPS 2024, OCTree).
    Returns an empty string on any failure so the rest of FE still proceeds.
    """
    if len(numeric) < 2 or len(df) < 50:
        return ""
    try:
        from sklearn.preprocessing import LabelEncoder as _LE
        from sklearn.tree import DecisionTreeClassifier as _DTC, DecisionTreeRegressor as _DTR

        X_dt = df[numeric].fillna(df[numeric].median())
        y_dt = y.copy()

        if task_type == "classification" or not pd.api.types.is_numeric_dtype(y_dt) or y_dt.nunique() <= 15:
            if not pd.api.types.is_numeric_dtype(y_dt):
                y_dt = pd.Series(_LE().fit_transform(y_dt.astype(str)), index=y_dt.index)
            dt = _DTC(max_depth=4, random_state=42, min_samples_leaf=max(5, len(df) // 50))
        else:
            dt = _DTR(max_depth=4, random_state=42, min_samples_leaf=max(5, len(df) // 50))

        dt.fit(X_dt, y_dt)

        # Top features by impurity reduction
        importances = dt.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_features = [
            (numeric[i], round(float(importances[i]), 3))
            for i in top_idx if importances[i] > 0.01
        ]
        if not top_features:
            return ""

        # Key splitting rules from the first 2 tree levels
        tree = dt.tree_
        rules: list[str] = []

        def _extract(node: int, depth: int = 0) -> None:
            if depth >= 2 or len(rules) >= 5 or tree.feature[node] < 0:
                return
            fname = numeric[tree.feature[node]]
            thr = tree.threshold[node]
            n = int(tree.n_node_samples[node])
            rules.append(f"{fname} <= {thr:.3g}  ({n} samples split here)")
            _extract(tree.children_left[node], depth + 1)
            _extract(tree.children_right[node], depth + 1)

        _extract(0)

        lines = [
            "DT_REASONING (shallow decision tree on cleaned data - use this to ground your proposals):",
            f"  Most predictive features: {top_features}",
        ]
        if rules:
            lines.append(f"  Key decision boundaries: {rules}")
        lines.append(
            "  Propose features that sharpen these boundaries or reveal patterns the tree misses."
        )
        return "\n".join(lines)
    except Exception as _e:
        logger.debug("_dt_reasoning_block failed (non-fatal): %s", _e)
        return ""


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "feature_engineering", **entry})


def _corr_with_target(feat: pd.Series, y: pd.Series) -> float:
    # Encode string/categorical target to numeric so Pearson corr works.
    # Churn='Yes'/'No', Survived=0/1, etc. — only the numeric case passes through unchanged.
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0].astype(float), index=y.index)
    sub = pd.DataFrame({"f": feat, "y": y}).dropna()
    if len(sub) < 3:
        return 0.0
    if sub["f"].nunique() <= 1 or sub["y"].nunique() <= 1:
        return 0.0
    return float(sub["f"].corr(sub["y"]))


def engineer_features(
    df: pd.DataFrame, session: SessionState, df_test: pd.DataFrame | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 5: LLM proposes features, code runs leakage check before each creation.

    Proposal types: interaction, ratio, datetime, domain_combination
    Scenarios:
    - dataset < 200 rows → skip entirely, no LLM call
    - proposals reference dropped columns → silently discard + log
    - all proposals fail leakage check → log, add note to eda_narrative

    Every accept/reject decision and every NaN-fill value is computed from `df`
    (train) only. When `df_test` is given, the identical accepted formulas and
    fill values — never recomputed from test — are applied to it too, and
    `(out_train, out_test)` is returned instead of a single frame.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if not target_s or target_s not in df.columns:
        session.feature_engineering_log = []
        return (df, df_test) if df_test is not None else df

    n_rows = len(df)

    # ── Decision Router gate — 3-tier check before LLM call ──────────────────
    _router_stats = RouterStats()
    _should_fe, _skip_reason = should_engineer_features(df, session, _router_stats)
    if not _should_fe:
        session.feature_engineering_log = [{"status": "skipped", "reason": _skip_reason}]
        _log(session, {"action": "fe_skipped", "reason": _skip_reason, "n_rows": n_rows})
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="skipped_by_router", n_rows=n_rows, reason=_skip_reason)
        return (df, df_test) if df_test is not None else df

    y = df[target_s]
    task_type = str(task.get("task_type") or "classification").lower()
    numeric = [c for c in df.columns if c != target_s and pd.api.types.is_numeric_dtype(df[c])]
    dt_cols: list[str] = []
    for c in df.columns:
        if c == target_s:
            continue
        try:
            if np.issubdtype(df[c].dtype, np.datetime64):
                dt_cols.append(c)
        except (TypeError, ValueError):
            pass
        if "date" in str(c).lower() and c not in dt_cols:
            dt_cols.append(c)

    # ── A: Decision-tree reasoning block (NeurIPS 2024 — OCTree technique) ─────
    # Grounds the LLM's proposals in actual data splits rather than EDA narrative alone.
    dt_block = _dt_reasoning_block(df, numeric, y, task_type)
    dt_section = f"\n\n{dt_block}" if dt_block else ""

    client = LLMClient()
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    prompt = f"""Phase 3 Step 5 — Feature Engineering. Propose new derived features.{revision_block}{dt_section}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2000]}
DATASET_ROWS: {n_rows}

NUMERIC_COLUMNS (use ONLY these for interaction/ratio/domain_combination): {numeric[:40]}
DATETIME_LIKE: {dt_cols}

Propose up to 8 candidate features. For each provide:
- name: short snake_case name
- type: "interaction" | "ratio" | "datetime" | "domain_combination"
- left: column name (MUST be from NUMERIC_COLUMNS)
- right: column name (MUST be from NUMERIC_COLUMNS, empty for datetime or domain_combination)
- formula: Python expression using ONLY names from NUMERIC_COLUMNS (e.g. "SibSp + Parch + 1")
- rationale: one sentence explaining predictive value

Types:
- interaction: left * right  (two NUMERIC columns only)
- ratio: left / right  (two NUMERIC columns only, handles division-by-zero)
- datetime: extract year/month/day/hour/dayofweek from left
- domain_combination: arithmetic expression using ONLY NUMERIC_COLUMNS names

CRITICAL: Do NOT reference string/categorical columns. Only use names from NUMERIC_COLUMNS.
Return JSON only:
{{"features": [{{"name": "...", "type": "...", "left": "...", "right": "...", "formula": "...", "rationale": "..."}}]}}"""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
    parsed = extract_json_object(raw) if raw else None
    proposals: list[dict[str, Any]] = []
    if parsed and isinstance(parsed.get("features"), list):
        proposals = [x for x in parsed["features"] if isinstance(x, dict)]

    out = df.copy()
    out_test = df_test.copy() if df_test is not None else None
    log: list[dict[str, Any]] = []
    leakage_count = 0
    low_corr_count = 0

    for p in proposals[:8]:
        name = str(p.get("name") or "derived")
        typ = str(p.get("type") or "ratio")
        left = p.get("left")
        right = p.get("right")
        formula = str(p.get("formula") or "").strip()
        rationale = str(p.get("rationale") or "")

        # Scenario: references a dropped column → silently discard + log
        referenced_cols = [c for c in [left, right] if c]
        missing_refs = [c for c in referenced_cols if c and c not in out.columns]
        if missing_refs:
            entry = {"feature": name, "status": "discarded_missing_column",
                     "missing": missing_refs, "rationale": rationale}
            log.append(entry)
            _log(session, {"action": "discard_feature_missing_cols", "feature": name,
                           "missing": missing_refs})
            continue

        try:
            if typ == "datetime" and left and left in out.columns:
                s = pd.to_datetime(out[left], errors="coerce")
                candidates = {
                    f"{name}_year": s.dt.year,
                    f"{name}_month": s.dt.month,
                    f"{name}_day": s.dt.day,
                    f"{name}_hour": s.dt.hour,
                    f"{name}_dayofweek": s.dt.dayofweek,
                }
                s_test = (
                    pd.to_datetime(out_test[left], errors="coerce")
                    if out_test is not None and left in out_test.columns else None
                )
                candidates_test = {
                    f"{name}_year": s_test.dt.year,
                    f"{name}_month": s_test.dt.month,
                    f"{name}_day": s_test.dt.day,
                    f"{name}_hour": s_test.dt.hour,
                    f"{name}_dayofweek": s_test.dt.dayofweek,
                } if s_test is not None else {}
                for c, vals in candidates.items():
                    vals = pd.to_numeric(vals, errors="coerce")
                    fill_val = None
                    if vals.isna().any():
                        fill_val = float(vals.median())
                        vals = vals.fillna(fill_val)
                    cc = float(abs(_corr_with_target(vals, y)))
                    if cc > _LEAKAGE_CORR_THRESHOLD:
                        leakage_count += 1
                        log.append({"feature": c, "status": "rejected_leakage",
                                    "corr": cc, "rationale": rationale})
                        _log(session, {"action": "reject_leakage", "feature": c, "corr": cc})
                    elif cc < _MIN_USEFUL_CORR:
                        low_corr_count += 1
                        log.append({"feature": c, "status": "rejected_low_corr",
                                    "corr": cc, "rationale": rationale})
                        _log(session, {"action": "reject_low_corr", "feature": c, "corr": cc})
                    else:
                        out[c] = vals
                        log.append({"feature": c, "status": "added", "rationale": rationale})
                        _log(session, {"action": "add_feature", "feature": c, "rationale": rationale})
                        if out_test is not None:
                            if c in candidates_test:
                                vals_test = pd.to_numeric(candidates_test[c], errors="coerce")
                                vals_test = vals_test.fillna(fill_val if fill_val is not None else 0.0)
                            else:
                                vals_test = fill_val if fill_val is not None else 0.0
                            out_test[c] = vals_test
                continue

            feat_test = None
            can_apply_test = out_test is not None

            if typ == "domain_combination" and formula:
                # Evaluate formula using only numeric columns — categorical columns
                # are intentionally excluded to prevent string-arithmetic errors.
                local_ns = {c: out[c] for c in out.columns if pd.api.types.is_numeric_dtype(out[c])}
                feat = eval(formula, {"__builtins__": {}}, local_ns)  # noqa: S307
                feat = pd.Series(feat, index=out.index, name=name)
                # Guard: if formula produced non-numeric output (e.g. string Series),
                # skip — attempting median/corr on strings raises ValueError.
                if not pd.api.types.is_numeric_dtype(feat):
                    logger.debug("engineer_features: skip %s — formula returned non-numeric dtype %s", name, feat.dtype)
                    log.append({"feature": name, "status": "skipped_non_numeric_formula"})
                    continue
                if can_apply_test:
                    try:
                        local_ns_test = {c: out_test[c] for c in out_test.columns if pd.api.types.is_numeric_dtype(out_test[c])}
                        feat_test = eval(formula, {"__builtins__": {}}, local_ns_test)  # noqa: S307
                        feat_test = pd.Series(feat_test, index=out_test.index, name=name)
                    except Exception:
                        feat_test = None
            elif (
                typ in ("ratio", "interaction")
                and left in out.columns
                and right in out.columns
                and pd.api.types.is_numeric_dtype(out[left])
                and pd.api.types.is_numeric_dtype(out[right])
            ):
                a = pd.to_numeric(out[left], errors="coerce")
                b = pd.to_numeric(out[right], errors="coerce")
                feat = a / (b.replace(0, np.nan) + 1e-12) if typ == "ratio" else a * b
                if can_apply_test and left in out_test.columns and right in out_test.columns:
                    a_test = pd.to_numeric(out_test[left], errors="coerce")
                    b_test = pd.to_numeric(out_test[right], errors="coerce")
                    feat_test = a_test / (b_test.replace(0, np.nan) + 1e-12) if typ == "ratio" else a_test * b_test
            else:
                log.append({"feature": name, "status": "skipped_unknown_type", "type": typ})
                continue

            # Fill NaN created by ratio (zero denominator) or domain_combination formulas
            fill_val = None
            if feat.isna().any():
                fill_val = float(feat.median())
                feat = feat.fillna(fill_val)
            if feat_test is not None and feat_test.isna().any() and fill_val is not None:
                feat_test = feat_test.fillna(fill_val)

            cc = float(abs(_corr_with_target(feat, y)))
            if cc > _LEAKAGE_CORR_THRESHOLD:
                leakage_count += 1
                log.append({"feature": name, "status": "rejected_leakage",
                            "corr": cc, "rationale": rationale})
                _log(session, {"action": "reject_leakage", "feature": name, "corr": cc})
            elif cc < _MIN_USEFUL_CORR:
                low_corr_count += 1
                log.append({"feature": name, "status": "rejected_low_corr",
                            "corr": cc, "rationale": rationale})
                _log(session, {"action": "reject_low_corr", "feature": name, "corr": cc})
            else:
                out[name] = feat
                log.append({"feature": name, "status": "added", "rationale": rationale})
                _log(session, {"action": "add_feature", "feature": name, "rationale": rationale})
                if out_test is not None:
                    out_test[name] = feat_test if feat_test is not None else (fill_val if fill_val is not None else 0.0)

        except Exception as e:
            logger.warning("engineer_features: skip %s: %s", name, e)
            log.append({"feature": name, "status": "error", "detail": str(e)})

    # ── Scenario: all proposals failed leakage check ──────────────────────────
    added = [e for e in log if e.get("status") == "added"]
    if proposals and not added and leakage_count > 0:
        note = "Feature engineering did not yield usable features due to high correlation with target."
        session.eda_narrative = (session.eda_narrative or "") + "\n\n" + note
        _log(session, {"action": "fe_all_leakage", "n_proposals": len(proposals)})
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="all_proposals_failed_leakage", n_proposals=len(proposals))

    session.feature_engineering_log = log
    session.dataset_profile = compute_dataset_profile(
        out, target_column=target_s)
    session.dataset_state_pointer = "after_feature_engineering"
    log_decision(session, step="Phase 3 - Feature Engineering",
                 action="complete", added=len(added), total_proposals=len(proposals))

    if out_test is not None:
        # Defensive: guarantee identical column set/order to train.
        out_test = out_test.reindex(columns=out.columns, fill_value=0.0)
        return out, out_test
    return out


