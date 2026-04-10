"""Phase 3 Step 5: LLM-proposed feature engineering with leakage guard."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

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


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "feature_engineering", **entry})


def _corr_with_target(feat: pd.Series, y: pd.Series) -> float:
    sub = pd.DataFrame({"f": feat, "y": y}).dropna()
    if len(sub) < 3:
        return 0.0
    if sub["f"].nunique() <= 1 or sub["y"].nunique() <= 1:
        return 0.0
    return float(sub["f"].corr(sub["y"]))


def engineer_features(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Step 5: LLM proposes features, code runs leakage check before each creation.

    Proposal types: interaction, ratio, datetime, domain_combination
    Scenarios:
    - dataset < 200 rows → skip entirely, no LLM call
    - proposals reference dropped columns → silently discard + log
    - all proposals fail leakage check → log, add note to eda_narrative
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if not target_s or target_s not in df.columns:
        session.feature_engineering_log = []
        return df

    n_rows = len(df)

    # ── Scenario: dataset too small → skip entirely ───────────────────────────
    if n_rows < _MIN_ROWS_FOR_FE:
        note = f"Dataset too small for reliable feature engineering ({n_rows} rows < {_MIN_ROWS_FOR_FE} threshold)."
        session.feature_engineering_log = [{"status": "skipped", "reason": note}]
        _log(session, {"action": "fe_skipped", "reason": note, "n_rows": n_rows})
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="skipped_too_small", n_rows=n_rows)
        return df

    y = df[target_s]
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

    client = LLMClient()
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    prompt = f"""Phase 3 Step 5 — Feature Engineering. Propose new derived features.{revision_block}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2000]}
DATASET_ROWS: {n_rows}

NUMERIC_COLUMNS: {numeric[:40]}
DATETIME_LIKE: {dt_cols}
ALL_COLUMNS: {list(df.columns)[:80]}

Propose up to 8 candidate features. For each provide:
- name: short snake_case name
- type: "interaction" | "ratio" | "datetime" | "domain_combination"
- left: column name (or formula for domain_combination)
- right: column name (empty for datetime or domain_combination)
- formula: e.g. "SibSp + Parch + 1" for domain_combination (optional, leave empty if using left/right)
- rationale: one sentence explaining predictive value

Types:
- interaction: left * right  (two numeric columns)
- ratio: left / right  (two numeric columns, handles division-by-zero)
- datetime: extract year/month/day/hour/dayofweek from left
- domain_combination: arbitrary expression (e.g. family_size = SibSp + Parch + 1, bmi_age = BMI * Age)

IMPORTANT: Only reference columns that exist in ALL_COLUMNS.
Return JSON only:
{{"features": [{{"name": "...", "type": "...", "left": "...", "right": "...", "formula": "...", "rationale": "..."}}]}}"""

    raw = client.call(prompt)
    parsed = extract_json_object(raw) if raw else None
    proposals: list[dict[str, Any]] = []
    if parsed and isinstance(parsed.get("features"), list):
        proposals = [x for x in parsed["features"] if isinstance(x, dict)]

    out = df.copy()
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
                for c, vals in candidates.items():
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
                continue

            if typ == "domain_combination" and formula:
                # Evaluate formula in namespace of current columns
                local_ns = {c: out[c] for c in out.columns if pd.api.types.is_numeric_dtype(out[c])}
                feat = eval(formula, {"__builtins__": {}}, local_ns)  # noqa: S307
                feat = pd.Series(feat, index=out.index, name=name)
            elif typ in ("ratio", "interaction") and left in out.columns and right in out.columns:
                a = pd.to_numeric(out[left], errors="coerce")
                b = pd.to_numeric(out[right], errors="coerce")
                feat = a / (b.replace(0, np.nan) + 1e-12) if typ == "ratio" else a * b
            else:
                log.append({"feature": name, "status": "skipped_unknown_type", "type": typ})
                continue

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
    return out


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "feature_engineering", **entry})


def _corr_with_target(feat: pd.Series, y: pd.Series) -> float:
    sub = pd.DataFrame({"f": feat, "y": y}).dropna()
    if len(sub) < 3:
        return 0.0
    if sub["f"].nunique() <= 1 or sub["y"].nunique() <= 1:
        return 0.0
    return float(sub["f"].corr(sub["y"]))


def engineer_features(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Decision Router gates the LLM call: if the dataset is too small, has too few
    numeric columns, or the preprocessing plan flags FE as unlikely, the LLM call
    is skipped entirely and df is returned unchanged.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if not target_s or target_s not in df.columns:
        session.feature_engineering_log = []
        return df

    # ── Tier 1 gate: should we even attempt feature engineering? ───────────────
    fe_stats = RouterStats()
    proceed, gate_reason = should_engineer_features(df, session, fe_stats)
    _log(session, {"action": "fe_gate", "proceed": proceed, "reason": gate_reason})
    logger.info("engineer_features gate: proceed=%s | %s", proceed, gate_reason)
    if not proceed:
        session.feature_engineering_log = [{"status": "skipped", "reason": gate_reason}]
        return df

    y = df[target_s]
    numeric = [c for c in df.columns if c != target_s and pd.api.types.is_numeric_dtype(df[c])]
    dt_cols: list[str] = []
    for c in df.columns:
        if c == target_s:
            continue
        if "date" in str(c).lower():
            dt_cols.append(c)
            continue
        try:
            if np.issubdtype(df[c].dtype, np.datetime64):
                dt_cols.append(c)
        except (TypeError, ValueError):
            pass

    client = LLMClient()
    prompt = f"""Propose derived features for tabular ML.

TASK: {task}
EDA (truncated): {(session.eda_narrative or '')[:2000]}

NUMERIC_COLUMNS: {numeric[:40]}
DATETIME_LIKE: {dt_cols}

Return JSON:
{{"features": [
  {{"name": "feat_name", "type": "ratio"|"interaction"|"datetime", "left": "colA", "right": "colB", "rationale": "one sentence"}},
  ...
]}}

Max 8 features. Ratios: left/right for numeric. Interaction: multiply two numerics. Datetime: set left to datetime column, right empty — we expand to year/month/day.
Skip if columns missing."""

    raw = client.call(prompt)
    parsed = extract_json_object(raw) if raw else None
    proposals: list[dict[str, Any]] = []
    if parsed and isinstance(parsed.get("features"), list):
        proposals = [x for x in parsed["features"] if isinstance(x, dict)]

    out = df.copy()
    log: list[dict[str, Any]] = []

    for p in proposals[:8]:
        name = str(p.get("name") or "derived")
        typ = str(p.get("type") or "ratio")
        left = p.get("left")
        right = p.get("right")
        rationale = str(p.get("rationale") or "")
        try:
            if typ == "datetime" and left and left in out.columns:
                s = pd.to_datetime(out[left], errors="coerce")
                new = pd.DataFrame(
                    {
                        f"{name}_year": s.dt.year,
                        f"{name}_month": s.dt.month,
                        f"{name}_day": s.dt.day,
                    },
                    index=out.index,
                )
                for c in new.columns:
                    cc = float(abs(_corr_with_target(new[c], y)))
                    if cc > _LEAKAGE_CORR_THRESHOLD:
                        log.append(
                            {
                                "feature": c,
                                "status": "rejected_leakage",
                                "corr": cc,
                                "rationale": rationale,
                            }
                        )
                        _log(session, {"action": "reject_feature", "feature": c, "corr": cc})
                    elif cc < _MIN_USEFUL_CORR:
                        log.append({"feature": c, "status": "rejected_low_corr", "corr": cc, "rationale": rationale})
                        _log(session, {"action": "reject_feature_low_corr", "feature": c, "corr": cc})
                    else:
                        out[c] = new[c]
                        log.append({"feature": c, "status": "added", "rationale": rationale})
                continue

            if typ in ("ratio", "interaction") and left in out.columns and right in out.columns:
                a = pd.to_numeric(out[left], errors="coerce")
                b = pd.to_numeric(out[right], errors="coerce")
                if typ == "ratio":
                    feat = a / (b.replace(0, np.nan) + 1e-12)
                else:
                    feat = a * b
                cc = float(abs(_corr_with_target(feat, y)))
                if cc > _LEAKAGE_CORR_THRESHOLD:
                    log.append(
                        {
                            "feature": name,
                            "status": "rejected_leakage",
                            "corr": cc,
                            "rationale": rationale,
                        }
                    )
                    _log(session, {"action": "reject_feature", "feature": name, "corr": cc})
                    continue
                if cc < _MIN_USEFUL_CORR:
                    log.append({"feature": name, "status": "rejected_low_corr", "corr": cc, "rationale": rationale})
                    _log(session, {"action": "reject_feature_low_corr", "feature": name, "corr": cc})
                    continue
                out[name] = feat
                log.append({"feature": name, "status": "added", "rationale": rationale})
                _log(session, {"action": "add_feature", "feature": name, "rationale": rationale})
        except Exception as e:
            logger.warning("engineer_features: skip %s: %s", name, e)
            log.append({"feature": name, "status": "error", "detail": str(e)})

    session.feature_engineering_log = log
    session.dataset_profile = compute_dataset_profile(out)
    session.dataset_state_pointer = "after_feature_engineering"
    return out
