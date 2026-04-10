"""Phase 3 Step 3: feature selection — code flags, one LLM call for all decisions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_CORR_FLAG_THRESHOLD = 0.95
_VARIANCE_DOMINANCE_THRESHOLD = 0.98   # flag when top-value share >= 98%
_ID_NAME_TOKENS = ("id", "index", "key", "uuid", "guid", "rowid", "row_id")


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "feature_selection", **entry})


def _code_flag_candidates(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """
    Pure code checks — no drops yet. Returns flags only.

    1. Variance thresholding: columns where >=98% of values are the same
    2. Correlation thresholding: feature pairs with |corr| > 0.95
    3. Identifier detection: cardinality == row count OR name contains id/index/key/uuid
    """
    n = len(df)
    low_variance: list[dict[str, Any]] = []
    for c in df.columns:
        if c == target:
            continue
        vc = df[c].value_counts(dropna=False, normalize=True)
        if len(vc) > 0 and float(vc.iloc[0]) >= _VARIANCE_DOMINANCE_THRESHOLD:
            low_variance.append({
                "column": c,
                "kind": "low_variance",
                "top_value_share": round(float(vc.iloc[0]), 4),
                "top_value": str(vc.index[0]),
            })

    id_like: list[dict[str, Any]] = []
    for c in df.columns:
        if c == target:
            continue
        nu = df[c].nunique(dropna=True)
        name_l = str(c).lower()
        by_card = (n > 0 and nu >= n)
        by_name = any(tok == name_l or name_l.startswith(tok + "_") or name_l.endswith("_" + tok)
                      for tok in _ID_NAME_TOKENS)
        if by_card or by_name:
            id_like.append({
                "column": c,
                "kind": "identifier",
                "reason": "cardinality_equals_row_count" if by_card else "name_pattern",
                "nunique": nu,
            })

    corr_pairs: list[dict[str, Any]] = []
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) >= 2:
        sub = df[numeric].dropna()
        if len(sub) > 2:
            cm = sub.corr()
            seen: set[tuple[str, str]] = set()
            for i, a in enumerate(numeric):
                for b in numeric[i + 1:]:
                    if a not in cm.index or b not in cm.columns:
                        continue
                    r = abs(float(cm.loc[a, b]))
                    if r > _CORR_FLAG_THRESHOLD:
                        key = tuple(sorted((a, b)))
                        if key in seen:
                            continue
                        seen.add(key)
                        va = float(df[a].var(skipna=True) or 0)
                        vb = float(df[b].var(skipna=True) or 0)
                        corr_pairs.append({
                            "col_a": a, "col_b": b,
                            "correlation": round(r, 4),
                            "lower_variance_col": a if va <= vb else b,
                        })

    return {
        "low_variance": low_variance,
        "identifier_like": id_like,
        "high_correlation_pairs": corr_pairs,
    }


def select_features(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Step 3: Code flags candidates, then ONE LLM call decides drop/keep for each
    flagged column. LLM is also asked about any unflagged columns that should be
    dropped for domain reasons.

    Scenarios:
    - LLM drops >50% of columns → unusual flag logged, user notified.
    - LLM keeps a code-flagged identifier → allowed with rationale logged.
    - After selection, fewer than 2 feature columns remain → pipeline stop.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if not target_s:
        target_s = None

    flags = _code_flag_candidates(df, target_s)
    n_cols = df.shape[1]

    _log(session, {
        "action": "code_flags",
        "low_variance": [f["column"] for f in flags["low_variance"]],
        "identifier_like": [f["column"] for f in flags["identifier_like"]],
        "high_corr_pairs": len(flags["high_correlation_pairs"]),
    })

    # Build flat flagged-column list for LLM
    all_flags: list[dict[str, Any]] = []
    for f in flags["low_variance"]:
        all_flags.append({
            "column": f["column"],
            "flag": "low_variance",
            "detail": f"top value '{f['top_value']}' covers {f['top_value_share']*100:.1f}% of rows",
        })
    for f in flags["identifier_like"]:
        all_flags.append({
            "column": f["column"],
            "flag": "identifier",
            "detail": f"reason: {f['reason']}, nunique={f['nunique']}",
        })
    for p in flags["high_correlation_pairs"]:
        all_flags.append({
            "column": p["lower_variance_col"],
            "flag": "high_correlation",
            "detail": f"corr={p['correlation']} with {p['col_a'] if p['lower_variance_col'] == p['col_b'] else p['col_b']}",
        })

    client = LLMClient()
    feature_cols = [c for c in df.columns if c != target_s]
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    prompt = f"""Phase 3 Step 3 — Feature Selection.{revision_block}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2500]}

ALL_FEATURE_COLUMNS: {feature_cols[:80]}

FLAGGED_COLUMNS (code found these — decide drop or keep for each):
{all_flags[:60]}

For each flagged column, decide: DROP or KEEP. Provide one-sentence rationale.
Also list any unflagged columns you think should be dropped (domain reasoning, e.g. name fields, free text, obvious non-predictors).

IMPORTANT:
- Never drop the target column: {target_s!r}
- For high_correlation pairs, drop at most ONE column per pair
- You MAY keep a code-flagged identifier if it carries genuine signal — explain why

Return JSON only:
{{
  "flagged_decisions": [{{"column": "...", "decision": "drop"|"keep", "rationale": "one sentence"}}],
  "additional_drops": [{{"column": "...", "rationale": "one sentence"}}]
}}"""

    raw = client.call(prompt)
    parsed = extract_json_object(raw) if raw else None

    drop_cols: list[str] = []
    log_entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _record_drop(col: str, rationale: str, tier: str) -> None:
        if col in seen or col not in df.columns:
            return
        if target_s and col == target_s:
            return
        seen.add(col)
        drop_cols.append(col)
        log_entries.append({"column": col, "rationale": rationale, "tier": tier})
        _log(session, {"action": "drop_feature", "column": col, "rationale": rationale, "tier": tier})

    if parsed:
        flagged_decisions = parsed.get("flagged_decisions") or []
        for d in flagged_decisions:
            if not isinstance(d, dict):
                continue
            col = str(d.get("column") or "")
            decision = str(d.get("decision") or "keep").lower()
            rationale = str(d.get("rationale") or "LLM decision.")
            if not col:
                continue

            # Log if LLM keeps a code-flagged identifier (override noted)
            is_id_flagged = any(f["column"] == col and f["flag"] == "identifier"
                                for f in all_flags)
            if is_id_flagged and decision == "keep":
                _log(session, {
                    "action": "llm_keeps_identifier",
                    "column": col,
                    "rationale": rationale,
                })
                logger.info("select_features: LLM kept identifier-flagged column %r: %s", col, rationale)

            if decision == "drop":
                _record_drop(col, rationale, "llm_flagged")

        additional = parsed.get("additional_drops") or []
        for d in additional:
            if not isinstance(d, dict):
                continue
            col = str(d.get("column") or "")
            rationale = str(d.get("rationale") or "LLM domain reasoning.")
            if col:
                _record_drop(col, rationale, "llm_domain")
    else:
        # LLM unavailable — fall back to dropping all low-variance + identifier flag columns
        logger.warning("select_features: LLM parse failed; applying fallback drops for low-variance+identifier")
        for f in flags["low_variance"]:
            _record_drop(f["column"], "Low variance fallback (LLM unavailable).", "fallback")
        for f in flags["identifier_like"]:
            if f["reason"] == "cardinality_equals_row_count":
                _record_drop(f["column"], "Identifier fallback (LLM unavailable).", "fallback")

    # ── Scenario: LLM drops >50% of feature columns ───────────────────────────
    n_features = len(feature_cols)
    if n_features > 0 and len(drop_cols) > n_features * 0.5:
        pct = len(drop_cols) / n_features * 100
        warning_msg = (
            f"Feature selection dropped {len(drop_cols)} of {n_features} feature columns "
            f"({pct:.0f}%). This is unusually high and may indicate a data quality issue."
        )
        _log(session, {"action": "high_drop_warning", "dropped": len(drop_cols),
                       "total_features": n_features, "pct": pct})
        session.preprocessing_meta["feature_selection_high_drop_warning"] = warning_msg
        logger.warning(warning_msg)

    out = df.drop(columns=drop_cols, errors="ignore")

    # ── Scenario: fewer than 2 feature columns remain ─────────────────────────
    remaining_features = [c for c in out.columns if c != target_s]
    if len(remaining_features) < 2:
        msg = (
            f"After removing non-predictive columns, your dataset has only "
            f"**{len(remaining_features)}** usable feature(s). "
            "This is insufficient for a reliable model. Please review your dataset."
        )
        session.task_config["_pipeline_stop"] = "too_few_features"
        log_decision(session, step="Phase 3 - Feature Selection",
                     action="too_few_features_pipeline_stop",
                     remaining_features=remaining_features)
        session.feature_selection_log = log_entries
        session.preprocessing_meta["feature_selection_stop_message"] = msg
        return out   # orchestrator will catch _pipeline_stop

    session.feature_selection_log = log_entries
    log_decision(session, step="Phase 3 - Feature Selection",
                 action="selection_complete",
                 dropped_count=len(drop_cols), remaining=list(out.columns))
    session.dataset_state_pointer = "after_feature_selection"
    session.preprocessing_meta["feature_selection_columns"] = list(out.columns)
    logger.info("select_features: dropped %d columns, %d remain", len(drop_cols), len(out.columns))
    return out
