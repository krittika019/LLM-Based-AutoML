"""Phase 6: diagnose why threshold was not met and propose targeted revision."""

from __future__ import annotations

import logging
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.memory import semantic as semantic_memory
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import build_semantic_profile_vector
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_ALLOWED_FAILURE_TYPES = {
    "data_quality_issue",
    "feature_issue",
    "model_issue",
    "fundamental_data_insufficiency",
}

_ALLOWED_RESPONSIBLE_PHASES = {
    "basic_cleaning",
    "deep_cleaning",
    "feature_selection",
    "feature_engineering",
    "phase4_planning",
}


def _compact_table(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for r in rows[:limit]:
        out.append(
            {
                "model_name": r.get("model_name"),
                "cv_score": r.get("cv_score"),
                "test_score": r.get("test_score"),
                "fold_variance": r.get("fold_variance"),
                "status": r.get("status"),
            }
        )
    return out


def _compact_decisions(entries: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "phase": e.get("phase"),
            "action": e.get("action"),
            "note": str(e.get("text") or e.get("summary") or e.get("message") or "")[:220],
        }
        for e in entries[-limit:]
    ]


def _fallback_diagnosis(session: SessionState, results: dict[str, Any]) -> dict[str, Any]:
    table = list(results.get("comparison_table") or [])
    if not table:
        target = str((session.task_config or {}).get("target_column") or "").strip()
        if not target:
            return {
                "failure_type": "fundamental_data_insufficiency",
                "responsible_phase": "phase4_planning",
                "proposed_fix": "Provide a valid target column to enable supervised training.",
                "should_exit": True,
            }
        return {
            "failure_type": "model_issue",
            "responsible_phase": "phase4_planning",
            "proposed_fix": "Broaden model shortlist and search space, and align metric with business objective.",
            "should_exit": False,
        }

    high_var = any(
        isinstance(r.get("fold_variance"), (int, float)) and float(r.get("fold_variance") or 0) > 0.02
        for r in table
    )
    large_gap = any(
        isinstance(r.get("cv_score"), (int, float))
        and isinstance(r.get("test_score"), (int, float))
        and (float(r["cv_score"]) - float(r["test_score"]) > 0.15)
        for r in table
    )

    if large_gap:
        return {
            "failure_type": "feature_issue",
            "responsible_phase": "feature_engineering",
            "proposed_fix": "Tighten leakage guard and remove unstable derived features with high train-CV/test divergence.",
            "should_exit": False,
        }
    if high_var:
        return {
            "failure_type": "data_quality_issue",
            "responsible_phase": "deep_cleaning",
            "proposed_fix": "Strengthen imputation/outlier strategy for high-missing and heavy-tail columns before retraining.",
            "should_exit": False,
        }
    return {
        "failure_type": "model_issue",
        "responsible_phase": "phase4_planning",
        "proposed_fix": "Retune model family choices and hyperparameter ranges using a stronger non-linear baseline.",
        "should_exit": False,
    }


def _sanitize_diagnosis(parsed: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)
    if not parsed:
        return out

    ft = str(parsed.get("failure_type") or "").strip()
    if ft in _ALLOWED_FAILURE_TYPES:
        out["failure_type"] = ft

    rp = str(parsed.get("responsible_phase") or "").strip()
    if rp in _ALLOWED_RESPONSIBLE_PHASES:
        out["responsible_phase"] = rp

    fix = str(parsed.get("proposed_fix") or "").strip()
    if fix:
        out["proposed_fix"] = fix

    if isinstance(parsed.get("should_exit"), bool):
        out["should_exit"] = bool(parsed["should_exit"])

    # Hard rule from spec.
    if out.get("failure_type") == "fundamental_data_insufficiency":
        out["should_exit"] = True

    return out


def diagnose_failure(session: SessionState, results: dict[str, Any]) -> dict[str, Any]:
    """
    Diagnose failure reason and route targeted revision.

    Returns dict with: failure_type, responsible_phase, proposed_fix, should_exit.
    """
    decisions = episodic.get_log(session.decisions_log_id) if session.decisions_log_id else []
    fallback = _fallback_diagnosis(session, results)

    # Semantic memory: retrieve similar failures and their fixes (reference Phase 6 Step 2)
    soft_past: list[dict] = []
    try:
        profile = session.dataset_profile or {}
        pv = build_semantic_profile_vector(profile)
        soft_past = semantic_memory.retrieve_similar(pv, top_k=3)
    except Exception as _sem_err:
        logger.debug("semantic retrieve_similar skipped in failure_diagnosis: %s", _sem_err)

    prompt = f"""You are a senior ML engineer diagnosing why an AutoML pipeline failed to meet its threshold.

Inputs available to you:
- COMPARISON_TABLE: {_compact_table(list(results.get('comparison_table') or []), limit=8)}
- EXECUTION_INTERPRETATION: {(results.get('interpretation') or session.execution_interpretation or '')[:1200]}
- DECISIONS_LOG_RECENT: {_compact_decisions(decisions, limit=12)}
- EDA_NARRATIVE: {(session.eda_narrative or '')[:800]}
- ERROR_MESSAGES: {results.get('errors') or results.get('status') or ''}
- PAST_RUN_SIMILARITIES (advisory — fixes that worked on similar datasets): {soft_past}

Classify the failure into EXACTLY one of these four types based on the signals described:

1. data_quality_issue
   Signals: all models performing near or at a random baseline; high variance across CV folds
   (fold_variance > 0.03); performance far below what similar datasets typically achieve;
   large CV-to-test gap suggesting noise in features.
   Responsible phase: basic_cleaning OR deep_cleaning

2. feature_issue
   Signals: all models converging to the same mediocre score regardless of complexity;
   no feature shows meaningful correlation with target in EDA narrative;
   after feature engineering was applied performance did not improve;
   high cook's distance / constant features still present.
   Responsible phase: feature_selection OR feature_engineering

3. model_issue
   Signals: one model significantly outperformed others but is still below threshold
   (search space may be too narrow); all tried models are from the same family; 
   model family is mismatched to the data structure (e.g. linear model on non-linear data).
   Responsible phase: phase4_planning

4. fundamental_data_insufficiency
   Signals: dataset is too small for the task complexity (< 100 rows after cleaning);
   target has no meaningful statistical relationship with ANY feature;
   data appears to be noise or random labels.
   Action: EXIT the loop immediately. Do NOT propose a fix — there is no fix.
   should_exit must be true.

Rules:
- For non-fundamental failures, provide ONE specific actionable proposed_fix.
- For fundamental_data_insufficiency, set proposed_fix to an honest explanation for the user and should_exit=true.
- responsible_phase must be one of: basic_cleaning, deep_cleaning, feature_selection, feature_engineering, phase4_planning

Return ONLY valid JSON:
{{
  "failure_type": "...",
  "responsible_phase": "...",
  "proposed_fix": "...",
  "should_exit": false
}}"""

    raw = LLMClient().call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed = extract_json_object(raw) if raw else None
    diagnosis = _sanitize_diagnosis(parsed, fallback)

    session.last_failure_diagnosis = diagnosis
    if session.decisions_log_id:
        episodic.append_decision(
            session.decisions_log_id,
            {
                "phase": "iteration",
                "action": "diagnose_failure",
                "diagnosis": diagnosis,
            },
        )

    if not parsed:
        logger.warning("diagnose_failure: LLM output unavailable or invalid; fallback diagnosis used")

    return diagnosis
