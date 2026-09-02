"""Phase 2: Planning Mode 1 — preprocessing plan from task + dataset profile."""

from __future__ import annotations

import logging
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.llm_json import extract_json_object
from automl_system.utils.profile_conditioner import condition_profile_for_preprocessing

logger = logging.getLogger(__name__)

_DEFAULT_PLAN = {
    "null_strategy_notes": {},
    "encoding_candidates": [],
    "scaling_needed": True,
    "feature_engineering_recommended": False,
    "leakage_risks": [],
    "skewed_columns": [],
    "notes": [],
}


def _deterministic_findings(profile: dict[str, Any]) -> dict[str, Any]:
    """Extract key findings from profile using conditioned view."""
    conditioned = condition_profile_for_preprocessing(profile)
    
    # Convert structured findings back to list format for backward compatibility
    high_null = list(conditioned.get("high_null_columns", {}).keys())
    categorical = list(conditioned.get("categorical_columns", {}).keys())
    skewed = list(conditioned.get("skewed_columns", {}).keys())
    leakage_id = conditioned.get("identifier_candidates", [])
    n_rows = int(profile.get("n_rows") or 0)
    n_feat = int(profile.get("n_columns") or 0)
    
    return {
        "high_null_columns": high_null,
        "categorical_columns": categorical,
        "skewed_numeric": skewed,
        "identifier_like_columns": leakage_id,
        "n_rows": n_rows,
        "n_features": n_feat,
    }


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "planning", **entry})


def make_preprocessing_plan(session: SessionState) -> dict[str, Any]:
    """
    LLM reads task_config and dataset_profile; returns structured plan dict.
    Stored fields: null_strategy, encoding_strategy, scaling_needed,
    feature_engineering_likely, leakage_risks, notes.
    Also sets session.preprocessing_plan and session.user_summary_planning.
    """
    profile = session.dataset_profile or {}
    task = session.task_config or {}
    findings = _deterministic_findings(profile)
    
    n_rows = findings["n_rows"]
    n_feat = findings["n_features"]

    client = LLMClient()
    prompt = f"""You are planning a tabular ML preprocessing pipeline (Mode 1 — single call, advisory plan).
The plan is a guide; downstream agents may deviate when evidence warrants it.

TASK_CONFIG (JSON):
{task}

DATASET_PROFILE_SUMMARY:
- rows: {n_rows}, columns: {n_feat}
- columns with >30% null: {findings['high_null_columns']}
- categorical-like columns: {findings['categorical_columns'][:40]}{'...' if len(findings['categorical_columns']) > 40 else ''}
- skewed numeric (|skew|>1): {findings['skewed_numeric'][:40]}
- potential ID / leakage (cardinality == row count): {findings['identifier_like_columns']}

Return ONE JSON object — keys are ALL required:
- null_strategy_notes: dict mapping each column with nulls to a short imputation note (e.g. {{"Age": "median imputation — numeric with 5% missing", "Cabin": "drop — >77% missing"}})
- encoding_candidates: array of {{"column": "...", "method": "one_hot|ordinal|target|binary"}} for columns that need encoding
- scaling_needed: boolean (true if numeric features span very different ranges or distance-based models are likely)
- feature_engineering_recommended: boolean — true only if row count >= 500 AND feature count >= 3; explain in notes
- leakage_risks: array of column name strings that are potential leakage risks (IDs, post-event fields, etc.)
- skewed_columns: array of column names that likely need log/power transformation
- notes: array of short bullet strings for any other observations
- user_summary: 2–4 plain-English sentences for an end user (no JSON, no raw column dumps)

JSON only, no markdown outside the object."""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
    parsed = extract_json_object(raw) if raw else None
    plan: dict[str, Any] = dict(_DEFAULT_PLAN)
    user_summary = ""

    if parsed:
        for k in (
            "null_strategy_notes",
            "encoding_candidates",
            "scaling_needed",
            "feature_engineering_recommended",
            "leakage_risks",
            "skewed_columns",
            "notes",
        ):
            if k in parsed:
                plan[k] = parsed[k]
        user_summary = str(parsed.get("user_summary") or "").strip()

    if not user_summary:
        user_summary = (
            f"The plan targets {n_rows} rows and {n_feat} columns. "
            f"High-null columns will use robust imputation; "
            f"{'encoding and scaling are recommended' if plan.get('scaling_needed') else 'encoding is prioritized'}. "
            f"{'Feature engineering may help given dataset size.' if plan.get('feature_engineering_recommended') else 'Feature engineering is deferred given size or simplicity.'}"
        )

    if not raw or not parsed:
        plan["notes"] = list(plan.get("notes") or []) + [
            "LLM plan unavailable; applied default heuristics from deterministic findings."
        ]
        logger.warning("make_preprocessing_plan: LLM parse failed; using defaults")

    session.preprocessing_plan = plan
    session.user_summary_planning = user_summary
    _log(
        session,
        {
            "action": "make_preprocessing_plan",
            "summary": user_summary[:500],
            "findings": findings,
        },
    )
    return plan
