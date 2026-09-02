"""Phase 1 Step 1: parse user intent and extract target column when inferable."""

from __future__ import annotations

import logging
import re
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_STATE_KEYWORDS = ("in_scope", "ambiguous", "out_of_scope", "unrecognizable")


def _fallback_parse_state(raw: str) -> str:
    """Extract state from raw LLM text when JSON parsing fails."""
    lower = raw.lower()
    for kw in _STATE_KEYWORDS:
        if kw.replace("_", " ") in lower or kw in lower:
            return kw
    # Keyword-based heuristics
    if any(w in lower for w in ("predict", "classify", "regression", "classification")):
        return "in_scope"
    if any(w in lower for w in ("clarify", "unclear", "which column", "what do you mean")):
        return "ambiguous"
    if any(w in lower for w in ("not support", "cannot", "out of scope", "visualization")):
        return "out_of_scope"
    return "unrecognizable"


def parse_user_intent(user_instruction: str, column_names: list[str], session: SessionState) -> dict[str, Any]:
    """
    LLM classifies state and extracts target column when possible.
    Returns dict with keys matching the Phase 1 spec schema:
      state, task_type, goal, target_variable, user_constraints, message,
      metric_hint, clarifying_question, target_column, target_status, llm_unavailable.
    """
    cols_preview = column_names[:120]
    prompt = f"""You classify a tabular ML prediction request.

USER_INSTRUCTION:
{user_instruction}

DATASET_COLUMNS (names only):
{cols_preview}

Return JSON only — no markdown, no explanation:
{{
  "state": "in_scope" | "ambiguous" | "out_of_scope" | "unrecognizable",
  "task_type": "classification" | "regression" | null,
  "goal": "<plain English restatement of the prediction goal>",
  "target_variable": "<exact column name from DATASET_COLUMNS> | deferred",
  "user_constraints": ["<list of constraints mentioned by the user>"],
  "message": "<one sentence: what to show the user if not in_scope>",
  "metric_hint": "<any metric preference mentioned, e.g. accuracy, AUC> | null"
}}

Rules:
- "in_scope": user wants classification or regression on tabular data.
- If the user names a target explicitly (e.g. "predict Survived"), set target_variable to that exact column name.
- If you cannot map to a specific column, set target_variable to "deferred".
- "ambiguous": goal is unclear — set message to one clarifying question.
- "out_of_scope": not a supervised ML task on tabular data — explain in message.
- "unrecognizable": instruction is incomprehensible — set message to one clarifying question."""

    raw = LLMClient().call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)

    # LLM unavailable (empty response after all retries)
    if not raw:
        logger.warning("Intent parser: LLM returned empty - marking unavailable.")
        return {
            "state": "in_scope",
            "task_type": None,
            "goal": user_instruction[:500],
            "target_variable": "deferred",
            "user_constraints": [],
            "message": "The AI service is temporarily unavailable. Please try again in a moment.",
            "metric_hint": None,
            "target_column": None,
            "target_status": "deferred",
            "llm_unavailable": True,
        }

    # Try structured JSON parse first; fall back to string matching
    parsed = extract_json_object(raw)
    if not parsed:
        logger.warning("Intent parser: JSON parse failed; using string fallback on raw=%r", raw[:200])
        fallback_state = _fallback_parse_state(raw)
        parsed = {"state": fallback_state}
        log_decision(session, step="Phase 1 - Intent parsing", action="json_parse_fallback",
                     rationale="LLM returned non-JSON", raw_snippet=raw[:300])

    out: dict[str, Any] = {
        "state": "in_scope",
        "task_type": None,
        "goal": user_instruction[:500],
        "target_variable": "deferred",
        "user_constraints": [],
        "message": "",
        "metric_hint": None,
        "target_column": None,
        "target_status": "deferred",
        "llm_unavailable": False,
    }
    for k in ("state", "task_type", "goal", "target_variable", "user_constraints",
              "message", "metric_hint"):
        if k in parsed and parsed[k] is not None:
            out[k] = parsed[k]

    # Normalise state (accept legacy "scope" key if LLM slips back)
    if "scope" in parsed and out["state"] == "in_scope" and parsed.get("scope") in _STATE_KEYWORDS:
        out["state"] = parsed["scope"]

    # Resolve target_variable → target_column + target_status (case-insensitive match)
    tv = out.get("target_variable")
    col_lower_map = {c.lower(): c for c in column_names}  # lowercase → original name
    if tv and tv != "deferred":
        tv_str = str(tv).strip()
        matched = col_lower_map.get(tv_str.lower())
        if matched:
            out["target_column"] = matched  # use original casing from dataset
            out["target_status"] = "inferred"
        else:
            # LLM proposed a column name not in dataset — warn and defer
            logger.warning("Intent parser proposed unknown column %r; deferring.", tv_str)
            out["target_variable"] = "deferred"
            out["target_status"] = "deferred"
    else:
        out["target_status"] = "deferred"

    # Populate session
    session.task_config = dict(session.task_config or {})
    session.task_config["goal"] = str(out.get("goal") or user_instruction)
    session.task_config["task_type"] = out.get("task_type") or "unknown"
    session.task_config["intent_state"] = out.get("state")
    session.task_config["user_constraints"] = out.get("user_constraints") or []
    if out.get("metric_hint"):
        session.task_config["metric_hint"] = str(out["metric_hint"])

    tc = out.get("target_column")
    if tc:
        session.task_config["target_column"] = str(tc)

    log_decision(
        session,
        step="Phase 1 - Intent parsing",
        action="parse_user_intent",
        rationale=str(out.get("goal") or ""),
        state=out.get("state"),
        target_status=out.get("target_status"),
        target_column=out.get("target_column"),
    )
    return out
