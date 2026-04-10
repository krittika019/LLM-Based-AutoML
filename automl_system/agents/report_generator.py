"""Phase 7: produce a plain-English markdown report for the full run."""

from __future__ import annotations

import logging
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import merge_logs_for_report
from automl_system.utils.llm_json import extract_json_object
from automl_system.utils.profile_conditioner import condition_profile_for_report
from automl_system.utils.token_tracker import track_profile_usage

logger = logging.getLogger(__name__)


def _compact_results(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(all_results, start=1):
        table = []
        for row in (r.get("comparison_table") or [])[:8]:
            table.append(
                {
                    "model_name": row.get("model_name"),
                    "cv_score": row.get("cv_score"),
                    "test_score": row.get("test_score"),
                    "status": row.get("status"),
                }
            )
        out.append({"iteration": i, "best_model_name": r.get("best_model_name"), "table": table})
    return out


def _compact_decisions(entries: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for e in entries[-limit:]:
        out.append(
            {
                "step": e.get("step") or e.get("phase"),
                "action": e.get("action"),
                "rationale": str(e.get("rationale") or "")[:200],
            }
        )
    return out


def generate_report(session: SessionState, all_results: list[dict[str, Any]]) -> str:
    """
    Generate a markdown narrative covering data, decisions, model performance, iterations, and recommendation.
    """
    decisions = merge_logs_for_report(session)
    print("[REPORT] Full decisions log (generate_report):")
    for i, e in enumerate(decisions, 1):
        print(f"  {i}. step={e.get('step') or e.get('phase')} action={e.get('action')} ts={e.get('timestamp') or e.get('ts')}")
    if len(decisions) < 5:
        print(
            "WARNING: decisions log is suspiciously short — report may be incomplete "
            f"({len(decisions)} entries)."
        )

    profile = session.dataset_profile or {}
    task = session.task_config or {}
    
    # Track profile compression for this agent
    compression_metrics = track_profile_usage("report_generator", profile, session.eda_results)
    if compression_metrics:
        logger.info(f"report_generator: {compression_metrics['compression_ratio_percent']}% profile compression")

    prompt = f"""Write a concise but complete markdown report for a tabular AutoML run.

USER_CONSTRAINTS_AND_GOAL:
{task}

INITIAL_DATA_PROFILE:
{condition_profile_for_report(profile, session.eda_results)}

ALL_ITERATION_RESULTS:
{_compact_results(all_results)}

COMPLETE_DECISIONS_LOG (ground truth — base the report ONLY on these entries plus results above):
{_compact_decisions(decisions, limit=80)}

The report must cover:
1) What the data looked like initially.
2) Every significant preprocessing decision and why (cite the log).
3) Which models were tried and their performance (use iteration results).
4) What changed across iterations (if any).
5) Final recommendation grounded in user constraints.

Output requirements:
- plain-English markdown
- use short sections with headings
- no JSON, no code blocks"""

    text = LLMClient().call(
        prompt,
        temperature=config.LLM_TEMPERATURE_NARRATIVE,
        max_tokens=config.LLM_MAX_TOKENS_NARRATIVE,
    ).strip()
    if text:
        return text

    # Deterministic fallback.
    best = None
    for r in all_results:
        for row in r.get("comparison_table") or []:
            if isinstance(row.get("test_score"), (int, float)):
                if not best or float(row["test_score"]) > float(best["test_score"]):
                    best = {"model": row.get("model_name"), "test_score": row.get("test_score")}

    return (
        "## AutoML Run Summary\n\n"
        f"- Goal: {task.get('goal') or 'Not specified'}\n"
        f"- Target column: {task.get('target_column') or 'Not provided'}\n"
        f"- Rows: {profile.get('n_rows')}, Columns: {profile.get('n_columns')}\n\n"
        "## Preprocessing Decisions\n\n"
        "Key cleaning, feature selection, and feature engineering decisions were applied based on data profile and EDA findings.\n\n"
        "## Model Evaluation\n\n"
        f"{len(all_results)} execution iteration(s) were run. "
        f"Best observed model: {best['model'] if best else 'N/A'} "
        f"with test score {best['test_score'] if best else 'N/A'}.\n\n"
        "## Recommendation\n\n"
        "Adopt the best-performing model and validate it on a holdout or production-like slice before deployment."
    )


def generate_stakeholder_report_text(
    session: SessionState, all_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Generate non-technical structured content for the stakeholder PDF.
    Returns a dict with keys: one_sentence, confidence_plain, data_needed,
    example_predictions, limitations, recommendation.
    """
    task = session.task_config or {}
    exec_results = session.execution_results or {}
    best_model = exec_results.get("best_model_name") or "the model"
    best_score = exec_results.get("best_test_score")
    target = task.get("target_column") or "the outcome"
    goal = task.get("goal") or f"predict {target}"
    task_type = (
        exec_results.get("task_type") or task.get("task_type") or "classification"
    ).lower()
    profile = session.dataset_profile or {}
    cols = list((profile.get("columns") or {}).keys())[:20]
    score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else "N/A"

    prompt = f"""You are writing a non-technical AI model summary for a business executive audience.

CONTEXT:
- Task: {goal}
- Target to predict: {target}
- Best model: {best_model}
- Performance score: {score_str}
- Task type: {task_type}
- Available input columns: {cols}

Write a JSON object with EXACTLY these keys:
{{
  "one_sentence": "One plain sentence stating what the model predicts. No ML terms.",
  "confidence_plain": "1-2 sentences about reliability in plain language. No AUC, RMSE, percentages. Use words like reliably, most of the time, with high confidence.",
  "data_needed": ["item1", "item2", "item3"],
  "example_predictions": [
    {{"scenario": "plain real-world case description", "prediction": "what model predicts", "explanation": "one-sentence plain reason"}},
    {{"scenario": "...", "prediction": "...", "explanation": "..."}},
    {{"scenario": "...", "prediction": "...", "explanation": "..."}}
  ],
  "limitations": ["limitation1", "limitation2", "limitation3"],
  "recommendation": "1-2 sentences for the decision-maker."
}}

Rules: no ML jargon, write for a CEO, be concise. Return ONLY the JSON object."""

    raw = LLMClient().call(
        prompt,
        max_tokens=config.LLM_MAX_TOKENS_NARRATIVE,
        temperature=config.LLM_TEMPERATURE_NARRATIVE,
    )
    parsed = extract_json_object(raw) if raw else None
    if not parsed:
        return {
            "one_sentence": f"This model predicts {target} using {len(cols)} input data fields.",
            "confidence_plain": (
                "The model performs reliably on the data it was trained on "
                "and should generalise well to similar cases."
            ),
            "data_needed": cols[:5] if cols else ["Input data"],
            "example_predictions": [],
            "limitations": [
                "May not generalise to cases very different from the training data.",
                "Performance depends on the quality and completeness of input data.",
                "Should be validated on new data before full deployment.",
            ],
            "recommendation": (
                "Validate the model on a representative holdout sample before deployment. "
                "Establish a monitoring process to track performance over time."
            ),
        }
    return parsed
