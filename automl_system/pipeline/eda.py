"""Phase 3: exploratory data analysis + LLM narrative."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.pipeline.eda_visuals import build_eda_chat_messages
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import log_decision
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "eda", **entry})


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _distribution_shapes(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in _numeric_cols(df):
        s = df[c].dropna()
        if len(s) < 3:
            out[c] = "insufficient_data"
            continue
        sk = float(s.skew())
        ku = float(s.kurtosis())  # excess kurtosis (>0 = heavy-tailed, <0 = flat)
        if ku > 3.5:
            out[c] = "heavy_tailed"
        elif ku < -1.2:
            # Flat distribution — uniform if symmetric, bimodal if off-centre
            out[c] = "uniform" if abs(sk) < 0.5 else "bimodal"
        elif abs(sk) < 0.5:
            out[c] = "approx_normal"
        elif sk > 0.5:
            out[c] = "right_skewed"
        else:
            out[c] = "left_skewed"
    return out


def _full_correlation_matrix(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 2:
        return {}
    sub = df[num].dropna()
    if len(sub) < 2:
        return {}
    corr = sub.corr()
    return {r: {c: float(corr.loc[r, c]) for c in corr.columns} for r in corr.index}


def _outlier_flags(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = {}
    for c in _numeric_cols(df):
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = int(((df[c] < low) | (df[c] > high)).sum())
        mapped = df[c].map(lambda x: np.nan if pd.isna(x) else abs((x - s.mean()) / (s.std() + 1e-12)))
        z_out = int((mapped > 3).sum())
        flags[c] = {"iqr_outlier_count": iqr_out, "zscore_gt3_count": z_out}
    return flags


def _missing_patterns(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {}
    return {c: float(df[c].isna().mean()) for c in df.columns}


def _target_correlations(df: pd.DataFrame, target: str) -> dict[str, float]:
    if target not in df.columns:
        return {}
    if not pd.api.types.is_numeric_dtype(df[target]):
        # encode for correlation if few categories
        t = pd.Categorical(df[target]).codes
        if len(np.unique(t)) < 2:
            return {}
        out: dict[str, float] = {}
        for c in df.columns:
            if c == target:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                sub = pd.DataFrame({"x": df[c], "y": t}).dropna()
                if len(sub) > 2:
                    out[c] = float(np.corrcoef(sub["x"], sub["y"])[0, 1])
        return out
    out = {}
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            sub = df[[c, target]].dropna()
            if len(sub) > 2:
                out[c] = float(sub[c].corr(sub[target]))
    return out


def _target_imbalance(df: pd.DataFrame, target: str) -> dict[str, Any]:
    if target not in df.columns:
        return {"severity": "unknown", "note": "target missing"}
    vc = df[target].value_counts(normalize=True)
    if len(vc) <= 1:
        return {"severity": "none", "top_class_share": 1.0}
    top = float(vc.iloc[0])
    minority_pct = float(vc.iloc[-1]) * 100
    if minority_pct < 5.0:
        sev = "severe"
    elif top > 0.9:
        sev = "high"
    elif top > 0.7:
        sev = "moderate"
    else:
        sev = "low"
    return {
        "severity": sev,
        "top_class_share": top,
        "minority_pct": round(minority_pct, 2),
        "n_classes": int(vc.shape[0]),
    }


def run_eda(df: pd.DataFrame, session: SessionState) -> dict[str, Any]:
    """
    Compute EDA bundle; LLM writes narrative (heavy model). Stores results and summaries in session.
    """
    task = session.task_config or {}
    target = str(task.get("target_column") or "")

    results: dict[str, Any] = {
        "distribution_shapes": _distribution_shapes(df),
        "correlation_matrix": _full_correlation_matrix(df),
        "outlier_flags": _outlier_flags(df),
        "missing_patterns": _missing_patterns(df),
        "target_correlations": _target_correlations(df, target) if target else {},
        "target_imbalance": _target_imbalance(df, target) if target else {},
        "target_column": target,
    }

    # Near-redundant pairs: |corr| > 0.95 off-diagonal
    near_redundant: list[tuple[str, str, float]] = []
    cm = results["correlation_matrix"]
    cols = list(cm.keys())
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = abs(cm[a].get(b, 0.0))
            if v > 0.95:
                near_redundant.append((a, b, float(cm[a][b])))
    results["near_redundant_pairs"] = [{"a": x[0], "b": x[1], "corr": x[2]} for x in near_redundant]

    # ── Derived flags used to steer the LLM narrative ───────────────────────
    # Severe imbalance flag (<5% minority class)
    imbalance = results["target_imbalance"]
    severe_imbalance = imbalance.get("severity") == "severe"
    if severe_imbalance:
        minority_pct = imbalance.get("minority_pct", 0)
        session.task_config["severe_class_imbalance"] = True
        session.task_config["minority_class_pct"] = minority_pct

    # Near-zero target correlation flag
    tc = results["target_correlations"]
    max_abs_corr = max((abs(v) for v in tc.values()), default=0.0) if tc else 0.0
    all_near_zero_corr = bool(tc) and max_abs_corr < 0.05
    results["all_near_zero_target_corr"] = all_near_zero_corr

    # Few features flag
    feature_cols = [c for c in df.columns if c != target]
    few_features = len(feature_cols) <= 3
    results["few_features"] = few_features

    session.eda_results = results

    # ── Build scenario-specific narrative instructions ────────────────────────
    scenario_notes = []
    if severe_imbalance:
        scenario_notes.append(
            f"IMPORTANT: Explicitly flag severe class imbalance — the minority class is only "
            f"{imbalance.get('minority_pct', '?')}% (<5%). Discuss its implications for metric choice and model training."
        )
    if all_near_zero_corr:
        scenario_notes.append(
            "IMPORTANT: All features show near-zero correlation with the target (<0.05). "
            "Explicitly flag this as a potential data quality risk. The pipeline will continue but this is logged."
        )
    if few_features:
        scenario_notes.append(
            f"IMPORTANT: Only {len(feature_cols)} features available after removing the target. "
            "Note this is a very small feature set; feature engineering will be prioritized."
        )

    client = LLMClient()
    compact = {
        "distribution_shapes": results["distribution_shapes"],
        "outlier_summary": {k: v for k, v in list(results["outlier_flags"].items())[:30]},
        "missing_top": sorted(results["missing_patterns"].items(), key=lambda x: -x[1])[:20],
        "target_correlations_top": sorted(
            results["target_correlations"].items(), key=lambda x: -abs(x[1])
        )[:25],
        "near_redundant_pairs": results["near_redundant_pairs"][:20],
        "target_imbalance": results["target_imbalance"],
    }
    column_names = list(df.columns)
    scenario_block = ("\n\nSCENARIO INSTRUCTIONS:\n" + "\n".join(f"- {s}" for s in scenario_notes)) if scenario_notes else ""
    prompt = f"""Write a concise EDA analysis in JSON. The narrative MUST cover ALL five points below.

1) Top 3 most predictive features and why (use target_correlations_top)
2) Distribution problems that will affect modelling (skewed, heavy-tailed, bimodal columns)
3) Target imbalance severity and its implications for model training and metric selection
4) Near-redundant feature pairs — high mutual correlation suggesting possible removal
5) Domain observations inferrable from column names (e.g. medical, financial, behavioural)

Also write:
- user_summary: 2–3 plain-English sentences for a non-technical stakeholder.{scenario_block}

COLUMN_NAMES: {column_names[:60]}{'...' if len(column_names) > 60 else ''}

EDA_SUMMARY: {compact}

JSON only: {{"narrative": "...", "user_summary": "..."}}"""

    # Use LLM_FLASH for EDA narrative — it writes well and is faster
    model = config.LLM_FLASH
    raw = client.call(prompt, model=model, max_tokens=config.LLM_MAX_TOKENS_NARRATIVE)
    parsed = extract_json_object(raw) if raw else None
    narrative = ""
    user_summary = ""
    if parsed:
        narrative = str(parsed.get("narrative") or "").strip()
        user_summary = str(parsed.get("user_summary") or "").strip()
    if not narrative:
        narrative = (
            "EDA highlights correlations with the target, missingness patterns, and skewed distributions. "
            "Review near-redundant pairs before modeling."
        )
    if not user_summary:
        user_summary = "Exploratory analysis is complete; key drivers and data-quality issues were summarized."

    session.eda_narrative = narrative
    session.user_summary_eda = user_summary
    _log(
        session,
        {
            "action": "run_eda",
            "user_summary_excerpt": user_summary[:500],
        },
    )
    log_decision(
        session,
        step="Phase 3 - EDA",
        action="run_eda_complete",
        rationale=user_summary[:1200],
    )
    try:
        session.eda_chat_messages = build_eda_chat_messages(df, session, compact)
    except Exception as e:
        logger.warning("EDA chat visuals failed: %s", e)
        session.eda_chat_messages = []
    return results
