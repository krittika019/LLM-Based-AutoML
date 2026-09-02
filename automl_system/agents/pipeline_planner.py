"""Phase 4: Planning Mode 2 — two-LLM-call execution planning from actual preprocessing outcome."""

from __future__ import annotations

import logging
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import compact_decisions
from automl_system.utils.llm_json import extract_json_object
from automl_system.utils.profile_conditioner import condition_profile_for_execution_planning

logger = logging.getLogger(__name__)

_ALLOWED_MODEL_FAMILIES = [
    # Linear / regularised — fast, interpretable, good baselines
    "LogisticRegression",   # classification baseline
    "Ridge",                # regression baseline (L2)
    "Lasso",                # regression with feature selection (L1)
    "ElasticNet",           # regression L1+L2 mix
    "LinearRegression",     # unregularised regression
    # Tree ensembles — robust, handle mixed types
    "RandomForest",         # bagging; low variance, high diversity
    "ExtraTrees",           # extra-randomised trees; fast, low-bias
    "GradientBoosting",     # sklearn classic GBM; strong on small-medium data
    "HistGradientBoosting", # histogram GBM; fast, handles missingness natively
    # Modern boosting — usually top performers on tabular data
    "XGBoost",              # regularised GBM; strong on regression and structured data
    "LightGBM",             # leaf-wise fast GBM; excels on large/imbalanced datasets
    "CatBoost",             # handles categoricals natively; strong when many cat features
    # Kernel / instance-based — useful for small datasets
    "SVC",                  # classification only; kernel SVM; strong when n < 10k
    "SVR",                  # regression only; kernel SVM; strong when n < 10k
    "KNeighbors",           # instance-based; no training, useful for local patterns
    # Neural
    "MLP",                  # multi-layer perceptron; good on dense numeric features
]



def _infer_task_type(session: SessionState) -> str:
    # Phase 1 already classified — trust it first.
    task_type = str((session.task_config or {}).get("task_type") or "").strip().lower()
    if task_type in ("classification", "regression"):
        return task_type
    # Fallback: infer from target column profile when Phase 1 didn't set it.
    target = str((session.task_config or {}).get("target_column") or "").strip()
    profile = session.dataset_profile or {}
    t_meta = (profile.get("columns") or {}).get(target)
    if not target or not t_meta:
        return "classification"
    dtype = str(t_meta.get("dtype") or "").lower()
    nunique = int(t_meta.get("nunique") or 0)
    if "object" in dtype or "bool" in dtype or "category" in dtype:
        return "classification"
    if nunique <= 20:
        return "classification"
    return "regression"


def _compute_dataset_context(session: SessionState) -> dict[str, Any]:
    """
    Pure code — assembled before any LLM call.
    Captures actual post-preprocessing state of the dataset.
    """
    profile = session.dataset_profile or {}
    task = session.task_config or {}
    target = str(task.get("target_column") or "")

    n_rows = int(profile.get("n_rows") or 0)
    n_cols = int(profile.get("n_columns") or 0)
    n_features = max(0, n_cols - (1 if target else 0))

    cols_meta = profile.get("columns") or {}
    n_numeric, n_categorical = 0, 0
    for c, m in cols_meta.items():
        if c == target:
            continue
        dtype_s = str((m or {}).get("dtype") or "").lower()
        if any(k in dtype_s for k in ("int", "float")):
            n_numeric += 1
        else:
            n_categorical += 1

    imbalance = (session.eda_results or {}).get("target_imbalance") or {}
    imbalance_severity = str(imbalance.get("severity") or "none")
    imbalance_present = imbalance_severity in {"moderate", "high", "severe"}

    assignments = session.deep_cleaning_assignments or {}
    scaling_applied = any(
        str((a or {}).get("scaling") or "none") not in ("none", "")
        for a in assignments.values() if isinstance(a, dict)
    )

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "imbalance_present": imbalance_present,
        "imbalance_severity": imbalance_severity,
        "scaling_applied": scaling_applied,
        "high_dimensionality": n_features > n_rows if n_rows > 0 else False,
    }


def _compute_mi_scores(df: Any, target: str, task_type: str, top_n: int = 10) -> list[dict[str, Any]]:
    """Compute mutual information scores between features and target.

    Gives the LLM statistical evidence of feature-target relationship strength
    and linearity, enabling more informed model family selection.
    Returns top_n features sorted by MI score descending.
    """
    try:
        import pandas as _pd
        import numpy as _np
        from sklearn.feature_selection import (
            mutual_info_classif, mutual_info_regression,
        )
        if df is None or target not in df.columns:
            return []
        y = df[target].dropna()
        X = df.drop(columns=[target]).select_dtypes(include=[_np.number])
        common = X.index.intersection(y.index)
        if len(common) < 20 or X.shape[1] == 0:
            return []
        X = X.loc[common].fillna(X.loc[common].median())
        y = y.loc[common]
        fn = mutual_info_classif if task_type == "classification" else mutual_info_regression
        scores = fn(X, y, random_state=42)
        total = float(scores.sum()) or 1.0
        pairs = sorted(
            zip(X.columns.tolist(), (scores / total * 100).tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return [{"feature": f, "mi_pct": round(s, 1)} for f, s in pairs[:top_n]]
    except Exception:
        return []


def _recall_past_model_wins(decisions_log: list[dict[str, Any]], task_type: str) -> str:
    """Scan episodic memory for past best-model outcomes on similar task type.

    Returns a short summary string injected into the LLM prompt so it can
    factor in what has historically worked rather than reasoning from scratch.
    """
    wins: dict[str, int] = {}
    for entry in decisions_log:
        if entry.get("phase") != "execution":
            continue
        action = str(entry.get("action") or "")
        if "best_model" in action or action == "execution_complete":
            best = str(entry.get("best_model") or entry.get("model") or "")
            t = str(entry.get("task_type") or "")
            if best and (not t or t == task_type):
                wins[best] = wins.get(best, 0) + 1
    if not wins:
        return ""
    top = sorted(wins.items(), key=lambda x: x[1], reverse=True)[:3]
    lines = ", ".join(f"{m}({n}x)" for m, n in top)
    return f"EPISODIC_MEMORY — past best models on {task_type} tasks: {lines}"


def _summarize_mode1_deviations(session: SessionState) -> list[str]:
    """Compare Mode 1 preprocessing plan intent to actual outcomes."""
    deviations: list[str] = []
    plan = session.preprocessing_plan or {}
    meta = session.preprocessing_meta or {}
    fe_log = session.feature_engineering_log or []

    fe_recommended = bool(plan.get("feature_engineering_recommended", False))
    fe_skipped = any(isinstance(e, dict) and e.get("status") == "skipped" for e in fe_log)
    fe_added_count = sum(1 for e in fe_log if isinstance(e, dict) and e.get("status") == "added")

    if fe_recommended and fe_skipped:
        deviations.append("Feature engineering was planned but skipped — dataset had fewer than 200 rows.")
    elif fe_recommended and fe_added_count == 0 and fe_log:
        deviations.append("Feature engineering was planned but no proposals survived the leakage/correlation check.")

    if meta.get("dimensionality_warning"):
        deviations.append(f"Post-encoding dimensionality: {meta['dimensionality_warning']}")
    if meta.get("feature_selection_high_drop_warning"):
        deviations.append(f"Feature selection high-drop: {meta['feature_selection_high_drop_warning']}")
    # KNN downgrade recorded as deviations in deep_cleaning decisions log
    assignments = session.deep_cleaning_assignments or {}
    knn_downgrades = [c for c, a in assignments.items()
                      if isinstance(a, dict) and "downgraded to median" in str(a.get("imputation_rationale","")).lower()]
    if knn_downgrades:
        deviations.append(f"KNN imputation downgraded to median for {len(knn_downgrades)} column(s) due to dataset size.")
    return deviations


def _parse_constraint_signals(session: SessionState) -> dict[str, Any]:
    """
    Scan user_constraints and goal text for high-level signals that should
    influence metric, model selection, and performance threshold.

    Recall-priority fires when the user indicates missing a positive case is
    the costly error (medical, fraud, safety). Precision-priority fires only
    when the user explicitly wants to minimise false alarms / false positives.
    Recall takes precedence when both signals appear.

    Numeric threshold extraction: parses phrases like "at least 90% accuracy",
    "achieve 0.85 AUC", "R2 above 0.75" and sets user_threshold accordingly.
    """
    import re as _re
    task = session.task_config or {}
    raw_constraints = list(task.get("user_constraints") or [])
    goal_text = str(task.get("goal") or "").lower()
    all_text = " ".join(str(c) for c in raw_constraints).lower() + " " + goal_text

    # Recall signals: missing the positive class is the costly error
    recall_triggers = (
        "sensitivity", "high recall", "minimize false negative", "minimise false negative",
        "missed diagnosis", "missed fraud", "missed case", "miss rate",
        "false negatives are costly", "false negative is costly",
        "must not miss", "cannot miss", "critical to catch",
        "regulatory penalt", "life-or-death", "safety-critical",
        "more costly than false alarm", "more costly than false positive",
        "cost of some false positive",
        "cost of false alarm",
    )
    # Precision signals: false positives are the costly error
    precision_triggers = (
        "minimize false positive", "minimise false positive",
        "reduce false alarm", "minimise false alarm",
        "false positives are costly", "false alarms are costly",
        "high precision required", "precision is critical",
        "too many false positives", "too many false alarms",
    )

    recall_priority    = any(k in all_text for k in recall_triggers)
    precision_priority = any(k in all_text for k in precision_triggers) and not recall_priority

    # Numeric threshold: parse "at least 85%", "above 0.9", ">= 0.80 R2", "90% accuracy" etc.
    user_threshold: float | None = None
    # Patterns: optional comparison word, then number (0-100 or 0.0-1.0)
    threshold_patterns = [
        r"(?:at least|above|over|minimum|min|>=|=>|>)\s*(\d+(?:\.\d+)?)\s*%",  # "at least 85%"
        r"(?:at least|above|over|minimum|min|>=|=>|>)\s*(0\.\d+)",              # "above 0.85"
        r"(\d+(?:\.\d+)?)\s*%\s*(?:accuracy|auc|f1|r2|recall|precision)",       # "90% accuracy"
        r"(?:accuracy|auc|f1|r2|recall|precision)\s*(?:of|>=|above|>)?\s*(\d+(?:\.\d+)?)\s*%",
        r"(?:r2|r-squared)\s*(?:of|>=|above|>)?\s*(0\.\d+)",                   # "R2 above 0.75"
    ]
    for pat in threshold_patterns:
        m = _re.search(pat, all_text)
        if m:
            val = float(m.group(1))
            user_threshold = val / 100.0 if val > 1.0 else val  # normalise % → fraction
            break

    return {
        "prioritize_recall":    recall_priority,
        "prioritize_precision": precision_priority,
        "user_threshold":       user_threshold,   # None if user didn't specify
    }


def _deterministic_default_plan(session: SessionState) -> dict[str, Any]:
    profile = session.dataset_profile or {}
    n_rows = int(profile.get("n_rows") or 0)
    task_type = _infer_task_type(session)
    target_imbalance = (session.eda_results or {}).get("target_imbalance") or {}
    imbalance = str(target_imbalance.get("severity") or "unknown")
    signals = _parse_constraint_signals(session)

    if task_type == "classification":
        if n_rows >= 20000:
            # Large datasets: GPU-accelerated boosters dominate; deprioritise sequential CPU models
            models = [
                {"family": "LightGBM",
                 "justification": "Leaf-wise fast GBM with GPU support; best speed/accuracy on large data."},
                {"family": "XGBoost",
                 "justification": "Regularised GBM with CUDA acceleration; strong structured classification."},
                {"family": "HistGradientBoosting",
                 "justification": "Histogram GBM; handles missingness natively, very fast on large datasets."},
                {"family": "RandomForest",
                 "justification": "Parallel bagging; reliable diverse baseline for large classification tasks."},
            ]
        else:
            # Small to medium: diverse mix covering linear, bagging, and boosting
            models = [
                {"family": "LogisticRegression",
                 "justification": "Linear baseline; fast, interpretable, strong on linearly separable data."},
                {"family": "RandomForest",
                 "justification": "Bagging ensemble; robust to noise, mixed feature types, and dataset scales."},
                {"family": "GradientBoosting",
                 "justification": "Sequential boosting on residuals; strong on small-medium tabular data."},
                {"family": "XGBoost",
                 "justification": "Regularised GBM; strong on structured tabular classification."},
                {"family": "LightGBM",
                 "justification": "Leaf-wise fast GBM; excels on imbalanced or medium-large datasets."},
            ]
        # Metric: recall-priority constraint overrides default F1/AUC choice
        if signals["prioritize_recall"]:
            metric = "recall" if int((session.dataset_profile or {}).get("n_classes") or 2) <= 2 else "recall_weighted"
            threshold = 0.60
            threshold_rationale = "Recall-focused metric selected: user constraints indicate false negatives are more costly than false positives."
        elif imbalance in {"moderate", "high", "severe"}:
            metric = "f1_weighted"
            threshold = 0.70
            threshold_rationale = "Threshold derived from inferred class balance and typical improvement over majority-class baseline."
        else:
            metric = "roc_auc"
            threshold = 0.75
            threshold_rationale = "Threshold derived from inferred class balance and typical improvement over majority-class baseline."
    else:
        if n_rows >= 20000:
            models = [
                {"family": "LightGBM",
                 "justification": "Leaf-wise fast GBM with GPU support; best speed/accuracy for large regression."},
                {"family": "XGBoost",
                 "justification": "Regularised GBM with CUDA; strong for large structured regression."},
                {"family": "HistGradientBoosting",
                 "justification": "Histogram GBM; handles missingness, very fast on large datasets."},
                {"family": "RandomForest",
                 "justification": "Parallel bagging; reliable baseline for large regression tasks."},
            ]
        else:
            models = [
                {"family": "Ridge",
                 "justification": "L2-regularised linear baseline; stable and fast."},
                {"family": "HistGradientBoosting",
                 "justification": "Histogram GBM; fast, handles missing values natively, strong on medium-large regression."},
                {"family": "RandomForest",
                 "justification": "Captures non-linear patterns without strict distribution assumptions."},
                {"family": "XGBoost",
                 "justification": "Regularised GBM; strong for structured regression including ordinal targets."},
                {"family": "LightGBM",
                 "justification": "Leaf-wise fast GBM; often best regression performance on large datasets."},
            ]
        metric = "r2"
        threshold = 0.60
        threshold_rationale = "R² ≥ 0.60 means the model explains at least 60% of target variance — a meaningful baseline for regression regardless of data scale."

    if n_rows < 5000:
        search = {"name": "GridSearchCV"}
    elif n_rows <= 50000:
        search = {"name": "RandomizedSearchCV", "n_iter": 20}
    else:
        search = {"name": "RandomizedSearchCV", "n_iter": 30}

    folds = 3 if n_rows <= 500 else 5
    cv_name = "StratifiedKFold" if task_type == "classification" else "KFold"

    return {
        "task_type": task_type,
        "models": models,
        "search_strategy": search,
        "cv_design": {"type": cv_name, "folds": folds, "shuffle": True, "random_state": 42},
        "metric": metric,
        "threshold": threshold,
        "threshold_rationale": threshold_rationale,
    }


def _sanitize_plan(plan: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Merge LLM plan output into fallback, validating each field."""
    out = dict(fallback)

    models = plan.get("models")
    if isinstance(models, list):
        cleaned: list[dict[str, str]] = []
        for item in models:
            if isinstance(item, dict):
                fam = str(item.get("family") or "").strip()
                rationale = str(item.get("rationale") or item.get("justification") or "").strip()
            else:
                fam = str(item).strip()
                rationale = ""
            if fam in _ALLOWED_MODEL_FAMILIES:
                cleaned.append({"family": fam, "rationale": rationale or "Selected by LLM for this dataset."})
        if cleaned:
            out["models"] = cleaned[:5]

    search = plan.get("search_strategy")
    if isinstance(search, dict) and search.get("name") in {"GridSearchCV", "RandomizedSearchCV"}:
        out["search_strategy"] = {
            "name": str(search["name"]),
            "n_trials": int(search.get("n_trials") or 50),
            "reason": str(search.get("reason") or ""),
        }

    cv = plan.get("cv_design")
    if isinstance(cv, dict):
        cv_type = str(cv.get("type") or out["cv_design"]["type"])
        if cv_type in {"StratifiedKFold", "KFold"}:
            out["cv_design"]["type"] = cv_type
        folds = int(cv.get("folds") or out["cv_design"]["folds"])
        out["cv_design"]["folds"] = 3 if folds < 4 else 5

    metric = str(plan.get("metric") or "").strip()
    if metric:
        # Reject raw-unit neg_* regression metrics — they make thresholds meaningless.
        # The LLM is instructed to use r2 for regression; enforce it here too.
        if metric.startswith("neg_") and out.get("task_type") == "regression":
            logger.warning(
                "_sanitize_plan: rejecting neg_* metric %r for regression; using r2 instead", metric
            )
            out["metric"] = "r2"
            if not isinstance(out.get("threshold"), float) or out.get("threshold", 0) < 0:
                out["threshold"] = 0.60
        else:
            out["metric"] = metric

    threshold = plan.get("threshold")
    if isinstance(threshold, (int, float)):
        t = float(threshold)
        # For neg_* sklearn metrics (neg_rmse etc.), threshold must be negative.
        # A value of 0.0 means "perfect score" — reject and keep fallback.
        if str(out.get("metric", "")).startswith("neg_") and t >= 0:
            logger.warning(
                "_sanitize_plan: LLM returned threshold=%.4f for neg_* metric %r — "
                "using fallback threshold=%.4f",
                t, out.get("metric"), out["threshold"],
            )
        else:
            out["threshold"] = t

    rationale = str(plan.get("threshold_rationale") or "").strip()
    if rationale:
        out["threshold_rationale"] = rationale

    return out


def make_execution_plan(session: SessionState, df: Any = None) -> dict[str, Any]:
    """Build execution plan from task config, EDA narrative, final profile, and decision logs.

    Returns dict with keys: models, search_strategy, cv_design, metric, threshold.
    """
    profile = session.dataset_profile or {}
    task_config = session.task_config or {}
    n_rows = int(profile.get("n_rows") or 0)
    task_type = _infer_task_type(session)
    difficulty = profile.get("difficulty") or {}
    difficulty_label = str(difficulty.get("label") or "moderate")
    difficulty_score = float(difficulty.get("overall") or 0.0)

    decisions_log: list[dict[str, Any]] = []
    if session.decisions_log_id:
        decisions_log = episodic.get_log(session.decisions_log_id)

    fallback = _deterministic_default_plan(session)
    client = LLMClient()

    # Preprocessing deviations influence model selection robustness
    prep_deviations = list(session.preprocessing_deviations or [])
    deviation_note = ""
    if prep_deviations:
        types = {d.get("type") for d in prep_deviations}
        if "knn_downgrade" in types or "ohe_cardinality_override" in types:
            deviation_note = (
                "NOTE: Preprocessing deviations occurred (KNN downgraded or high-cardinality encoding overridden). "
                "Prefer robust tree-based models (HistGradientBoosting, XGBoost, RandomForest) "
                "over models sensitive to encoding quality (MLP, SVR, ElasticNet)."
            )
        if "feature_engineering_skipped" in types:
            deviation_note += (
                " Feature engineering was skipped (dataset too small or LLM unavailable). "
                "Prefer models that handle raw features well without derived interaction terms."
            )

    # Resolve constraint signals before building the prompt so they appear explicitly.
    signals = _parse_constraint_signals(session)
    constraint_block = ""
    if signals["prioritize_recall"]:
        constraint_block = (
            "\nCONSTRAINT SIGNALS (extracted from user goal and constraints):\n"
            "  recall_priority = TRUE — the user indicated that false negatives are more costly\n"
            "  than false positives (e.g. medical diagnosis, fraud, safety-critical detection).\n"
            "  MANDATORY RULE: you MUST set metric to 'recall' (binary) or 'recall_weighted'\n"
            "  (multiclass). Do NOT use roc_auc or f1_weighted when recall_priority is TRUE.\n"
            "  Also prefer models that support class_weight (LogisticRegression, RandomForest,\n"
            "  GradientBoosting, LightGBM with is_unbalance=True) over models that do not.\n"
        )
    elif signals["prioritize_precision"]:
        constraint_block = (
            "\nCONSTRAINT SIGNALS (extracted from user goal and constraints):\n"
            "  precision_priority = TRUE — the user indicated that false positives are more\n"
            "  costly than false negatives (e.g. spam filter, recommendation system).\n"
            "  MANDATORY RULE: you MUST set metric to 'precision' (binary) or\n"
            "  'precision_weighted' (multiclass).\n"
        )

    # Statistical evidence to ground LLM model selection in actual data patterns
    target_col = str(task_config.get("target_column") or "")
    mi_scores = _compute_mi_scores(df, target_col, task_type)
    mi_block = ""
    if mi_scores:
        mi_lines = ", ".join(f"{r['feature']}={r['mi_pct']}%" for r in mi_scores)
        # Derive a signal for the LLM: high avg MI → strong predictive signal, use complex models
        avg_mi = sum(r["mi_pct"] for r in mi_scores) / len(mi_scores)
        signal = "strong non-linear signal" if avg_mi > 5 else "moderate signal" if avg_mi > 2 else "weak/noisy signal"
        mi_block = (
            f"\nFEATURE_TARGET_MI_SCORES (mutual information, % of total — higher = more predictive):\n"
            f"  Top features: {mi_lines}\n"
            f"  Signal strength: {signal} (avg MI%={avg_mi:.1f})\n"
            f"  Guidance: {('High MI → prefer non-linear models (XGBoost, LightGBM, CatBoost) that capture complex patterns.' if avg_mi > 5 else 'Low MI → data is noisy; prefer ensembles (RandomForest) that average out noise, or regularized linear models.')}\n"
        )

    # Memory-augmented reasoning: what has worked before on similar tasks
    memory_block = _recall_past_model_wins(decisions_log, task_type)
    memory_note = f"\n{memory_block}" if memory_block else ""

    prompt = f"""You are planning AutoML execution for tabular data.

Inputs:
- TASK_CONFIG: {task_config}
- TASK_TYPE (inferred): {task_type}
- DATASET_DIFFICULTY: {difficulty_label} (score={difficulty_score:.2f}) — components: {difficulty.get("components", {})}
- EDA_NARRATIVE: {(session.eda_narrative or '')[:5000]}
- FINAL_CLEANED_DATASET_PROFILE: {condition_profile_for_execution_planning(profile)}
- DECISIONS_LOG (recent entries): {compact_decisions(decisions_log, limit=10)}
- MODE1_PLAN_DEVIATIONS (what actually happened vs preprocessing plan): {_summarize_mode1_deviations(session) or ["None — preprocessing followed plan as expected."]}
- PREPROCESSING_DEVIATIONS (actual vs planned — {len(prep_deviations)} deviation(s)): {prep_deviations[:10]}
{mi_block}{memory_note}{deviation_note}{constraint_block}
Required reasoning:
1) Model selection: rank 3-5 model families from this allowed list only:
   {_ALLOWED_MODEL_FAMILIES}

   SIZE guidance (strong preference, not hard rules — apply alongside domain heuristics):
   - VERY SMALL (< 500 rows): prefer simpler models first — LogisticRegression/Ridge, RandomForest,
     GradientBoosting, SVC/SVR. Heavy boosters (XGBoost, LightGBM, CatBoost) can work but need
     strong regularisation and add little over well-tuned RandomForest at this scale.
   - SMALL (500–2k rows): RandomForest, GradientBoosting, and XGBoost all work well. LightGBM and
     CatBoost are acceptable if domain signals (many categoricals, imbalance) favour them.
   - MEDIUM (2k–20k rows): all families appropriate — apply domain heuristics below.
   - LARGE (> 20k rows): prefer LightGBM, XGBoost, HistGradientBoosting, RandomForest.
     Deprioritise GradientBoosting (sequential CPU-only — slow on large data) and
     SVC/SVR/KNeighbors (O(n²) or O(n) query complexity makes them impractical).

   DOMAIN heuristics:
   - IMBALANCED classes (moderate/high/severe): prefer LightGBM or XGBoost (handle imbalance well)
   - MANY CATEGORICAL features: prefer CatBoost or HistGradientBoosting
   - SPARSE / HIGH-DIMENSIONAL: prefer LogisticRegression, Lasso, or ElasticNet
   - REGRESSION with few features: Lasso or Ridge may beat tree models
   - ORDINAL regression (quality scores, ratings): XGBoost or GradientBoosting
   - NOISY data: prefer RandomForest or ExtraTrees (ensemble averaging reduces noise)
   - INTERPRETABILITY required: prefer LogisticRegression or Ridge

   Always ensure diversity: include at least one linear model AND at least one tree/boosting model.
   Give plain-English rationale for each shortlisted family.

2) Metric and threshold:
   - If CONSTRAINT SIGNALS above specify a mandatory metric, use it — no exceptions.
   - For REGRESSION (no constraint): prefer "r2" (scale-invariant, universally interpretable).
     Avoid "neg_root_mean_squared_error" — raw-unit values make thresholds meaningless.
   - For CLASSIFICATION (no constraint): prefer "roc_auc" (balanced) or "f1_weighted" (imbalanced).
   - Define a context-aware performance threshold (not a universal fixed value).
   - Provide a short rationale explaining why this threshold is appropriate for this domain.

Return JSON only in this exact shape:
{{
  "models": [{{"family": "...", "rationale": "..."}}, ...],
  "search_strategy": {{"n_trials": 20, "reason": "..."}},
  "metric": "...",
  "threshold": 0.0,
  "threshold_rationale": "..."
}}"""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
    parsed = extract_json_object(raw) if raw else None

    plan = _sanitize_plan(parsed or {}, fallback)

    # Enforce row-count strategy — LLM cannot override these.
    if n_rows < 5000:
        plan["search_strategy"]["name"] = "GridSearchCV"
    else:
        plan["search_strategy"]["name"] = "RandomizedSearchCV"

    plan["cv_design"]["type"] = "StratifiedKFold" if task_type == "classification" else "KFold"
    # 5-fold for most datasets — more stable HPO selection than 3-fold.
    # Only drop to 3-fold when n_rows is so small that each fold has < 50 samples.
    plan["cv_design"]["folds"] = 3 if n_rows < 150 else 5
    plan["difficulty"] = difficulty
    plan["dataset_context"] = _compute_dataset_context(session)

    # Constraint-signal enforcement: apply BEFORE benchmark override so the
    # benchmark can still force its own metric for fair comparison, but real
    # user runs always honour what the user actually asked for.
    # Only fires when task_config has no pre-set metric (i.e. not a benchmark run).
    forced_metric = str((task_config.get("metric") or "")).strip()
    metric_overridden_by_constraint = False
    if not forced_metric:
        n_classes = int((session.dataset_profile or {}).get("n_classes") or 2)
        if signals["prioritize_recall"] and task_type == "classification":
            constraint_metric = "recall" if n_classes <= 2 else "recall_weighted"
            if plan.get("metric") != constraint_metric:
                logger.info(
                    "make_execution_plan: recall_priority constraint overrides LLM metric %r → %r",
                    plan.get("metric"), constraint_metric,
                )
                plan["metric"] = constraint_metric
                plan["threshold_rationale"] = (
                    plan.get("threshold_rationale", "") +
                    " Metric overridden to recall: user constraints indicate false negatives are more costly than false positives."
                )
            metric_overridden_by_constraint = True
        elif signals["prioritize_precision"] and task_type == "classification":
            constraint_metric = "precision" if n_classes <= 2 else "precision_weighted"
            if plan.get("metric") != constraint_metric:
                logger.info(
                    "make_execution_plan: precision_priority constraint overrides LLM metric %r → %r",
                    plan.get("metric"), constraint_metric,
                )
                plan["metric"] = constraint_metric
            metric_overridden_by_constraint = True

        # Phase 1 already resolved and announced an evaluation metric to the user
        # (task_config["evaluation_metric"], distinct from the benchmark-only
        # "metric" key checked above). Honour it as a strong prior so Phase 4
        # doesn't silently pick a different metric than what the user was told,
        # unless an explicit recall/precision constraint just overrode it.
        if not metric_overridden_by_constraint:
            phase1_metric = str(task_config.get("evaluation_metric") or "").strip()
            if phase1_metric and plan.get("metric") != phase1_metric:
                logger.info(
                    "make_execution_plan: Phase 1 evaluation_metric %r overrides LLM metric %r",
                    phase1_metric, plan.get("metric"),
                )
                plan["metric"] = phase1_metric

    # User-specified numeric threshold: overrides LLM threshold when the user said
    # "at least 90% accuracy" or "R2 above 0.80" in their natural language instruction.
    user_thr = signals.get("user_threshold")
    if user_thr is not None and not forced_metric:
        old_thr = plan.get("threshold")
        plan["threshold"] = user_thr
        plan["threshold_rationale"] = (
            f"Threshold set to {user_thr:.2f} as explicitly requested by the user "
            f"(overrides LLM suggestion of {old_thr})."
        )
        logger.info(
            "make_execution_plan: user-specified threshold %.2f overrides LLM threshold %s",
            user_thr, old_thr,
        )

    # Benchmark override: when task_config pins an explicit metric, honour it so
    # every system is scored on the same metric (comparability). Normal runs do
    # not set this key, so production behaviour is unchanged.
    if forced_metric:
        plan["metric"] = forced_metric
        logger.info("make_execution_plan: metric forced to %r via task_config (benchmark)", forced_metric)

    session.execution_plan = plan
    if session.decisions_log_id:
        episodic.append_decision(
            session.decisions_log_id,
            {
                "phase": "execution_planning",
                "action": "make_execution_plan",
                "plan": {
                    "models": plan.get("models"),
                    "search_strategy": plan.get("search_strategy"),
                    "cv_design": plan.get("cv_design"),
                    "metric": plan.get("metric"),
                    "threshold": plan.get("threshold"),
                    "difficulty": difficulty_label,
                },
            },
        )

    if not parsed:
        logger.warning("make_execution_plan: LLM output unavailable or unparseable; fallback plan used")

    return plan
