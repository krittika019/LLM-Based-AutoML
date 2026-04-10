"""Phase 4: Planning Mode 2 — two-LLM-call execution planning from actual preprocessing outcome."""

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
from automl_system.utils.profile_conditioner import condition_profile_for_execution_planning
from automl_system.utils.token_tracker import track_profile_usage

logger = logging.getLogger(__name__)

_ALLOWED_MODEL_FAMILIES = [
    "LogisticRegression",
    "RandomForest",
    "HistGradientBoosting",
    "XGBoost",
    "LightGBM",
    "LinearRegression",
    "Ridge",
    "SVR",
]


def _compact_decisions(entries: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    out = []
    for e in entries[-limit:]:
        out.append({
            "phase": e.get("phase"),
            "action": e.get("action"),
            "note": str(e.get("text") or e.get("summary") or e.get("message") or "")[:200],
        })
    return out


def _infer_task_type(session: SessionState) -> str:
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


def _deterministic_default_plan(session: SessionState) -> dict[str, Any]:
    profile = session.dataset_profile or {}
    n_rows = int(profile.get("n_rows") or 0)
    task_type = _infer_task_type(session)
    target_imbalance = (session.eda_results or {}).get("target_imbalance") or {}
    imbalance = str(target_imbalance.get("severity") or "unknown")

    if task_type == "classification":
        models = [
            {"family": "LogisticRegression",
             "justification": "Strong baseline for tabular classification; interpretable and fast."},
            {"family": "RandomForest",
             "justification": "Robust to non-linear interactions and mixed scales."},
            {"family": "HistGradientBoosting",
             "justification": "Efficient boosting for tabular data, handles missingness natively."},
        ]
        metric = "f1_weighted" if imbalance in {"moderate", "high", "severe"} else "roc_auc"
        threshold = 0.70 if imbalance in {"moderate", "high", "severe"} else 0.75
        threshold_rationale = "Threshold derived from inferred class balance and typical improvement over majority-class baseline."
    else:
        models = [
            {"family": "Ridge",
             "justification": "Regularized linear baseline for regression; stable under many weak features."},
            {"family": "RandomForest",
             "justification": "Captures non-linear patterns without strict distribution assumptions."},
            {"family": "HistGradientBoosting",
             "justification": "Strong boosting regressor for medium-to-large tabular datasets."},
        ]
        metric = "neg_root_mean_squared_error"
        threshold = -0.60
        threshold_rationale = "Fallback regression threshold when LLM output is unavailable."

    if n_rows < 5000:
        search = {"name": "GridSearchCV"}
    elif n_rows <= 50000:
        search = {"name": "RandomizedSearchCV", "n_iter": 20}
    else:
        search = {"name": "Optuna", "n_trials": 30}

    folds = 3 if n_rows <= 500 else 5
    cv_name = "StratifiedKFold" if task_type == "classification" else "KFold"

    return {
        "task_type": task_type,
        "models": models[:3],
        "search_strategy": search,
        "cv_design": {"type": cv_name, "folds": folds, "shuffle": True, "random_state": 42},
        "metric": metric,
        "threshold": threshold,
        "threshold_rationale": threshold_rationale,
    }


def make_execution_plan(session: SessionState) -> dict[str, Any]:
    """
    Two LLM calls — reads actual preprocessing outcome, not the Mode 1 plan.

    Call 1 (model family selection): task_config, EDA, final profile, decisions log
      summary, semantic memory, deviations from Mode 1 plan.
      → ranked shortlist of 2-3 model families with justification.

    Call 2 (search + threshold): for each model, concrete parameter ranges;
      CV design; context-aware performance threshold reasoning.

    Search/CV rules are code-enforced after both calls (LLM cannot override).
    Execution plan is frozen into session and returned.
    """
    profile = session.dataset_profile or {}
    task_config = session.task_config or {}
    task_type = _infer_task_type(session)

    # ── Pure-code context assembly (before any LLM call) ─────────────────────
    ctx = _compute_dataset_context(session)
    n_rows = ctx["n_rows"]

    compression_metrics = track_profile_usage("pipeline_planner", profile)
    if compression_metrics:
        logger.info("pipeline_planner: %s%% profile compression",
                    compression_metrics.get("compression_ratio_percent", "?"))

    decisions_log: list[dict[str, Any]] = []
    if session.decisions_log_id:
        decisions_log = episodic.get_log(session.decisions_log_id)

    deviations = _summarize_mode1_deviations(session)
    fallback = _deterministic_default_plan(session)
    client = LLMClient()

    # Semantic memory
    soft_past: list[dict] = []
    try:
        pv = build_semantic_profile_vector(
            profile if profile else {"n_rows": n_rows, "n_columns": 0, "columns": {}})
        soft_past = semantic_memory.retrieve_similar(pv, top_k=2)
    except Exception as _sem_err:
        logger.debug("semantic retrieve_similar skipped: %s", _sem_err)

    # ── LLM Call 1: Model family selection ───────────────────────────────────
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    constraint_hints: list[str] = []
    user_constraints = str(task_config.get("constraints") or "").lower()
    if "explainable" in user_constraints or "interpretable" in user_constraints:
        constraint_hints.append("User requires explainability: LogisticRegression or a shallow DecisionTree must be in the shortlist.")
    if "fast prediction" in user_constraints or "fast_prediction" in user_constraints:
        constraint_hints.append("User requires fast prediction: prefer tree-based, avoid SVMs with large kernels.")

    call1_prompt = f"""Phase 4 Mode 2 — Step 1: Model Family Selection.{revision_block}

Inputs:
TASK_CONFIG: {task_config}
INFERRED_TASK_TYPE: {task_type}

FINAL DATASET (actual state after all preprocessing):
  rows            : {ctx["n_rows"]}
  features        : {ctx["n_features"]}  (numeric={ctx["n_numeric"]}, categorical={ctx["n_categorical"]})
  class imbalance : {ctx["imbalance_severity"]} (present={ctx["imbalance_present"]})
  scaling applied : {ctx["scaling_applied"]}
  high dimensionality (features > rows): {ctx["high_dimensionality"]}

EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2500]}

KEY PREPROCESSING DECISIONS:
{_compact_decisions(decisions_log, limit=8)}

DEVIATIONS FROM MODE 1 PLAN:
{deviations if deviations else ["None — preprocessing followed the original plan."]}

PAST SIMILAR RUNS (cosine similarity, advisory):
{soft_past}

Selection guidelines (apply based on the actual dataset context above):
- rows < 1000: prefer simpler models, avoid deep ensembles
- high_dimensionality: regularized models preferred (Ridge, LogisticRegression)
- imbalance_present: prefer models that support class_weight (LogisticRegression, RandomForest, HistGradientBoosting)
- mixed feature types (categorical > 0): tree-based models generally preferred over linear
{chr(10).join(constraint_hints) if constraint_hints else "- No special user constraints detected."}

ALLOWED families ONLY (no others): {_ALLOWED_MODEL_FAMILIES}

Return JSON only — ranked shortlist of 2-3 families:
{{"models": [{{"family": "...", "justification": "one concise sentence for THIS dataset and task"}}]}}"""

    raw1 = client.call(call1_prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed1 = extract_json_object(raw1) if raw1 else None

    selected_models: list[dict[str, Any]] = list(fallback["models"])
    if parsed1 and isinstance(parsed1.get("models"), list):
        cleaned: list[dict[str, str]] = []
        for item in parsed1["models"]:
            if isinstance(item, dict):
                fam = str(item.get("family") or "").strip()
                justification = str(item.get("justification") or "").strip()
                if fam in _ALLOWED_MODEL_FAMILIES:
                    cleaned.append({
                        "family": fam,
                        "justification": justification or "Selected for this dataset regime.",
                    })
        if cleaned:
            selected_models = cleaned[:3]

    logger.info("Phase 4 Call 1: selected models = %s", [m["family"] for m in selected_models])

    # ── Code-enforced search/CV rules (not overrideable by LLM) ──────────────
    if n_rows < 5000:
        search_name = "GridSearchCV"
        search_extra: dict[str, Any] = {}
    elif n_rows <= 50000:
        search_name = "RandomizedSearchCV"
        search_extra = {"n_iter": 20}
    else:
        search_name = "Optuna"
        search_extra = {"n_trials": 30}

    folds = 3 if n_rows <= 500 else 5
    cv_type = "StratifiedKFold" if task_type == "classification" else "KFold"

    # ── LLM Call 2: Parameter ranges + threshold ──────────────────────────────
    model_families = [m["family"] for m in selected_models]
    search_desc = (
        f"small explicit GridSearchCV (3-4 params per model, ~8-16 fits)"
        if search_name == "GridSearchCV"
        else f"RandomizedSearchCV with n_iter=20 (5-8 params per model, realistic ranges)"
        if search_name == "RandomizedSearchCV"
        else "Optuna with 30 trials (wide ranges)"
    )

    call2_prompt = f"""Phase 4 Mode 2 — Step 2: Parameter Ranges and Performance Threshold.{revision_block}

TASK_TYPE: {task_type}
SELECTED_MODELS: {model_families}
DATASET_ROWS: {n_rows}
SEARCH_STRATEGY: {search_name} ({search_desc})
CV: {cv_type} {folds}-fold

For each model, provide concrete parameter search space suitable for {search_name}.
- Use specific value lists, not vague descriptions
- Match the scale of the search to "{search_name}":
  {"GridSearchCV: keep it small, 2–4 values per param, 2-3 params" if search_name == "GridSearchCV" else "RandomizedSearchCV: 5–8 params with realistic min/max ranges or value lists" if search_name == "RandomizedSearchCV" else "Optuna: wide parameter bounds (sampled internally)"}

For the performance threshold:
- State what a naive baseline (majority class / zeroR for classification; mean predictor for regression) would score on the metric
- Set threshold as a meaningful, context-aware improvement above that baseline
- Justify it in one sentence referencing the specific dataset characteristics

Return JSON only:
{{
  "model_params": {{
    "ModelFamilyName": {{"param": ["val1", "val2"], ...}},
    ...
  }},
  "metric": "single sklearn scorer name (e.g. roc_auc, f1_weighted, neg_root_mean_squared_error)",
  "threshold": <number>,
  "threshold_rationale": "e.g. Majority-class baseline ROC-AUC ≈ 0.50; 0.65 represents meaningful skill given moderate imbalance (severity={ctx["imbalance_severity"]})."
}}"""

    raw2 = client.call(call2_prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed2 = extract_json_object(raw2) if raw2 else None

    # Merge results from Call 2
    metric = fallback["metric"]
    threshold = fallback["threshold"]
    threshold_rationale = fallback.get("threshold_rationale", "")
    if parsed2 and isinstance(parsed2, dict):
        model_params = parsed2.get("model_params") or {}
        for m in selected_models:
            fam = m.get("family")
            pg = (model_params or {}).get(fam)
            if isinstance(pg, dict) and pg:
                # Validate: values must be lists
                clean_pg = {k: v if isinstance(v, list) else [v] for k, v in pg.items()}
                m["param_grid"] = clean_pg

        m_str = str(parsed2.get("metric") or "").strip()
        if m_str:
            metric = m_str
        thr = parsed2.get("threshold")
        if isinstance(thr, (int, float)):
            threshold = float(thr)
        tr = str(parsed2.get("threshold_rationale") or "").strip()
        if tr:
            threshold_rationale = tr

    logger.info("Phase 4 Call 2: metric=%s threshold=%s", metric, threshold)

    # ── Assemble frozen execution plan ────────────────────────────────────────
    search_strategy: dict[str, Any] = {"name": search_name, **search_extra}
    plan: dict[str, Any] = {
        "task_type": task_type,
        "models": selected_models,
        "search_strategy": search_strategy,
        "cv_design": {
            "type": cv_type,
            "folds": folds,
            "shuffle": True,
            "random_state": 42,
        },
        "metric": metric,
        "threshold": threshold,
        "threshold_rationale": threshold_rationale,
        "dataset_context": ctx,
    }

    session.execution_plan = plan

    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {
            "phase": "execution_planning",
            "action": "make_execution_plan",
            "plan_summary": {
                "models": [m.get("family") for m in selected_models],
                "search_strategy": search_name,
                "metric": metric,
                "threshold": threshold,
                "cv": f"{cv_type} {folds}-fold",
            },
        })

    if not parsed1:
        logger.warning("make_execution_plan: LLM Call 1 unavailable; fallback model list used")
    if not parsed2:
        logger.warning("make_execution_plan: LLM Call 2 unavailable; fallback threshold used")

    return plan

_ALLOWED_MODEL_FAMILIES = [
    "LogisticRegression",
    "RandomForest",
    "HistGradientBoosting",
    "XGBoost",
    "LightGBM",
    "LinearRegression",
    "Ridge",
    "SVR",
]


def _compact_decisions(entries: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    out = []
    for e in entries[-limit:]:
        out.append(
            {
                "phase": e.get("phase"),
                "action": e.get("action"),
                "note": str(e.get("text") or e.get("summary") or e.get("message") or "")[:240],
            }
        )
    return out


def _infer_task_type(session: SessionState) -> str:
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


def _deterministic_default_plan(session: SessionState) -> dict[str, Any]:
    profile = session.dataset_profile or {}
    n_rows = int(profile.get("n_rows") or 0)
    task_type = _infer_task_type(session)
    target_imbalance = (session.eda_results or {}).get("target_imbalance") or {}
    imbalance = str(target_imbalance.get("severity") or "unknown")

    if task_type == "classification":
        models = [
            {
                "family": "LogisticRegression",
                "rationale": "Strong baseline with interpretable coefficients and fast convergence for tabular classification.",
            },
            {
                "family": "RandomForest",
                "rationale": "Robust to non-linear interactions and mixed feature scales, with limited preprocessing sensitivity.",
            },
            {
                "family": "HistGradientBoosting",
                "rationale": "Efficient boosting for larger tabular data; often improves decision boundaries over linear baselines.",
            },
        ]
        metric = "f1_weighted" if imbalance in {"moderate", "high"} else "roc_auc"
        threshold = 0.72 if imbalance in {"moderate", "high"} else 0.78
    else:
        models = [
            {
                "family": "Ridge",
                "rationale": "Stable linear baseline with regularization, useful when many features may be weakly informative.",
            },
            {
                "family": "RandomForest",
                "rationale": "Captures non-linear patterns and interaction effects without strict feature distribution assumptions.",
            },
            {
                "family": "HistGradientBoosting",
                "rationale": "Powerful tree-boosting regressor for medium-to-large tabular datasets.",
            },
        ]
        metric = "neg_root_mean_squared_error"
        threshold = -0.60

    if n_rows < 5000:
        search = {"name": "GridSearchCV", "reason": "Dataset is small enough for an explicit hand-reasoned grid."}
    elif n_rows <= 50000:
        search = {
            "name": "RandomizedSearchCV",
            "reason": "Dataset size favors broader but bounded exploration over exhaustive grids.",
        }
    else:
        search = {
            "name": "Optuna",
            "n_trials": 50,
            "reason": "Large dataset requires guided optimization to balance quality and runtime.",
        }

    folds = 3 if n_rows < 2000 else 5
    cv_name = "StratifiedKFold" if task_type == "classification" else "KFold"

    return {
        "task_type": task_type,
        "models": models[:3],
        "search_strategy": search,
        "cv_design": {
            "type": cv_name,
            "folds": folds,
            "shuffle": True,
            "random_state": 42,
        },
        "metric": metric,
        "threshold": threshold,
        "threshold_rationale": "Fallback threshold derived from dataset scale and inferred task type when LLM output is unavailable.",
    }


def _sanitize_plan(plan: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)

    models = plan.get("models")
    if isinstance(models, list):
        cleaned_models: list[dict[str, str]] = []
        for item in models:
            if isinstance(item, dict):
                fam = str(item.get("family") or "").strip()
                rationale = str(item.get("rationale") or "").strip()
            else:
                fam = str(item).strip()
                rationale = ""
            if fam in _ALLOWED_MODEL_FAMILIES:
                cleaned_models.append({"family": fam, "rationale": rationale or "Candidate selected by LLM based on dataset context."})
        if cleaned_models:
            out["models"] = cleaned_models[:3]

    search = plan.get("search_strategy")
    if isinstance(search, dict) and search.get("name") in {"GridSearchCV", "RandomizedSearchCV", "Optuna"}:
        out["search_strategy"] = {
            "name": str(search.get("name")),
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
        out["metric"] = metric

    threshold = plan.get("threshold")
    if isinstance(threshold, (int, float)):
        out["threshold"] = float(threshold)

    rationale = str(plan.get("threshold_rationale") or "").strip()
    if rationale:
        out["threshold_rationale"] = rationale

    return out


def make_execution_plan(session: SessionState) -> dict[str, Any]:
    """
    Build execution plan from task config, EDA narrative, final profile, and decision logs.

    Returns dict with keys: models, search_strategy, cv_design, metric, threshold.
    """
    profile = session.dataset_profile or {}
    task_config = session.task_config or {}
    n_rows = int(profile.get("n_rows") or 0)
    task_type = _infer_task_type(session)
    
    # Track profile compression for this agent
    compression_metrics = track_profile_usage("pipeline_planner", profile)
    if compression_metrics:
        logger.info(f"pipeline_planner: {compression_metrics['compression_ratio_percent']}% profile compression")

    decisions_log: list[dict[str, Any]] = []
    if session.decisions_log_id:
        decisions_log = episodic.get_log(session.decisions_log_id)

    fallback = _deterministic_default_plan(session)
    client = LLMClient()

    # Semantic memory: retrieve similar past runs to inform model selection (reference Phase 4)
    soft_past: list[dict] = []
    try:
        pv = build_semantic_profile_vector(profile if profile else {"n_rows": n_rows, "n_columns": 0, "columns": {}})
        soft_past = semantic_memory.retrieve_similar(pv, top_k=3)
    except Exception as _sem_err:
        logger.debug("semantic retrieve_similar skipped in pipeline_planner: %s", _sem_err)

    prompt = f"""You are planning AutoML execution for tabular data.

Inputs:
- TASK_CONFIG: {task_config}
- TASK_TYPE (inferred): {task_type}
- EDA_NARRATIVE: {(session.eda_narrative or '')[:5000]}
- FINAL_CLEANED_DATASET_PROFILE: {condition_profile_for_execution_planning(profile)}
- DECISIONS_LOG (recent entries): {_compact_decisions(decisions_log, limit=10)}
- SOFT_RECOMMENDATIONS_FROM_PAST_RUNS (cosine similarity; advisory only): {soft_past}

Required reasoning:
1) Model selection: rank 2-3 model families from this allowed list only:
   { _ALLOWED_MODEL_FAMILIES }
   Consider dataset size, feature types, class imbalance, and user constraints.
   Give plain-English rationale for each shortlisted family.

2) Search strategy by row count:
   - rows < 5000: GridSearchCV
   - rows 5000-50000: RandomizedSearchCV
   - rows > 50000: Optuna with 50 trials

3) CV design:
   - classification: StratifiedKFold
   - regression: KFold
   - folds = 3 when rows < 500 else 5

4) Metric and threshold:
   - choose one primary metric suitable for this task
   - define a context-aware performance threshold (not a universal fixed value)
   - provide a short rationale for that threshold.

Return JSON only in this exact shape:
{{
  "models": [{{"family": "...", "rationale": "..."}}, ...],
  "search_strategy": {{"name": "GridSearchCV|RandomizedSearchCV|Optuna", "n_trials": 50, "reason": "..."}},
  "cv_design": {{"type": "StratifiedKFold|KFold", "folds": 3|5, "shuffle": true, "random_state": 42}},
  "metric": "...",
  "threshold": 0.0,
  "threshold_rationale": "..."
}}"""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON)
    parsed = extract_json_object(raw) if raw else None
    plan = _sanitize_plan(parsed or {}, fallback)

    # Enforce row-count strategy if LLM deviates.
    if n_rows < 5000:
        plan["search_strategy"]["name"] = "GridSearchCV"
    elif n_rows <= 50000:
        plan["search_strategy"]["name"] = "RandomizedSearchCV"
    else:
        plan["search_strategy"]["name"] = "Optuna"
        plan["search_strategy"]["n_trials"] = 50

    plan["cv_design"]["type"] = "StratifiedKFold" if task_type == "classification" else "KFold"
    plan["cv_design"]["folds"] = 3 if n_rows < 2000 else 5

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
                },
            },
        )

    if not parsed:
        logger.warning("make_execution_plan: LLM output unavailable or unparseable; fallback plan used")

    return plan
