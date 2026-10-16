"""build_phase_details: real content from recorded session data, never raises."""

import json

from automl_system.state.session import SessionState
from automl_system.utils.phase_details import build_phase_details


def _full_session() -> SessionState:
    s = SessionState()
    s.task_config = {"target_column": "HeartDisease", "task_type": "classification"}
    s.problem_summary = {
        "task_type": "classification", "target": "HeartDisease", "rows": 918, "columns": 12,
        "numerical_count": 7, "categorical_count": 5, "missing_values": 0, "duplicate_rows": 0,
        "primary_metric_label": "ROC-AUC", "objective": "Catch every real case.",
        "table": [{"name": "Age", "dtype": "int64", "role": "feature"},
                  {"name": "HeartDisease", "dtype": "int64", "role": "target"}],
    }
    s.dataset_profile = {"columns": {"Cholesterol": {"null_pct": 18.7}, "Age": {"null_pct": 0}}}
    s.preprocessing_plan = {"encoding_candidates": ["Sex", "ChestPainType"], "skewed_columns": ["Oldpeak"],
                            "leakage_risks": [], "scaling_needed": True, "feature_engineering_recommended": True,
                            "notes": ["Cholesterol zeros look like missing values."]}
    s.user_summary_planning = "Encode the five categorical columns and scale numerics."
    s.cleaning_report = {"duplicate_rows_removed": 0, "coercion_log": [{"column": "Cholesterol"}]}
    s.eda_results = {"target_correlations": {"ST_Slope_Up": -0.62, "Oldpeak": 0.4, "Age": 0.28},
                     "target_imbalance": {"severity": "mild", "minority_pct": 44.7, "n_classes": 2}}
    s.feature_selection_log = [{"column": "PatientID", "rationale": "Identifier.", "tier": "rule"}]
    s.deep_cleaning_assignments = {
        "Age": {"imputation": "median", "outlier": "leave", "encoding": "none", "scaling": "standard", "source": "rule"},
        "Cholesterol": {"imputation": "knn", "outlier": "clip", "encoding": "none", "scaling": "robust", "source": "llm"},
    }
    s.feature_engineering_log = [
        {"feature": "age_x_oldpeak", "status": "added", "cv_gain": 0.0041, "describe": "Age × Oldpeak", "round": 1},
        {"feature": "chol_ratio", "status": "rejected_no_cv_gain", "cv_gain": -0.0002, "round": 1},
    ]
    s.last_verification = {"ok": True, "shape": [734, 15]}
    s.preprocessing_meta = {
        "row_preview": {"before": [{"column": "Age", "value": "40"}], "after": [{"column": "Age", "value": "-1.43"}],
                        "n_before": 12, "n_after": 15},
        "phase6_checks": [{"iteration": 0, "threshold": 0.9, "best": 0.93, "best_model": "CatBoost",
                           "outcome": "met", "reason": ""}],
    }
    s.execution_plan = {"models": [{"family": "catboost", "rationale": "Handles categoricals."},
                                   {"family": "logistic_regression", "rationale": "Strong linear baseline."}],
                        "search_strategy": "optuna_tpe", "cv_design": {"type": "stratified_kfold", "n_splits": 5},
                        "metric": "roc_auc", "threshold": 0.9, "threshold_rationale": "Typical for this data."}
    s.execution_results = {
        "comparison_table": [
            {"model_name": "CatBoost", "cv_score": 0.925, "test_score": 0.931, "status": "ok", "training_time": 4.1},
            {"model_name": "LogisticRegression", "cv_score": 0.91, "test_score": 0.917, "status": "ok"},
            {"model_name": "SVC", "cv_score": None, "test_score": None, "status": "failed"},
        ],
        "best_model_name": "CatBoost", "metric": "roc_auc", "metric_display": "ROC-AUC",
        "metric_is_lower_better": False, "best_test_score": 0.931,
        "best_model_metrics": {"accuracy": 0.88, "recall": 0.91, "note": "text is skipped"},
        "split_meta": {"n_train": 734, "n_test": 184, "split_type": "stratified"},
        "shap_summary": {"available": True, "features": [{"name": "ST_Slope_Up", "mean_abs_shap": 0.81}]},
        "elimination_audit": {"eliminated_family": "knn", "elimination_correct": True},
    }
    s.all_results = [dict(s.execution_results)]
    s.script_validation = {"status": "verified", "source": "assembled", "reported_score": 0.931,
                           "reproduced_score": 0.931, "gap": 0.0, "attempts": 1, "backend": "docker",
                           "history": [{"attempt": 1, "stage": "semantic", "problem": "", "reproduced_score": 0.931}]}
    s.final_pipeline_script = "import pandas as pd\nprint(1)\n"
    s.phase_timings = {"eda": 3.2, "deep_cleaning": 1.0, "train_evaluate": 40.0}
    s.token_cost_table = {"eda": 1, "deep_cleaning": 2, "model_training": 1}
    s.local_decisions_log = [{"step": "Phase 1", "action": "metric_selected"},
                             {"step": "Phase 5 - Execution", "action": "train_evaluate_complete"}]
    return s


def test_empty_session_marks_every_phase_unavailable():
    d = build_phase_details(SessionState())
    assert set(d) == {str(i) for i in range(1, 8)}
    assert not any(p["available"] for p in d.values())
    json.dumps(d, allow_nan=False)


def test_full_session_builds_real_content():
    d = build_phase_details(_full_session())
    assert all(d[str(i)]["available"] for i in range(1, 8))
    json.dumps(d, allow_nan=False)

    p3 = d["3"]
    assert p3["seconds"] == 4.2 and p3["llm_calls"] == 3
    types = {sec["type"] for sec in p3["sections"]}
    assert {"steps", "bars", "table", "features", "diff"} <= types
    recipe = next(sec for sec in p3["sections"] if sec["type"] == "table")
    assert ["Cholesterol", "knn", "none", "robust", "clip", "llm"] in recipe["rows"]
    assert any(dec["what"] == "Added feature age_x_oldpeak" and dec["by"] == "stat" for dec in p3["decisions"])

    p5 = d["5"]
    board = next(sec for sec in p5["sections"] if sec["type"] == "leaderboard")
    assert [i["label"] for i in board["items"]] == ["CatBoost", "LogisticRegression"]
    assert board["items"][0]["best"] is True
    assert p5["decisions"][0]["entry"] == 2
    assert d["1"]["decisions"][-1]["entry"] == 1

    assert d["6"]["stats"][2]["value"] == "+0.0300"
    assert "verified" in d["7"]["stats"][0]["value"]


def test_lower_is_better_metric_shows_positive_errors():
    s = _full_session()
    s.execution_results.update(metric_is_lower_better=True, comparison_table=[
        {"model_name": "A", "cv_score": -3.0, "test_score": -2.5, "status": "ok"},
        {"model_name": "B", "cv_score": -3.1, "test_score": -2.9, "status": "ok"}])
    board = next(sec for sec in build_phase_details(s)["5"]["sections"] if sec["type"] == "leaderboard")
    assert [i["value"] for i in board["items"]] == [2.5, 2.9]


def test_malformed_data_never_raises():
    s = _full_session()
    s.execution_results = {"comparison_table": "oops", "split_meta": None}
    s.deep_cleaning_assignments = {"x": "not-a-dict"}
    s.feature_engineering_log = [None, {"feature": "f", "cv_gain": float("nan")}]
    s.execution_plan = {"models": None, "cv_design": "5-fold", "threshold": "high"}
    s.last_failure_diagnosis = {"failure_type": "feature_issue", "evidence": "free text"}
    d = build_phase_details(s)
    assert set(d) == {str(i) for i in range(1, 8)}
    json.dumps(d, allow_nan=False)
