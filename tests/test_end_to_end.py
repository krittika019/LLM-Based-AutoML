from __future__ import annotations

import json

import pandas as pd
from sklearn.datasets import fetch_openml

from automl_system.graph.orchestrator import build_graph
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import compute_dataset_profile


_MEMORY: dict[str, list[dict]] = {}


def _mock_create_session() -> str:
    sid = "test-session"
    _MEMORY[sid] = []
    return sid


def _mock_append_decision(session_id: str, entry: dict) -> None:
    _MEMORY.setdefault(session_id, []).append(entry)


def _mock_get_log(session_id: str) -> list[dict]:
    return list(_MEMORY.get(session_id, []))


def _mock_llm_call(self, prompt: str, model: str | None = None, system_prompt: str | None = None, **kwargs) -> str:
    if "Return ONE JSON object with keys" in prompt:
        return json.dumps(
            {
                "null_strategy": "median_for_numeric_mode_for_categorical",
                "encoding_strategy": "one_hot_for_low_cardinality_target_else_ordinal",
                "scaling_needed": True,
                "feature_engineering_likely": True,
                "leakage_risks": ["passenger_id_like_fields"],
                "notes": ["Test mock plan"],
                "user_summary": "Prepared preprocessing plan for Titanic survival prediction.",
            }
        )

    if "Respond with JSON only:" in prompt and '"narrative"' in prompt:
        return json.dumps(
            {
                "narrative": "Age and fare appear informative; class imbalance exists but manageable.",
                "user_summary": "EDA complete with key feature and quality observations.",
            }
        )

    if "Return JSON only: {\"drops\"" in prompt:
        return json.dumps({"drops": [{"column": "name", "rationale": "High-cardinality identifier-like text."}]})

    if "Return JSON: {\"columns\"" in prompt:
        return json.dumps({"columns": {}})

    if "Propose derived features for tabular ML" in prompt:
        return json.dumps({"features": []})

    if "One short decision per item" in prompt:
        return "SET_NULL for invalid values where needed."

    if "Return JSON only in this exact shape" in prompt:
        return json.dumps(
            {
                "models": [
                    {
                        "family": "LogisticRegression",
                        "rationale": "Strong baseline for binary classification.",
                    },
                    {
                        "family": "RandomForest",
                        "rationale": "Captures non-linear interactions.",
                    },
                    {
                        "family": "HistGradientBoosting",
                        "rationale": "Efficient boosting for tabular features.",
                    },
                ],
                "search_strategy": {"name": "GridSearchCV", "n_trials": 50, "reason": "Small dataset"},
                "cv_design": {"type": "StratifiedKFold", "folds": 5, "shuffle": True, "random_state": 42},
                "metric": "f1_weighted",
                "threshold": 0.7,
                "threshold_rationale": "Balanced target and practical baseline.",
            }
        )

    if "Interpret this model comparison table" in prompt:
        return "RandomForest performed best with stable CV and test behavior."

    if "Classify failure into one of exactly" in prompt:
        return json.dumps(
            {
                "failure_type": "model_issue",
                "responsible_phase": "phase4_planning",
                "proposed_fix": "Adjust model ranking and regularization.",
                "should_exit": False,
            }
        )

    if "Write a concise but complete markdown report" in prompt:
        return "## Report\n\nTitanic run completed with model comparison and final recommendation."

    if "Generate a single Python script for this AutoML run" in prompt:
        return "import pandas as pd\nprint('pipeline script')\n"

    return ""


def _load_titanic() -> pd.DataFrame:
    ds = fetch_openml("titanic", version=1, as_frame=True)
    df = ds.frame.copy()
    if "survived" not in df.columns and ds.target is not None:
        df["survived"] = ds.target
    return df


def test_end_to_end() -> None:
    # Monkeypatch external dependencies for deterministic local test.
    episodic.create_session = _mock_create_session
    episodic.append_decision = _mock_append_decision
    episodic.get_log = _mock_get_log
    LLMClient.call = _mock_llm_call

    df = _load_titanic()

    session = SessionState()
    session.decisions_log_id = episodic.create_session()
    session.task_config = {
        "goal": "predict whether a passenger survived",
        "target_column": "survived",
    }
    session.dataset_profile = compute_dataset_profile(df)

    graph = build_graph()
    out = graph.invoke(
        {
            "session": session.to_dict(),
            "df": df,
            "messages": [],
            "checkpoints": {},
        }
    )

    final = SessionState.from_dict(out.get("session"))

    decisions = episodic.get_log(final.decisions_log_id or "")
    assert len(decisions) > 0, "Expected decisions log entries"

    comparison = list(final.final_comparison_table or [])
    assert len(comparison) >= 2, "Expected at least 2 model rows in final comparison table"

    script = str(final.final_pipeline_script or "")
    assert script.strip(), "Expected non-empty pipeline script artifact"
