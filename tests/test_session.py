"""SessionState.to_dict must never deep-copy trained model objects.

Regression: to_dict used dataclasses.asdict, which deep-copied the "_fitted_estimator"
objects kept in Phase 5 results. Deep-copying an XGBoost GPU booster from the server's
main thread crashed the backend natively, so the final event (script + dashboard) was
never sent to the UI.
"""

from __future__ import annotations

import copy

from automl_system.state.session import SessionState


class _ExplodingModel:
    def __deepcopy__(self, memo):
        raise AssertionError("trained models must not be deep-copied")


def test_to_dict_skips_models_and_copies_data():
    s = SessionState(
        execution_results={
            "comparison_table": [{"model_name": "XGBoost", "test_score": 0.9, "_fitted_estimator": _ExplodingModel()}],
            "best_test_score": 0.9,
        },
        all_results=[{"comparison_table": [{"model_name": "CatBoost", "_fitted_estimator": _ExplodingModel()}]}],
        token_cost_table={"_total": 5, "eda": 2},
    )
    d = s.to_dict()                                   # must not raise
    row = d["execution_results"]["comparison_table"][0]
    assert "_fitted_estimator" not in row and row["test_score"] == 0.9
    assert "_fitted_estimator" not in d["all_results"][0]["comparison_table"][0]
    assert d["token_cost_table"]["_total"] == 5       # private *data* keys are kept

    row["test_score"] = 0.1                           # copy is independent of the session
    assert s.execution_results["comparison_table"][0]["test_score"] == 0.9
    assert SessionState.from_dict(copy.deepcopy(d)).to_dict() == d
