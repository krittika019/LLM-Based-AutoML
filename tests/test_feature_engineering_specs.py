"""Feature engineering records a replayable spec for every feature it adds."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from automl_system.llm.client import LLMClient
from automl_system.pipeline import feature_engineering as fe
from automl_system.pipeline import replay
from automl_system.state.session import SessionState

PROPOSALS = {"features": [
    {"name": "a_per_b", "type": "ratio", "left": "a", "right": "b", "rationale": "r"},
    {"name": "a_times_b", "type": "interaction", "left": "a", "right": "b", "rationale": "r"},
    {"name": "combo", "type": "domain_combination", "formula": "a * 2 + b", "rationale": "r"},
    {"name": "escape", "type": "domain_combination", "formula": "a.__class__", "rationale": "r"},
]}


def test_added_features_have_replayable_specs(monkeypatch):
    rng = np.random.default_rng(3)
    n = 400
    a, b = rng.normal(size=n), rng.normal(size=n)
    b[:5] = 0.0
    df = pd.DataFrame({"a": a, "b": b, "y": a * 2 + b + rng.normal(0, 0.5, n)})
    train, test = df.iloc[:300].reset_index(drop=True), df.iloc[300:].reset_index(drop=True)

    monkeypatch.setattr(LLMClient, "call", lambda *_a, **_k: json.dumps(PROPOSALS))
    monkeypatch.setattr(fe, "should_engineer_features", lambda *_a, **_k: (True, ""))
    session = SessionState(task_config={"target_column": "y", "task_type": "regression"})
    out, out_test = fe.engineer_features(train, session, df_test=test)

    added = [e for e in session.feature_engineering_log if e["status"] == "added"]
    assert added and all("spec" in e for e in added)
    # Attribute access in a formula is rejected by the AST whitelist.
    assert "escape" not in out.columns

    for e in added:
        spec = e["spec"]
        replayed = replay.compute_raw_feature(out, spec)
        replayed_test = replay.compute_raw_feature(out_test, spec)
        if spec["fill"] is not None:
            replayed = replayed.fillna(spec["fill"])
            replayed_test = replayed_test.fillna(spec["test_fill"])
        pd.testing.assert_series_equal(replayed, out[spec["name"]], check_names=False)
        pd.testing.assert_series_equal(replayed_test, out_test[spec["name"]], check_names=False)
