"""CV-gated feature engineering keeps features that measurably help, rejects noise,
and records specs the generated script replays exactly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.pipeline import feature_engineering as fe
from automl_system.pipeline import replay
from automl_system.state.session import SessionState

PROPOSALS = {"features": [
    {"name": "a_per_b", "type": "ratio", "left": "a", "right": "b", "rationale": "true signal"},
    {"name": "c_times_d", "type": "interaction", "left": "c", "right": "d", "rationale": "noise"},
    {"name": "bad_ref", "type": "ratio", "left": "a", "right": "missing_col", "rationale": "typo"},
]}


def _data(n: int = 1500):
    rng = np.random.default_rng(5)
    a = rng.uniform(1, 100, n)
    b = rng.uniform(1, 100, n)
    c, d = rng.normal(size=n), rng.normal(size=n)
    y = np.where(a / b + rng.normal(0, 0.15, n) > 1.0, "yes", "no")
    raw = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "city": rng.choice(["x", "y"], n), "y": y})
    raw.loc[rng.choice(n, 40, replace=False), "b"] = np.nan
    train, test = raw.iloc[:1200].reset_index(drop=True), raw.iloc[1200:].reset_index(drop=True)
    # Stand-in for deep cleaning: scaled numerics, encoded category, imputed b
    def clean(f):
        out = pd.DataFrame({"y": f["y"]})
        for col in ("a", "b", "c", "d"):
            v = f[col].fillna(train[col].median())
            out[col] = (v - train[col].mean()) / train[col].std()
        out["city"] = (f["city"] == "x").astype(float)
        return out
    return train, test, clean(train), clean(test)


def test_cv_gate_accepts_signal_rejects_noise(monkeypatch):
    monkeypatch.setattr(config, "FE_SELECTION", "cv")
    monkeypatch.setattr(config, "FE_ROUNDS", 2)
    calls = []
    monkeypatch.setattr(LLMClient, "call", lambda self, prompt, **_k: calls.append(prompt) or json.dumps(PROPOSALS))
    raw_train, raw_test, train, test = _data()
    session = SessionState(task_config={"target_column": "y", "task_type": "classification"})

    out, out_test = fe.engineer_features(train, session, df_test=test, raw_train=raw_train, raw_test=raw_test)

    status = {e["feature"]: e["status"] for e in session.feature_engineering_log}
    assert status["a_per_b"] == "added"
    assert status["c_times_d"] in ("rejected_no_cv_gain", "rejected_redundant")
    assert status["bad_ref"] == "error"
    assert "a_per_b" in out.columns and "c_times_d" not in out.columns
    assert list(out_test.columns) == list(out.columns)
    summary = session.preprocessing_meta["fe_cv_summary"]
    assert summary["final_cv"] > summary["baseline_cv"]
    # Round 2 prompt carries the measured results back to the LLM.
    assert len(calls) == 2 and "PREVIOUS ROUND RESULTS" in calls[1] and "ACCEPTED a_per_b" in calls[1]

    # The recorded spec replays exactly on train and test.
    spec = next(e["spec"] for e in session.feature_engineering_log if e["status"] == "added")
    assert spec["source"] == "raw"
    rt, rte = replay.add_raw_features(raw_train.drop(columns=["y"]), raw_test.drop(columns=["y"]), [spec])
    np.testing.assert_array_equal(rt["a_per_b"].to_numpy(), out["a_per_b"].to_numpy())
    np.testing.assert_array_equal(np.asarray(rte["a_per_b"], dtype=float), out_test["a_per_b"].to_numpy())


def test_corr_mode_keeps_old_behaviour(monkeypatch):
    monkeypatch.setattr(config, "FE_SELECTION", "corr")
    monkeypatch.setattr(LLMClient, "call", lambda *_a, **_k: json.dumps(PROPOSALS))
    monkeypatch.setattr(fe, "should_engineer_features", lambda *_a, **_k: (True, ""))
    raw_train, raw_test, train, test = _data()
    session = SessionState(task_config={"target_column": "y", "task_type": "classification"})
    fe.engineer_features(train, session, df_test=test, raw_train=raw_train, raw_test=raw_test)
    assert "fe_cv_summary" not in session.preprocessing_meta
    assert all("cv_gain" not in e for e in session.feature_engineering_log)
