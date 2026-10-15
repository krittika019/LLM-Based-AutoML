"""The assembled script must reproduce, exactly, the steps recorded during a run."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from automl_system.pipeline import replay
from automl_system.pipeline.execution import _estimator_spec
from automl_system.pipeline.script_builder import UnsupportedRecipe, assemble_pipeline_script
from automl_system.pipeline.script_checks import static_check
from automl_system.sandbox import Sandbox
from automl_system.state.session import SessionState

ASSIGN = {
    "age": {"imputation": "median", "outlier": "iqr_cap", "scaling": "standard"},
    "income": {"imputation": "flag_as_missing", "outlier": "log", "scaling": "none"},
    "city": {"imputation": "mode", "encoding": "one_hot"},
    "plan": {"imputation": "mode", "encoding": "binary"},
}
FE = [{"type": "ratio", "left": "income", "right": "age", "formula": "", "name": "inc_per_age",
       "fill": None, "test_fill": None}]


def _raw(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "age": rng.integers(18, 80, n).astype(float),
        "income": rng.exponential(40_000, n),
        "city": rng.choice(["Pune", "Delhi", "Goa"], n),
        "plan": rng.choice(["basic", "pro"], n),
    })
    df.loc[rng.choice(n, 30, replace=False), "income"] = np.nan
    signal = (df["age"] / 80 + (df["plan"] == "pro") * 0.5 + rng.normal(0, 0.3, n))
    df["target"] = np.where(signal > 0.9, "churn", "stay") if task == "clf" else np.exp(signal * 3)
    # CSV round-trip so in-process frames have exactly the dtypes the script will read
    train, test = df.iloc[:300], df.iloc[300:]
    return (pd.read_csv(io.StringIO(train.to_csv(index=False))),
            pd.read_csv(io.StringIO(test.to_csv(index=False))))


def _preprocess(train: pd.DataFrame, test: pd.DataFrame):
    keep = ["age", "income", "city", "plan", "target"]
    tr, te = replay.apply_deep_clean(train[keep], test[keep], ASSIGN, "target")
    for spec in FE:
        tr[spec["name"]] = replay.compute_raw_feature(tr, spec)
        te[spec["name"]] = replay.compute_raw_feature(te, spec)
    te = te.reindex(columns=tr.columns, fill_value=0.0)
    x_tr = replay.safe_numeric_features(tr.drop(columns=["target"]))
    x_te = replay.safe_numeric_features(te.drop(columns=["target"]))
    return keep, x_tr, x_te, tr["target"]


def _session(keep, model_spec, feature_cols, task: str, log_target=False) -> SessionState:
    return SessionState(
        task_config={"target_column": "target"},
        preprocessing_meta={"feature_selection_columns": keep},
        deep_cleaning_assignments=ASSIGN,
        feature_engineering_log=[{"feature": s["name"], "status": "added", "spec": s} for s in FE],
        execution_results={
            "task_type": "classification" if task == "clf" else "regression",
            "best_model_name": "test-model", "metric": "accuracy",
            "replay": {"smote": None, "log_target": log_target, "has_nan": False,
                       "feature_columns": feature_cols, "model": model_spec},
        },
    )


def _run(script: str, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    assert static_check(script) == [], static_check(script)
    with Sandbox("local") as sb:
        sb.put_dataframe("train.csv", train)
        sb.put_dataframe("test.csv", test)
        res = sb.run(script, timeout=180)
        assert res.ok, res.stderr[-3000:]
        return pd.read_csv(io.StringIO(sb.read_text("predictions.csv")))


def test_classification_single_model_matches_in_process():
    train, test = _raw("clf")
    keep, x_tr, x_te, y = _preprocess(train, test)
    feats = [c for c in x_tr.columns if c != "plan"]  # pretend the MI filter dropped one
    le = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=2)
    model.fit(x_tr[feats], pd.Series(le.transform(y)))
    expected = le.inverse_transform(model.predict(x_te[feats]))

    spec = {"kind": "single", "family": "RandomForest", "estimator": _estimator_spec(model),
            "fit": {"mode": "full"}, "impute_median": False}
    assert "n_jobs" not in spec["estimator"]["params"]
    preds = _run(assemble_pipeline_script(_session(keep, spec, feats, "clf")), train, test)

    assert list(preds["prediction"]) == list(expected)
    np.testing.assert_allclose(preds[["proba_0", "proba_1"]].to_numpy(),
                               model.predict_proba(x_te[feats]), rtol=1e-9)


def test_regression_early_stopping_log_target_matches_in_process():
    train, test = _raw("reg")
    keep, x_tr, x_te, y = _preprocess(train, test)
    feats = list(x_tr.columns)
    y_fit = np.log1p(y)
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, verbosity=-1, n_jobs=2)
    rows = len(x_tr)
    idx = np.random.default_rng(43).choice(rows, size=rows, replace=False)
    x_r, y_r = x_tr.iloc[idx], y_fit.iloc[idx]
    x_a, x_v, y_a, y_v = train_test_split(x_r, y_r, test_size=0.1, random_state=42)
    import lightgbm as lgb
    model.fit(x_a, y_a, eval_set=[(x_v, y_v)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    expected = np.expm1(model.predict(x_te[feats]))

    spec = {"kind": "single", "family": "LightGBM", "estimator": _estimator_spec(model),
            "fit": {"mode": "early_stopping", "rows": rows, "seed": 43, "val_size": 0.1, "split_seed": 42},
            "impute_median": False}
    preds = _run(assemble_pipeline_script(_session(keep, spec, feats, "reg", log_target=True)), train, test)
    np.testing.assert_allclose(preds["prediction"].to_numpy(), expected, rtol=1e-9)


def _single(est, family, impute=False):
    return {"kind": "single", "family": family, "estimator": _estimator_spec(est),
            "fit": {"mode": "full"}, "impute_median": impute}


@pytest.mark.parametrize("kind", ["bag", "threshold", "average", "stacking"])
def test_ensemble_kinds_run(kind):
    train, test = _raw("clf")
    keep, x_tr, _, _ = _preprocess(train, test)
    feats = list(x_tr.columns)
    hgb = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    rf = RandomForestClassifier(n_estimators=40, random_state=42)
    lr = LogisticRegression(max_iter=500)
    spec = {
        "bag": {"kind": "bag", "base": _single(hgb, "HistGradientBoosting"),
                "member": _estimator_spec(HistGradientBoostingClassifier(max_iter=50, random_state=123)),
                "seed_key": "random_state", "seeds": [123, 456, 789]},
        "threshold": {"kind": "threshold", "base": _single(rf, "RandomForest"), "threshold": 0.35},
        "average": {"kind": "average", "members": [_single(hgb, "HistGradientBoosting"),
                                                   _single(rf, "RandomForest")],
                    "weights": [0.6, 0.4], "soft": True},
        "stacking": {"kind": "stacking", "members": [["HGB", _estimator_spec(hgb)], ["RF", _estimator_spec(rf)]],
                     "final_estimator": _estimator_spec(lr), "cv": 3},
    }[kind]
    preds = _run(assemble_pipeline_script(_session(keep, spec, feats, "clf")), train, test)
    assert len(preds) == len(test)
    assert set(preds["prediction"]) <= {"churn", "stay"}


def test_unrecorded_run_is_unsupported():
    with pytest.raises(UnsupportedRecipe):
        assemble_pipeline_script(SessionState(task_config={"target_column": "y"}))


def test_class_votes_never_invent_labels():
    # Two models say class 0, one says class 2: averaging labels would give 0.67 → 1 (wrong).
    preds = [np.array([0, 2]), np.array([0, 2]), np.array([2, 0])]
    pred, proba = replay.combine_class_votes(preds, None)
    assert list(pred) == [0, 2] and proba is None
    # Weighted: the heavier model wins a 1-vs-1 disagreement.
    pred_w, _ = replay.combine_class_votes([np.array([0]), np.array([2])], None, [0.3, 0.7])
    assert list(pred_w) == [2]
    # Soft vote when all probabilities are available.
    probas = [np.array([[0.6, 0.1, 0.3]]), np.array([[0.1, 0.1, 0.8]])]
    pred_s, proba_s = replay.combine_class_votes([np.array([0]), np.array([2])], probas)
    assert list(pred_s) == [2] and proba_s.shape == (1, 3)


def test_multiclass_bag_matches_in_process():
    train, test = _raw("clf")
    rng = np.random.default_rng(11)
    for frame in (train, test):  # three classes
        frame["target"] = rng.choice(["low", "mid", "high"], len(frame))
        frame.loc[frame["age"] > 60, "target"] = "high"
    keep, x_tr, x_te, y = _preprocess(train, test)
    feats = list(x_tr.columns)
    le = LabelEncoder().fit(y)
    y_enc = pd.Series(le.transform(y))
    base = HistGradientBoostingClassifier(max_iter=40, random_state=42).fit(x_tr[feats], y_enc)
    preds, probas = [base.predict(x_te[feats]).astype(float)], [base.predict_proba(x_te[feats])]
    for seed in (123, 456, 789):
        m = HistGradientBoostingClassifier(max_iter=40, random_state=seed).fit(x_tr[feats], y_enc)
        preds.append(m.predict(x_te[feats]).astype(float))
        probas.append(m.predict_proba(x_te[feats]))
    expected = le.inverse_transform(replay.combine_class_votes(preds, probas)[0])

    spec = {"kind": "bag", "base": _single(base, "HistGradientBoosting"),
            "member": _estimator_spec(HistGradientBoostingClassifier(max_iter=40, random_state=123)),
            "seed_key": "random_state", "seeds": [123, 456, 789]}
    out = _run(assemble_pipeline_script(_session(keep, spec, feats, "clf")), train, test)
    assert list(out["prediction"]) == list(expected)
    assert {"proba_0", "proba_1", "proba_2"} <= set(out.columns)


def test_raw_scale_features_replay_in_script():
    train, test = _raw("clf")
    keep, x_tr, x_te, y = _preprocess(train, test)
    spec = {"type": "ratio", "left": "income", "right": "age", "name": "income_per_age",
            "source": "raw", "fill": None, "test_fill": None}
    raw_tr = train[keep].drop(columns=["target"]).reset_index(drop=True)
    raw_te = test[keep].drop(columns=["target"]).reset_index(drop=True)
    vals = replay.finish_feature(replay.compute_raw_feature(raw_tr, spec), None)
    spec["fill"] = spec["test_fill"] = float(vals.median())
    raw_tr, raw_te = replay.add_raw_features(raw_tr, raw_te, [spec])
    x_tr["income_per_age"] = raw_tr["income_per_age"].to_numpy()
    x_te["income_per_age"] = np.asarray(raw_te["income_per_age"], dtype=float)
    feats = list(x_tr.columns)
    le = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(x_tr[feats], le.transform(y))
    expected = le.inverse_transform(model.predict(x_te[feats]))

    session = _session(keep, {"kind": "single", "family": "RandomForest", "estimator": _estimator_spec(model),
                              "fit": {"mode": "full"}, "impute_median": False}, feats, "clf")
    session.feature_engineering_log.append({"feature": spec["name"], "status": "added", "spec": spec})
    out = _run(assemble_pipeline_script(session), train, test)
    assert list(out["prediction"]) == list(expected)


def test_code_feature_replays_in_script():
    train, test = _raw("clf")
    keep, x_tr, x_te, y = _preprocess(train, test)
    code = ("def make_feature(train, test):\n"
            "    m = train.groupby('city')['income'].mean()\n"
            "    return train['income'] - train['city'].map(m), test['income'] - test['city'].map(m)\n")
    raw_tr = train[keep].drop(columns=["target"]).reset_index(drop=True)
    raw_te = test[keep].drop(columns=["target"]).reset_index(drop=True)
    spec = {"type": "code", "name": "income_vs_city", "code": code, "column": "0",
            "input_columns": list(raw_tr.columns), "source": "raw"}
    vals, _ = replay.run_code_feature(raw_tr, raw_te, code)
    spec["fill"] = spec["test_fill"] = float(vals["0"].median())
    raw_tr, raw_te = replay.add_raw_features(raw_tr, raw_te, [spec])
    x_tr["income_vs_city"] = raw_tr["income_vs_city"].to_numpy()
    x_te["income_vs_city"] = np.asarray(raw_te["income_vs_city"], dtype=float)
    feats = list(x_tr.columns)
    le = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(x_tr[feats], le.transform(y))
    expected = le.inverse_transform(model.predict(x_te[feats]))

    session = _session(keep, {"kind": "single", "family": "RandomForest", "estimator": _estimator_spec(model),
                              "fit": {"mode": "full"}, "impute_median": False}, feats, "clf")
    session.feature_engineering_log.append({"feature": spec["name"], "status": "added", "spec": spec})
    out = _run(assemble_pipeline_script(session), train, test)
    assert list(out["prediction"]) == list(expected)
