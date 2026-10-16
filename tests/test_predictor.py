"""Try the model: Predictor on raw rows == the run's model on the run's own preprocessing."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from automl_system.pipeline import replay
from automl_system.pipeline.predictor import PredictUnavailable, Predictor, bag_members, build_bundle
from automl_system.state.session import SessionState


def _data(n=240, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(20, 80, n).astype(float),
        "chol": rng.normal(220, 40, n),
        "sex": rng.choice(["M", "F"], n),
        "pain": rng.choice(["ASY", "ATA", "NAP", "TA"], n),
    })
    df.loc[rng.rand(n) < 0.1, "chol"] = np.nan
    logit = 0.05 * (df["age"] - 50) + (df["sex"] == "M") * 0.8 + (df["pain"] == "ASY") * 1.2
    df["target"] = np.where(logit + rng.normal(0, 1, n) > 0.8, "yes", "no")
    return df.iloc[:180].reset_index(drop=True), df.iloc[180:].reset_index(drop=True)


def _processed(train, test, assignments):
    tr, te = replay.apply_deep_clean(train, test.drop(columns=["target"]), assignments, "target")
    x_tr = replay.safe_numeric_features(tr.drop(columns=["target"]))
    x_te = replay.safe_numeric_features(te.drop(columns=["target"], errors="ignore"))
    imp = SimpleImputer(strategy="median").fit(x_tr)
    return x_tr, x_te, pd.DataFrame(imp.transform(x_tr), columns=x_tr.columns), \
        pd.DataFrame(imp.transform(x_te), columns=x_te.columns), tr["target"]


def _session(train, model_spec, assignments, feature_columns, task="classification"):
    s = SessionState()
    s.task_config = {"target_column": "target"}
    s.deep_cleaning_assignments = assignments
    s.execution_results = {"task_type": task, "best_model_name": "m", "replay": {
        "model": model_spec, "feature_columns": feature_columns, "log_target": False}}
    return s


@pytest.fixture
def clf_setup():
    train, test = _data()
    assignments = {c: replay.default_assignment(c, train[c]) for c in ["age", "chol", "sex", "pain"]}
    x_tr, x_te, x_tr_imp, x_te_imp, y = _processed(train, test, assignments)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    lr = LogisticRegression(max_iter=500).fit(x_tr_imp, y_enc)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(x_tr_imp, y_enc)
    rows = [
        {"model_name": "LogisticRegression", "status": "ok", "_fitted_estimator": lr,
         "model_spec": {"kind": "single", "family": "LogisticRegression", "impute_median": True}},
        {"model_name": "RandomForest", "status": "ok", "_fitted_estimator": rf,
         "model_spec": {"kind": "single", "family": "RandomForest", "impute_median": True}},
    ]
    return train, test, assignments, list(x_tr.columns), x_te_imp, le, lr, rf, rows


def test_single_model_matches_pipeline_and_survives_pickle(clf_setup, tmp_path):
    train, test, assignments, cols, x_te_imp, le, lr, _rf, rows = clf_setup
    spec = rows[0]["model_spec"]
    bundle = build_bundle(_session(train, spec, assignments, cols), rows, train)
    joblib.dump(bundle, tmp_path / "b.joblib")
    pred = Predictor(joblib.load(tmp_path / "b.joblib"))
    out = pred.predict(test.drop(columns=["target"]).to_dict("records"))
    expected = le.inverse_transform(lr.predict(x_te_imp))
    assert [o["prediction"] for o in out] == list(expected)
    np.testing.assert_allclose([o["probabilities"]["yes"] for o in out], lr.predict_proba(x_te_imp)[:, 1], atol=1e-4)
    assert out[0]["contributions"]["items"]  # linear explanation present


def test_one_row_with_blank_and_unseen_values(clf_setup):
    train, test, assignments, cols, x_te_imp, le, lr, _rf, rows = clf_setup
    pred = Predictor(build_bundle(_session(train, rows[0]["model_spec"], assignments, cols), rows, train))
    one = pred.predict([{"age": "61", "chol": "", "sex": "M", "pain": "UNSEEN"}])
    assert one[0]["prediction"] in ("yes", "no")
    assert abs(sum(one[0]["probabilities"].values()) - 1) < 1e-3


def test_threshold_bag_and_average_kinds(clf_setup):
    train, test, assignments, cols, x_te_imp, le, lr, rf, rows = clf_setup
    raw = test.drop(columns=["target"]).to_dict("records")
    lr_spec, rf_spec = rows[0]["model_spec"], rows[1]["model_spec"]

    thr = {"kind": "threshold", "base": lr_spec, "threshold": 0.3}
    out = Predictor(build_bundle(_session(train, thr, assignments, cols), rows, train)).predict(raw, explain=False)
    expected = le.inverse_transform((lr.predict_proba(x_te_imp)[:, 1] >= 0.3).astype(int))
    assert [o["prediction"] for o in out] == list(expected)

    avg = {"kind": "average", "members": [lr_spec, rf_spec], "weights": [0.5, 0.5], "soft": True}
    out = Predictor(build_bundle(_session(train, avg, assignments, cols), rows, train)).predict(raw, explain=False)
    proba = 0.5 * lr.predict_proba(x_te_imp) + 0.5 * rf.predict_proba(x_te_imp)
    assert [o["prediction"] for o in out] == list(le.inverse_transform(proba.argmax(1)))

    member = RandomForestClassifier(n_estimators=30, random_state=1).fit(_fit_frame(train, assignments), _fit_target(train, le))
    bag_rows = rows + [{"model_name": "Bag", "status": "ok", "_fitted_estimator": rf,
                        "_bag_members": bag_members([member]), "model_spec": {"kind": "bag", "base": rf_spec}}]
    bag = {"kind": "bag", "base": rf_spec}
    out = Predictor(build_bundle(_session(train, bag, assignments, cols), bag_rows, train)).predict(raw, explain=False)
    assert len(out) == len(raw) and all(o["prediction"] in ("yes", "no") for o in out)


def _fit_frame(train, assignments):
    tr, _ = replay.apply_deep_clean(train, None, assignments, "target")
    x = replay.safe_numeric_features(tr.drop(columns=["target"]))
    return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=x.columns)


def _fit_target(train, le):
    return le.transform(train["target"])


def test_regression_with_tree_explanation():
    train, test = _data()
    train["target"] = train["age"] * 2.0 + (train["sex"] == "M") * 10
    test["target"] = 0.0
    assignments = {c: replay.default_assignment(c, train[c]) for c in ["age", "chol", "sex", "pain"]}
    x_tr, x_te, x_tr_imp, x_te_imp, y = _processed(train, test, assignments)
    rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(x_tr, y)
    spec = {"kind": "single", "family": "RandomForest", "impute_median": False}
    rows = [{"status": "ok", "_fitted_estimator": rf, "model_spec": spec}]
    pred = Predictor(build_bundle(_session(train, spec, assignments, list(x_tr.columns), "regression"), rows, train))
    out = pred.predict(test.drop(columns=["target"]).head(5).to_dict("records"))
    np.testing.assert_allclose([o["prediction"] for o in out], rf.predict(x_te.head(5)), rtol=1e-6)
    assert out[0]["contributions"]["method"].startswith("SHAP")
    schema = pred.schema(test.head(3))
    assert {c["name"] for c in schema["columns"]} == {"age", "chol", "sex", "pain"}
    assert schema["examples"] and schema["classes"] is None


def test_missing_model_raises():
    train, _ = _data()
    s = _session(train, {"kind": "single", "family": "XGBoost"}, {}, [])
    with pytest.raises(PredictUnavailable):
        build_bundle(s, [], train)
