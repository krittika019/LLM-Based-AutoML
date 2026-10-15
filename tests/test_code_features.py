"""Option B: LLM-written feature functions — static checks, sandbox execution, CV gate, replay."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.pipeline import feature_engineering as fe
from automl_system.pipeline import replay
from automl_system.pipeline.code_features import check_feature_code
from automl_system.sandbox import docker_available
from automl_system.state.session import SessionState

GOOD = (
    "def make_feature(train, test):\n"
    "    m = train.groupby('shop')['price'].mean()\n"
    "    return train['price'] - train['shop'].map(m), test['price'] - test['shop'].map(m)\n"
)
CRASH = "def make_feature(train, test):\n    return train['no_such_column'], test['no_such_column']\n"
UNSAFE = "def make_feature(train, test):\n    open('x.txt', 'w')\n    return train['price'], test['price']\n"


def test_static_check_rules():
    assert check_feature_code(GOOD) is None
    assert "open" in check_feature_code(UNSAFE)
    assert "import" in check_feature_code("import os\ndef make_feature(train, test):\n    return 1, 1")
    assert "dunder" in check_feature_code("def make_feature(train, test):\n    return train.__class__, 1")
    assert "make_feature" in check_feature_code("def other(train, test):\n    return 1, 1")
    assert "SyntaxError" in check_feature_code("def make_feature(:\n")


def test_run_code_feature_normalizes_output():
    train = pd.DataFrame({"shop": ["a", "a", "b"], "price": [1.0, 3.0, 5.0]})
    test = pd.DataFrame({"shop": ["a", "c"], "price": [2.0, 9.0]})
    tr, te = replay.run_code_feature(train, test, GOOD)
    assert list(tr.columns) == list(te.columns) and tr.shape[1] == 1
    col = tr.columns[0]
    assert tr[col].tolist() == [-1.0, 1.0, 0.0]
    assert te[col].iloc[0] == 0.0 and np.isnan(te[col].iloc[1])  # unseen shop → NaN


def _data(n: int = 1600):
    rng = np.random.default_rng(9)
    shops = [f"shop_{i}" for i in range(25)]
    shop = rng.choice(shops, n)
    base = dict(zip(shops, rng.uniform(10, 500, len(shops))))
    price = np.array([base[s] for s in shop]) * rng.uniform(0.6, 1.4, n)
    y = np.where(price > np.array([base[s] for s in shop]), "expensive", "cheap")
    raw = pd.DataFrame({"shop": shop, "price": price, "qty": rng.integers(1, 9, n), "y": y})
    train, test = raw.iloc[:1300].reset_index(drop=True), raw.iloc[1300:].reset_index(drop=True)
    shop_codes = {s: i for i, s in enumerate(sorted(shops))}

    def clean(f):  # stand-in for deep cleaning: scaled numerics, ordinal-coded shop
        return pd.DataFrame({
            "y": f["y"],
            "price": (f["price"] - train["price"].mean()) / train["price"].std(),
            "qty": f["qty"].astype(float),
            "shop": f["shop"].map(shop_codes).astype(float),
        })
    return train, test, clean(train), clean(test)


@pytest.mark.skipif(not docker_available(), reason="Docker sandbox required for code features")
def test_code_features_end_to_end(monkeypatch):
    monkeypatch.setattr(config, "FE_SELECTION", "cv")
    monkeypatch.setattr(config, "FE_ROUNDS", 2)
    monkeypatch.setattr(config, "FE_CODE_FEATURES", "auto")
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "auto")
    proposals = {"features": [
        {"name": "price_vs_shop_mean", "type": "code", "code": GOOD, "rationale": "price relative to shop"},
        {"name": "broken", "type": "code", "code": CRASH, "rationale": "typo"},
        {"name": "sneaky", "type": "code", "code": UNSAFE, "rationale": "writes a file"},
    ]}
    prompts = []
    monkeypatch.setattr(LLMClient, "call", lambda self, prompt, **_k: prompts.append(prompt) or json.dumps(proposals))
    raw_train, raw_test, train, test = _data()
    session = SessionState(task_config={"target_column": "y", "task_type": "classification"})

    out, out_test = fe.engineer_features(train, session, df_test=test, raw_train=raw_train, raw_test=raw_test)

    by_name = {e["feature"]: e for e in session.feature_engineering_log if e.get("round") == 1}
    assert by_name["price_vs_shop_mean"]["status"] == "added", by_name["price_vs_shop_mean"]
    assert by_name["broken"]["status"] == "error" and "no_such_column" in by_name["broken"]["detail"]
    assert by_name["sneaky"]["status"] == "rejected_unsafe_code"
    assert "CATEGORICAL_COLUMNS" in prompts[0]
    assert "ERROR broken" in prompts[1]  # the crash is fed back for repair

    # The recorded spec replays to the same values (trusted test code may run on the host here).
    spec = by_name["price_vs_shop_mean"]["spec"]
    assert spec["type"] == "code" and spec["source"] == "raw"
    rt, rte = replay.add_raw_features(
        raw_train.drop(columns=["y"]), raw_test.drop(columns=["y"]), [spec]
    )
    np.testing.assert_allclose(rt["price_vs_shop_mean"].to_numpy(), out["price_vs_shop_mean"].to_numpy())
    np.testing.assert_allclose(np.asarray(rte["price_vs_shop_mean"], dtype=float),
                               out_test["price_vs_shop_mean"].to_numpy())


def test_code_features_skipped_without_docker(monkeypatch):
    monkeypatch.setattr(config, "FE_SELECTION", "cv")
    monkeypatch.setattr(config, "FE_ROUNDS", 1)
    monkeypatch.setattr(config, "FE_CODE_FEATURES", "off")
    proposals = {"features": [{"name": "price_vs_shop_mean", "type": "code", "code": GOOD, "rationale": "r"}]}
    monkeypatch.setattr(LLMClient, "call", lambda *_a, **_k: json.dumps(proposals))
    raw_train, raw_test, train, test = _data(600)
    session = SessionState(task_config={"target_column": "y", "task_type": "classification"})
    out, _ = fe.engineer_features(train, session, df_test=test, raw_train=raw_train, raw_test=raw_test)
    assert "price_vs_shop_mean" not in out.columns
    assert session.feature_engineering_log[0]["status"] == "skipped_code_disabled"


def test_double_escaped_code_is_repaired():
    from automl_system.pipeline.code_features import is_safety_problem, normalize_feature_code
    one_line = GOOD.replace("\n", "\\n")
    assert check_feature_code(one_line) is not None          # broken as received
    assert check_feature_code(normalize_feature_code(one_line)) is None
    assert check_feature_code(normalize_feature_code("```python\n" + GOOD + "```")) is None
    assert not is_safety_problem(check_feature_code("def make_feature(:\n"))
    assert is_safety_problem(check_feature_code(UNSAFE))
