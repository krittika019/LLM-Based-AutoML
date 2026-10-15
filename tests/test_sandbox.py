from __future__ import annotations

import pandas as pd
import pytest

from automl_system.sandbox import Sandbox, docker_available

BACKENDS = ["local", pytest.param("docker", marks=pytest.mark.skipif(
    not docker_available(), reason="Docker not running or sandbox image not built"))]


@pytest.fixture(params=BACKENDS)
def sb(request):
    with Sandbox(request.param) as box:
        yield box


def test_runs_code(sb):
    res = sb.run("print('hello from sandbox')")
    assert res.ok, res.stderr
    assert "hello from sandbox" in res.stdout


def test_reads_input_data(sb):
    sb.put_dataframe("train.csv", pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}))
    res = sb.run("import pandas as pd; print(pd.read_csv('train.csv').shape)")
    assert res.ok, res.stderr
    assert "(3, 2)" in res.stdout


def test_trains_model(sb):
    code = (
        "from sklearn.datasets import make_classification\n"
        "from lightgbm import LGBMClassifier\n"
        "X, y = make_classification(300, random_state=0)\n"
        "print('acc', LGBMClassifier(verbose=-1).fit(X, y).score(X, y))\n"
    )
    res = sb.run(code, timeout=120)
    assert res.ok, res.stderr
    assert "acc" in res.stdout


def test_error_is_captured(sb):
    res = sb.run("x = 1 / 0")
    assert not res.ok
    assert "ZeroDivisionError" in res.stderr


def test_timeout_kills_script(sb):
    res = sb.run("import time; time.sleep(60)", timeout=3)
    assert res.timed_out
    assert res.duration_s < 30


def test_files_persist_between_runs(sb):
    assert sb.run("open('state.txt', 'w').write('kept')").ok
    res = sb.run("print(open('state.txt').read())")
    assert "kept" in res.stdout


def test_secrets_not_visible(sb, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "should-not-leak")
    res = sb.run("import os; print(os.environ.get('GROQ_API_KEY'))")
    assert "should-not-leak" not in res.stdout


@pytest.mark.skipif(not docker_available(), reason="Docker not running")
def test_docker_has_no_network():
    with Sandbox("docker") as box:
        res = box.run("import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)")
        assert not res.ok
