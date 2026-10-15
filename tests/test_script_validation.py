from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_system import config
from automl_system.pipeline import script_validation as sv
from automl_system.pipeline.code_generator import fallback_pipeline_script
from automl_system.sandbox import docker_available
from automl_system.state.session import SessionState

HEADER = (
    "import pandas as pd\n"
    "TRAIN_PATH = 'train.csv'\n"
    "TEST_PATH = 'test.csv'\n"
    "TARGET_COLUMN = 'y'\n"
    "PREDICTIONS_PATH = 'predictions.csv'\n"
    "test = pd.read_csv(TEST_PATH)\n"
)
# Copies the true labels from test.csv → always a perfect score.
PERFECT = HEADER + "pd.DataFrame({'prediction': test['y']}).to_csv(PREDICTIONS_PATH, index=False)\n"
CONSTANT = HEADER + "pd.DataFrame({'prediction': ['no'] * len(test)}).to_csv(PREDICTIONS_PATH, index=False)\n"
CRASH = HEADER + "raise ValueError('boom')\n" + PERFECT.split("\n")[-2] + "\n"


def _split() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "cat": rng.choice(["a", "b"], 200), "y": np.where(x > 0, "yes", "no")})
    return df.iloc[:150].reset_index(drop=True), df.iloc[150:].reset_index(drop=True)


def _session(best_score: float = 1.0) -> SessionState:
    return SessionState(
        task_config={"target_column": "y", "task_type": "classification"},
        execution_results={
            "metric": "accuracy", "metric_display": "Accuracy", "task_type": "classification",
            "best_test_score": best_score, "best_model_name": "RandomForest",
        },
    )


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "local")
    monkeypatch.setattr(config, "SCRIPT_VALIDATION_MAX_REPAIRS", 1)
    # No LLM in tests: repairs return nothing unless a test overrides this.
    monkeypatch.setattr(sv, "repair_pipeline_script", lambda *_a, **_k: "")


def test_reproduced_score_is_verified():
    train, test = _split()
    _, rep = sv.validate_and_repair_script(_session(1.0), PERFECT, train, test)
    assert rep["status"] == sv.STATUS_VERIFIED, rep
    assert rep["reproduced_score"] == pytest.approx(1.0)
    assert rep["first_attempt_ok"]


def test_score_gap_is_reported_as_mismatch():
    train, test = _split()
    _, rep = sv.validate_and_repair_script(_session(1.0), CONSTANT, train, test)
    assert rep["status"] == sv.STATUS_MISMATCH, rep
    assert rep["gap"] > config.SCRIPT_VALIDATION_TOLERANCE


def test_crash_is_reported_as_failed():
    train, test = _split()
    code, rep = sv.validate_and_repair_script(_session(), CRASH, train, test)
    assert rep["status"] == sv.STATUS_FAILED
    assert rep["history"][0]["stage"] == "runtime"
    assert "boom" in rep["history"][0]["problem"]
    assert code == CRASH


def test_dropped_test_rows_are_rejected():
    train, test = _split()
    code = PERFECT.replace("test['y']", "test['y'].iloc[:10]")
    _, rep = sv.validate_and_repair_script(_session(), code, train, test)
    assert rep["status"] == sv.STATUS_FAILED
    assert "rows" in rep["history"][0]["problem"]


def test_encoded_labels_are_rejected():
    train, test = _split()
    code = PERFECT.replace("test['y']", "(test['y'] == 'yes').astype(int) + 5")
    _, rep = sv.validate_and_repair_script(_session(), code, train, test)
    assert rep["status"] == sv.STATUS_FAILED
    assert "original target labels" in rep["history"][0]["problem"]


def test_repair_loop_fixes_script(monkeypatch):
    train, test = _split()
    monkeypatch.setattr(sv, "repair_pipeline_script", lambda *_a, **_k: PERFECT)
    code, rep = sv.validate_and_repair_script(_session(1.0), CRASH, train, test)
    assert rep["status"] == sv.STATUS_VERIFIED
    assert rep["attempts"] == 2
    assert not rep["first_attempt_ok"]
    assert code == PERFECT


def test_static_issue_skips_sandbox_and_repairs(monkeypatch):
    train, test = _split()
    bad = PERFECT.replace(
        "import pandas as pd\n",
        "import pandas as pd\nfrom sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder(sparse=False)\n",
    )
    monkeypatch.setattr(sv, "repair_pipeline_script", lambda *_a, **_k: PERFECT)
    code, rep = sv.validate_and_repair_script(_session(1.0), bad, train, test)
    first = rep["history"][0]
    assert first["stage"] == "static" and first["exit_code"] is None
    assert "sparse" in first["problem"]
    assert rep["status"] == sv.STATUS_VERIFIED and code == PERFECT


def test_sandbox_off_skips(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "off")
    train, test = _split()
    _, rep = sv.validate_and_repair_script(_session(), PERFECT, train, test)
    assert rep["status"] == sv.STATUS_SKIPPED


@pytest.mark.skipif(not docker_available(), reason="Docker not running or sandbox image not built")
def test_fallback_script_runs_in_docker(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "docker")
    train, test = _split()
    session = _session(0.9)
    _, rep = sv.validate_and_repair_script(session, fallback_pipeline_script(session), train, test, "fallback")
    assert rep["backend"] == "docker"
    assert rep["source"] == "fallback"
    assert rep["status"] in (sv.STATUS_VERIFIED, sv.STATUS_MISMATCH), rep
    assert rep["reproduced_score"] is not None


def test_docker_hang_falls_back_to_local(monkeypatch):
    import subprocess

    from automl_system.sandbox import runner

    monkeypatch.setattr(config, "SANDBOX_BACKEND", "auto")
    monkeypatch.setattr(runner, "docker_available", lambda: True)

    def hang(self):
        if self.backend == "docker":
            raise subprocess.TimeoutExpired(["docker", "run"], config.SANDBOX_START_TIMEOUT_SEC)

    monkeypatch.setattr(runner.Sandbox, "start", hang)
    train, test = _split()
    _, rep = sv.validate_and_repair_script(_session(1.0), PERFECT, train, test)
    assert rep["status"] == sv.STATUS_VERIFIED, rep
    assert rep["backend"] == "local"
    assert "local process" in rep["message"]
