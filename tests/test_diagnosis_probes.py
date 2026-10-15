"""Phase 6 diagnosis experiments find planted causes; measured evidence beats a wrong LLM guess."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from automl_system.agents import failure_diagnosis as fd
from automl_system.llm.client import LLMClient
from automl_system.pipeline.diagnosis_probes import format_evidence, run_probes
from automl_system.state.session import SessionState


def _session(keep, task_type="classification"):
    return SessionState(task_config={"target_column": "y", "task_type": task_type},
                        preprocessing_meta={"feature_selection_columns": keep})


def test_probe_finds_column_dropped_by_feature_selection():
    rng = np.random.default_rng(0)
    n = 1500
    raw = pd.DataFrame({"key": rng.normal(size=n), "noise1": rng.normal(size=n),
                        "noise2": rng.normal(size=n), "city": rng.choice(["a", "b"], n)})
    raw["y"] = np.where(raw["key"] + rng.normal(0, 0.3, n) > 0, "yes", "no")
    keep = ["noise1", "noise2", "city", "y"]          # feature selection wrongly dropped "key"
    processed = pd.DataFrame({"noise1": raw["noise1"], "noise2": raw["noise2"],
                              "city": (raw["city"] == "a").astype(float), "y": raw["y"]})
    report = run_probes(processed, raw, _session(keep), {"metric": "roc_auc", "task_type": "classification"})
    by = {p["probe"]: p for p in report["probes"]}
    assert by["feature_selection"]["significant"] and by["feature_selection"]["gain"] > 0.1
    assert report["best"]["responsible_phase"] == "feature_selection"
    assert "key" in report["best"]["fix"]
    assert "SIGNIFICANT" in format_evidence(report)


def test_probe_finds_class_imbalance_on_f1():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=(n, 4))
    # ~6% minority; "yes" sorts second, so it is the positive class for binary F1 (as in Phase 5)
    y = np.where(x[:, 0] + 0.8 * rng.normal(size=n) > 1.9, "yes", "no")
    processed = pd.DataFrame(x, columns=["a", "b", "c", "d"]).assign(y=y)
    report = run_probes(processed, None, _session([]), {"metric": "f1", "task_type": "classification"})
    by = {p["probe"]: p for p in report["probes"]}
    assert "class_weight" in by and by["class_weight"]["significant"], by.get("class_weight")
    assert by["class_weight"]["responsible_phase"] == "phase4_planning"


def test_evidence_overrides_contradicting_llm(monkeypatch):
    rng = np.random.default_rng(0)
    n = 1500
    raw = pd.DataFrame({"key": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    raw["y"] = np.where(raw["key"] > 0, "yes", "no")
    processed = raw[["noise1", "noise2", "y"]].copy()
    session = _session(["noise1", "noise2", "y"])
    results = {"metric": "roc_auc", "task_type": "classification", "comparison_table": []}
    report = run_probes(processed, raw, session, results)

    wrong = {"failure_type": "model_issue", "responsible_phase": "phase4_planning",
             "proposed_fix": "Try a bigger model.", "should_exit": False}
    prompts = []
    monkeypatch.setattr(LLMClient, "call", lambda self, prompt, **_k: prompts.append(prompt) or json.dumps(wrong))
    diagnosis = fd.diagnose_failure(session, results, report)

    assert "EXPERIMENT_EVIDENCE" in prompts[0]
    assert diagnosis["overridden_by_evidence"]
    assert (diagnosis["failure_type"], diagnosis["responsible_phase"]) == ("feature_issue", "feature_selection")
    assert "key" in diagnosis["proposed_fix"]
    assert diagnosis["llm_diagnosis"]["failure_type"] == "model_issue"


def test_no_significant_evidence_keeps_llm_diagnosis():
    llm = {"failure_type": "model_issue", "responsible_phase": "phase4_planning", "proposed_fix": "x"}
    report = {"probes": [{"probe": "preprocessing", "gain": -0.01, "se": 0.01, "significant": False}],
              "best": None, "baseline": 0.8, "metric": "roc_auc"}
    out = fd._apply_evidence(llm, report)
    assert out["failure_type"] == "model_issue" and not out.get("overridden_by_evidence")
    assert out["evidence"]["probes"][0]["probe"] == "preprocessing"
