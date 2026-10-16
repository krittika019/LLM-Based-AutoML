"""Analytics curves: ROC + threshold sweep counts are consistent with the data."""

import numpy as np
import pandas as pd

from automl_system.pipeline.execution import _binary_curves, _compute_full_metrics


def test_sweep_counts_and_roc_shape():
    rng = np.random.RandomState(0)
    y = pd.Series(rng.choice(["no", "yes"], size=400))
    p = np.clip((y == "yes").astype(float) * 0.4 + rng.rand(400) * 0.6, 0, 1)
    c = _binary_curves(y, p)
    assert c["positive_label"] == "yes" and c["negative_label"] == "no"
    assert len(c["roc"]) <= 80 and c["roc"][0] == [0.0, 0.0] and c["roc"][-1] == [1.0, 1.0]
    n_pos = int((y == "yes").sum())
    for s in c["sweep"]:
        assert s["tp"] + s["fn"] == n_pos and s["tp"] + s["fp"] + s["fn"] + s["tn"] == 400
    at_half = next(s for s in c["sweep"] if s["t"] == 0.5)
    assert at_half["tp"] == int(((p >= 0.5) & (y == "yes")).sum())


def test_full_metrics_include_curves_only_for_binary_proba():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.45, 0.55], [0.9, 0.1]])
    m = _compute_full_metrics("classification", y, proba.argmax(1), proba)
    assert "curves" in m and m["curves"]["positive_label"] == "1"
    assert "curves" not in _compute_full_metrics("classification", y, proba.argmax(1), None)
