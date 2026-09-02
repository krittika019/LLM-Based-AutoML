"""
Regenerate the Phase 7 report from a saved case-study run, without re-running
the pipeline.

extract_case_study.py already saves every input the report is built from, so the
narrative can be rebuilt for the cost of one LLM call instead of a full run. Use
this when iterating on the report prompt or the faithfulness rules; use
extract_case_study.py when the pipeline itself changed.

The regenerated report is written back into the same JSON and the .txt case
study, so generate_report_figure.py picks it up with no extra step.

Usage:
    python experiments/regenerate_report.py --dataset heart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_RESULTS_DIR = Path(__file__).parent / "results"


def regenerate(ds_name: str) -> None:
    json_path = _RESULTS_DIR / f"case_study_{ds_name}.json"
    if not json_path.exists():
        print(f"No saved run at {json_path}. Run extract_case_study.py first.")
        return

    raw = json.loads(json_path.read_text(encoding="utf-8"))

    from automl_system.agents.report_generator import (
        _eda_execution_conflicts,
        _eda_top_features,
        _majority_baseline,
        _mi_dropped_features,
        _parse_eda_narrative,
        _ranked_models,
        audit_report_text,
        generate_report,
    )
    from automl_system.state.session import SessionState

    session = SessionState()
    session.task_config = raw.get("task_config") or {}
    session.initial_dataset_profile = raw.get("initial_dataset_profile") or {}
    session.eda_results = raw.get("eda_results") or {}
    session.eda_narrative = raw.get("eda_narrative") or ""
    session.deep_cleaning_assignments = raw.get("deep_cleaning_assignments") or {}
    session.feature_selection_log = raw.get("feature_selection_log") or []
    session.feature_engineering_log = raw.get("feature_engineering_log") or []
    session.feature_importances = raw.get("feature_importances") or {}
    session.execution_results = {
        "best_model_name": raw.get("best_model"),
        "best_test_score": raw.get("best_score"),
    }
    # Replayed through the local mirror, not Mongo: the saved decisions are already
    # the merged result, and going back to episodic storage would either duplicate
    # them or return nothing when Mongo is unreachable.
    session.local_decisions_log = raw.get("decisions") or []

    all_results = [{"comparison_table": raw.get("comparison_table") or []}]

    print(f"Regenerating report for '{ds_name}' from {json_path.name}...")
    report = generate_report(session, all_results)

    best = {"model": raw.get("best_model"), "test_score": raw.get("best_score")}
    eda_sections = _parse_eda_narrative(session.eda_narrative)
    eda_top = _eda_top_features(session, eda_sections)
    conflicts = _eda_execution_conflicts(eda_top, _mi_dropped_features(session.local_decisions_log))
    top_feats = (session.feature_importances or {}).get("top_features") or []

    audit = audit_report_text(
        report,
        expected_models=[n for n, _t, _c in _ranked_models(all_results)[:3]],
        conflicts=conflicts,
        expected_features=[str(f[0]) for f in top_feats[:3] if f],
        baseline=_majority_baseline(all_results, best),
    )

    raw["report"] = report
    raw["report_audit"] = audit
    json_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

    # Swap the report into the .txt case study in place, so the two artifacts do
    # not disagree about what the system produced.
    txt_path = _RESULTS_DIR / f"case_study_{ds_name}.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
        marker = "[FINAL REPORT]"
        if marker in text:
            head = text.split(marker)[0]
            indented = report.replace("\n", "\n  ")
            audit_lines = [f"  FAIL  {k}: {v}" for k, v in audit.items() if v] or [
                "  PASS  all checks clean"
            ]
            txt_path.write_text(
                f"{head}{marker}\n  {indented}\n\n[REPORT AUDIT]\n"
                + "\n".join(audit_lines)
                + "\n\n"
                + "=" * 72
                + "\n",
                encoding="utf-8",
            )

    print("\n" + report)
    failed = {k: v for k, v in audit.items() if v}
    print("\n" + "-" * 72)
    print("AUDIT:", json.dumps(failed, indent=2) if failed else "PASS — all checks clean")
    print(f"words: {len(report.split())}")
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name of a saved case study")
    regenerate(parser.parse_args().dataset)
