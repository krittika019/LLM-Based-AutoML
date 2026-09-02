"""
Extract a formatted case study from one AutoML pipeline run.
Used to generate the qualitative example section of the research paper.

Usage:
    python experiments/extract_case_study.py --dataset titanic
    python experiments/extract_case_study.py --dataset insurance

Output:
    experiments/results/case_study_<dataset>.txt   (paste into paper)
    experiments/results/case_study_<dataset>.json  (full raw log)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from _datasets import KAGGLE_DATASETS, load_raw

_RESULTS_DIR  = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

_SPECS = KAGGLE_DATASETS


def run_and_extract(ds_name: str) -> None:
    spec = _SPECS.get(ds_name)
    if spec is None:
        print(f"Unknown dataset: {ds_name}. Choose from: {list(_SPECS)}")
        return

    loaded = load_raw(ds_name)
    if loaded is None:
        print(f"CSV not found for '{ds_name}' in experiments/datasets/")
        return

    from automl_system.graph.orchestrator import build_graph
    from automl_system.state.session import SessionState
    from automl_system.llm.client import LLMClient
    # Reused rather than re-implemented so the case study audits the report against
    # the same rules the generator was written under; two copies would drift.
    from automl_system.agents.report_generator import (
        _BENCHMARK_ROW_MARKERS,
        audit_report_text,
    )

    df, task_type, metric = loaded   # df has feature columns + 'target'
    target = spec["target"].strip()  # original name, for display only

    session = SessionState()
    user_goal = spec.get("user_goal", f"Predict {target} for {ds_name} dataset")
    session.task_config = {
        "goal": user_goal,
        "target_column": "target",
        "task_type": task_type,
        "constraints": "",
        "metric": metric,
    }

    print(f"Running pipeline on '{ds_name}' ({len(df)} rows × {df.shape[1]-1} features)...")
    t0 = time.perf_counter()
    graph  = build_graph()
    result = graph.invoke({"session": session.to_dict(), "df": df, "messages": [], "checkpoints": {}})
    elapsed = time.perf_counter() - t0
    print(f"Pipeline completed in {elapsed:.1f}s")

    final_session = SessionState.from_dict(result.get("session") or {})
    exec_res      = result.get("execution_results") or {}

    # Pull decision log — merges MongoDB episodic log with the always-populated
    # session.local_decisions_log fallback (see decision_log.py). Using only the
    # Mongo path silently returns nothing when MongoDB isn't reachable.
    from automl_system.utils.decision_log import merge_logs_for_report
    decisions = merge_logs_for_report(final_session)

    # Save raw JSON
    raw = {
        "dataset": ds_name,
        "elapsed_s": round(elapsed, 1),
        "best_model": exec_res.get("best_model_name"),
        "best_score": exec_res.get("best_test_score"),
        "decisions": decisions,
        "eda_narrative": final_session.eda_narrative,
        "report": final_session.final_report_markdown,
        "pipeline_script": final_session.final_pipeline_script,
        # Inputs the Phase 7 report is built from. Saved so a claim in the report
        # can be checked against what the pipeline actually did without re-running.
        "deep_cleaning_assignments": final_session.deep_cleaning_assignments,
        "feature_selection_log": final_session.feature_selection_log,
        "feature_engineering_log": final_session.feature_engineering_log,
        "feature_importances": final_session.feature_importances,
        "comparison_table": exec_res.get("comparison_table"),
        "task_config": final_session.task_config,
        "initial_dataset_profile": final_session.initial_dataset_profile,
        "eda_results": final_session.eda_results,
    }
    json_out = _RESULTS_DIR / f"case_study_{ds_name}.json"
    json_out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    print(f"Raw log saved: {json_out}")

    # Format paper-ready case study
    lines = []
    lines.append("=" * 72)
    lines.append(f"CASE STUDY: {ds_name.upper()} ({spec['task_type']})")
    lines.append("=" * 72)
    lines.append(f"Dataset    : {ds_name}  ({len(df)} rows × {df.shape[1]-1} features)")
    lines.append(f"Task       : {spec['task_type'].capitalize()}  |  Target: {target}")
    lines.append(f"Runtime    : {elapsed:.1f}s  |  LLM calls: {LLMClient.call_count}")
    lines.append(f"Best model : {exec_res.get('best_model_name', 'N/A')}")
    lines.append(f"Test score : {exec_res.get('best_test_score', 'N/A')}")
    lines.append("")
    lines.append("USER PROMPT")
    lines.append("-" * 72)
    lines.append(f'  "{user_goal}"')
    lines.append("")

    # Phase-by-phase decision summary
    phase_map: dict[str, list] = {}
    for d in decisions:
        ph = str(d.get("phase") or d.get("step") or "other")
        phase_map.setdefault(ph, []).append(d)

    phase_order = [
        "intent_parsing", "task_classification",
        "eda", "preprocessing_plan", "basic_cleaning", "deep_cleaning",
        "feature_selection", "feature_engineering",
        "pipeline_planning", "execution", "failure_diagnosis",
        "report_generation",
    ]
    all_phases = phase_order + [p for p in phase_map if p not in phase_order]

    lines.append("PHASE-BY-PHASE DECISION TRACE")
    lines.append("-" * 72)
    for phase in all_phases:
        entries = phase_map.get(phase, [])
        if not entries:
            continue
        lines.append(f"\n[{phase.upper().replace('_',' ')}]")
        for e in entries[:4]:  # max 4 decisions per phase
            action   = e.get("action") or e.get("step") or ""
            note     = e.get("note") or e.get("rationale") or e.get("text") or ""
            col      = e.get("column") or ""
            col_part = f" | column={col}" if col else ""
            note_part = f"\n    → {note[:200]}" if note else ""
            lines.append(f"  • {action}{col_part}{note_part}")
        if len(entries) > 4:
            lines.append(f"  ... (+{len(entries)-4} more decisions)")

    # Dropped features summary. Uses .get() throughout: a log entry written by an
    # earlier pipeline version can be missing either key, and a KeyError here
    # would lose the whole case study after the run had already been paid for.
    fs_log = final_session.feature_selection_log or []
    if fs_log:
        lines.append(f"\n[FEATURE SELECTION SUMMARY]")
        lines.append(f"  Dropped {len(fs_log)} feature(s):")
        for e in fs_log[:6]:
            col = e.get("column", "?")
            why = str(e.get("rationale") or "no rationale recorded")[:120]
            lines.append(f"  • {col}: {why}")
        if len(fs_log) > 6:
            lines.append(f"  ... (+{len(fs_log)-6} more; full list in the JSON)")

    # Model comparison. AutoGluon/FLAML rows are appended by execution.py as
    # external reference points, not as candidates this pipeline chose — they are
    # marked so a table pasted into the paper cannot imply we selected one.
    comp = exec_res.get("comparison_table") or []
    ranked: list[dict] = []
    if comp:
        own_rows = [
            r for r in comp
            if not any(m in str(r.get("model_name", "")).lower() for m in _BENCHMARK_ROW_MARKERS)
        ]
        ranked = sorted(
            (r for r in own_rows if isinstance(r.get("test_score"), (int, float))),
            key=lambda r: float(r["test_score"]),
            reverse=True,
        )
        if ranked:
            lines.append(f"\n[TOP 3 MODELS]")
            for rank, row in enumerate(ranked[:3], start=1):
                cv = row.get("cv_score")
                cv_text = f"{float(cv):.4f}" if isinstance(cv, (int, float)) else "N/A"
                lines.append(
                    f"  #{rank} {row.get('model_name','?'):<26} "
                    f"test={float(row['test_score']):.4f}  cv={cv_text}"
                )
            if len(ranked) > 1:
                gap = float(ranked[0]["test_score"]) - float(ranked[1]["test_score"])
                lines.append(f"  Gap #1 to #2: {gap:.4f}")

        lines.append(f"\n[MODEL COMPARISON TABLE]")
        lines.append(f"  {'Model':<28} {'CV Score':>10} {'Test Score':>12} {'Eliminated':>12}  Source")
        lines.append(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*12}  {'-'*9}")
        for row in comp:
            raw_name = str(row.get("model_name", ""))
            name  = raw_name[:27]
            cv    = f"{float(row['cv_score']):.4f}"  if isinstance(row.get("cv_score"), (int, float))  else "N/A"
            test  = f"{float(row['test_score']):.4f}" if isinstance(row.get("test_score"), (int, float)) else "N/A"
            elim  = "YES" if str(row.get("status") or "").startswith("elim") else "no"
            src   = "benchmark" if any(m in raw_name.lower() for m in _BENCHMARK_ROW_MARKERS) else "ours"
            lines.append(f"  {name:<28} {cv:>10} {test:>12} {elim:>12}  {src}")

    # Feature importances — the basis for the report's takeaway section, so a
    # reviewer can check the named features against the measured ranking.
    fi = final_session.feature_importances or {}
    top_feats = fi.get("top_features") or []
    if top_feats:
        lines.append(f"\n[FEATURE IMPORTANCES] (method: {fi.get('method', 'unknown')})")
        for i, item in enumerate(top_feats[:10], start=1):
            try:
                fname, pct = item
                lines.append(f"  {i:>2}. {str(fname):<28} {float(pct):>6.2f}%")
            except (TypeError, ValueError):
                lines.append(f"  {i:>2}. {item}")

    # EDA narrative, one finding per line. Stored as a stringified dict, so it is
    # parsed back rather than dumped as a repr.
    narrative = (final_session.eda_narrative or "").strip()
    if narrative:
        lines.append(f"\n[EDA NARRATIVE]")
        rendered = None
        if narrative.startswith("{"):
            try:
                parsed_narr = ast.literal_eval(narrative)
                if isinstance(parsed_narr, dict):
                    rendered = [
                        f"  {k.replace('_',' ')}: {str(v).strip()}"
                        for k, v in parsed_narr.items() if str(v).strip()
                    ]
            except (ValueError, SyntaxError):
                rendered = None
        lines.append("\n".join(rendered) if rendered else f"  {narrative}")

    # Full final report — no truncation, this is the primary artifact for the case study.
    if final_session.final_report_markdown:
        lines.append(f"\n[FINAL REPORT]")
        lines.append(f"  {final_session.final_report_markdown.replace(chr(10), chr(10) + '  ')}")

        # Audit the report against the constraints it was generated under. Printed
        # into the case study itself so a claim in the paper about report fidelity
        # is backed by a check the reader can see, not just asserted.
        expected_models = [
            str(r.get("model_name")) for r in ranked[:3]
        ] if comp and ranked else []
        audit = audit_report_text(
            final_session.final_report_markdown, expected_models=expected_models
        )
        failed = {check: items for check, items in audit.items() if items}
        lines.append(f"\n[REPORT AUDIT]")
        if failed:
            for check, items in failed.items():
                lines.append(f"  FAIL  {check}: {items}")
        else:
            lines.append("  PASS  all checks (sections present, no hedges, no unsupported")
            lines.append("        caveats, no directional claims, top-3 models all named)")
        raw["report_audit"] = audit
        json_out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

    lines.append("\n" + "=" * 72)

    txt_out = _RESULTS_DIR / f"case_study_{ds_name}.txt"
    txt_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Case study saved: {txt_out}")
    print("\n" + "\n".join(lines[:30]) + "\n...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(_SPECS), help="Dataset name")
    args = parser.parse_args()
    run_and_extract(args.dataset)
