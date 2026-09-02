"""
LLM Elimination Accuracy Analysis.

Validates the Progressive Model Elimination step: after Round-1 subsample
screening, the LLM eliminates one model family. This script measures whether
that elimination was a good decision.

Two correctness metrics:
  strict  — eliminated model had the single lowest subsample score (rank 1)
  lenient — eliminated model was within 0.01 of the worst subsample score
             (practically equivalent; avoids penalising near-ties)

Note: runs are deterministic (same data + seed 42 + same LLM plan), so
running --runs > 1 produces identical results per dataset. Use --runs 1.

Usage:
    # Collect fresh audit data (runs full pipeline once per dataset)
    python experiments/elimination_accuracy.py --collect --datasets titanic,heart,insurance,churn,bank,house_prices,diamonds,life_expectancy,wine

    # Analyse previously saved audits
    python experiments/elimination_accuracy.py --from-json experiments/results/elimination_audits.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXPERIMENTS))

import pandas as pd
import numpy as np


def _run_and_collect(ds_name: str, df, task_type: str) -> dict | None:
    """Run pipeline once and return the elimination_audit dict."""
    from automl_system.graph.orchestrator import build_graph
    from automl_system.state.session import SessionState
    from automl_system.llm.client import LLMClient
    from _datasets import KAGGLE_DATASETS

    LLMClient.call_count = 0
    LLMClient.cache_hits = 0

    user_goal = KAGGLE_DATASETS.get(ds_name, {}).get("user_goal", f"Predict target for {ds_name}")
    session = SessionState()
    session.task_config = {
        "goal": user_goal,
        "target_column": "target",
        "task_type": task_type,
        "constraints": "",
        "metric": "f1_weighted" if task_type == "classification" else "r2",
    }

    try:
        graph = build_graph()
        final_state = graph.invoke({
            "session": session.to_dict(),
            "df": df.copy(),
            "messages": [],
            "checkpoints": {},
        })
        # elimination_audit is stored at both top-level state and inside session
        exec_res = (
            final_state.get("execution_results")
            or (final_state.get("session") or {}).get("execution_results")
            or {}
        )
        audit = exec_res.get("elimination_audit")
        return audit
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None


def collect_audits(datasets: list[str]) -> list[dict]:
    from _datasets import load_raw

    audits = []
    for ds_name in datasets:
        loaded = load_raw(ds_name)
        if loaded is None:
            print(f"SKIP {ds_name}: CSV not found in experiments/datasets/")
            continue
        df, task_type, _ = loaded

        print(f"\nDataset: {ds_name} ({task_type}, {len(df):,} rows) ...")
        audit = _run_and_collect(ds_name, df, task_type)
        if audit:
            audit["dataset"] = ds_name
            audits.append(audit)
            elim = audit.get("eliminated_family", "—")
            rank = audit.get("subsample_rank_of_eliminated")
            total = audit.get("total_candidates")
            correct = audit.get("elimination_correct")
            print(f"  eliminated={elim}  rank={rank}/{total}  strict_correct={correct}")
        else:
            print("  No elimination event (fewer than 3 model candidates or error)")

    return audits


def _lenient_correct(row: pd.Series, tol: float = 0.01) -> bool:
    """True if eliminated model score is within `tol` of the actual worst score."""
    try:
        scores = eval(row["all_subsample_scores"]) if isinstance(row["all_subsample_scores"], str) else row["all_subsample_scores"]
        elim_score = row["eliminated_subsample_score"]
        worst_score = min(scores.values())
        return float(elim_score) <= float(worst_score) + tol
    except Exception:
        return bool(row["elimination_correct"])


def analyse_audits(audits: list[dict]) -> None:
    if not audits:
        print("No elimination audits to analyse.")
        return

    df = pd.DataFrame(audits)
    total = len(df)
    ran = df[df["eliminated_family"].notna()].copy()

    print("\n" + "=" * 65)
    print("PROGRESSIVE ELIMINATION ACCURACY ANALYSIS")
    print("=" * 65)
    print(f"Datasets analysed    : {total}")
    print(f"With elimination     : {len(ran)} ({100 * len(ran) / max(total, 1):.0f}%)")

    if len(ran) == 0:
        print("No elimination events found.")
        return

    ran["lenient_correct"] = ran.apply(_lenient_correct, axis=1)

    strict_correct = ran["elimination_correct"].sum()
    lenient_correct = ran["lenient_correct"].sum()
    mean_rank = ran["subsample_rank_of_eliminated"].mean()
    rank1_pct = 100 * (ran["subsample_rank_of_eliminated"] == 1).sum() / len(ran)

    print(f"\nStrict correct  (rank-1 only) : {int(strict_correct)}/{len(ran)} ({100 * strict_correct / len(ran):.1f}%)")
    print(f"Lenient correct (within 0.01) : {int(lenient_correct)}/{len(ran)} ({100 * lenient_correct / len(ran):.1f}%)")
    print(f"Mean rank of eliminated       : {mean_rank:.2f}  (1 = worst = ideal)")
    print(f"Rank-1 eliminations           : {rank1_pct:.1f}%")

    print("\nPer-dataset breakdown:")
    for _, row in ran.sort_values("dataset").iterrows():
        flag = "✓" if row["elimination_correct"] else ("~" if row["lenient_correct"] else "✗")
        print(
            f"  [{flag}] {row['dataset']:<20} eliminated={row['eliminated_family']:<22} "
            f"rank={int(row['subsample_rank_of_eliminated'])}/{int(row['total_candidates'])}  "
            f"score={row['eliminated_subsample_score']:.4f}"
        )

    print("\nEliminated families:")
    print(ran["eliminated_family"].value_counts().to_string())

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "elimination_results.csv"
    ran.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Run pipeline to collect fresh audit data (one run per dataset)")
    parser.add_argument("--datasets", default="titanic,heart,insurance,churn,bank,house_prices,diamonds,life_expectancy,wine",
                        help="Comma-separated dataset names")
    parser.add_argument("--from-json", default="", help="Load audits from a previously saved JSON file")
    args = parser.parse_args()

    audits = []

    if args.from_json and Path(args.from_json).exists():
        with open(args.from_json) as f:
            audits = json.load(f)
        print(f"Loaded {len(audits)} audits from {args.from_json}")

    if args.collect:
        ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        fresh = collect_audits(ds_list)
        audits.extend(fresh)

        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        save_path = out_dir / "elimination_audits.json"
        with open(save_path, "w") as f:
            json.dump(audits, f, indent=2, default=str)
        print(f"\nRaw audits saved to {save_path}")

    analyse_audits(audits)
