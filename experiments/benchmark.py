"""
Benchmark harness: compare the AutoML system against AutoML competitors and
single-model baselines on tabular datasets.

Datasets (place CSVs in experiments/datasets/ — see _datasets.py for filenames):
    Classification: titanic, heart, churn, bank, creditcard_fraud
    Regression:     house_prices, insurance, diamonds, life_expectancy, wine
    Note: diamonds is auto-generated from seaborn (no CSV needed).
    Note: creditcard_fraud requires creditcard.csv from Kaggle (skipped if absent).

Systems compared:
  AutoML competitors (main comparison):
    - AutoGluon  (TabularPredictor, time-limited)
    - FLAML      (time-limited)
    - AutoML (ours, full LangGraph pipeline)
  Single-model baselines (lower bound / reference):
    - RandomForest (default, 100 trees)
    - Linear (LogisticRegression / Ridge)

Metrics (forced identical across systems for comparability):
    Classification → F1 (weighted)
    Regression     → R²  (headline); RMSE + MAE also reported per dataset

Note on our system:
    Only the metric is pre-set for fair comparison. Target column, task type,
    drop decisions, and all preprocessing are determined by the LLM pipeline
    from the raw CSV + user instruction alone.

    Novel contributions tracked per run:
      smote_applied      — SMOTE was triggered (minority class < 25%)
      diversity_pruned   — At least one model pruned from ensemble for redundancy

Output: experiments/results/benchmark_results.csv  +  table to stdout.

Usage:
    python experiments/benchmark.py                          # full run
    python experiments/benchmark.py --datasets titanic,heart
    python experiments/benchmark.py --skip-automl            # skip our pipeline (fast)
    python experiments/benchmark.py --only-ours              # skip competitors, run only our pipeline
    python experiments/benchmark.py --automl-time 90         # AutoGluon/FLAML budget (s)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["AUTOGLUON_TEXT_TRAIN_WITHOUT_GPU"] = "1"
os.environ["MPLBACKEND"] = "Agg"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from _datasets import KAGGLE_DATASETS, DATASETS_DIR, RESULTS_DIR, load_raw, metric_for, metric_label_for

_SEED = 42


def _adaptive_test_size(n_rows: int) -> float:
    """Consistent test fraction used by ALL systems in the benchmark.

    Small datasets get a larger test fraction to ensure enough evaluation
    samples; large datasets shrink the holdout to maximise training data.
    Applied identically to every system so comparisons remain fair.
    """
    if n_rows < 500:
        return 0.25
    if n_rows < 5_000:
        return 0.20
    return 0.15


# ── Feature preparation for single-model baselines ────────────────────────────

def _prep_features(X: pd.DataFrame) -> np.ndarray:
    """Encode categoricals (one-hot) and median-impute — sklearn baselines need numeric input."""
    X = X.copy()
    # One-hot encode object/category columns; numeric columns pass through.
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    return X.values.astype(float)


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "test_score": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


# ── Single-model baselines (RandomForest, Linear) ─────────────────────────────

def _baseline_score(X: pd.DataFrame, y: pd.Series, task_type: str, n_folds: int = 5) -> dict:
    X_np = _prep_features(X)
    y_np = y.values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_np)

    cv = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_SEED)
          if task_type == "classification"
          else KFold(n_splits=n_folds, shuffle=True, random_state=_SEED))
    scoring = "f1_weighted" if task_type == "classification" else "r2"

    test_size = _adaptive_test_size(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_np, test_size=test_size, random_state=_SEED)

    results = {}
    for name, clf in [
        ("RandomForest (default)",
         RandomForestClassifier(n_estimators=100, random_state=_SEED) if task_type == "classification"
         else RandomForestRegressor(n_estimators=100, random_state=_SEED)),
        ("Linear baseline",
         LogisticRegression(max_iter=1000, random_state=_SEED) if task_type == "classification"
         else Ridge()),
    ]:
        t0 = time.perf_counter()
        cv_mean = float(np.mean(cross_val_score(clf, X_s, y_np, cv=cv, scoring=scoring, n_jobs=-1)))
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        if task_type == "classification":
            row = {"test_score": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
                   "rmse": None, "mae": None}
        else:
            row = _reg_metrics(y_te, y_pred)
        row["cv_score"] = round(cv_mean, 4)
        row["time_s"] = round(time.perf_counter() - t0, 2)
        results[name] = row
    return results


# ── AutoML competitor: AutoGluon ──────────────────────────────────────────────

def _autogluon_score(X: pd.DataFrame, y: pd.Series, task_type: str, time_limit: int) -> dict:
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        return {"test_score": None, "rmse": None, "mae": None, "cv_score": None,
                "time_s": 0.0, "status": f"unavailable: {exc}"}

    test_size = _adaptive_test_size(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=_SEED)
    train_df = X_tr.copy(); train_df["target"] = y_tr.values
    test_df = X_te.copy()

    t0 = time.perf_counter()
    try:
        problem = "binary" if (task_type == "classification" and pd.Series(y).nunique() == 2) \
            else "multiclass" if task_type == "classification" else "regression"
        eval_metric = "f1_weighted" if task_type == "classification" else "r2"
        predictor = TabularPredictor(label="target", problem_type=problem, eval_metric=eval_metric,
                                     verbosity=0).fit(train_df, time_limit=time_limit)
        y_pred = predictor.predict(test_df).values
        if task_type == "classification":
            row = {"test_score": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
                   "rmse": None, "mae": None}
        else:
            row = _reg_metrics(y_te.values, y_pred)
        row["cv_score"] = None
        row["time_s"] = round(time.perf_counter() - t0, 2)
        row["status"] = "ok"
        return row
    except Exception as exc:
        return {"test_score": None, "rmse": None, "mae": None, "cv_score": None,
                "time_s": round(time.perf_counter() - t0, 2), "status": f"error: {exc}"}


# ── AutoML competitor: FLAML ───────────────────────────────────────────────────

def _flaml_score(X: pd.DataFrame, y: pd.Series, task_type: str, time_limit: int) -> dict:
    try:
        from flaml import AutoML
    except Exception as exc:
        return {"test_score": None, "rmse": None, "mae": None, "cv_score": None,
                "time_s": 0.0, "status": f"unavailable: {exc}"}

    # FLAML accepts categoricals as 'category' dtype; coerce object → category.
    Xc = X.copy()
    for c in Xc.columns:
        if Xc[c].dtype == object:
            Xc[c] = Xc[c].astype("category")
    test_size = _adaptive_test_size(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, y, test_size=test_size, random_state=_SEED)

    t0 = time.perf_counter()
    try:
        automl = AutoML()
        automl.fit(X_tr, y_tr,
                   task="classification" if task_type == "classification" else "regression",
                   metric="f1" if task_type == "classification" else "r2",
                   time_budget=time_limit, verbose=0, seed=_SEED)
        y_pred = automl.predict(X_te)
        if task_type == "classification":
            row = {"test_score": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
                   "rmse": None, "mae": None}
        else:
            row = _reg_metrics(y_te.values, y_pred)
        row["cv_score"] = None
        row["time_s"] = round(time.perf_counter() - t0, 2)
        row["status"] = "ok"
        return row
    except Exception as exc:
        return {"test_score": None, "rmse": None, "mae": None, "cv_score": None,
                "time_s": round(time.perf_counter() - t0, 2), "status": f"error: {exc}"}


# ── AutoML (ours) ──────────────────────────────────────────────────────────────

def _run_automl(ds_name: str, task_type: str) -> dict:
    """
    Run our full LangGraph pipeline on the raw CSV.

    Phase 1 (intent parsing + task classification) runs here before the graph,
    mirroring what server.py does. Only the metric is pre-set for fair comparison;
    target_column and task_type are inferred by the LLM from user_goal + column names.
    """
    from automl_system.graph.orchestrator import build_graph
    from automl_system.state.session import SessionState
    from automl_system.llm.client import LLMClient
    from automl_system.agents.intent_parser import parse_user_intent
    from automl_system.agents.task_classifier import resolve_target_from_data, run_task_classification
    from automl_system.utils.dataset_profile import compute_dataset_profile

    spec = KAGGLE_DATASETS[ds_name]
    csv_path = DATASETS_DIR / spec["file"]

    # Load raw CSV — no target renaming, no drops, no encoding
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Apply row cap only (keeps HPO tractable for large datasets)
    if spec.get("max_rows") and len(df) > spec["max_rows"]:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=spec["max_rows"], replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    user_goal = spec.get("user_goal", f"Predict the target variable in {ds_name}.")
    metric = metric_for(task_type)
    test_size = _adaptive_test_size(len(df))

    LLMClient.call_count = 0
    LLMClient.cache_hits = 0

    # ── Phase 1: LLM infers target_column + task_type (mirrors server.py) ────────
    session = SessionState()
    session.dataset_profile = compute_dataset_profile(df)
    session.task_config = {
        "goal": user_goal,
        "metric": metric,       # forced for fair cross-system comparison
        "test_size": test_size,
    }

    # Step 1: intent parser → extracts target_column candidate from user goal + columns
    intent = parse_user_intent(user_goal, list(df.columns), session)
    session.intent_result = intent

    # Step 2: resolve target if intent parser didn't identify it
    if not session.task_config.get("target_column") or \
            str(session.task_config.get("target_column", "")) not in df.columns:
        resolve_target_from_data(session, df)

    target_col = session.task_config.get("target_column")
    if not target_col or target_col not in df.columns:
        return {
            "test_score": None, "rmse": None, "mae": None, "cv_score": None,
            "best_model": None, "time_s": 0.0,
            "llm_calls": LLMClient.call_count, "cache_hits": LLMClient.cache_hits,
            "rule_decisions": 0, "stat_decisions": 0, "llm_decisions": 0, "llm_avoided_pct": 0.0,
            "smote_applied": False, "diversity_pruned": False,
            "status": "error: Phase 1 could not resolve target_column from user goal",
        }

    # Step 3: validate target + determine task_type
    run_task_classification(session, df)

    # Re-enforce forced metric AND task_type after task classification.
    # The LLM may override regression → classification for low-cardinality targets
    # (e.g. wine_quality with 6 unique values). Forcing both keeps the benchmark fair:
    # classifiers produce discrete predictions with lower R² than regression models.
    session.task_config["metric"] = metric
    session.task_config["task_type"] = task_type
    session.task_config["test_size"] = test_size

    # Step 4: recompute profile with resolved target (same as server.py)
    session.dataset_profile = compute_dataset_profile(df, target_column=target_col)

    print(f"    [Phase 1] target={target_col!r}  task_type={session.task_config.get('task_type')!r}")

    # ── Phases 2–7: LangGraph pipeline ────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        graph = build_graph()
        final_state = graph.invoke({"session": session.to_dict(), "df": df, "messages": [], "checkpoints": {}})
        elapsed = time.perf_counter() - t0

        final_sess = final_state.get("session") or {}
        exec_res = final_sess.get("execution_results") or {}
        prep_meta = final_sess.get("preprocessing_meta") or {}
        router = prep_meta.get("router_stats") or {}
        best_metrics = exec_res.get("best_model_metrics") or {}

        best_test = exec_res.get("best_test_score")
        status = exec_res.get("status") or "ok"

        # ── Novel-contribution flags ──────────────────────────────────────────
        # Read from session.local_decisions_log — both SMOTE and diversity pruning
        # call log_decision() which appends to that list.
        local_log = final_sess.get("local_decisions_log") or []
        smote_applied = any(e.get("action") == "smote_applied" for e in local_log)
        diversity_pruned = any(e.get("action") == "diversity_pruning" for e in local_log)

        # Progressive-elimination audit — already computed inside this same pipeline
        # run (execution.py writes it into execution_results). No second run needed.
        elimination_audit = exec_res.get("elimination_audit") or None

        return {
            "test_score": round(float(best_test), 4) if isinstance(best_test, (int, float)) else None,
            "rmse": best_metrics.get("rmse") if task_type == "regression" else None,
            "mae": best_metrics.get("mae") if task_type == "regression" else None,
            "cv_score": None,
            "best_model": exec_res.get("best_model_name") or "?",
            "time_s": round(elapsed, 2),
            "llm_calls": LLMClient.call_count,
            "cache_hits": LLMClient.cache_hits,
            "rule_decisions": router.get("rule_decisions", 0),
            "stat_decisions": router.get("stat_decisions", 0),
            "llm_decisions": router.get("llm_decisions", 0),
            "llm_avoided_pct": router.get("avoided_pct", 0.0),
            "smote_applied": smote_applied,
            "diversity_pruned": diversity_pruned,
            "status": status,
            "elimination_audit": elimination_audit,
        }
    except Exception as exc:
        return {
            "test_score": None, "rmse": None, "mae": None, "cv_score": None,
            "best_model": None, "time_s": round(time.perf_counter() - t0, 2),
            "llm_calls": LLMClient.call_count, "cache_hits": LLMClient.cache_hits,
            "rule_decisions": 0, "stat_decisions": 0, "llm_decisions": 0, "llm_avoided_pct": 0.0,
            "smote_applied": False, "diversity_pruned": False,
            "status": f"error: {exc}",
            "elimination_audit": None,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def run_benchmark(dataset_filter: list[str] | None = None, skip_automl: bool = False,
                  only_ours: bool = False, automl_time: int = 90) -> pd.DataFrame:
    names = dataset_filter or list(KAGGLE_DATASETS)
    rows: list[dict] = []
    elimination_audits: list[dict] = []
    RESULTS_DIR.mkdir(exist_ok=True)

    for ds_name in names:
        loaded = load_raw(ds_name)
        if loaded is None:
            print(f"  SKIP {ds_name}: CSV not found in experiments/datasets/")
            continue
        df, task_type, _ = loaded
        X = df.drop(columns=["target"])
        y = df["target"]
        metric_label = metric_label_for(task_type)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(X)} rows × {X.shape[1]} features, {task_type})")
        print(f"{'='*60}")

        def _append(system: str, r: dict, **extra):
            rows.append({
                "dataset": ds_name, "n_rows": len(X), "n_features": X.shape[1],
                "task_type": task_type, "metric": metric_label, "system": system,
                "cv_score": r.get("cv_score"), "test_score": r.get("test_score"),
                "rmse": r.get("rmse"), "mae": r.get("mae"),
                "time_s": r.get("time_s"), "status": r.get("status", "ok"), **extra,
            })

        if not only_ours:
            # Single-model baselines
            print("  Running single-model baselines...")
            for bl_name, bl in _baseline_score(X, y, task_type).items():
                print(f"  {bl_name}: test={_fmt(bl['test_score'])}  cv={_fmt(bl['cv_score'])}  t={bl['time_s']}s")
                _append(bl_name, bl, llm_calls=0, best_model="—")

            # AutoML competitors
            print("  Running AutoGluon...")
            ag = _autogluon_score(X, y, task_type, automl_time)
            print(f"  AutoGluon: test={_fmt(ag['test_score'])}  t={ag['time_s']}s  status={ag['status']}")
            _append("AutoGluon", ag, llm_calls=0, best_model="—")

            print("  Running FLAML...")
            fl = _flaml_score(X, y, task_type, automl_time)
            print(f"  FLAML: test={_fmt(fl['test_score'])}  t={fl['time_s']}s  status={fl['status']}")
            _append("FLAML", fl, llm_calls=0, best_model="—")

        # AutoML (ours)
        if not skip_automl:
            print("  Running AutoML (ours)...")
            am = _run_automl(ds_name, task_type)
            print(f"  AutoML (ours): test={_fmt(am['test_score'])}  model={am['best_model']}  "
                  f"t={am['time_s']}s  llm_calls={am['llm_calls']}  llm_avoided={am['llm_avoided_pct']}%  "
                  f"smote={am['smote_applied']}  diversity_pruned={am['diversity_pruned']}  status={am['status']}")
            _append("AutoML (ours)", am,
                    llm_calls=am["llm_calls"], best_model=am["best_model"],
                    rule_decisions=am["rule_decisions"], stat_decisions=am["stat_decisions"],
                    llm_decisions=am["llm_decisions"], llm_avoided_pct=am["llm_avoided_pct"],
                    smote_applied=am["smote_applied"], diversity_pruned=am["diversity_pruned"])

            audit = am.get("elimination_audit")
            if audit:
                audit = {**audit, "dataset": ds_name}
                elimination_audits.append(audit)
                elim = audit.get("eliminated_family", "—")
                rank = audit.get("subsample_rank_of_eliminated")
                total = audit.get("total_candidates")
                correct = audit.get("elimination_correct")
                print(f"  [Elimination audit] eliminated={elim}  rank={rank}/{total}  strict_correct={correct}")

    out = pd.DataFrame(rows)

    # If a single dataset was run, save to a per-dataset file so parallel
    # terminal runs don't overwrite each other. A combined file is written
    # only when multiple datasets are present in this run.
    if len(names) == 1:
        csv_path = RESULTS_DIR / f"result_{names[0]}.csv"
    else:
        csv_path = RESULTS_DIR / "benchmark_results.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    cols = [c for c in ["dataset", "system", "test_score", "rmse", "mae", "time_s", "llm_calls", "best_model"]
            if c in out.columns]
    print(out[cols].to_string(index=False))

    if elimination_audits:
        elim_csv_path = (
            RESULTS_DIR / f"elimination_result_{names[0]}.csv"
            if len(names) == 1 else RESULTS_DIR / "elimination_results.csv"
        )
        _print_elimination_summary(elimination_audits, elim_csv_path)

    return out


def _print_elimination_summary(audits: list[dict], csv_path: Path) -> None:
    """Print + save the progressive-elimination accuracy analysis.

    Reuses the same audit data collected in this run's AutoML pipeline calls —
    no second pipeline run needed. Mirrors elimination_accuracy.py's analysis.
    """
    from elimination_accuracy import _lenient_correct

    df = pd.DataFrame(audits)
    ran = df[df["eliminated_family"].notna()].copy()

    print("\n" + "="*65)
    print("PROGRESSIVE ELIMINATION ACCURACY ANALYSIS")
    print("="*65)
    print(f"Datasets with elimination : {len(ran)}/{len(df)}")

    if len(ran) == 0:
        print("No elimination events found.")
        return

    ran["lenient_correct"] = ran.apply(_lenient_correct, axis=1)
    strict_correct = ran["elimination_correct"].sum()
    lenient_correct = ran["lenient_correct"].sum()
    mean_rank = ran["subsample_rank_of_eliminated"].mean()
    rank1_pct = 100 * (ran["subsample_rank_of_eliminated"] == 1).sum() / len(ran)

    print(f"Strict correct  (rank-1 only) : {int(strict_correct)}/{len(ran)} ({100 * strict_correct / len(ran):.1f}%)")
    print(f"Lenient correct (within 0.01) : {int(lenient_correct)}/{len(ran)} ({100 * lenient_correct / len(ran):.1f}%)")
    print(f"Mean rank of eliminated       : {mean_rank:.2f}  (1 = worst = ideal)")
    print(f"Rank-1 eliminations           : {rank1_pct:.1f}%")

    print("\nPer-dataset breakdown:")
    for _, row in ran.sort_values("dataset").iterrows():
        flag = "correct" if row["elimination_correct"] else ("near" if row["lenient_correct"] else "wrong")
        print(
            f"  [{flag}] {row['dataset']:<20} eliminated={row['eliminated_family']:<22} "
            f"rank={int(row['subsample_rank_of_eliminated'])}/{int(row['total_candidates'])}  "
            f"score={row['eliminated_subsample_score']:.4f}"
        )

    csv_path.parent.mkdir(exist_ok=True)
    ran.to_csv(csv_path, index=False)
    print(f"\nElimination results saved to {csv_path}")


def merge_results() -> pd.DataFrame:
    """Merge all per-dataset result_*.csv files into benchmark_results.csv."""
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(RESULTS_DIR.glob("result_*.csv"))
    if not files:
        print("No per-dataset result files found in", RESULTS_DIR)
        return pd.DataFrame()
    parts = [pd.read_csv(f) for f in files]
    combined = pd.concat(parts, ignore_index=True)
    out_path = RESULTS_DIR / "benchmark_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"Merged {len(files)} files → {out_path}  ({len(combined)} rows)")
    cols = [c for c in ["dataset", "system", "test_score", "rmse", "mae", "time_s", "llm_calls", "best_model"]
            if c in combined.columns]
    print(combined[cols].to_string(index=False))

    elim_files = sorted(RESULTS_DIR.glob("elimination_result_*.csv"))
    if elim_files:
        elim_parts = [pd.read_csv(f) for f in elim_files]
        elim_combined = pd.concat(elim_parts, ignore_index=True)
        elim_out_path = RESULTS_DIR / "elimination_results.csv"
        elim_combined.to_csv(elim_out_path, index=False)
        print(f"Merged {len(elim_files)} elimination files → {elim_out_path}  ({len(elim_combined)} rows)")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names (default: all)")
    parser.add_argument("--skip-automl", action="store_true", help="Skip our pipeline (competitors + baselines only)")
    parser.add_argument("--only-ours", action="store_true", help="Skip competitors/baselines, run only our pipeline")
    parser.add_argument("--automl-time", type=int, default=90, help="Time budget (s) for AutoGluon/FLAML")
    parser.add_argument("--merge", action="store_true", help="Merge all result_*.csv files into benchmark_results.csv")
    args = parser.parse_args()

    if args.merge:
        merge_results()
    else:
        ds_filter = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None
        run_benchmark(
            dataset_filter=ds_filter,
            skip_automl=args.skip_automl,
            only_ours=args.only_ours,
            automl_time=args.automl_time,
        )
