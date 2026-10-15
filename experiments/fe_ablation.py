"""
Resumable feature-engineering ablation: FE_SELECTION=corr (old) vs cv (CV-gated).

Runs the full pipeline once per (dataset, mode, repeat) job, one job per subprocess,
and appends a JSON line per finished run to results/fe_ablation.jsonl. Built for a
daily LLM token quota (e.g. Groq free tier):

  - finished jobs are skipped, so just run it again the next day to continue;
  - jobs are ordered round-robin (repeat 1 of every dataset/mode first) so partial
    results stay balanced;
  - if the LLM provider hits its rate limit during a run, that run is discarded and
    the batch stops — rate-limited runs fall back to defaults and are not comparable.

Usage:
  python experiments/fe_ablation.py                 # run pending jobs until the quota is used up
  python experiments/fe_ablation.py --max-runs 3    # at most 3 runs now
  python experiments/fe_ablation.py --report        # mean ± std per dataset and mode

Teammates can run it on their own account and send their fe_ablation.jsonl; merge
files by concatenating them (jobs are identified by dataset, mode and repeat).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
RESULTS_FILE = _HERE / "results" / "fe_ablation.jsonl"
LOG_DIR = _HERE / "results" / "fe_ablation_logs"

DATASETS = ["titanic", "heart", "churn", "bank", "insurance", "house_prices", "life_expectancy", "wine"]
MODES = ["corr", "cv"]
_RATE_LIMIT_MARKERS = ("Error code: 429", "rate_limit_exceeded", "Rate limit reached")


def _load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return [json.loads(line) for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pending_jobs(repeats: int) -> list[tuple[str, str, int]]:
    done = {(r["dataset"], r["mode"], r["repeat"]) for r in _load_results()}
    jobs = [(ds, mode, rep) for rep in range(1, repeats + 1) for ds in DATASETS for mode in MODES]
    return [j for j in jobs if j not in done]


def _run_one(dataset: str, mode: str) -> dict:
    """Worker (runs in a subprocess): one full pipeline run, returns a result row."""
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_HERE))
    from automl_system import config

    config.AUTOML_BENCHMARK_TIME_LIMIT = 0     # no in-pipeline AutoGluon/FLAML
    config.FE_SELECTION = mode
    config.FE_CODE_FEATURES = "off"

    import automl_system.graph.orchestrator as orch

    captured: dict = {}
    build = orch.build_graph

    def _capturing_build():
        graph = build()

        class _Graph:
            def invoke(self, state, *a, **k):
                captured["state"] = graph.invoke(state, *a, **k)
                return captured["state"]

        return _Graph()

    orch.build_graph = _capturing_build
    import benchmark

    t0 = time.perf_counter()
    summary = benchmark._run_automl(dataset, benchmark.KAGGLE_DATASETS[dataset]["task_type"])
    sess = (captured.get("state") or {}).get("session") or {}
    log = sess.get("feature_engineering_log") or []
    return {
        "dataset": dataset,
        "mode": mode,
        "status": summary.get("status"),
        "test_score": summary.get("test_score"),
        "best_model": summary.get("best_model"),
        "llm_calls": summary.get("llm_calls"),
        "fe_added": [e.get("feature") for e in log if e.get("status") == "added"],
        "fe_skipped": next((e.get("reason") for e in log if e.get("status") == "skipped"), None),
        "fe_seconds": (sess.get("phase_timings") or {}).get("feature_engineering"),
        "script_status": (sess.get("script_validation") or {}).get("status"),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def run_pending(repeats: int, max_runs: int | None) -> None:
    jobs = _pending_jobs(repeats)
    print(f"{len(jobs)} job(s) pending (repeats={repeats}).")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for n, (dataset, mode, rep) in enumerate(jobs, start=1):
        if max_runs is not None and n > max_runs:
            break
        log_path = LOG_DIR / f"{dataset}_{mode}_r{rep}.log"
        print(f"[{n}/{len(jobs)}] {dataset} / {mode} / repeat {rep} … ", end="", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", dataset, mode],
                cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if any(m in output for m in _RATE_LIMIT_MARKERS):
            print("LLM rate limit hit - run discarded. Stopping; run again after the quota resets.")
            return
        result_line = next((ln for ln in reversed(output.splitlines()) if ln.startswith("RESULT_JSON ")), None)
        if proc.returncode != 0 or result_line is None:
            print(f"failed (exit {proc.returncode}) - see {log_path.name}")
            continue
        row = json.loads(result_line[len("RESULT_JSON "):])
        if not isinstance(row.get("test_score"), (int, float)):
            print(f"no score ({row.get('status')}) - not recorded")
            continue
        row["repeat"] = rep
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"score {row['test_score']}  ({row['wall_s']:.0f}s)")
    print("All pending jobs finished." if not _pending_jobs(repeats) else "Stopped early.")


def report() -> None:
    rows = _load_results()
    if not rows:
        print("No results yet.")
        return
    print(f"{'dataset':16s} {'corr (old)':>22s} {'cv (new)':>22s} {'diff':>8s}")
    diffs = []
    for ds in DATASETS:
        cells, means = [], {}
        for mode in MODES:
            scores = [r["test_score"] for r in rows if r["dataset"] == ds and r["mode"] == mode]
            if scores:
                means[mode] = statistics.mean(scores)
                sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
                cells.append(f"{means[mode]:.4f} +/- {sd:.4f} (n={len(scores)})")
            else:
                cells.append("-")
        diff = means["cv"] - means["corr"] if len(means) == 2 else None
        if diff is not None:
            diffs.append(diff)
        print(f"{ds:16s} {cells[0]:>22s} {cells[1]:>22s} {(f'{diff:+.4f}' if diff is not None else '-'):>8s}")
    if diffs:
        better = sum(d > 0 for d in diffs)
        print(f"\nMean difference (cv - corr) over {len(diffs)} dataset(s): {statistics.mean(diffs):+.4f}; "
              f"cv better on {better}/{len(diffs)}.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        result = _run_one(sys.argv[2], sys.argv[3])
        print("RESULT_JSON " + json.dumps(result, default=str))
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=2, help="runs per dataset and mode (default 2)")
    parser.add_argument("--max-runs", type=int, default=None, help="stop after this many runs")
    parser.add_argument("--report", action="store_true", help="print the summary table and exit")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        run_pending(args.repeats, args.max_runs)
