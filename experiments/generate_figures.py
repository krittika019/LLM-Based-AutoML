"""
Generate all figures for the research paper from experiment CSVs.
Run after benchmark.py --merge has been executed.

Output: experiments/results/figures/
"""

from __future__ import annotations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
FIGS    = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Linear baseline":        "#82c08e",
    "RandomForest (default)": "#5b8dd9",
    "AutoML (ours)":          "#e05c5c",
    "AutoGluon":              "#f5a623",
    "FLAML":                  "#9b59b6",
}
SYSTEM_ORDER = ["Linear baseline", "RandomForest (default)", "FLAML", "AutoGluon", "AutoML (ours)"]

ROUTER_COLORS = {
    "Rule-based":  "#5b8dd9",
    "Stat-based":  "#f5a623",
    "LLM":         "#e05c5c",
}


def _load_benchmark() -> pd.DataFrame | None:
    path = RESULTS / "benchmark_results.csv"
    if not path.exists():
        print(f"  benchmark_results.csv not found at {path}")
        print("  Run: python experiments/benchmark.py --merge")
        return None
    df = pd.read_csv(path)
    df = df[df["test_score"].notna()]
    return df


# ── Figure 1: Performance comparison — split by task type ─────────────────────

CLS_DATASETS = ["titanic", "heart", "churn", "bank", "creditcard_fraud"]
REG_DATASETS = ["house_prices", "insurance", "diamonds", "life_expectancy", "wine"]


def _draw_group(ax, df, datasets, systems, ylabel, ylim):
    x      = np.arange(len(datasets))
    width  = 0.14
    n      = len(systems)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, sys in enumerate(systems):
        sub    = df[df["system"] == sys].set_index("dataset")
        scores = [float(sub.loc[d, "test_score"]) if d in sub.index else 0 for d in datasets]
        bars   = ax.bar(x + offsets[i], scores, width * 0.88,
                        label=sys, color=COLORS.get(sys, "#aaa"), zorder=3)
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{score:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig_performance_comparison():
    df = _load_benchmark()
    if df is None:
        print("SKIP fig1")
        return

    systems = [s for s in SYSTEM_ORDER if s in df["system"].unique()]

    cls_ds = [d for d in CLS_DATASETS if d in df["dataset"].unique()]
    reg_ds = [d for d in REG_DATASETS if d in df["dataset"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6),
                                    gridspec_kw={"width_ratios": [len(cls_ds), len(reg_ds)]})

    _draw_group(ax1, df, cls_ds, systems, "F1 Score (weighted)", (0.60, 1.15))
    ax1.set_title("Classification", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    _draw_group(ax2, df, reg_ds, systems, "R² Score", (0.40, 1.12))
    ax2.set_title("Regression", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = FIGS / "fig1_performance_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(FIGS / "fig1_performance_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")




# ── Figure 3: Decision Router — LLM avoidance rate per dataset ────────────────
# Two colors (Avoided vs LLM), not three (Rule/Stat/LLM) -- the rule-vs-stat
# split is already table-level detail (Table I, Table VI); this figure only
# needs to carry the sentence-level claim: how much each dataset needed the LLM.

def fig_decision_router():
    df = _load_benchmark()
    if df is None:
        print("SKIP fig3")
        return

    ours = df[df["system"] == "AutoML (ours)"].copy()
    ours = ours[ours["rule_decisions"].notna()].copy()
    if ours.empty:
        print("SKIP fig3: no router decision data in benchmark_results.csv")
        return

    ours["total"] = (
        ours["rule_decisions"].astype(float)
        + ours["stat_decisions"].astype(float)
        + ours["llm_decisions"].astype(float)
    )
    ours = ours[ours["total"] > 0].copy()
    ours["llm_pct"]     = ours["llm_decisions"].astype(float) / ours["total"] * 100
    ours["avoided_pct"] = 100 - ours["llm_pct"]
    ours = ours.sort_values("avoided_pct", ascending=True)  # hardest datasets (most LLM) at top

    datasets     = ours["dataset"].tolist()
    avoided_vals = ours["avoided_pct"].values
    llm_vals     = ours["llm_pct"].values
    mean_avoided = avoided_vals.mean()  # matches Table VI's own Mean row (71.7%)
    y = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(y, avoided_vals, color=ROUTER_COLORS["Rule-based"],
            label="Avoided (Rule + Stat)", height=0.6, zorder=3)
    ax.barh(y, llm_vals, left=avoided_vals,
            color=ROUTER_COLORS["LLM"], label="LLM", height=0.6, zorder=3)

    for i, (a, l) in enumerate(zip(avoided_vals, llm_vals)):
        ax.text(a / 2, i, f"{a:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(a + l / 2, i, f"{l:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.axvline(mean_avoided, color="#333", linewidth=1.2, linestyle="--", zorder=4)
    ax.text(mean_avoided, len(datasets) - 0.55, f" Mean: {mean_avoided:.1f}%",
            va="bottom", ha="left", fontsize=9, color="#333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(datasets) - 1 + 1.1)  # headroom for the mean label above the top bar
    ax.set_xlabel("Percentage of Preprocessing Decisions (%)", fontsize=10)
    ax.set_title("Decision Router: LLM Avoidance Rate per Dataset", fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    out = FIGS / "fig3_decision_router.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(FIGS / "fig3_decision_router.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Win / loss margin vs best competitor ────────────────────────────

def fig_win_loss_margin():
    df = _load_benchmark()
    if df is None:
        print("SKIP fig4")
        return

    competitors = ["RandomForest (default)", "Linear baseline", "AutoGluon", "FLAML"]
    ours_df = df[df["system"] == "AutoML (ours)"].set_index("dataset")
    comp_df = df[df["system"].isin(competitors)]

    # Best competitor score per dataset
    best_comp = comp_df.groupby("dataset")["test_score"].max()

    datasets = sorted(set(ours_df.index) & set(best_comp.index))
    margins  = [float(ours_df.loc[d, "test_score"]) - float(best_comp[d]) for d in datasets]

    colors = ["#5b8dd9" if m >= 0 else "#e05c5c" for m in margins]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(datasets, margins, color=colors, zorder=3, height=0.55)

    for bar, margin in zip(bars, margins):
        sign  = "+" if margin >= 0 else ""
        xpos  = margin + 0.0005 if margin >= 0 else margin - 0.0005
        ha    = "left" if margin >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{sign}{margin:.4f}", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="#333", linewidth=1.2, zorder=4)
    ax.set_xlabel("Score Margin vs Best Competitor", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    win_patch  = mpatches.Patch(color="#5b8dd9", label="Our system wins")
    loss_patch = mpatches.Patch(color="#e05c5c", label="Our system loses")
    ax.legend(handles=[win_patch, loss_patch], fontsize=9)

    plt.tight_layout()
    out = FIGS / "fig4_win_loss_margin.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(FIGS / "fig4_win_loss_margin.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: Progressive Elimination — dot strip + accuracy summary ──────────

def fig_elimination_accuracy():
    import ast
    from matplotlib.lines import Line2D

    path = RESULTS / "elimination_results.csv"
    if not path.exists():
        print("SKIP fig5: elimination_results.csv not found")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("SKIP fig5: elimination_results.csv is empty")
        return

    df = df.sort_values("dataset").reset_index(drop=True)
    datasets  = df["dataset"].tolist()
    n         = len(datasets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: dot strip — all candidate scores, eliminated model highlighted ──
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            scores = ast.literal_eval(row["all_subsample_scores"]) \
                     if isinstance(row["all_subsample_scores"], str) \
                     else row["all_subsample_scores"]
        except Exception:
            scores = {}
        if not scores:
            continue

        elim_family  = row["eliminated_family"]
        worst_family = min(scores, key=scores.get)

        # All other candidates — grey dots
        for family, score in scores.items():
            if family == elim_family:
                continue
            ax1.scatter(score, i, color="#bbbbbb", s=55, zorder=3, alpha=0.85)

        # Eliminated model — coloured diamond
        if row["elimination_correct"]:
            elim_color = "#5b8dd9"   # blue  = strict correct
        elif row["lenient_correct"]:
            elim_color = "#f5a623"   # amber = lenient correct
        else:
            elim_color = "#e05c5c"   # red   = wrong

        elim_score = scores.get(elim_family, float(row["eliminated_subsample_score"]))
        ax1.scatter(elim_score, i, color=elim_color, s=130, zorder=5, marker="D")

        # If the LLM missed the actual worst, mark it with ×
        if not row["elimination_correct"] and worst_family != elim_family:
            ax1.scatter(scores[worst_family], i,
                        color="#333", s=90, zorder=4, marker="x", linewidths=2)

    ax1.set_yticks(range(n))
    ax1.set_yticklabels(datasets, fontsize=10)
    ax1.set_xlabel("Round-1 Subsample Score", fontsize=10)
    ax1.set_title("LLM Model Elimination Decisions\n(diamond = eliminated, × = missed worst)",
                  fontsize=11, fontweight="bold")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#5b8dd9",
               markersize=9, label="Eliminated — strict correct"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f5a623",
               markersize=9, label="Eliminated — lenient correct (≤0.01 gap)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e05c5c",
               markersize=9, label="Eliminated — wrong"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
               markersize=8, label="Other candidates"),
        Line2D([0], [0], marker="x", color="#333", markersize=8,
               linewidth=0, markeredgewidth=2, label="Actual worst (when missed)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # ── Right: accuracy summary bars ─────────────────────────────────────────
    strict   = int(df["elimination_correct"].sum())
    lenient  = int(df["lenient_correct"].sum())
    total_ds = len(df)

    categories = ["Strict\n(rank-1)", "Lenient\n(±0.01)"]
    values     = [strict / total_ds * 100, lenient / total_ds * 100]
    bar_colors = ["#5b8dd9", "#f5a623"]

    b = ax2.bar(categories, values, color=bar_colors, width=0.4, zorder=3)
    for bar, val, count in zip(b, values, [strict, lenient]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{count}/{total_ds}\n({val:.0f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.set_ylim(0, 120)
    ax2.set_ylabel("Elimination Accuracy (%)", fontsize=10)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Accuracy Summary", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = FIGS / "fig5_elimination_accuracy.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(FIGS / "fig5_elimination_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    fig_performance_comparison()
    fig_decision_router()
    fig_win_loss_margin()
    fig_elimination_accuracy()
    print(f"\nAll figures saved to: {FIGS}")
