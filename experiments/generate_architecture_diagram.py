"""Generate the system architecture diagram for the research paper.

Grounded directly in automl_system/graph/orchestrator.py's actual LangGraph
node/edge structure (14 nodes, verification retry loop, failure-diagnosis
feedback loop, unsupervised fork after basic_cleaning) -- not a stylized
approximation.

Output: experiments/results/figures/fig0_system_architecture.png/.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

FIGS = Path(__file__).parent / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_INOUT   = "#4a4a4a"   # input/output
C_CONTROL = "#5b8dd9"   # phase 1, 7 (control / decision points)
C_PREP    = "#82c08e"   # phase 2-3 (planning + preprocessing)
C_EXEC    = "#e0a458"   # phase 4-5 (planning + execution)
C_PHASE6  = "#3a9599"   # phase 6 (threshold check + failure diagnosis + targeted revision)
C_INFRA   = "#9b9b9b"   # cross-cutting infrastructure
TXT_LIGHT = "white"
TXT_DARK  = "#222222"


def box(ax, x, y, w, h, text, facecolor, textcolor=TXT_LIGHT, fontsize=9,
        weight="bold", edgecolor="none", zorder=3, linestyle="-"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, weight=weight, zorder=zorder + 1,
             linespacing=1.35)
    return b


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.4, style="-|>",
          connectionstyle="arc3,rad=0.0", zorder=2, linestyle="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        color=color, lw=lw, connectionstyle=connectionstyle, zorder=zorder,
        linestyle=linestyle,
    )
    ax.add_patch(a)
    return a


def icon_scale(ax, cx, cy, s=0.16, color="#5b4fa8"):
    """Simple balance-scale glyph for the rule-based tier."""
    ax.add_line(Line2D([cx, cx], [cy - s, cy + s], color=color, lw=1.3, zorder=6))
    ax.add_line(Line2D([cx - s, cx + s], [cy + s, cy + s], color=color, lw=1.3, zorder=6))
    ax.add_patch(Circle((cx - s, cy + s * 0.55), s * 0.32, facecolor="none",
                         edgecolor=color, lw=1.1, zorder=6))
    ax.add_patch(Circle((cx + s, cy + s * 0.55), s * 0.32, facecolor="none",
                         edgecolor=color, lw=1.1, zorder=6))
    ax.add_patch(Polygon([(cx - s * 0.3, cy - s), (cx + s * 0.3, cy - s), (cx, cy - s * 1.7)],
                          closed=True, facecolor=color, edgecolor="none", zorder=6))


def icon_bars(ax, cx, cy, s=0.16, color="#5b4fa8"):
    """Ascending bar-chart glyph for the statistical tier."""
    heights = [0.6 * s, 1.0 * s, 1.5 * s]
    for i, h in enumerate(heights):
        bx = cx - s * 1.1 + i * s * 1.1
        ax.add_patch(Rectangle((bx, cy - s), s * 0.7, h * 2, facecolor=color,
                                edgecolor="none", zorder=6))


def icon_network(ax, cx, cy, s=0.16, color="#5b4fa8"):
    """Simple connected-node glyph for the LLM tier."""
    pts = [(cx, cy + s), (cx - s, cy - s * 0.6), (cx + s, cy - s * 0.6), (cx, cy - s * 1.0)]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]:
        ax.add_line(Line2D([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                            color=color, lw=0.9, zorder=6))
    for (px, py) in pts:
        ax.add_patch(Circle((px, py), s * 0.28, facecolor=color, edgecolor="none", zorder=7))


def icon_database(ax, cx, cy, w=0.4, h=0.42, color="#555555"):
    """Classic database-cylinder glyph: bottom ellipse, body, top ellipse."""
    rw, rh = w / 2, h * 0.16
    ax.add_patch(Ellipse((cx, cy - h / 2 + rh * 0.3), w, rh * 2,
                          facecolor="none", edgecolor=color, lw=1.1, zorder=6))
    ax.add_patch(Rectangle((cx - rw, cy - h / 2 + rh * 0.3), w, h - rh * 0.6,
                            facecolor="white", edgecolor=color, lw=0, zorder=6))
    ax.add_line(Line2D([cx - rw, cx - rw], [cy - h / 2 + rh * 0.3, cy + h / 2 - rh * 0.3],
                        color=color, lw=1.1, zorder=7))
    ax.add_line(Line2D([cx + rw, cx + rw], [cy - h / 2 + rh * 0.3, cy + h / 2 - rh * 0.3],
                        color=color, lw=1.1, zorder=7))
    ax.add_patch(Ellipse((cx, cy + h / 2 - rh * 0.3), w, rh * 2,
                          facecolor="white", edgecolor=color, lw=1.1, zorder=7))


# Canvas extended 2.75 units to the left (figsize widened to match, keeping
# 1 unit per inch) to host the unsupervised branch beside the spine -- the
# right column is fully occupied by the router, the log and the Phase 6 branch.
fig, ax = plt.subplots(figsize=(18.25, 10))
ax.set_xlim(-3.25, 15)
ax.set_ylim(0, 10.8)
ax.axis("off")

# ── Main pipeline spine (x = 0.6 to 7.4) ───────────────────────────────────────
W = 6.8
X0 = 0.6

# Input -- moved up slightly (9.9 -> 10.05) to lengthen the arrow into
# Phase 1, which was too short at the old spacing.
box(ax, X0, 10.05, W, 0.5, "User Input — Raw Dataset + Natural-Language User Goal + Task Mode",
    C_INOUT, fontsize=9.5)

# Phase 1 -- target/metric resolution only runs for supervised tasks; the
# unsupervised mode (set by the UI toggle) skips it entirely.
box(ax, X0, 9.05, W, 0.6,
    "Phase 1 — Intent Parsing & Task Classification\n"
    "(task type; target column & metric resolution for supervised runs)",
    C_CONTROL, fontsize=8.7)

# Phase 2
box(ax, X0, 8.15, W, 0.6,
    "Phase 2 — Preprocessing Planning\n"
    "(null strategy, encoding, scaling plan)",
    C_PREP, fontsize=8.7)

# Phase 3 (grouped preprocessing block)
# p3_h trimmed from 1.55 to restore a normal ~0.3 gap to Phase 2 above
# (it was 0.15, half the gap size everywhere else in the spine).
p3_top, p3_h = 6.45, 1.40
box(ax, X0, p3_top, W, p3_h,
    "", C_PREP, fontsize=9.2)
ax.text(X0 + W / 2, p3_top + p3_h - 0.22,
        "Phase 3 — Data Cleaning & Feature Engineering",
        ha="center", va="center", fontsize=9.2, color=TXT_LIGHT, weight="bold")
# 7 sub-steps now (Train-Test Split added after Basic Cleaning, matching
# orchestrator.py's basic_cleaning -> split_data -> eda edge order) -- boxes
# narrowed accordingly (width divisor 6 -> 7, font trimmed so longer labels
# still fit). Margin subtracted from each box's width is 0.10 for visible
# spacing between adjacent boxes.
sub_labels = ["Basic\nCleaning", "Train-Test\nSplit", "EDA", "Feature\nSelection",
              "Deep\nCleaning", "Feature\nEngineering", "Verification\n(rule-based check)"]
n_sub = len(sub_labels)
sub_w = (W - 0.3) / n_sub - 0.10
sub_y = p3_top + 0.18
for i, lab in enumerate(sub_labels):
    sx = X0 + 0.15 + i * ((W - 0.3) / n_sub)
    box(ax, sx, sub_y, sub_w, 0.55, lab, "#ffffff", textcolor=TXT_DARK,
        fontsize=5.4, edgecolor="#4c9463", weight="normal", zorder=4)
    if i < n_sub - 1:
        arrow(ax, sx + sub_w, sub_y + 0.275, sx + (W - 0.3) / n_sub + 0.15,
              sub_y + 0.275, color="#4c9463", lw=0.9)

# Solid black covering border around the 6 preprocessing sub-steps (excludes
# Verification, which is the rewind SOURCE, not part of the rewind target).
cover_pad = 0.07
cover_left = X0 + 0.15
cover_right = X0 + 0.15 + 5 * ((W - 0.3) / n_sub) + sub_w
cover_x0 = cover_left - cover_pad
cover_y0 = sub_y - cover_pad
cover_w = (cover_right - cover_left) + 2 * cover_pad
cover_h = 0.55 + 2 * cover_pad
ax.add_patch(FancyBboxPatch(
    (cover_x0, cover_y0), cover_w, cover_h,
    boxstyle="round,pad=0.01,rounding_size=0.04",
    facecolor="none", edgecolor="#8a8a8a", linewidth=1.3,
    linestyle="-", zorder=6,
))

# Rewind arrow: a long, gentle path (not a tight auto-curve, whose height
# scales with distance and looked cramped) from above Verification, across
# nearly the full width of the covering border, landing near its left end
# (close to Basic Cleaning). Explicit peak height keeps it clear of both
# the sub-box row below and the Phase 3 title above.
verif_sx = X0 + 0.15 + (n_sub - 1) * ((W - 0.3) / n_sub)
verif_center_x = verif_sx + sub_w / 2
peak_y = 7.35
rw_start = (verif_center_x, sub_y + 0.55 + 0.02)
rw_end = (cover_x0 + 0.5, cover_y0 + cover_h + 0.02)
rw_path2 = MplPath(
    [rw_start, (rw_start[0], peak_y), (rw_end[0], peak_y), rw_end],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
)
ax.add_patch(FancyArrowPatch(path=rw_path2, arrowstyle="-|>", mutation_scale=12,
                              color="#c0392b", lw=1.2, linestyle="--", zorder=6))
ax.text((rw_start[0] + rw_end[0]) / 2, peak_y + 0.03,
        "re-wind to failed step if preprocessing output is invalid",
        ha="center", va="bottom", fontsize=6.2, color="#c0392b", style="italic",
        zorder=6)

# Phase 4
box(ax, X0, 5.55, W, 0.6,
    "Phase 4 — Execution Planning\n"
    "(model shortlist, search strategy, CV design, threshold)",
    C_EXEC, fontsize=8.7)

# Phase 5
box(ax, X0, 4.4, W, 0.75,
    "Phase 5 — Model Training & Evaluation\n"
    "Progressive elimination → Optuna TPE + LLM HPO priors →\n"
    "diversity-aware ensembling → threshold optimization → SHAP attribution",
    C_EXEC, fontsize=8.2)

# Phase 6 -- threshold check + failure diagnosis + targeted revision all share
# the C_PHASE6 teal so the three boxes read as one phase (the code tags all
# three "Phase 6" in orchestrator.py), distinct from Phase 1/7's blue.
box(ax, X0, 3.35, W, 0.65,
    "Phase 6 — Threshold Check",
    C_PHASE6, fontsize=8.7)

# Failure diagnosis + targeted revision (branch, right of Phase 6, in the
# gap before the infrastructure column so its feedback line never crosses
# any pipeline box). FX pushed further out than before to give the feedback
# labels enough breathing room -- they were cramped into a 0.23-unit gap.
FX = X0 + W + 0.95
FW = 2.65
box(ax, FX, 3.35, FW, 0.65,
    "Phase 6 — Failure Diagnosis\n(data / feature / model /\nfundamental-insufficiency)",
    C_PHASE6, fontsize=7.2)
box(ax, FX, 2.35, FW, 0.6,
    "Phase 6 — Targeted Revision\n(re-enters affected phase only)",
    C_PHASE6, fontsize=7.4)

arrow(ax, X0 + W, 3.675, FX, 3.675, color="#333333")
ax.text((X0 + W + FX) / 2, 3.78, "Fail", ha="center", va="bottom",
        fontsize=7.5, color="#333333", weight="bold")
arrow(ax, FX + FW / 2, 3.35, FX + FW / 2, 2.95, color="#333333")

# Feedback loop: Targeted Revision -> back into Phase 3/4, routed straight
# up through the empty gap, then a short horizontal jog left -- never
# crosses a box. Widened from +0.22 to +0.45 to match FX's wider gap.
feedback_x = X0 + W + 0.45
verts = [
    (FX, 2.65),                 # start: left edge of Targeted Revision box
    (feedback_x, 2.65),         # jog left into the gap
    (feedback_x, p3_top + 0.4),  # straight up through the gap
    (X0 + W, p3_top + 0.4),      # into Phase 3's right edge, NOT Phase 2 --
                                 # route_after_targeted_revision() only ever
                                 # returns basic_cleaning / deep_cleaning /
                                 # feature_selection / feature_engineering
                                 # (all inside Phase 3) or phase4_planning.
]
codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
path = MplPath(verts, codes)
patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                         color="#c0392b", lw=1.3, linestyle="--", zorder=2)
ax.add_patch(patch)

# Second branch: straight short arrow into Phase 4 (the model_issue path)
arrow(ax, feedback_x, 5.85, X0 + W, 5.85, color="#c0392b", lw=1.3,
      style="-|>", linestyle="--")

# One combined label beside the vertical segment of the dashed line,
# covering both branch points, now that the narrower Decision Router
# (moved further right) leaves enough room for it.
ax.text(feedback_x + 0.12, 6.5, "re-enters Phase 3 or 4", ha="left", va="center",
        fontsize=7.2, color="#c0392b", weight="bold", style="italic")

# Phase 7
box(ax, X0, 2.35, W, 0.65,
    "Phase 7 — Report Generation & Final Output\n"
    "(stakeholder report, analytics dashboard, model recommendation, pipeline code)",
    C_CONTROL, fontsize=8.5)

# Vertical connectors down the spine (terminates at Phase 7 -- the
# "Trained Model + Report + Code" output box was removed per request)
spine_edges = [
    (10.05, 9.65), (9.05, 8.75), (8.15, p3_top + p3_h), (p3_top, 5.55 + 0.6),
    (5.55, 4.4 + 0.75), (4.4, 3.35 + 0.65), (3.35, 2.35 + 0.65),
]
cx = X0 + W / 2
for y_from, y_to in spine_edges:
    arrow(ax, cx, y_from, cx, y_to, color="#333333")

# Label the Phase 6 -> Phase 7 spine arrow as the "pass" branch, matching
# the "Fail" label on the horizontal arrow into Failure Diagnosis.
ax.text(cx + 0.15, 3.175, "Pass", ha="left", va="center",
        fontsize=7.5, color="#333333", weight="bold")

# ── Big wrapping box around Phase 1 -- Phase 7 (drawn behind everything) ────
# Represents "the system" as a whole, for the Episodic Decision Log's
# read/write arrow to point at. Does not include Input (raw dataset) since
# that is external to the system, not one of the seven phases.
wrap_pad = 0.15
wrap_x0 = X0 - wrap_pad
wrap_y0 = 2.35 - wrap_pad
wrap_w = W + 2 * wrap_pad
wrap_h = (9.05 + 0.6) - (2.35 - wrap_pad) + wrap_pad
ax.add_patch(FancyBboxPatch(
    (wrap_x0, wrap_y0), wrap_w, wrap_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor="none", edgecolor="#8a8a8a", linewidth=1.3,
    linestyle="-", zorder=1,
))

# ── Unsupervised branch (left of the spine) ───────────────────────────────────
# route_after_basic_cleaning() forks clustering / dimensionality-reduction
# runs straight to unsupervised_train and then phase7_final_output -- they
# never enter the train/test split, Phase 4, Phase 5 (supervised) or the
# Phase 6 threshold loop, since there is no held-out label to score against.
# Drawn on the left, mirroring the Phase 6 branch on the right, so the two
# connectors cross nothing.
UW = 2.85
UX = wrap_x0 - 0.55 - UW
ubx = UX + UW / 2                     # branch column x
box(ax, UX, 4.4, UW, 0.75,
    "Phase 5 — Unsupervised Execution\n"
    "(K-Means / PCA on the full dataset;\n"
    "silhouette, Davies-Bouldin, cluster profiles)",
    C_EXEC, fontsize=7.2)

# In: from the Basic Cleaning sub-box, left through the Phase 3 and wrapping
# borders, then down the branch column into the box.
bc_y = sub_y + 0.275
in_path = MplPath(
    [(X0 + 0.15, bc_y), (ubx, bc_y), (ubx, 4.4 + 0.75)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO],
)
ax.add_patch(FancyArrowPatch(path=in_path, arrowstyle="-|>", mutation_scale=13,
                              color="#333333", lw=1.4, zorder=2))
ax.text((wrap_x0 + ubx) / 2, bc_y + 0.1, "Unsupervised (no target)",
        ha="center", va="bottom", fontsize=7.5, color="#333333", weight="bold")

# Out: down the branch column, then right into Phase 7 -- bypassing 4-6.
out_path = MplPath(
    [(ubx, 4.4), (ubx, 2.675), (X0, 2.675)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO],
)
ax.add_patch(FancyArrowPatch(path=out_path, arrowstyle="-|>", mutation_scale=13,
                              color="#333333", lw=1.4, zorder=2))
ax.text(ubx + 0.12, 3.55, "bypasses Phases 4–6", ha="left", va="center",
        fontsize=7.2, color="#333333", style="italic")

# ── Cross-cutting infrastructure (right column, x = 8.5 to 14.6) ──────────────
IX, IW = 8.7, 5.9

# -- 3-Tier Decision Router (stacked-tier layout). Narrower than the full
# column width and centered within it, since the full-width version looked
# oversized. --
router_w = 3.5
router_top = 8.4  # keep the top fixed, close to the Episodic Log above
# Tier boxes shrunk (0.62 -> 0.42 height, 0.28 -> 0.18 gap) since a single
# centered line of text left a lot of empty space at the old size. Router_h
# shrinks to match instead of leaving dead space at the bottom.
tier_h_new, gap_new = 0.42, 0.18
router_h = 0.55 + 3 * tier_h_new + 2 * gap_new + 0.25  # title + tiers + gaps + bottom pad
router_y0 = router_top - router_h
router_x = IX + (IW - router_w) / 2
ax.add_patch(FancyBboxPatch(
    (router_x, router_y0), router_w, router_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor="#efecf9", edgecolor="#5b4fa8", linewidth=1.4, zorder=3,
))
ax.text(router_x + router_w / 2, router_y0 + router_h - 0.28,
        "3-Tier Decision Router",
        ha="center", va="center", fontsize=10, color="#5b4fa8", weight="bold", zorder=4)

tier_w, tier_h = router_w - 0.5, tier_h_new
tier_x = router_x + 0.25
tiers = ["Tier 1: Rule-based", "Tier 2: Statistical", "Tier 3: LLM"]
tier_top = router_y0 + router_h - 0.55
for i, label in enumerate(tiers):
    ty = tier_top - i * (tier_h + gap_new) - tier_h
    ax.add_patch(FancyBboxPatch(
        (tier_x, ty), tier_w, tier_h,
        boxstyle="round,pad=0.015,rounding_size=0.05",
        facecolor="white", edgecolor="#5b4fa8", linewidth=1.1, zorder=4,
    ))
    ax.text(tier_x + tier_w / 2, ty + tier_h / 2, label, ha="center", va="center",
            fontsize=9, color="#3a3170", weight="bold", zorder=5)
    if i < len(tiers) - 1:
        arrow(ax, tier_x + tier_w / 2, ty, tier_x + tier_w / 2, ty - gap_new + tier_h,
              color="#5b4fa8", lw=1.2)

# Direct arrow into Phase 3 (near the Deep Cleaning / Feature Engineering
# sub-boxes it actually serves) instead of a caption explaining the link.
# Raised from y=6.9 to 7.15 to put clear space between this and the red
# dashed feedback arrow just below it at y=6.85 -- they were too close.
arrow(ax, router_x, 7.15, X0 + W, 7.15, color="#5b4fa8", lw=1.3)

# -- Episodic Decision Log, positioned above the router at the top of the
# infra column so its read/write arrow (drawn below) stays clear of the
# y=[2.65, 6.85] band already used by the red targeted-revision feedback
# lines, avoiding any crossing. Box narrowed and shifted right (no longer
# spanning the full IW) so the longer arrow + label have room and don't
# overlap the box, as they did before.
log_w = IW - 1.6
log_x = IX + (IW - log_w)
log_y, log_h = 8.9, 0.8
box(ax, log_x, log_y, log_w, log_h,
    "Episodic Decision Log\nEvery routed decision + rationale, timestamped",
    C_INFRA, fontsize=8.8)

# Read/write arrow from the log into the Phase 1-7 wrapping box, routed at
# y=9.25 -- above the red feedback lines' max height (6.85), so it never
# crosses them.
rw_y = log_y + log_h / 2
# Bidirectional -- decisions are both written to the log and read back from
# it, not a one-way flow.
arrow(ax, log_x, rw_y, wrap_x0 + wrap_w, rw_y, color="#555555", lw=1.3,
      style="<|-|>")
ax.text((log_x + wrap_x0 + wrap_w) / 2, rw_y + 0.12, "Read and write decisions",
        ha="center", va="bottom", fontsize=7.2, color="#555555", style="italic")

plt.tight_layout()
out_png = FIGS / "fig0_system_architecture.png"
out_pdf = FIGS / "fig0_system_architecture.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
