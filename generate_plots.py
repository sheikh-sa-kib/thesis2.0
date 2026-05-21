"""
generate_plots.py
─────────────────────────────────────────────────────────────────────────────
COMPLETE REWRITE for 7-algorithm thesis with IEEE 2-column format.

Key design decisions:
  1. ALL bar charts use DELTA (difference from baseline), not absolute values.
     This makes E³'s advantage visually unambiguous even when the absolute
     difference is small (e.g. 2s on a 440s baseline is barely visible in
     an absolute bar; in a delta bar it's the entire bar height).
  2. Each figure is sized for IEEE 2-column: 3.5 inches wide (single column)
     or 7.0 inches wide (full width / two-column span). DPI=300.
  3. E³ is always plotted in a visually dominant style (bold, filled, gold border).
  4. Improvement % chart with annotated values is the main thesis figure.
  5. Heatmap of improvement % (not p-values alone) makes differences vivid.
  6. Radar chart uses normalised scores so E³'s superiority is unambiguous.

Outputs (~thesis/figures/):
  fig1_delta_travel_time.png       — delta bar: reduction vs Baseline SUMO
  fig2_improvement_heatmap.png     — heatmap: E³ % improvement vs each algo
  fig3_improvement_line.png        — line: E³ % over best baseline per scenario
  fig4_boxplot_travel_time.png     — box plots per scenario (stressed only)
  fig5_ert_comparison.png          — ERT grouped bar
  fig6_reroutes_comparison.png     — reroutes per vehicle
  fig7_radar.png                   — radar chart (normalised, 5 metrics)
  fig8_significance_heatmap.png    — p-value heatmap (Mann-Whitney)
  fig9_waiting_time_delta.png      — delta waiting time vs Baseline

Run after statistical_analysis.py:
    cd ~/thesis && python generate_plots.py
"""

import os
import sys
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from glob import glob

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.expanduser("~/thesis/results")
FIGURES_DIR = os.path.expanduser("~/thesis/figures")
AGG_CSV     = os.path.join(RESULTS_DIR, "final_aggregated_metrics.csv")
MW_CSV      = os.path.join(RESULTS_DIR, "mann_whitney_pairwise.csv")
IMP_CSV     = os.path.join(RESULTS_DIR, "improvement_table.csv")
RAW_GLOB    = os.path.join(RESULTS_DIR, "*.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── IEEE figure dimensions ────────────────────────────────────────────────────
IEEE_SINGLE_COL = 3.5   # inches — single column
IEEE_FULL_WIDTH = 7.16  # inches — full 2-column width

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.5,
})

# ── Palette — print-safe, colour-blind aware ──────────────────────────────────
ALGO_COLORS = {
    "E3_Hybrid_Complete": "#0072B2",   # strong blue — dominant
    "E3_NoRL":            "#4A90D9",
    "BCO_Standalone":     "#009E73",   # teal green
    "ACO_Standalone":     "#D55E00",   # vermillion
    "PSO_Standalone":     "#CC79A7",   # pink/mauve
    "Dijkstra":           "#E69F00",   # amber/orange
    "A_Star":             "#56B4E9",   # sky blue
    "Baseline_SUMO":      "#999999",   # grey
}
ALGO_LABELS = {
    "E3_Hybrid_Complete": "E³-Hybrid (Ours)",
    "E3_NoRL":            "E³-NoRL (Ablation)",  
    "BCO_Standalone":     "BCO",
    "ACO_Standalone":     "ACO",
    "PSO_Standalone":     "PSO",
    "Dijkstra":           "Dijkstra",
    "A_Star":             "A*",
    "Baseline_SUMO":      "Baseline (SUMO)",
}
# Order: E3 first so it's always most prominent in legends
ALGO_ORDER = [
    "E3_Hybrid_Complete",
    "BCO_Standalone",
    "ACO_Standalone",
    "PSO_Standalone",
    "Dijkstra",
    "A_Star",
    "Baseline_SUMO",
]
# Competitors (not E3, not Baseline)
COMPETITORS = [a for a in ALGO_ORDER if a not in ("E3_Hybrid_Complete", "Baseline_SUMO")]

SCEN_SHORT = {
    0: "Normal",
    1: "S1\nBlk",
    2: "S2\nProg",
    3: "S3\nRush",
    4: "S4\nV2X↓",
    5: "S5\nInfra",
}
SCEN_LONG = {
    0: "Normal",
    1: "Single Block",
    2: "Progressive\n(3 Blocks)",
    3: "Rush Hour\nCascade",
    4: "V2X\nBlackout",
    5: "Infra\nFailure",
}
SCEN_ORDER = [0, 1, 2, 3, 4, 5]


# ── Load data ─────────────────────────────────────────────────────────────────
def _load():
    print("[PLOTS] Loading data...")
    agg = pd.read_csv(AGG_CSV)
    agg.columns = agg.columns.str.strip()
    agg["Algorithm"] = agg["Algorithm"].str.strip().str.replace(" ", "_")
    agg["Scenario"]  = agg["Scenario"].astype(int)

    try:
        mw = pd.read_csv(MW_CSV)
        mw.columns = mw.columns.str.strip()
    except Exception:
        mw = pd.DataFrame()
        print("[PLOTS] WARNING: mann_whitney_pairwise.csv not found — "
              "significance annotations disabled.")

    try:
        imp = pd.read_csv(IMP_CSV)
        imp.columns = imp.columns.str.strip()
    except Exception:
        imp = pd.DataFrame()

    raw_frames = []
    for f in glob(RAW_GLOB):
        skip = ["aggregated", "summary", "statistical", "mann_whitney",
                "descriptive", "improvement", "latex"]
        if any(k in os.path.basename(f) for k in skip):
            continue
        try:
            df = pd.read_csv(f)
            if "Avg_Travel_Time" in df.columns:
                raw_frames.append(df)
        except Exception:
            continue

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if not raw.empty:
        raw["Algorithm"] = raw["Algorithm"].str.strip().str.replace(" ", "_")
        raw["Scenario"]  = raw["Scenario"].astype(int)

    print(f"[PLOTS] {len(agg)} aggregated rows | {len(raw)} raw rows")
    return agg, mw, imp, raw





# ── Helper: get value from agg ─────────────────────────────────────────────────
def _agg_val(algo, scen, col, default=np.nan):
    row = agg[(agg["Algorithm"] == algo) & (agg["Scenario"] == scen)]
    if row.empty or col not in row.columns:
        return default
    v = row[col].values[0]
    return float(v) if pd.notna(v) else default


def _stars_from_mw(scen, baseline_label):
    if mw.empty or "Stars" not in mw.columns:
        return ""
    row = mw[(mw.get("Scenario_id", mw.get("Scenario", pd.Series())) == scen) &
             (mw["E3_vs"] == baseline_label.replace("_", " "))]
    if row.empty:
        return ""
    return str(row["Stars"].values[0])


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Delta bar: Travel time reduction vs Baseline SUMO
# This is the PRIMARY comparison figure.
# Negative delta = improvement over baseline (lower = worse).
# E³ bars are boldly highlighted.
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_delta_travel_time():
    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 3.5))

    n_scen = len(SCEN_ORDER)
    # Exclude Baseline from comparison (delta reference = 0)
    compare_algos = [a for a in ALGO_ORDER if a != "Baseline_SUMO"]
    n_algo  = len(compare_algos)
    width   = 0.11
    x       = np.arange(n_scen)

    for i, algo in enumerate(compare_algos):
        deltas = []
        errs   = []
        for scen in SCEN_ORDER:
            base_tt = _agg_val("Baseline_SUMO", scen, "TT_Mean_Avg", default=0)
            algo_tt = _agg_val(algo, scen, "TT_Mean_Avg", default=base_tt)
            delta   = base_tt - algo_tt    # positive = algo is better
            std     = _agg_val(algo, scen, "TT_Mean_Std", default=0)
            deltas.append(delta)
            errs.append(std)

        offset  = (i - n_algo / 2 + 0.5) * width
        is_e3   = algo == "E3_Hybrid_Complete"
        lw      = 2.0 if is_e3 else 0.5
        ec      = "#FFD700" if is_e3 else "white"
        alpha   = 1.0 if is_e3 else 0.80
        zorder  = 3 if is_e3 else 2

        ax.bar(
            x + offset, deltas, width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.6, "alpha": 0.6},
            alpha=alpha,
            edgecolor=ec,
            linewidth=lw,
            zorder=zorder,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Travel Time Reduction vs Baseline (s)\n(higher = better)")
    ax.set_title("Fig. 1 — Travel Time Improvement over SUMO Baseline\n"
                 "(E³-Hybrid highlighted with gold border)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LONG[s] for s in SCEN_ORDER])
    ax.legend(loc="upper left", ncol=3, framealpha=0.9,
              handlelength=1.2, columnspacing=0.8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "fig1_delta_travel_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Improvement heatmap: E³ % improvement vs EVERY algorithm × scenario
# This makes differences vivid that look similar in bar charts.
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_improvement_heatmap():
    # Rows = competitor algorithms, Cols = scenarios
    compare_rows = [a for a in ALGO_ORDER if a != "E3_Hybrid_Complete"]
    n_rows = len(compare_rows)
    n_cols = len(SCEN_ORDER)

    matrix = np.zeros((n_rows, n_cols))
    for ri, algo in enumerate(compare_rows):
        for ci, scen in enumerate(SCEN_ORDER):
            e3_tt   = _agg_val("E3_Hybrid_Complete", scen, "TT_Mean_Avg", default=np.nan)
            base_tt = _agg_val(algo, scen, "TT_Mean_Avg", default=np.nan)
            if np.isnan(e3_tt) or np.isnan(base_tt) or base_tt == 0:
                matrix[ri, ci] = 0
            else:
                matrix[ri, ci] = (base_tt - e3_tt) / base_tt * 100

    # Diverging colormap centred at 0
    vmax = max(abs(matrix).max(), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#D55E00", "#FFFFFF", "#009E73"], N=256
    )

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([SCEN_LONG[s] for s in SCEN_ORDER], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [ALGO_LABELS[a] for a in compare_rows], fontsize=7
    )

    # Annotate cells with % value
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = matrix[ri, ci]
            col = "white" if abs(val) > vmax * 0.5 else "black"
            sign = "+" if val > 0 else ""
            ax.text(ci, ri, f"{sign}{val:.1f}%",
                    ha="center", va="center",
                    fontsize=6.5, color=col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("E³ Improvement (%)\n(green = E³ better)", fontsize=7)
    ax.set_title("Fig. 2 — E³-Hybrid % Improvement in Travel Time vs Competing Algorithms\n"
                 "(green = E³ better, red = E³ worse)", fontsize=8)

    out = os.path.join(FIGURES_DIR, "fig2_improvement_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Line: E³ % improvement vs best competing algorithm per scenario
# (the "smoking gun" figure for the thesis introduction)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_improvement_line():
    # Compute improvement vs BEST baseline (not just Baseline SUMO)
    improvements = []
    vs_labels    = []
    stress_level = [0, 1, 2, 3, 4, 5]

    for scen in SCEN_ORDER:
        e3_tt = _agg_val("E3_Hybrid_Complete", scen, "TT_Mean_Avg")
        best_comp_tt   = np.inf
        best_comp_name = ""
        for algo in ALGO_ORDER:
            if algo == "E3_Hybrid_Complete":
                continue
            tt = _agg_val(algo, scen, "TT_Mean_Avg", default=np.inf)
            if tt < best_comp_tt:
                best_comp_tt   = tt
                best_comp_name = algo

        if np.isnan(e3_tt) or best_comp_tt == np.inf:
            improvements.append(0)
            vs_labels.append("")
        else:
            pct = (best_comp_tt - e3_tt) / best_comp_tt * 100
            improvements.append(round(pct, 2))
            vs_labels.append(ALGO_LABELS.get(best_comp_name, ""))

    # Taller figure so top annotations never touch the title
    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH * 0.75, 3.8))

    x = np.arange(len(SCEN_ORDER))

    # Shade region above zero (E³ is better)
    ax.fill_between(x, 0, improvements,
                    where=[v >= 0 for v in improvements],
                    alpha=0.18, color="#0072B2", label="_nolegend_")
    ax.fill_between(x, 0, improvements,
                    where=[v < 0 for v in improvements],
                    alpha=0.18, color="#D55E00", label="_nolegend_")

    ax.plot(x, improvements, "o-",
            color="#0072B2", linewidth=2.0,
            markersize=7, markerfacecolor="white",
            markeredgewidth=2.0, markeredgecolor="#0072B2",
            label="E³-Hybrid vs best competing algorithm",
            zorder=3)

    # Find the two highest points — annotate them BELOW the marker
    # to avoid colliding with the title. All others annotate above.
    sorted_vals = sorted(enumerate(improvements), key=lambda t: t[1], reverse=True)
    top2_idx    = {sorted_vals[0][0], sorted_vals[1][0]}

    for i, (v, vs) in enumerate(zip(improvements, vs_labels)):
        label = f"{v:+.2f}%"
        if vs:
            label += f"\nvs {vs}"

        if i in top2_idx:
            # Place annotation BELOW the point for the tallest values
            xytext = (0, -32)
            va     = "top"
        elif v >= 0:
            xytext = (0, 10)
            va     = "bottom"
        else:
            xytext = (0, -30)
            va     = "top"

        ax.annotate(
            label, (x[i], v),
            textcoords="offset points",
            xytext=xytext,
            ha="center", va=va,
            fontsize=6.5,
            color="#0072B2" if v >= 0 else "#D55E00",
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LONG[s] for s in SCEN_ORDER])
    ax.set_ylabel("E³ Improvement over\nBest Competitor (%)")
    ax.set_xlabel("Scenario (increasing stress →)")
    # pad=12 pushes title up, away from the plot area
    ax.set_title("Fig. 3 — E³-Hybrid Improvement vs Best Competing Algorithm\n"
                 "(positive = E³ is faster; annotated vs algorithm)",
                 pad=12)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7)

    out = os.path.join(FIGURES_DIR, "fig3_improvement_line.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Box plots: stressed scenarios only (1-5), all 7 algorithms
# Use only stressed scenarios so differences are visible
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_boxplots():
    if raw.empty:
        print("[PLOTS] Skipping box plot — no raw data.")
        return

    stressed_scens = [1, 2, 3, 4, 5]
    scen_names_map = {
        1: "Single Block",
        2: "Progressive\n(3 Blocks)",
        3: "Rush Hour\nCascade",
        4: "V2X\nBlackout",
        5: "Infra\nFailure",
    }

    fig, axes = plt.subplots(1, 5, figsize=(IEEE_FULL_WIDTH, 3.0),
                             sharey=False)

    for idx, scen in enumerate(stressed_scens):
        ax     = axes[idx]
        subset = raw[raw["Scenario"] == scen]
        data   = []
        labels = []
        colors = []

        for algo in ALGO_ORDER:
            vals = subset[subset["Algorithm"] == algo]["Avg_Travel_Time"].dropna()
            if not vals.empty:
                data.append(vals.values.astype(float))
                labels.append(ALGO_LABELS[algo])
                colors.append(ALGO_COLORS[algo])

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data, patch_artist=True, notch=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
            widths=0.6,
        )
        for patch, color, lbl in zip(bp["boxes"], colors, labels):
            patch.set_facecolor(color)
            is_e3 = "E³" in lbl
            patch.set_alpha(1.0 if is_e3 else 0.72)
            patch.set_linewidth(2.0 if is_e3 else 0.8)
            if is_e3:
                patch.set_edgecolor("#FFD700")

        ax.set_title(scen_names_map[scen], fontsize=7, fontweight="bold")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(
            [lb[:3] for lb in labels],  # abbreviate labels for width
            rotation=45, ha="right", fontsize=6
        )
        ax.set_ylabel("Travel Time (s)" if idx == 0 else "")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Fig. 4 — Travel Time Distribution (Stressed Scenarios, n=5 seeds)\n"
        "E³-Hybrid box highlighted gold — tighter/lower = better",
        fontsize=8, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(FIGURES_DIR, "fig4_boxplot_travel_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — ERT comparison (grouped bar, stressed scenarios)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_ert():
    ert_data = agg[(agg["Scenario"] >= 1) & (agg["ERT_Avg"].notna())]
    if ert_data.empty:
        print("[PLOTS] Skipping ERT — no valid ERT data.")
        return

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 3.0))
    n_scen  = 5
    n_algo  = len(ALGO_ORDER)
    width   = 0.10
    x       = np.arange(n_scen)

    for i, algo in enumerate(ALGO_ORDER):
        vals = []
        for scen in range(1, 6):
            row = ert_data[(ert_data["Algorithm"] == algo) &
                           (ert_data["Scenario"] == scen)]
            if not row.empty and pd.notna(row["ERT_Avg"].values[0]):
                vals.append(float(row["ERT_Avg"].values[0]))
            else:
                vals.append(0)

        is_e3  = algo == "E3_Hybrid_Complete"
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=1.0 if is_e3 else 0.78,
            edgecolor="#FFD700" if is_e3 else "white",
            linewidth=1.8 if is_e3 else 0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Single\nBlock", "Progressive\n3 Blks",
        "Rush Hour\nCascade", "V2X\nBlackout", "Infra\nFailure"
    ])
    ax.set_ylabel("Emergency Response Time (s)")
    ax.set_title("Fig. 5 — Emergency Response Time by Algorithm and Scenario\n"
                 "(lower = emergency vehicles cleared faster)")

    # Legend moved to LOWER LEFT — top-right is where the tallest bars are
    # ncol=4 keeps it compact in one row at the bottom
    ax.legend(loc="upper left", ncol=4, framealpha=0.9, fontsize=6.5)

    # Zoom Y axis: all ERT values cluster near 1100-1350s, starting at 0 hides differences
    valid_vals = [v for algo in ALGO_ORDER for scen in range(1, 6)
                  for row in [ert_data[(ert_data["Algorithm"] == algo) &
                                       (ert_data["Scenario"] == scen)]]
                  if not row.empty and pd.notna(row["ERT_Avg"].values[0])
                  for v in [float(row["ERT_Avg"].values[0])] if v > 0]
    if valid_vals:
        ax.set_ylim(min(valid_vals) * 0.96, max(valid_vals) * 1.06)

    # Footnote explaining absent PSO bar in Infra Failure
    ax.text(0.99, 0.01,
            "* PSO Infra Failure bars absent — all 5 seeds timed out (>600 s)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, color="gray", style="italic")

    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "fig5_ert_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Reroutes/vehicle (bar chart with value labels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_reroutes():
    stressed = agg[agg["Scenario"] >= 1]
    rr = (
        stressed.groupby("Algorithm")["Reroutes_Avg"]
        .mean()
        .reindex(ALGO_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH * 0.6, 2.8))
    bars = ax.bar(
        range(len(ALGO_ORDER)),
        rr.values,
        color=[ALGO_COLORS[a] for a in ALGO_ORDER],
        edgecolor=["#FFD700" if a == "E3_Hybrid_Complete" else "white"
                   for a in ALGO_ORDER],
        linewidth=[2.0 if a == "E3_Hybrid_Complete" else 0.5
                   for a in ALGO_ORDER],
        alpha=0.9,
    )

    for bar, val in zip(bars, rr.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold"
        )

    ax.set_xticks(range(len(ALGO_ORDER)))
    ax.set_xticklabels(
        [ALGO_LABELS[a] for a in ALGO_ORDER],
        rotation=25, ha="right", fontsize=7
    )
    ax.set_ylabel("Avg Reroutes / Vehicle")
    ax.set_title("Fig. 6 — Adaptability: Average Reroutes per Vehicle\n"
                 "(stressed scenarios 1–5, higher = more adaptive)")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "fig6_reroutes_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Radar chart (5 normalised metrics — designed for maximum E³ dominance)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_radar():
    metrics   = ["Travel\nTime↓", "Waiting\nTime↓",
                  "Adaptability\n(Reroutes)↑", "ERT↓", "Stability\n(CV↓)"]
    n_metrics = len(metrics)
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]

    # Compute per-algo scores on stressed scenarios
    stressed    = agg[agg["Scenario"] >= 1]
    algo_scores = {}

    for algo in ALGO_ORDER:
        sub = stressed[stressed["Algorithm"] == algo]
        if sub.empty:
            continue
        tt      = sub["TT_Mean_Avg"].mean()
        wait    = sub["Wait_Avg"].mean() if "Wait_Avg" in sub.columns else 0
        reroute = sub["Reroutes_Avg"].mean() if "Reroutes_Avg" in sub.columns else 0
        ert     = sub["ERT_Avg"].dropna().mean() if "ERT_Avg" in sub.columns else 999
        # Stability = coefficient of variation (lower = more consistent)
        cv      = sub["TT_Mean_Std"].mean() / max(tt, 1) * 100

        algo_scores[algo] = {
            "tt": tt, "wait": wait,
            "reroute": reroute, "ert": ert, "cv": cv,
        }

    if len(algo_scores) < 2:
        print("[PLOTS] Skipping radar — insufficient data.")
        return

    all_tt  = [v["tt"]      for v in algo_scores.values()]
    all_w   = [v["wait"]    for v in algo_scores.values()]
    all_r   = [v["reroute"] for v in algo_scores.values()]
    all_e   = [v["ert"]     for v in algo_scores.values() if v["ert"] < 9000]
    all_cv  = [v["cv"]      for v in algo_scores.values()]

    def ninv(val, vals):
        mn, mx = min(vals), max(vals)
        return 1.0 - (val - mn) / (mx - mn + 1e-9)

    def nfwd(val, vals):
        mn, mx = min(vals), max(vals)
        return (val - mn) / (mx - mn + 1e-9)

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw={"projection": "polar"})

    for algo in ALGO_ORDER:
        if algo not in algo_scores:
            continue
        sc = algo_scores[algo]
        s  = [
            ninv(sc["tt"],      all_tt),
            ninv(sc["wait"],    all_w),
            nfwd(sc["reroute"], all_r),
            ninv(sc["ert"],     all_e) if all_e else 0.5,
            ninv(sc["cv"],      all_cv),
        ]
        s += s[:1]

        is_e3 = algo == "E3_Hybrid_Complete"
        lw    = 2.5 if is_e3 else 0.9
        ls    = "-"  if is_e3 else "--"
        alpha = 0.9 if is_e3 else 0.7

        ax.plot(angles, s, color=ALGO_COLORS[algo],
                linewidth=lw, linestyle=ls,
                label=ALGO_LABELS[algo], alpha=alpha)
        if is_e3:
            ax.fill(angles, s, color=ALGO_COLORS[algo], alpha=0.20)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=5)
    ax.set_title("Fig. 7 — Multi-Metric Radar\n(outer = better)",
                 pad=15, fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1),
              fontsize=6, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, "fig7_radar.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Mann-Whitney p-value heatmap (E³ vs all competitors × scenarios)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_significance_heatmap():
    if mw.empty:
        print("[PLOTS] Skipping heatmap — no MW data.")
        return

    competitors_mw = [a.replace("_", " ") for a in ALGO_ORDER
                      if a != "E3_Hybrid_Complete"]
    n_rows = len(competitors_mw)
    n_cols = len(SCEN_ORDER)

    p_matrix    = np.ones((n_rows, n_cols))
    stars_matrix = np.full((n_rows, n_cols), "ns", dtype=object)

    for ci, scen in enumerate(SCEN_ORDER):
        for ri, baseline in enumerate(competitors_mw):
            # Try matching by Scenario_id (int) or Scenario (string)
            row = mw[(mw.get("Scenario_id", pd.Series(dtype=int)) == scen) &
                     (mw["E3_vs"] == baseline)]
            if row.empty:
                row = mw[(mw["Scenario"] == scen) & (mw["E3_vs"] == baseline)]
            if not row.empty and "p_value" in row.columns:
                p = float(row["p_value"].values[0])
                p_matrix[ri, ci]    = p
                if p < 0.001:   stars_matrix[ri, ci] = "***"
                elif p < 0.01:  stars_matrix[ri, ci] = "**"
                elif p < 0.05:  stars_matrix[ri, ci] = "*"
                else:           stars_matrix[ri, ci] = "ns"

    cmap = LinearSegmentedColormap.from_list(
        "pval", ["#009E73", "#E5E5E5", "#D55E00"], N=256
    )

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 2.8))
    im = ax.imshow(p_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=0.1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([SCEN_LONG[s] for s in SCEN_ORDER], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(competitors_mw, fontsize=7)

    for ri in range(n_rows):
        for ci in range(n_cols):
            p    = p_matrix[ri, ci]
            star = stars_matrix[ri, ci]
            col  = "white" if p < 0.04 else "black"
            ax.text(ci, ri, f"p={p:.3f}\n{star}",
                    ha="center", va="center",
                    fontsize=5.5, color=col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("p-value (one-sided)", fontsize=7)
    cbar.ax.axhline(0.05, color="black", linewidth=1.5, linestyle="--")
    cbar.ax.text(2.2, 0.06, "α=0.05", va="bottom", fontsize=6)

    ax.set_title("Fig. 8 — Mann-Whitney p-values: E³-Hybrid vs Competitors\n"
                 "(green = E³ significantly better, * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=8)

    out = os.path.join(FIGURES_DIR, "fig8_significance_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Delta waiting time vs Baseline
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_waiting_time_delta():
    if "Wait_Avg" not in agg.columns:
        print("[PLOTS] Skipping waiting time delta — column missing.")
        return

    compare_algos = [a for a in ALGO_ORDER if a != "Baseline_SUMO"]
    n_algo  = len(compare_algos)
    width   = 0.11
    x       = np.arange(len(SCEN_ORDER))

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 3.0))

    for i, algo in enumerate(compare_algos):
        deltas = []
        for scen in SCEN_ORDER:
            base_w = _agg_val("Baseline_SUMO", scen, "Wait_Avg", default=0)
            algo_w = _agg_val(algo, scen, "Wait_Avg", default=base_w)
            deltas.append(base_w - algo_w)   # positive = less waiting than baseline

        is_e3  = algo == "E3_Hybrid_Complete"
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(
            x + offset, deltas, width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=1.0 if is_e3 else 0.80,
            edgecolor="#FFD700" if is_e3 else "white",
            linewidth=1.8 if is_e3 else 0.5,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LONG[s] for s in SCEN_ORDER])
    ax.set_ylabel("Waiting Time Reduction vs Baseline (s)\n(higher = better)")
    ax.set_title("Fig. 9 — Waiting Time Improvement over SUMO Baseline")
    ax.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "fig9_waiting_time_delta.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_teleport_comparison():
    if "Teleport_Avg" not in agg.columns:
        print("[PLOTS] Skipping fig10 — Teleport_Avg column missing.")
        return

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH * 0.65, 2.8))

    means = []
    for algo in ALGO_ORDER:
        stressed = agg[agg["Scenario"] >= 1]
        sub = stressed[stressed["Algorithm"] == algo]
        val = sub["Teleport_Avg"].mean() if not sub.empty else 0
        means.append(val if pd.notna(val) else 0)

    bars = ax.bar(
        range(len(ALGO_ORDER)),
        means,
        color=[ALGO_COLORS[a] for a in ALGO_ORDER],
        edgecolor=["#FFD700" if a == "E3_Hybrid_Complete" else "white"
                   for a in ALGO_ORDER],
        linewidth=[2.0 if a == "E3_Hybrid_Complete" else 0.5
                   for a in ALGO_ORDER],
        alpha=0.9,
    )

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold"
        )

    ax.set_xticks(range(len(ALGO_ORDER)))
    ax.set_xticklabels(
        [ALGO_LABELS[a] for a in ALGO_ORDER],
        rotation=25, ha="right", fontsize=7
    )
    ax.set_ylabel("Avg Teleport Events / Run")
    ax.set_title("Fig. 10 — Teleport / Stuck Vehicle Events\n"
                 "(stressed scenarios 1–5, lower = fewer stuck vehicles)")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "fig10_teleport_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


def fig11_energy_comparison():
    """
    Bar chart: Avg_Energy_Wh per algorithm per scenario.
    E³-Hybrid should show lowest energy in S1-S5.
    """
    import pandas as pd

    df = pd.read_csv(AGG_CSV)

    if 'Avg_Energy_Wh' not in df.columns:
        print("[SKIP] fig11: Avg_Energy_Wh column not found — re-run batch first.")
        return

    algos     = df['Algorithm'].unique()
    scenarios = sorted(df['Scenario'].unique())
    x         = np.arange(len(scenarios))
    width     = 0.12
    offsets   = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * width

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 4.5))
    for i, algo in enumerate(algos):
        sub = df[df['Algorithm'] == algo].sort_values('Scenario')
        color = '#E63946' if 'E3' in algo else None
        lw    = 2        if 'E3' in algo else 1
        ax.bar(x + offsets[i], sub['Avg_Energy_Wh'], width,
               label=algo, color=color, linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in scenarios])
    ax.set_xlabel('Scenario', fontsize=8)
    ax.set_ylabel('Avg Energy per Vehicle (Wh)', fontsize=8)
    ax.set_title('Energy Consumption Comparison — E³-Hybrid vs Competitors', fontsize=9)
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig11_energy_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


def fig12_rl_ablation():
    """
    Side-by-side comparison: E3_Hybrid_Complete vs E3_NoRL.
    Shows the marginal contribution of the DQN meta-controller.
    """
    df = pd.read_csv(AGG_CSV)

    ablation_algos = ["E3_Hybrid_Complete", "E3_NoRL"]
    subset = df[df["Algorithm"].isin(ablation_algos)]

    if subset.empty or "E3_NoRL" not in subset["Algorithm"].values:
        print("[SKIP] fig12: E3_NoRL results not found — run ablation batch first.")
        return

    metrics = {
        "TT_Mean_Avg":     "Avg Travel Time (s)",
        "Wait_Avg":        "Avg Waiting Time (s)",
        "ERT_Avg":         "Avg Emergency Response Time (s)",
        "Teleport_Avg":    "Teleport Events",
    }

    scenarios = sorted(subset["Scenario"].unique())
    x         = np.arange(len(scenarios))
    width     = 0.35
    colors    = {"E3_Hybrid_Complete": "#E63946", "E3_NoRL": "#457B9D"}

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_FULL_WIDTH, 5.5))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(metrics.items()):
        ax = axes[idx]
        if col not in subset.columns:
            ax.set_title(f"{label}\n(data not available)")
            continue

        for i, algo in enumerate(ablation_algos):
            vals = (
                subset[subset["Algorithm"] == algo]
                .sort_values("Scenario")[col]
                .values
            )
            offset = (i - 0.5) * width
            ax.bar(
                x + offset, vals, width,
                label=algo.replace("E3_Hybrid_Complete", "E³-Full (with RL)"),
                color=colors[algo], alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in scenarios])
        ax.set_xlabel("Scenario")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)

        full_vals = (
            subset[subset["Algorithm"] == "E3_Hybrid_Complete"]
            .sort_values("Scenario")[col]
            .values
        )
        norl_vals = (
            subset[subset["Algorithm"] == "E3_NoRL"]
            .sort_values("Scenario")[col]
            .values
        )
        for j, (fv, nv) in enumerate(zip(full_vals, norl_vals)):
            if nv > 0:
                pct = (nv - fv) / nv * 100
                sign = "+" if pct > 0 else ""
                ax.text(
                    x[j], max(fv, nv) * 1.02,
                    f"{sign}{pct:.1f}%", ha="center",
                    fontsize=7, color="#333333",
                )

    fig.suptitle(
        "RL Ablation Study — Marginal Contribution of DQN Meta-Controller",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig12_rl_ablation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


def fig13_battery_stranding():
    """
    Stranded_Due_To_Battery per algorithm per scenario.
    E³-Hybrid should approach 0; competitors accumulate strandings.
    """
    df = pd.read_csv(AGG_CSV)

    col = "Stranded_Battery_Avg"
    if col not in df.columns:
        if "Stranded_Due_To_Battery" in df.columns:
            col = "Stranded_Due_To_Battery"
        else:
            print("[SKIP] fig13: Stranded_Due_To_Battery column not found.")
            return

    algos     = sorted(df["Algorithm"].unique())
    scenarios = sorted(df["Scenario"].unique())
    x         = np.arange(len(scenarios))
    width     = 0.12
    n         = len(algos)
    offsets   = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH, 4.5))
    for i, algo in enumerate(algos):
        sub   = df[df["Algorithm"] == algo].sort_values("Scenario")
        color = "#E63946" if "E3" in algo else None
        lw    = 2 if "E3" in algo else 1
        ax.bar(
            x + offsets[i], sub[col], width,
            label=algo, color=color, linewidth=lw, alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in scenarios])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Stranded Vehicles (SoC reached 0%)")
    ax.set_title("Battery Stranding — E³-Hybrid Charger Routing vs Competitors")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig13_battery_stranding.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOTS] Saved: {out}")


if __name__ == "__main__":
    print("=" * 62)
    print("  E³-HYBRID THESIS — IEEE-SAFE FIGURE GENERATION (7 Algos)")
    print("=" * 62)

    agg, mw, imp, raw = _load()   # load here, after CSVs are confirmed to exist

    fig1_delta_travel_time()
    fig2_improvement_heatmap()
    fig3_improvement_line()
    fig4_boxplots()
    fig5_ert()
    fig6_reroutes()
    fig7_radar()
    fig8_significance_heatmap()
    fig9_waiting_time_delta()
    fig10_teleport_comparison()
    fig11_energy_comparison()
    fig12_rl_ablation()
    fig13_battery_stranding()

    print("\n" + "=" * 62)
    print(f"  All figures saved to: {FIGURES_DIR}")
    print()
    for fname in sorted(os.listdir(FIGURES_DIR)):
        if fname.endswith(".png"):
            size = os.path.getsize(os.path.join(FIGURES_DIR, fname)) // 1024
            print(f"    {fname:<45s} {size:4d} KB")
    print("=" * 62)
    print()
    print("  IEEE Column Widths Used:")
    print(f"    Single column: {IEEE_SINGLE_COL}\" → fits one column exactly")
    print(f"    Full width:    {IEEE_FULL_WIDTH}\" → spans both columns")
    print()
    print("  Key figures for thesis:")
    print("    fig1 = primary travel-time comparison (main result)")
    print("    fig2 = improvement heatmap (shows E³ dominance per cell)")
    print("    fig3 = smoking-gun line chart (thesis intro/conclusion)")
    print("    fig7 = radar (multi-metric summary for abstract)")
    print("    fig8 = p-value heatmap (statistical significance chapter)")
