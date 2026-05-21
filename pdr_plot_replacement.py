"""
Reusable PDR comparison plot.

The figure is intentionally data-driven: pass in the PDR values computed from
the latest simulation logs instead of editing a hard-coded table.
"""

import os
import re
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCENARIOS_ALL      = ["Normal", "Single\nBlock", "Progressive",
                       "Rush\nHour", "V2X\nBlackout", "Infra\nFailure"]
SCENARIOS_NO_BLK   = ["Normal", "Single\nBlock", "Progressive",
                       "Rush\nHour", "Infra\nFailure"]
INDICES_NO_BLK     = [0, 1, 2, 3, 5]   # column indices excluding V2X Blackout (col 4)
BLACKOUT_IDX       = 4                  # V2X Blackout column

COLORS = {
    "E\u00b3-Hybrid": "#E84545",
    "Dijkstra":       "#2B9EB3",
    "A*":             "#F5A623",
    "BCO":            "#7B68EE",
    "ACO":            "#4CAF50",
    "PSO":            "#26C6DA",
    "Baseline":       "#9E9E9E",
}
MARKERS = {
    "E\u00b3-Hybrid": "o",
    "Dijkstra":       "s",
    "A*":             "^",
    "BCO":            "D",
    "ACO":            "v",
    "PSO":            "P",
    "Baseline":       "X",
}
LINESTYLES = {
    "E\u00b3-Hybrid": "-",
    "Dijkstra":       "--",
    "A*":             "-.",
    "BCO":            ":",
    "ACO":            (0, (3, 1, 1, 1)),
    "PSO":            (0, (5, 2)),
    "Baseline":       (0, (1, 1)),
}

IEEE_FULL = 7.16    # inches — full two-column IEEE width
FONT_TITLE  = 9
FONT_LABEL  = 8
FONT_TICK   = 7
FONT_LEGEND = 7
FONT_ANNOT  = 6.5
LW = 1.6            # line width
MS = 5              # marker size


def _finite(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def pdr_data_from_analysis(data, algos, scenarios, seeds, labels, compute_pdr, seed_mean):
    """Convert analyze_v2x.py's raw log rows into {label: [PDR%, ...]}."""
    pdr_data = {}
    for algo in algos:
        values = []
        for scen in scenarios:
            seed_values = [
                compute_pdr(data[algo][scen][seed])
                for seed in seeds
                if data[algo][scen][seed]
            ]
            values.append(seed_mean(seed_values) * 100 if seed_values else np.nan)
        pdr_data[labels[algo]] = values
    return pdr_data


def load_pdr_data_from_sca(sca_path: str):
    """Read pdr_mean values from an OMNeT++ .sca file, returned as percentages."""
    pdr_data = {}
    pattern = re.compile(r"^scalar V2X\.(?P<algo>.+)\.Scen(?P<scen>\d+) pdr_mean\s+(?P<value>[0-9.]+)")

    with open(sca_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            algo = match.group("algo").replace("_", " ")
            scen = int(match.group("scen"))
            value = float(match.group("value")) * 100
            pdr_data.setdefault(algo, [np.nan] * len(SCENARIOS_ALL))
            if scen < len(SCENARIOS_ALL):
                pdr_data[algo][scen] = value

    return pdr_data


def plot_pdr_comparison(out_path: str, pdr_data: dict[str, list[float]]) -> None:
    """
    Two-panel figure:
      Left  — PDR for all scenarios EXCEPT V2X Blackout.
              Y-axis zooms to the latest data range.
      Right — PDR in V2X Blackout only, with an axis based on latest data.
              Horizontal bars show each algorithm's value with an annotation.
    """
    if not pdr_data:
        raise ValueError("No PDR data available to plot.")

    missing = {
        algo: len(values)
        for algo, values in pdr_data.items()
        if len(values) != len(SCENARIOS_ALL)
    }
    if missing:
        raise ValueError(
            "Each algorithm must provide one PDR value per scenario; "
            f"bad lengths: {missing}"
        )

    plt.rcParams.update({
        "font.family":  "serif",
        "font.size":    FONT_TICK,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.dpi":   300,
    })

    fig = plt.figure(figsize=(IEEE_FULL, 3.4))
    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[3.2, 1],   # left panel wider (5 scenarios vs 1)
        wspace=0.38,
    )
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ── LEFT PANEL: line chart, non-blackout scenarios ────────────────────
    x = np.arange(len(SCENARIOS_NO_BLK))

    for algo, values in pdr_data.items():
        y = [values[i] for i in INDICES_NO_BLK]
        lw_use = 2.2 if algo == "E\u00b3-Hybrid" else LW
        zord   = 5   if algo == "E\u00b3-Hybrid" else 2
        ax_left.plot(
            x, y,
            color     = COLORS.get(algo, "#333333"),
            linestyle = LINESTYLES.get(algo, "-"),
            linewidth = lw_use,
            marker    = MARKERS.get(algo, "o"),
            markersize= MS,
            label     = algo,
            zorder    = zord,
        )

    non_blackout_values = _finite(
        value
        for values in pdr_data.values()
        for i, value in enumerate(values)
        if i in INDICES_NO_BLK
    )
    if non_blackout_values:
        y_min = max(0, min(non_blackout_values) - 1.0)
        y_max = min(100, max(non_blackout_values) + 1.0)
        if y_max - y_min < 5:
            mid = (y_min + y_max) / 2
            y_min = max(0, mid - 2.5)
            y_max = min(100, mid + 2.5)
    else:
        y_min, y_max = 0, 100

    ax_left.set_xlim(-0.4, len(SCENARIOS_NO_BLK) - 0.6)
    ax_left.set_ylim(y_min, y_max)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(SCENARIOS_NO_BLK, fontsize=FONT_TICK)
    ax_left.set_ylabel("Packet Delivery Ratio (%)", fontsize=FONT_LABEL)
    ax_left.set_title(
        "(a)  PDR - Normal Operating Scenarios\n"
        "[Y-axis zoomed  |  V2X Blackout excluded]",
        fontsize=FONT_TITLE - 0.5, pad=4,
    )
    ax_left.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.1f%%")
    )
    ax_left.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax_left.set_axisbelow(True)

    # Broken-axis indicator (zigzag at bottom)
    d = 0.015
    kwargs = dict(transform=ax_left.transAxes, color="k",
                  clip_on=False, linewidth=0.8)
    ax_left.plot((-d, +d), (-d, +d), **kwargs)
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)

    # ── RIGHT PANEL: horizontal bar chart, V2X Blackout only ─────────────
    algos  = list(pdr_data.keys())
    blk_vals = [pdr_data[a][BLACKOUT_IDX] for a in algos]
    y_pos  = np.arange(len(algos))
    colors = [COLORS.get(a, "#333333") for a in algos]

    bars = ax_right.barh(
        y_pos, blk_vals,
        color  = colors,
        height = 0.62,
        edgecolor = "white",
        linewidth = 0.5,
    )

    # Gold edge highlight for E3
    if "E\u00b3-Hybrid" in algos:
        e3_idx = algos.index("E\u00b3-Hybrid")
        bars[e3_idx].set_edgecolor("#FFD700")
        bars[e3_idx].set_linewidth(1.8)

    # Value annotations
    for i, val in enumerate(blk_vals):
        ax_right.text(
            val + 0.15, i, f"{val:.1f}%",
            va="center", ha="left",
            fontsize=FONT_ANNOT,
            fontweight="bold" if algos[i] == "E\u00b3-Hybrid" else "normal",
            color=COLORS.get(algos[i], "#333333"),
        )

    blackout_values = _finite(blk_vals)
    if blackout_values:
        x_min = max(0, min(blackout_values) - 2)
        x_max = min(100, max(blackout_values) + 3)
        if x_max - x_min < 10:
            mid = (x_min + x_max) / 2
            x_min = max(0, mid - 5)
            x_max = min(100, mid + 5)
    else:
        x_min, x_max = 0, 100

    ax_right.set_xlim(x_min, x_max)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(algos, fontsize=FONT_TICK)
    ax_right.set_xlabel("PDR (%)", fontsize=FONT_LABEL)
    ax_right.set_title(
        "(b)  V2X Blackout\n[all algorithms  |  95% packet loss]",
        fontsize=FONT_TITLE - 0.5, pad=4,
    )
    ax_right.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.0f%%")
    )
    ax_right.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax_right.set_axisbelow(True)

    # Annotation: highest PDR in blackout
    if blackout_values:
        best_idx = int(np.nanargmax(blk_vals))
        best_algo = algos[best_idx]
        best_val = blk_vals[best_idx]
        ax_right.annotate(
            f"Highest PDR\n{best_algo}: {best_val:.1f}%",
            xy=(best_val, best_idx),
            xytext=(max(x_min, best_val - 0.3), best_idx + 1.3),
            fontsize=FONT_ANNOT - 0.5,
            color=COLORS.get(best_algo, "#333333"),
            arrowprops=dict(arrowstyle="->", color=COLORS.get(best_algo, "#333333"), lw=0.8),
        )

    # ── Shared legend below both panels ──────────────────────────────────
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc            = "lower center",
        ncol           = 7,
        fontsize       = FONT_LEGEND,
        frameon        = True,
        framealpha     = 0.9,
        edgecolor      = "#cccccc",
        bbox_to_anchor = (0.5, -0.08),
    )

    fig.suptitle(
        "V2X Packet Delivery Ratio — All Algorithms × All Scenarios",
        fontsize=FONT_TITLE + 0.5,
        fontweight="bold",
        y=1.01,
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


# ── Run standalone to test ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from omnet_analysis.analyze_v2x import (
            ALGOS,
            SCENARIOS,
            SEEDS,
            ALGO_LABELS,
            compute_pdr,
            load_logs,
            seed_mean,
        )

        logs = load_logs()
        data = pdr_data_from_analysis(
            logs, ALGOS, SCENARIOS, SEEDS, ALGO_LABELS, compute_pdr, seed_mean
        )
        if not any(_finite(values) for values in data.values()):
            raise ValueError("simulation log CSVs did not contain PDR values")
    except Exception as exc:
        sca_path = os.path.join("omnet_analysis", "v2x_results.sca")
        if not os.path.exists(sca_path):
            raise
        print(f"[WARN] Could not build from CSV logs ({exc}); using {sca_path}.")
        data = load_pdr_data_from_sca(sca_path)

    plot_pdr_comparison("pdr_comparison.png", data)
