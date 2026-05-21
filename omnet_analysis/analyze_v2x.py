"""
analyze_v2x.py — E³-Hybrid Thesis V2X Analysis
================================================
Loads all 210 omnet_logs CSVs and produces:
  1. Packet Delivery Ratio (PDR) per algo per scenario
  2. Message propagation timeline (notified over steps) — Scen 1 vs Scen 4
  3. Channel load heatmap (dropped messages) — all algos × all scenarios
  4. OMNeT++ .sca scalar output for IDE result analyzer

Output directory: ~/thesis/omnet_analysis/
Figures:         pdr_comparison.png
                 propagation_timeline.png
                 channel_load_heatmap.png
Scalar file:     v2x_results.sca

Usage:
    cd ~/thesis && source venv/bin/activate
    python omnet_analysis/analyze_v2x.py
"""

import os
import glob
import csv
import math
import shutil
from collections import defaultdict

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pdr_plot_replacement import (
    load_pdr_data_from_sca,
    pdr_data_from_analysis,
    plot_pdr_comparison,
)

# ── paths ────────────────────────────────────────────────────────────────────
LOG_DIR     = os.path.expanduser("~/thesis/omnet_logs")
OUT_DIR     = os.path.expanduser("~/thesis/omnet_analysis")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = [
    "E3_Hybrid_Complete",
    "Dijkstra",
    "A_Star",
    "BCO_Standalone",
    "ACO_Standalone",
    "PSO_Standalone",
    "Baseline_SUMO",
]
ALGO_LABELS = {
    "E3_Hybrid_Complete": "E³-Hybrid",
    "Dijkstra":           "Dijkstra",
    "A_Star":             "A*",
    "BCO_Standalone":     "BCO",
    "ACO_Standalone":     "ACO",
    "PSO_Standalone":     "PSO",
    "Baseline_SUMO":      "Baseline",
}
SCENARIOS   = list(range(6))
SCEN_LABELS = [
    "Normal", "Single Block", "Progressive",
    "Rush Hour", "V2X Blackout", "Infra Failure",
]
SEEDS = [42, 123, 456, 789, 1337]

COLORS = {
    "E3_Hybrid_Complete": "#E84545",
    "Dijkstra":           "#2B9EB3",
    "A_Star":             "#F5A623",
    "BCO_Standalone":     "#7B68EE",
    "ACO_Standalone":     "#4CAF50",
    "PSO_Standalone":     "#26C6DA",
    "Baseline_SUMO":      "#6C757D",
}

# ── load all logs ─────────────────────────────────────────────────────────────
def load_logs():
    """Returns dict: data[algo][scen][seed] = list of row dicts."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    missing = []
    for algo in ALGOS:
        for scen in SCENARIOS:
            for seed in SEEDS:
                path = os.path.join(LOG_DIR, f"{algo}_scen{scen}_seed{seed}.csv")
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data[algo][scen][seed].append({
                            "step":      int(row["step"]),
                            "loss_rate": float(row["loss_rate"]),
                            "notified":  int(row["notified"]),
                            "dropped":   int(row["dropped"]),
                        })
    if missing:
        print(f"[WARN] {len(missing)} log files missing — results may be partial.")
    total = sum(
        len(data[a][s][sd])
        for a in data for s in data[a] for sd in data[a][s]
    )
    total_expected = len(ALGOS) * len(SCENARIOS) * len(SEEDS)
    print(f"[INFO] Loaded {total:,} rows from {total_expected - len(missing)}/{total_expected} log files.")
    return data


# ── metric helpers ────────────────────────────────────────────────────────────
def compute_pdr(rows):
    """Packet Delivery Ratio across all rows in one run."""
    total_notified = sum(r["notified"] for r in rows)
    total_dropped  = sum(r["dropped"]  for r in rows)
    total          = total_notified + total_dropped
    return total_notified / total if total > 0 else 0.0

def compute_channel_load(rows):
    """Mean dropped messages per step (channel congestion proxy)."""
    if not rows:
        return 0.0
    return sum(r["dropped"] for r in rows) / len(rows)

def seed_mean(values):
    return sum(values) / len(values) if values else 0.0

def seed_std(values):
    if len(values) < 2:
        return 0.0
    m = seed_mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


# ── Figure 1: PDR comparison ─────────────────────────────────────────────────
def plot_pdr(data):
    pdr_data = pdr_data_from_analysis(
        data, ALGOS, SCENARIOS, SEEDS, ALGO_LABELS, compute_pdr, seed_mean
    )
    has_values = any(
        value is not None and np.isfinite(value)
        for values in pdr_data.values()
        for value in values
    )
    sca_path = os.path.join(OUT_DIR, "v2x_results.sca")
    if not has_values and os.path.exists(sca_path):
        print(f"[WARN] No PDR values found in CSV logs; using {sca_path}.")
        pdr_data = load_pdr_data_from_sca(sca_path)

    out = os.path.join(OUT_DIR, "pdr_comparison.png")
    plot_pdr_comparison(out, pdr_data)

    for mirror in (
        os.path.join(PROJECT_DIR, "pdr_comparison.png"),
        os.path.join(PROJECT_DIR, "figures", "pic", "pdr_comparison.png"),
    ):
        if os.path.abspath(out) == os.path.abspath(mirror):
            continue
        if os.path.isdir(os.path.dirname(mirror)):
            shutil.copy2(out, mirror)
            print(f"[OK] Updated: {mirror}")


# ── Figure 2: Message propagation timeline Scen 1 vs Scen 4 ──────────────────
def plot_propagation_timeline(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    compare_scens = [(1, "Scenario 1 — Single Block (12% loss)"),
                     (4, "Scenario 4 — V2X Blackout (95% loss)")]

    for ax, (scen, title) in zip(axes, compare_scens):
        for algo in ALGOS:
            # Average notified per step across seeds
            step_notified = defaultdict(list)
            for seed in SEEDS:
                for row in data[algo][scen][seed]:
                    step_notified[row["step"]].append(row["notified"])
            steps  = sorted(step_notified.keys())
            means  = [seed_mean(step_notified[s]) for s in steps]
            stds   = [seed_std(step_notified[s])  for s in steps]

            ax.plot(steps, means, label=ALGO_LABELS[algo],
                    color=COLORS[algo], linewidth=1.6, alpha=0.9)
            ax.fill_between(
                steps,
                [m - e for m, e in zip(means, stds)],
                [m + e for m, e in zip(means, stds)],
                color=COLORS[algo], alpha=0.12,
            )

        ax.axvline(100, color="gray", linestyle="--", linewidth=1,
                   label="Block event (t=100)")
        ax.set_xlabel("Simulation Step", fontsize=11)
        ax.set_ylabel("Avg Vehicles Notified / Step", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("V2X Message Propagation Timeline",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "propagation_timeline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


# ── Figure 3: Channel load heatmap ───────────────────────────────────────────
def plot_channel_load_heatmap(data):
    # Matrix: rows=algos, cols=scenarios, value=mean dropped/step
    matrix = np.zeros((len(ALGOS), len(SCENARIOS)))
    for i, algo in enumerate(ALGOS):
        for j, scen in enumerate(SCENARIOS):
            loads = [
                compute_channel_load(data[algo][scen][seed])
                for seed in SEEDS
                if data[algo][scen][seed]
            ]
            matrix[i, j] = seed_mean(loads)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(SCEN_LABELS, fontsize=10)
    ax.set_yticks(range(len(ALGOS)))
    ax.set_yticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=10)
    ax.set_title("V2X Channel Load — Mean Dropped Messages per Step",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(ALGOS)):
        for j in range(len(SCENARIOS)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean dropped / step", fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "channel_load_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


# ── OMNeT++ .sca scalar output ────────────────────────────────────────────────
def export_sca(data):
    """
    Writes v2x_results.sca in OMNeT++ scalar result format.
    Load in OMNeT++ IDE via: File > Open > Browse Workspace > v2x_results.sca
    Then use the Result Analyzer (Anf) to plot scalars.
    """
    out = os.path.join(OUT_DIR, "v2x_results.sca")
    with open(out, "w") as f:
        f.write("version 2\n")
        f.write('run v2x-e3hybrid-analysis r0-20260513\n')
        f.write('attr configname V2XAnalysis\n')
        f.write('attr experiment "E3-Hybrid V2X"\n')
        f.write('attr measurement "150-run batch"\n\n')

        for algo in ALGOS:
            for scen in SCENARIOS:
                pdrs   = []
                loads  = []
                for seed in SEEDS:
                    rows = data[algo][scen][seed]
                    if rows:
                        pdrs.append(compute_pdr(rows))
                        loads.append(compute_channel_load(rows))

                if not pdrs:
                    continue

                module = f"V2X.{ALGO_LABELS[algo].replace(' ','_')}.Scen{scen}"
                f.write(f"scalar {module} pdr_mean {seed_mean(pdrs):.6f}\n")
                f.write(f"scalar {module} pdr_std  {seed_std(pdrs):.6f}\n")
                f.write(f"scalar {module} channel_load_mean {seed_mean(loads):.4f}\n")
                f.write(f"scalar {module} channel_load_std  {seed_std(loads):.4f}\n")

    print(f"[OK] Saved: {out}")


# ── summary table (console) ───────────────────────────────────────────────────
def print_summary_table(data):
    print("\n" + "=" * 72)
    print(f"{'ALGO':<22} {'SCEN':<18} {'PDR%':>7} {'LOAD':>7}")
    print("=" * 72)
    for algo in ALGOS:
        for scen in SCENARIOS:
            pdrs  = [compute_pdr(data[algo][scen][s])  for s in SEEDS if data[algo][scen][s]]
            loads = [compute_channel_load(data[algo][scen][s]) for s in SEEDS if data[algo][scen][s]]
            if not pdrs:
                continue
            label = ALGO_LABELS[algo]
            scen_label = SCEN_LABELS[scen]
            print(f"{label:<22} {scen_label:<18} {seed_mean(pdrs)*100:>6.1f}% {seed_mean(loads):>7.2f}")
        print("-" * 72)
    print()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  E³-Hybrid V2X Analysis — OMNeT++ Log Processing")
    print("=" * 55)

    data = load_logs()
    print_summary_table(data)

    print("[*] Generating figures...")
    plot_pdr(data)
    plot_propagation_timeline(data)
    plot_channel_load_heatmap(data)
    export_sca(data)

    print("\n[DONE] All outputs in ~/thesis/omnet_analysis/")
    print("  pdr_comparison.png       — Figure for thesis Chapter 5")
    print("  propagation_timeline.png — Figure for thesis Chapter 5")
    print("  channel_load_heatmap.png — Figure for thesis Chapter 5")
    print("  v2x_results.sca          — Load in OMNeT++ IDE Result Analyzer")
