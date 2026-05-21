"""
statistical_analysis.py
─────────────────────────────────────────────────────────────────────────────
Computes Mann-Whitney U tests comparing E³-Hybrid vs every other algorithm
for each scenario, plus a full descriptive statistics table.

Outputs:
  ~/thesis/results/mann_whitney_pairwise.csv    ← required by generate_plots.py
  ~/thesis/results/descriptive_statistics.csv  ← summary table
  ~/thesis/results/improvement_table.csv        ← % improvement E³ vs each algo

Run AFTER aggregate_data.py:
    cd ~/thesis && python statistical_analysis.py
"""
import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RESULTS_DIR  = os.path.expanduser("~/thesis/results")
RAW_GLOB     = os.path.join(RESULTS_DIR, "*.csv")

MW_OUTPUT    = os.path.join(RESULTS_DIR, "mann_whitney_pairwise.csv")
DESC_OUTPUT  = os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
IMP_OUTPUT   = os.path.join(RESULTS_DIR, "improvement_table.csv")

ALGORITHMS = [
    "Baseline_SUMO",
    "Dijkstra",
    "A_Star",
    "BCO_Standalone",
    "ACO_Standalone",
    "PSO_Standalone",
    "E3_Hybrid_Complete",
    "E3_NoRL",
]
SCENARIOS = [0, 1, 2, 3, 4, 5]
SEEDS = [
    42, 123, 456, 789, 1337,
    2024, 3001, 5555, 7777, 9999,
    1111, 2222, 3333, 4444, 6666,
]

EXPECTED_SEEDS = 15
EXPECTED_ALGOS = 8

SCEN_LABELS = {
    0: "Normal",
    1: "Single Block",
    2: "Progressive (3 Blocks)",
    3: "Rush Hour Cascade",
    4: "V2X Blackout",
    5: "Infra Failure",
}


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_raw_data():
    """Load all per-run CSVs into a single DataFrame."""
    frames = []
    for f in glob.glob(RAW_GLOB):
        basename = os.path.basename(f)
        skip_keywords = [
            "aggregated", "summary", "statistical",
            "mann_whitney", "descriptive", "improvement",
            "latex"
        ]
        if any(k in basename for k in skip_keywords):
            continue
        try:
            df = pd.read_csv(f)
            if "Avg_Travel_Time" in df.columns:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        print("[STAT] ERROR: No per-run CSVs found. Run batch_evaluator.py first.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["Algorithm"] = raw["Algorithm"].str.strip().str.replace(" ", "_")
    raw["Scenario"]  = raw["Scenario"].astype(int)
    print(f"[STAT] Loaded {len(raw)} per-run rows from "
          f"{len(frames)} CSV files.")
    return raw


def compute_descriptive_stats(raw):
    """Compute N, mean, std, min, median, max, CV per algo × scenario."""
    records = []
    scen_name_map = {
        0: "Normal", 1: "Single Block", 2: "Progressive (3 Blocks)",
        3: "Rush Hour Cascade", 4: "V2X Blackout", 5: "Infra Failure",
    }
    for algo in ALGORITHMS:
        algo_label = algo.replace("_", " ")
        # Try both formats
        subset_algo = raw[
            (raw["Algorithm"] == algo) |
            (raw["Algorithm"] == algo.replace("_", " "))
        ]
        for scen in SCENARIOS:
            subset = subset_algo[subset_algo["Scenario"] == scen]
            vals   = subset["Avg_Travel_Time"].dropna().astype(float)
            if vals.empty:
                continue
            n      = len(vals)
            mean   = vals.mean()
            std    = vals.std(ddof=1) if n > 1 else 0.0
            mn     = vals.min()
            med    = vals.median()
            mx     = vals.max()
            cv     = (std / mean * 100) if mean > 0 else 0.0
            records.append({
                "Algorithm":  algo_label,
                "Scenario":   scen_name_map.get(scen, str(scen)),
                "N":          n,
                "Mean":       round(mean, 3),
                "Std":        round(std, 3),
                "Min":        round(mn, 3),
                "Median":     round(med, 3),
                "Max":        round(mx, 3),
                "CV_%":       round(cv, 3),
            })

    df = pd.DataFrame(records)
    df.to_csv(DESC_OUTPUT, index=False)
    print(f"[STAT] Descriptive statistics saved: {DESC_OUTPUT}")
    return df


def compute_mann_whitney(raw):
    """
    Mann-Whitney U test: E³-Hybrid vs every other algorithm,
    per scenario, on Avg_Travel_Time.
    """
    records  = []
    e3_key   = "E3_Hybrid_Complete"
    baselines = [a for a in ALGORITHMS if a != e3_key]

    for scen in SCENARIOS:
        scen_label = SCEN_LABELS[scen]
        e3_subset  = raw[
            ((raw["Algorithm"] == e3_key) |
             (raw["Algorithm"] == e3_key.replace("_", " "))) &
            (raw["Scenario"] == scen)
        ]["Avg_Travel_Time"].dropna().astype(float).values

        for baseline in baselines:
            base_subset = raw[
                ((raw["Algorithm"] == baseline) |
                 (raw["Algorithm"] == baseline.replace("_", " "))) &
                (raw["Scenario"] == scen)
            ]["Avg_Travel_Time"].dropna().astype(float).values

            if len(e3_subset) < 2 or len(base_subset) < 2:
                records.append({
                    "Scenario":  scen_label,
                    "Scenario_id": scen,
                    "E3_vs":     baseline.replace("_", " "),
                    "n_e3":      len(e3_subset),
                    "n_base":    len(base_subset),
                    "mean_e3":   round(e3_subset.mean(), 3) if len(e3_subset) > 0 else None,
                    "mean_base": round(base_subset.mean(), 3) if len(base_subset) > 0 else None,
                    "improvement_pct": None,
                    "U_stat":    None,
                    "p_value":   None,
                    "Stars":     "insufficient data",
                })
                continue

            U, p = stats.mannwhitneyu(
                e3_subset, base_subset, alternative="less"
            )  # one-sided: E3 < baseline (lower travel time = better)

            mean_e3   = e3_subset.mean()
            mean_base = base_subset.mean()
            improv    = ((mean_base - mean_e3) / mean_base * 100
                         if mean_base > 0 else 0.0)

            records.append({
                "Scenario":        scen_label,
                "Scenario_id":     scen,
                "E3_vs":           baseline.replace("_", " "),
                "n_e3":            len(e3_subset),
                "n_base":          len(base_subset),
                "mean_e3":         round(mean_e3, 3),
                "mean_base":       round(mean_base, 3),
                "improvement_pct": round(improv, 2),
                "U_stat":          round(U, 3),
                "p_value":         round(p, 6),
                "Stars":           _stars(p),
            })

    df = pd.DataFrame(records)
    df.to_csv(MW_OUTPUT, index=False)
    print(f"[STAT] Mann-Whitney results saved: {MW_OUTPUT}")
    return df


def compute_improvement_table(raw):
    """
    Full improvement % table: E³-Hybrid vs every competitor,
    every scenario, every metric.
    """
    records = []
    e3_key  = "E3_Hybrid_Complete"

    for scen in SCENARIOS:
        e3_sub = raw[
            ((raw["Algorithm"] == e3_key) |
             (raw["Algorithm"] == e3_key.replace("_", " "))) &
            (raw["Scenario"] == scen)
        ]

        for algo in ALGORITHMS:
            if algo == e3_key:
                continue
            base_sub = raw[
                ((raw["Algorithm"] == algo) |
                 (raw["Algorithm"] == algo.replace("_", " "))) &
                (raw["Scenario"] == scen)
            ]
            if e3_sub.empty or base_sub.empty:
                continue

            def pct_improvement(col):
                e3_v   = e3_sub[col].dropna().astype(float).mean()
                base_v = base_sub[col].dropna().astype(float).mean()
                if base_v == 0:
                    return 0.0
                return round((base_v - e3_v) / base_v * 100, 2)

            records.append({
                "Scenario":            SCEN_LABELS[scen],
                "vs_Algorithm":        algo.replace("_", " "),
                "TravelTime_Improv_%": pct_improvement("Avg_Travel_Time"),
                "WaitTime_Improv_%":   pct_improvement("Avg_Waiting_Time"),
                "Reroutes_Improv_%":   -pct_improvement("Avg_Reroutes"),  # higher reroutes = more adaptive, invert
                "StrandedEV_Improv_%": pct_improvement("Stranded_EVs"),
            })

    df = pd.DataFrame(records)
    df.to_csv(IMP_OUTPUT, index=False)
    print(f"[STAT] Improvement table saved: {IMP_OUTPUT}")
    return df


def print_summary(mw_df):
    print("\n" + "=" * 70)
    print("  E³-HYBRID MANN-WHITNEY SIGNIFICANCE SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<25} {'vs Algorithm':<22} "
          f"{'Improv%':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)
    for _, row in mw_df.iterrows():
        if row["p_value"] is None:
            continue
        print(f"{str(row['Scenario']):<25} {str(row['E3_vs']):<22} "
              f"{row['improvement_pct']:>+7.2f}% "
              f"{row['p_value']:>10.6f} {row['Stars']:>5}")
    print("=" * 70)
    # Count significant results
    sig = mw_df[mw_df["Stars"].isin(["*", "**", "***"])]
    total_valid = mw_df[mw_df["p_value"].notna()]
    print(f"\n  Significant results: {len(sig)} / {len(total_valid)} "
          f"({100*len(sig)/max(len(total_valid),1):.1f}%)")
    print()


def verify_seed_coverage(raw):
    """Warn if any algorithm/scenario combo has fewer seeds than expected."""
    seed_counts = raw.groupby(["Algorithm", "Scenario"]).size()

    min_seeds = seed_counts.min()
    max_seeds = seed_counts.max()

    if min_seeds < EXPECTED_SEEDS:
        print(
            f"[WARNING] Some algorithm/scenario combos have only {min_seeds} seeds."
        )
        print(f"          Expected {EXPECTED_SEEDS}. Incomplete combos:")
        print(seed_counts[seed_counts < EXPECTED_SEEDS].to_string())
        print("          Run batch_evaluator.py to completion before re-running stats.")
    else:
        print(f"[OK] All combos have {min_seeds}–{max_seeds} seeds. Proceeding.")


if __name__ == "__main__":
    print("=" * 60)
    print("  E³-HYBRID STATISTICAL ANALYSIS")
    print("=" * 60)

    raw = load_raw_data()
    if raw.empty:
        print("[STAT] No data to analyse. Exiting.")
        raise SystemExit(1)

    verify_seed_coverage(raw)

    desc_df = compute_descriptive_stats(raw)
    mw_df   = compute_mann_whitney(raw)
    imp_df  = compute_improvement_table(raw)

    print_summary(mw_df)

    n_seeds = raw["Seed"].nunique() if "Seed" in raw.columns else len(SEEDS)
    total_valid = mw_df[mw_df["p_value"].notna()]
    n_sig = (total_valid["p_value"] < 0.05).sum()
    n_total = len(total_valid)
    if n_total > 0:
        ratio = n_sig / n_total
        power = "HIGH" if ratio > 0.6 else "MODERATE" if ratio > 0.3 else "LOW"
        print("\n=== STATISTICAL SUMMARY ===")
        print(f"Seeds per combo      : {n_seeds}")
        print(
            f"Significant pairs (Bonferroni p<0.05): {n_sig} / {n_total} "
            f"({100 * ratio:.1f}%)"
        )
        print(f"Power estimate       : {power}")

    print("\n[DONE] Statistical analysis complete.")
    print(f"  {MW_OUTPUT}")
    print(f"  {DESC_OUTPUT}")
    print(f"  {IMP_OUTPUT}")
