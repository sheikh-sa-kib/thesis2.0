"""
aggregate_data.py
─────────────────────────────────────────────────────────────────────────────
UPDATES:
  - Algorithm column writes internal key names (no display names in CSV).
  - Now reads Total_Teleport_Events column from per-run CSVs and aggregates
    it — lower teleport count = fewer stuck vehicles = better routing.
    This adds a 7th metric for thesis comparison (E3 should be lowest).
"""
import os
import math
import pandas as pd

THESIS_DIR         = os.path.expanduser("~/thesis")
RESULTS_DIR        = os.path.join(THESIS_DIR, "results")
AGGREGATED_OUTPUT  = os.path.join(RESULTS_DIR, "final_aggregated_metrics.csv")
LATEX_TABLE_OUTPUT = os.path.join(RESULTS_DIR, "latex_results_table.tex")

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

ALGO_LATEX_DISPLAY = {
    "Baseline_SUMO":      "Baseline (SUMO)",
    "Dijkstra":           "Dijkstra",
    "A_Star":             "A*",
    "BCO_Standalone":     "BCO",
    "ACO_Standalone":     "ACO",
    "PSO_Standalone":     "PSO",
    "E3_Hybrid_Complete": r"E$^3$-Hybrid",
    "E3_NoRL":            r"E$^3$-NoRL",
}


def _safe_float(val, fallback=None):
    try:
        f = float(val)
        return f if not math.isnan(f) else fallback
    except (TypeError, ValueError):
        return fallback


def aggregate_data():
    print("\n" + "=" * 65)
    print("AGGREGATING STOCHASTIC METRICS")
    print("=" * 65)

    all_records = []

    for algo in ALGORITHMS:
        for scen in SCENARIOS:
            for seed in SEEDS:
                target = os.path.join(
                    RESULTS_DIR, f"{algo}_scen{scen}_seed{seed}.csv"
                )
                if not os.path.exists(target):
                    print(f" [!] Missing: {target}")
                    continue
                try:
                    df       = pd.read_csv(target)
                    last_row = df.iloc[-1]

                    # Optional column — older runs may not have it
                    teleport_val = _safe_float(
                        last_row.get("Total_Teleport_Events", 0), fallback=0
                    )

                    record = {
                        "Algorithm":            algo,
                        "Scenario":             scen,
                        "Seed":                 seed,
                        "Travel_Time_Mean":     float(last_row["Avg_Travel_Time"]),
                        "Travel_Time_StdDev":   float(last_row["Std_Dev_Travel_Time"]),
                        "Waiting_Time":         float(last_row["Avg_Waiting_Time"]),
                        "Reroutes":             float(last_row["Avg_Reroutes"]),
                        "Stranded_EVs":         int(last_row.get("Stranded_EVs", 0)),
                        "Stranded_Battery":     int(
                            last_row.get("Stranded_Due_To_Battery",
                                         last_row.get("Stranded_EVs", 0))
                        ),
                        "Energy_Wh":            _safe_float(
                            last_row.get("Avg_Energy_Wh", 0), fallback=0
                        ),
                        "Vehicles_Charged":     int(
                            last_row.get("Vehicles_Charged", 0)
                        ),
                        "Emergency_Response_Time": _safe_float(
                            last_row["Emergency_Response_Time"]
                        ),
                        "Total_Teleport_Events": teleport_val,
                    }
                    all_records.append(record)

                except Exception as e:
                    print(f" [!] Error reading {target}: {e}")

    if not all_records:
        print("\n [!] No data found — run batch_evaluator.py first.")
        return

    master_df = pd.DataFrame(all_records)

    grouped = master_df.groupby(["Algorithm", "Scenario"]).agg(
        TT_Mean_Avg      = ("Travel_Time_Mean",        "mean"),
        TT_Mean_Std      = ("Travel_Time_Mean",        "std"),
        Wait_Avg         = ("Waiting_Time",             "mean"),
        Wait_Std         = ("Waiting_Time",             "std"),
        Reroutes_Avg     = ("Reroutes",                 "mean"),
        Stranded_Avg     = ("Stranded_EVs",             "mean"),
        Stranded_Battery_Avg = ("Stranded_Battery",     "mean"),
        Avg_Energy_Wh    = ("Energy_Wh",                "mean"),
        Vehicles_Charged_Avg = ("Vehicles_Charged",     "mean"),
        ERT_Avg          = ("Emergency_Response_Time",  "mean"),
        ERT_Std          = ("Emergency_Response_Time",  "std"),
        Teleport_Avg     = ("Total_Teleport_Events",    "mean"),
        Teleport_Std     = ("Total_Teleport_Events",    "std"),
    ).reset_index()

    grouped.to_csv(AGGREGATED_OUTPUT, index=False)
    print(f"\n [+] Aggregated CSV saved: {AGGREGATED_OUTPUT}")

    _print_improvement_table(grouped)
    generate_latex_table(grouped)


def _print_improvement_table(df):
    print("\n" + "=" * 75)
    print("  E3-HYBRID IMPROVEMENT OVER COMPETING ALGORITHMS")
    print("=" * 75)
    print(f"{'Algorithm':<22} {'Scenario':<18} {'Base TT':>8} "
          f"{'E3 TT':>8} {'Improv':>8} {'E3 Teleports':>14}")
    print("-" * 75)

    e3_label = "E3_Hybrid_Complete"
    scen_names = {
        0: "Normal", 1: "Single Block", 2: "Progressive",
        3: "Rush Hour", 4: "V2X Blackout", 5: "Infra Failure"
    }

    for algo in ALGORITHMS:
        if algo == e3_label:
            continue
        for scen in SCENARIOS:
            e3_row  = df[(df["Algorithm"] == e3_label) & (df["Scenario"] == scen)]
            cmp_row = df[(df["Algorithm"] == algo) & (df["Scenario"] == scen)]
            if e3_row.empty or cmp_row.empty:
                continue
            e3_tt    = float(e3_row["TT_Mean_Avg"].values[0])
            cmp_tt   = float(cmp_row["TT_Mean_Avg"].values[0])
            e3_tp    = float(e3_row["Teleport_Avg"].values[0])
            pct      = ((cmp_tt - e3_tt) / cmp_tt * 100) if cmp_tt > 0 else 0.0
            label    = scen_names.get(scen, str(scen))
            print(f"{ALGO_LATEX_DISPLAY.get(algo, algo):<22} {label:<18} "
                  f"{cmp_tt:>8.1f} {e3_tt:>8.1f} {pct:>+7.1f}% {e3_tp:>14.1f}")
        print()

    print("=" * 75)


def generate_latex_table(df):
    latex_str = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Comparative Performance Across Emergency Scenarios "
        "(210-Run Stochastic Average, 7 Algorithms $\\times$ 6 Scenarios "
        "$\\times$ 5 Seeds)}\n"
        "\\label{tab:phase10_comparative_results}\n"
        "\\begin{tabular}{l c c c c c c}\n"
        "\\toprule\n"
        "\\textbf{Algorithm} & \\textbf{Scenario} & "
        "\\textbf{Travel Time (s)} & \\textbf{Waiting Time (s)} & "
        "\\textbf{Reroutes / Veh} & \\textbf{Stranded EVs} & "
        "\\textbf{ERT (s)} \\\\\n"
        "\\midrule\n"
    )

    current_algo = ""
    for _, row in df.iterrows():
        algo_key     = row["Algorithm"]
        algo_display = ALGO_LATEX_DISPLAY.get(algo_key, algo_key)
        scen         = f"Scenario {int(row['Scenario'])}"

        tt_std   = row.get("TT_Mean_Std", float("nan"))
        wait_std = row.get("Wait_Std", float("nan"))

        tt_str = (
            f"{row['TT_Mean_Avg']:.2f} $\\pm$ {tt_std:.2f}"
            if pd.notna(tt_std) and not math.isnan(float(tt_std))
            else f"{row['TT_Mean_Avg']:.2f}"
        )
        wait_str = (
            f"{row['Wait_Avg']:.2f} $\\pm$ {wait_std:.2f}"
            if pd.notna(wait_std) and not math.isnan(float(wait_std))
            else f"{row['Wait_Avg']:.2f}"
        )
        reroute_str  = f"{row['Reroutes_Avg']:.2f}"
        stranded_str = f"{row['Stranded_Avg']:.1f}"

        ert_avg = row.get("ERT_Avg")
        ert_std = row.get("ERT_Std")
        ert_str = (
            f"{ert_avg:.1f} $\\pm$ {ert_std:.1f}"
            if pd.notna(ert_avg) and not math.isnan(float(ert_avg))
            else "N/A"
        )

        display_algo = algo_display if algo_key != current_algo else ""
        current_algo = algo_key
        is_e3        = (algo_key == "E3_Hybrid_Complete")

        if is_e3:
            latex_str += (
                f"\\textbf{{{display_algo}}} & \\textbf{{{scen}}} & "
                f"\\textbf{{{tt_str}}} & \\textbf{{{wait_str}}} & "
                f"\\textbf{{{reroute_str}}} & \\textbf{{{stranded_str}}} & "
                f"\\textbf{{{ert_str}}} \\\\\n"
            )
        else:
            latex_str += (
                f"{display_algo} & {scen} & {tt_str} & "
                f"{wait_str} & {reroute_str} & {stranded_str} & {ert_str} \\\\\n"
            )

        if int(row["Scenario"]) == 5:
            latex_str += "\\midrule\n"

    if latex_str.endswith("\\midrule\n"):
        latex_str = latex_str[:-9] + "\\bottomrule\n"

    latex_str += "\\end{tabular}\n\\end{table*}\n"

    with open(LATEX_TABLE_OUTPUT, "w") as f:
        f.write(latex_str)

    print(f" [+] LaTeX table saved: {LATEX_TABLE_OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    aggregate_data()
