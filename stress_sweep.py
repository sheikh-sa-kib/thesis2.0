"""
stress_sweep.py
─────────────────────────────────────────────────────────────────────────────
Single-seed demand-density sweep for locating the saturated-capacity boundary.
"""
import os
import subprocess
import sys

import pandas as pd


PERIODS = ["0.40", "0.35", "0.30", "0.25", "0.22", "0.18"]
TARGET_ALGOS = ["Dijkstra", "E3_Hybrid_Complete"]
SCENARIO_ID = "3"
SEED = "42"
THESIS_DIR = os.path.expanduser("~/thesis")


def main():
    print(f"{'Period':<10} {'Algo':<22} {'Avg Travel Time':<18} {'Teleports':<10}")
    print("-" * 65)

    for period in PERIODS:
        demand_cmd = [
            sys.executable,
            os.path.join(THESIS_DIR, "generate_dense_demand.py"),
            "--period",
            period,
        ]
        demand_result = subprocess.run(demand_cmd, capture_output=True, text=True)
        if demand_result.returncode != 0:
            print(f"{period:<10} {'DEMAND_GENERATION':<22} {'FAILED':<18} {'-':<10}")
            continue

        for algo in TARGET_ALGOS:
            run_cmd = [
                sys.executable,
                os.path.join(THESIS_DIR, "run_sim.py"),
                "--algo",
                algo,
                "--scenario",
                SCENARIO_ID,
                "--seed",
                SEED,
                "--nogui",
            ]
            subprocess.run(run_cmd, capture_output=True, text=True)

            output_file = os.path.join(
                THESIS_DIR, "results", f"{algo}_scen{SCENARIO_ID}_seed{SEED}.csv"
            )
            if not os.path.exists(output_file):
                print(f"{period:<10} {algo:<22} {'NO OUTPUT':<18} {'-':<10}")
                continue

            df = pd.read_csv(output_file)
            avg_tt = float(df["Avg_Travel_Time"].iloc[-1])
            teleports = int(df["Total_Teleport_Events"].iloc[-1])
            print(f"{period:<10} {algo:<22} {avg_tt:<18.2f} {teleports:<10}")


if __name__ == "__main__":
    main()
