"""
batch_evaluator.py
─────────────────────────────────────────────────────────────────────────────
FIX A6: Added timeout=600 (10 minutes) to process.wait().
Strategy A: 8 algorithms × 6 scenarios × 3 seeds = 144 runs (sequential execution).
"""
import os
import sys
import time
import subprocess
from datetime import datetime

THESIS_DIR             = os.path.expanduser("~/thesis")
RESULTS_DIR            = os.path.join(THESIS_DIR, "results")
LOGS_DIR               = os.path.join(RESULTS_DIR, "logs")
SUMMARY_CSV            = os.path.join(RESULTS_DIR, "master_144run_summary.csv")
SIMULATION_ENTRY_POINT = os.path.join(THESIS_DIR, "run_sim.py")

ALGORITHMS = [
    "Baseline_SUMO",
    "Dijkstra",
    "A_Star",
    "BCO_Standalone",
    "ACO_Standalone",      # NEW
    "PSO_Standalone",      # NEW
    "E3_Hybrid_Complete",
    "E3_NoRL",             # NEW — ablation control: E³ without DQN layer
]
SCENARIOS = [0, 1, 2, 3, 4, 5]
# Minimum seeds for standard deviation / confidence bounds on weak hardware
SEEDS = [42, 123, 456]

RUN_TIMEOUT_SEC = 3600   # 30 minutes — covers dense S5 worst-case

SKIP_COMBOS = set()


def force_flush_memory():
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "sumo"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    time.sleep(1.5)


def run_evaluation_matrix(skip_existing=True):
    """
    skip_existing=True: skip any run where the output CSV already exists.
    This lets you re-run the batch without repeating completed CSV outputs.
    """
    total_planned = len(ALGORITHMS) * len(SCENARIOS) * len(SEEDS)
    print("\n" + "=" * 65)
    print("🚀 E³-HYBRID THESIS: FULL EVALUATION MATRIX")
    print(
        f"   Scheduled: {len(ALGORITHMS)} Algos × {len(SCENARIOS)} Scenarios "
        f"× {len(SEEDS)} Seeds = {total_planned} Runs"
    )
    print(f"   Per-run timeout: {RUN_TIMEOUT_SEC}s")
    print(f"   Skip existing results: {skip_existing}")
    print("=" * 65 + "\n")

    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w") as f:
            f.write(
                "timestamp,algorithm,scenario,seed,status,"
                "execution_time_sec,log_file\n"
            )

    total_start  = time.time()
    run_counter  = 0
    runs_ok      = 0
    runs_failed  = 0
    runs_timeout = 0
    runs_skipped = 0

    for algo in ALGORITHMS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                if (algo, scenario, seed) in SKIP_COMBOS:
                    print(
                        f"---> [RUN {run_counter + 1}/{total_planned}] "
                        f"SKIP (known timeout): [{algo}] Scen {scenario} Seed {seed}"
                    )
                    continue

                run_counter  += 1
                log_filename  = f"run_{algo}_scen{scenario}_seed{seed}.log"
                log_filepath  = os.path.join(LOGS_DIR, log_filename)

                # Skip if result CSV already exists and is non-empty
                result_csv = os.path.join(
                    RESULTS_DIR, f"{algo}_scen{scenario}_seed{seed}.csv"
                )
                if skip_existing and os.path.exists(result_csv):
                    try:
                        size = os.path.getsize(result_csv)
                        if size > 50:   # non-empty CSV
                            runs_skipped += 1
                            runs_ok      += 1   # count as success for summary
                            print(
                                f"---> [RUN {run_counter}/{total_planned}] "
                                f"SKIP (exists): [{algo}] Scen {scenario} Seed {seed}"
                            )
                            continue
                    except OSError:
                        pass

                print(
                    f"---> [RUN {run_counter}/{total_planned}]: "
                    f"[{algo}] | Scen: [{scenario}] | Seed: [{seed}]"
                )

                # Force Windows cmd to use your isolated virtual environment engine
                python_bin = os.path.join(THESIS_DIR, "venv", "Scripts", "python.exe")
                if not os.path.exists(python_bin):
                    python_bin = sys.executable  # safe fallback if local path isn't mapped
                    
                cmd = [
                    python_bin,
                    SIMULATION_ENTRY_POINT,
                    "--algo",     algo,
                    "--scenario", str(scenario),
                    "--seed",     str(seed),
                    "--nogui",
                ]

                start_time = time.time()
                status     = "SUCCESS"

                with open(log_filepath, "w") as log_file:
                    log_file.write(
                        f"--- START: {datetime.now()} ---\n"
                        f"--- CMD: {' '.join(cmd)} ---\n\n"
                    )
                    log_file.flush()

                    process = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=THESIS_DIR,
                    )

                    try:
                        process.wait(timeout=RUN_TIMEOUT_SEC)
                        if process.returncode != 0:
                            status = f"FAILED (Code {process.returncode})"
                            runs_failed += 1
                            print(f"     [!] Exit code {process.returncode}.")
                        else:
                            runs_ok += 1
                            elapsed = time.time() - start_time
                            print(f"     [+] Completed in {elapsed:.2f}s.")

                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        status = "TIMEOUT"
                        runs_timeout += 1
                        print(
                            f"     [!] TIMEOUT after {RUN_TIMEOUT_SEC}s. "
                            f"Process killed."
                        )

                exec_time = time.time() - start_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(SUMMARY_CSV, "a") as f:
                    f.write(
                        f"{timestamp},{algo},{scenario},{seed},"
                        f"{status},{exec_time:.2f},{log_filename}\n"
                    )

                force_flush_memory()

    elapsed_min = (time.time() - total_start) / 60.0
    print("\n" + "=" * 65)
    print(
        f"🏆 MATRIX COMPLETE: {runs_ok} OK (incl {runs_skipped} skipped) | "
        f"{runs_failed} FAILED | {runs_timeout} TIMEOUT "
        f"(of {total_planned} total)"
    )
    print(f"   Total time: {elapsed_min:.2f} minutes")
    print(f"   Summary: {SUMMARY_CSV}")
    print("=" * 65)


def run_ablation_only(skip_existing=True):
    """
    Runs only E3_Hybrid_Complete and E3_NoRL across 6 scenarios × len(SEEDS) runs.
    Use this after the main batch is already complete.
    skip_existing=True means it won't re-run if CSV already exists.
    """
    ABLATION_ALGOS = ["E3_Hybrid_Complete", "E3_NoRL"]
    total = len(ABLATION_ALGOS) * len(SCENARIOS) * len(SEEDS)
    run_num = 0
    runs_ok = 0
    runs_failed = 0
    runs_timeout = 0

    print("\n" + "=" * 65)
    print("🧪 E³-HYBRID THESIS: ABLATION STUDY (RL Contribution)")
    print(f"   Scheduled: 2 Algos × 6 Scenarios × {len(SEEDS)} Seeds = {total} Runs")
    print(f"   Per-run timeout: {RUN_TIMEOUT_SEC}s")
    print("=" * 65 + "\n")

    total_start = time.time()

    for algo in ABLATION_ALGOS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                run_num += 1
                out_file = f"results/{algo}_scen{scenario}_seed{seed}.csv"
                log_filename = f"run_{algo}_scen{scenario}_seed{seed}.log"
                log_filepath = os.path.join(LOGS_DIR, log_filename)

                if skip_existing and os.path.exists(out_file):
                    try:
                        size = os.path.getsize(out_file)
                        if size > 50:
                            print(
                                f"---> [RUN {run_num}/{total}] SKIP (exists): "
                                f"[{algo}] Scen {scenario} Seed {seed}"
                            )
                            runs_ok += 1
                            continue
                    except OSError:
                        pass

                print(
                    f"---> [RUN {run_num}/{total}]: [{algo}] | "
                    f"Scen: {scenario} | Seed: {seed}"
                )

                cmd = [
                    sys.executable,
                    SIMULATION_ENTRY_POINT,
                    "--algo",     algo,
                    "--scenario", str(scenario),
                    "--seed",     str(seed),
                    "--nogui",
                ]

                start_time = time.time()
                status = "SUCCESS"

                with open(log_filepath, "w") as log_file:
                    log_file.write(
                        f"--- START: {datetime.now()} ---\n"
                        f"--- CMD: {' '.join(cmd)} ---\n\n"
                    )
                    log_file.flush()

                    process = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=THESIS_DIR,
                    )

                    try:
                        process.wait(timeout=RUN_TIMEOUT_SEC)
                        if process.returncode != 0:
                            status = f"FAILED (Code {process.returncode})"
                            runs_failed += 1
                            print(f"     [!] Exit code {process.returncode}.")
                        else:
                            runs_ok += 1
                            elapsed = time.time() - start_time
                            print(f"     [+] Completed in {elapsed:.2f}s.")

                    except subprocess.TimeoutExpired:
                        process.kill()
                        status = "TIMEOUT"
                        runs_timeout += 1
                        print(f"     [!] TIMEOUT after {RUN_TIMEOUT_SEC}s.")

                force_flush_memory()

    total_elapsed = time.time() - total_start
    elapsed_min = total_elapsed / 60.0

    print("\n" + "=" * 65)
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  {runs_ok} OK | {runs_failed} FAILED | {runs_timeout} TIMEOUT "
          f"(of {total} total)")
    print(f"   Total time: {elapsed_min:.2f} minutes")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    if "--ablation-only" in sys.argv:
        # Ablation only (E3_Full vs E3_NoRL)
        run_ablation_only(skip_existing=True)
    else:
        # Full 144-run evaluation matrix (sequential, one SUMO instance at a time).
        run_evaluation_matrix(skip_existing=True)
