"""
setup_project.py
────────────────────────────────────────────────────────────────────────────
Run ONCE before everything else:
    cd ~/thesis && python setup_project.py
"""
import os

THESIS = os.path.expanduser("~/thesis")

PACKAGES = [
    "algorithms",
    "communication",
    "constraints",
    "metrics",
    "scenarios",
]

DIRS = [
    "results",
    "results/logs",
    "figures",
    "omnet_logs",
    "omnet_analysis",
    "network",
    "demand",
    "scenarios",
    "rl",
]

print("=" * 55)
print("  E3-Hybrid Project Setup")
print("=" * 55)

for pkg in PACKAGES:
    pkg_dir   = os.path.join(THESIS, pkg)
    init_file = os.path.join(pkg_dir, "__init__.py")
    os.makedirs(pkg_dir, exist_ok=True)
    if not os.path.exists(init_file):
        open(init_file, "w").close()
        print(f"  [+] Created: {init_file}")
    else:
        print(f"  [=] Exists:  {init_file}")

for d in DIRS:
    full = os.path.join(THESIS, d)
    os.makedirs(full, exist_ok=True)
    print(f"  [+] Dir: {full}")

print()
print("  Run in this exact order:")
print("  1.  python setup_project.py              (this file, once only)")
print("  2.  python select_important_edges.py     (once only)")
print("  3.  python generate_dense_demand.py      (once only)")
print("  4.  python batch_evaluator.py            (210 runs, ~hours)")
print("  5.  python aggregate_data.py")
print("  6.  python statistical_analysis.py")
print("  7.  python generate_plots.py             (10 figures)")
print("  8.  python omnet_analysis/analyze_v2x.py")
print("  9.  python thesis_verification.py        (final proof table)")
print("=" * 55)
