"""
generate_dense_demand.py
─────────────────────────────────────────────────────────────────────────────
Generates a high-density demand file so ~200 vehicles are active at step 100.
Run ONCE before batch_evaluator.py:
    cd ~/thesis && python generate_dense_demand.py
"""
import os
import sys
import argparse
import subprocess
import xml.etree.ElementTree as ET

# Dynamically identify the project base folder path based on this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import sumolib

THESIS_DIR  = os.path.abspath(CURRENT_DIR)
NET_FILE    = os.path.join(THESIS_DIR, "network", "midtown.net.xml")
OUTPUT_FILE = os.path.join(THESIS_DIR, "demand", "dense_final.rou.xml")

# Force a clean, fully qualified absolute path string format for the XML parser
SUMOCFG     = os.path.abspath(os.path.join(THESIS_DIR, "simulation.sumocfg"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--period",
    default="0.35",
    help="randomTrips.py departure period; lower values generate denser demand.",
)
args = parser.parse_args()

print(f"Generating dense random trips with period={args.period}...")
trips_file = os.path.join(THESIS_DIR, "demand", "dense_trips.xml")
cmd = [
    sys.executable,
    os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
    "-n", NET_FILE,
    "-o", trips_file,
    "--end",    "3400",
    "--period", args.period,
    "--min-distance", "200",
    "--seed",   "42",
    "--trip-attributes", 'type="ev_swarm"',
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"randomTrips.py error:\n{result.stderr}")
    sys.exit(1)
print(f"Trips generated: {trips_file}")

print("Routing trips with duarouter...")
routed_file = os.path.join(THESIS_DIR, "demand", "dense_routed.rou.xml")
cmd = [
    "duarouter",
    "--net-file",      NET_FILE,
    "--route-files",   trips_file,
    "--output-file",   routed_file,
    "--ignore-errors", "true",
    "--repair",        "true",
    "--remove-loops",  "true",
    "--seed",          "42",
    "--no-warnings",   "true",
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"duarouter error:\n{result.stderr}")
    sys.exit(1)
print(f"Routed demand: {routed_file}")

print("Validating route connectivity...")
net = sumolib.net.readNet(NET_FILE)

valid_connections = set()
for edge in net.getEdges():
    for conn in edge.getOutgoing():
        valid_connections.add((edge.getID(), conn.getID()))

tree    = ET.parse(routed_file)
root    = tree.getroot()
removed = 0
kept    = 0

for vehicle in list(root.findall("vehicle")):
    route = vehicle.find("route")
    if route is None:
        root.remove(vehicle); removed += 1; continue
    edges = route.get("edges", "").split()
    if len(edges) < 2:
        kept += 1; continue
    broken = any(
        (edges[i], edges[i + 1]) not in valid_connections
        for i in range(len(edges) - 1)
    )
    if broken:
        root.remove(vehicle); removed += 1
    else:
        kept += 1

tree.write(OUTPUT_FILE, encoding="unicode", xml_declaration=True)
print(f"Removed {removed} broken vehicles. Kept {kept} clean vehicles.")
print(f"Dense demand file: {OUTPUT_FILE}")

# Parse and link the configuration map targets cleanly via absolute paths
if os.path.exists(SUMOCFG):
    cfg_tree = ET.parse(SUMOCFG)
    cfg_root = cfg_tree.getroot()
    updated  = False
    for elem in cfg_root.iter():
        val = elem.get("value", "")
        if val.endswith(".rou.xml") or val.endswith(".rou.alt.xml"):
            elem.set("value", "demand/dense_final.rou.xml")
            updated = True
            
    if updated:
        print("Updated sumocfg → demand/dense_final.rou.xml")
    cfg_tree.write(SUMOCFG, encoding="utf-8", xml_declaration=True)
else:
    print(f"[WARNING] Could not locate simulation file descriptor configuration at: {SUMOCFG}")

print("Done. Run: python batch_evaluator.py")