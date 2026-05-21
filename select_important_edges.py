"""
select_important_edges.py
─────────────────────────────────────────────────────────────────────────────
Selects 10 spatially distributed chokepoint edge IDs from the midtown network
and writes them to scenarios/important_edges.txt.
Run once before any simulation:
    cd ~/thesis && python select_important_edges.py
"""
import math
import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import sumolib

net_path   = os.path.expanduser("~/thesis/network/midtown.net.xml")
output_dir = os.path.expanduser("~/thesis/scenarios")
output_file = os.path.join(output_dir, "important_edges.txt")

os.makedirs(output_dir, exist_ok=True)
print("Reading Midtown network geometry...")

net   = sumolib.net.readNet(net_path)
edges = [e for e in net.getEdges() if not e.getID().startswith(":")]

candidate_edges = sorted(
    edges, key=lambda e: (e.getLaneNumber(), e.getLength()), reverse=True
)


def get_distance(edge1, edge2):
    bb1 = edge1.getBoundingBox()
    bb2 = edge2.getBoundingBox()
    c1  = ((bb1[0] + bb1[2]) / 2, (bb1[1] + bb1[3]) / 2)
    c2  = ((bb2[0] + bb2[2]) / 2, (bb2[1] + bb2[3]) / 2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


selected_edges      = []
min_separation_buffer = 300.0

for edge in candidate_edges:
    if len(selected_edges) >= 10:
        break
    too_close = any(
        get_distance(edge, sel) < min_separation_buffer
        for sel in selected_edges
    )
    if not too_close:
        selected_edges.append(edge)

top_10_ids = [e.getID() for e in selected_edges]

with open(output_file, "w") as f:
    for edge_id in top_10_ids:
        f.write(f"{edge_id}\n")

print(f"Extracted {len(top_10_ids)} chokepoint edges → {output_file}")
