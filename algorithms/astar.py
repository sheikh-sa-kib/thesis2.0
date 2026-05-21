"""
algorithms/astar.py
───────────────────
A* router using geographic heuristic (Euclidean distance to goal).
Separates from Dijkstra by using a distance-based heuristic to
prioritize edges that move toward the destination, not just minimize
cumulative cost — produces different routes under congestion.
"""
import os
import sys
import math

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import networkx as nx
import sumolib
import traci


class AStarRouter:

    def __init__(self, net_file):
        self.net_file   = os.path.expanduser(net_file)
        self.graph      = nx.DiGraph()
        self.node_pos   = {}  # node_id -> (x, y) for heuristic
        self._build_graph()

    def _build_graph(self):
        print("[A*] Building routing graph with node positions...")
        net = sumolib.net.readNet(self.net_file)
        for node in net.getNodes():
            x, y = node.getCoord()
            self.node_pos[node.getID()] = (x, y)
        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            u      = edge.getFromNode().getID()
            v      = edge.getToNode().getID()
            length = edge.getLength()
            speed  = edge.getSpeed()
            weight = length / speed if speed > 0 else length
            self.graph.add_edge(
                u, v,
                edge_id=edge.getID(),
                weight=weight,
            )
        print(f"[A*] Graph built: {len(self.graph.nodes)} nodes.")

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    def _heuristic(self, node, goal_node):
        """Euclidean distance heuristic — makes A* prefer goal direction."""
        if node not in self.node_pos or goal_node not in self.node_pos:
            return 0.0
        x1, y1 = self.node_pos[node]
        x2, y2 = self.node_pos[goal_node]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2) / 13.9  # normalize by max speed

    def compute_astar_path(self, start_edge, end_edge, blocked_edges=None):
        """A* path using geographic heuristic + live travel times."""
        blocked_edges = blocked_edges or []

        removed = []
        for be in blocked_edges:
            u, v = self._edge_to_nodes(be)
            if u and v:
                data = self.graph.get_edge_data(u, v)
                if data:
                    self.graph.remove_edge(u, v)
                    removed.append((u, v, data))

        def weight_fn(u, v, data):
            edge_id = data.get("edge_id")
            try:
                tt = traci.edge.getTraveltime(edge_id)
                return tt if tt > 0 else data.get("weight", 10.0)
            except Exception:
                return data.get("weight", 10.0)

        route = []
        try:
            su, _  = self._edge_to_nodes(start_edge)
            _, ev  = self._edge_to_nodes(end_edge)
            if su and ev:
                node_path = nx.astar_path(
                    self.graph,
                    source=su,
                    target=ev,
                    heuristic=lambda n, g: self._heuristic(n, g),
                    weight=weight_fn,
                )
                for i in range(len(node_path) - 1):
                    ed = self.graph.get_edge_data(node_path[i], node_path[i+1])
                    if ed and "edge_id" in ed:
                        route.append(ed["edge_id"])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        for u, v, data in removed:
            self.graph.add_edge(u, v, **data)

        return route

    def reroute_vehicle(self, v_id, blocked_edges=None, logger=None):
        """Reroutes a single vehicle using A* path."""
        try:
            curr_edge = traci.vehicle.getRoadID(v_id)
            dest_edge = traci.vehicle.getRoute(v_id)[-1]
            if curr_edge.startswith(":"):
                return False
            new_route = self.compute_astar_path(
                curr_edge, dest_edge, blocked_edges=blocked_edges
            )
            if new_route:
                traci.vehicle.setRoute(v_id, new_route)
                if logger:
                    logger.record_reroute(v_id)
                return True
        except traci.exceptions.TraCIException:
            pass
        return False
