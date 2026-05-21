"""
algorithms/dijkstra.py
──────────────────────
Dijkstra-based single-source shortest path router.
Uses networkx for graph traversal with live SUMO travel times.
One-shot rerouting triggered by V2X alert at first block event.
"""
import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import networkx as nx
import sumolib
import traci


class DijkstraRouter:

    def __init__(self, net_file):
        self.net_file = os.path.expanduser(net_file)
        self.graph    = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        print("[DIJKSTRA] Building static routing graph...")
        net = sumolib.net.readNet(self.net_file)
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
        print(f"[DIJKSTRA] Graph built: {len(self.graph.nodes)} nodes.")

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    def compute_shortest_path(self, start_edge, end_edge, blocked_edges=None):
        """
        Computes shortest path using Dijkstra with live SUMO travel times.
        Blocked edges are temporarily removed from graph.
        """
        blocked_edges = blocked_edges or []

        # Temporarily remove blocked edges
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
            su, _ = self._edge_to_nodes(start_edge)
            _, ev = self._edge_to_nodes(end_edge)
            if su and ev:
                node_path = nx.shortest_path(
                    self.graph, source=su, target=ev, weight=weight_fn
                )
                for i in range(len(node_path) - 1):
                    ed = self.graph.get_edge_data(node_path[i], node_path[i+1])
                    if ed and "edge_id" in ed:
                        route.append(ed["edge_id"])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # Restore blocked edges
        for u, v, data in removed:
            self.graph.add_edge(u, v, **data)

        return route

    def reroute_vehicle(self, v_id, blocked_edges=None, logger=None):
        """Reroutes a single vehicle using Dijkstra shortest path."""
        try:
            curr_edge = traci.vehicle.getRoadID(v_id)
            dest_edge = traci.vehicle.getRoute(v_id)[-1]
            if curr_edge.startswith(":"):
                return False
            new_route = self.compute_shortest_path(
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
