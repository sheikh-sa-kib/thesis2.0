"""
algorithms/bco.py
─────────────────────────────────────────────────────────────────────────────
Bee Colony Optimizer — standalone swarm routing layer.

FIX A2 (run_sim.py side): The call site in run_sim.py used the wrong name
  bco_engine.process_onlookers() — corrected in run_sim.py to
  bco_engine.process_swarm_onlookers().

FIX (scout guard): process_swarm_onlookers() now auto-triggers scout
  exploration if called before scouts have run for this blocked edge,
  preventing silent no-ops when the call order is unexpected.
"""
import math
import os
import random
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import networkx as nx
import sumolib
import traci

from algorithms.route_validator import (
    validate_detour_entry,
    find_safe_intermediate_target,
    find_dynamic_bypass_targets,
)

# Optional charger network injected from E³-Hybrid


class BeeColonyOptimizer:

    def __init__(self, net_file):
        self.net_file     = os.path.expanduser(net_file)
        self.graph        = nx.DiGraph()
        self.scout_reports = {}   # edge_id -> list of route lists
        self.best_detours  = {}   # edge_id -> single best route list

        self.ratio_scouts    = 0.15
        self.ratio_employed  = 0.55
        self.ratio_onlookers = 0.30

        # Deferred re-assertion: {v_id: {final_edge, intermediate_edge, since_step}}
        self._pending_reassertion = {}
        self.charging_network = None   # injected from E³-Hybrid when used jointly

        self._build_static_graph()

    # ------------------------------------------------------------------ #
    # Graph construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_static_graph(self):
        print("[BCO BACKEND] Building static routing graph in memory...")
        net = sumolib.net.readNet(self.net_file)

        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            u         = edge.getFromNode().getID()
            v         = edge.getToNode().getID()
            length    = edge.getLength()
            speed     = edge.getSpeed()
            base_time = length / speed if speed > 0 else length

            self.graph.add_edge(
                u, v,
                edge_id=edge.getID(),
                weight=base_time,
                length=length,
            )

        print(
            f"[BCO BACKEND] Graph built: {len(self.graph.nodes)} nodes, "
            f"{len(self.graph.edges)} edges."
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    def _scout_score(self, edge, veh_id=None, battery_tracker=None):
        """
        BCO scout evaluates an edge.
        Adds charger bonus for low-battery vehicles near available chargers.
        """
        try:
            base_score = traci.edge.getTraveltime(edge)
        except traci.exceptions.TraCIException:
            base_score = 10.0

        if (veh_id is not None and battery_tracker is not None
                and self.charging_network is not None):
            soc = battery_tracker.get_soc(veh_id)
            if soc < 0.30:
                station = self.charging_network.station_at_edge(edge)
                if station and station.is_available:
                    charger_bonus = (0.30 - soc) * 200
                    base_score -= charger_bonus

        return base_score

    def compute_fitness(self, route_edges, current_step, veh_id=None,
                        battery_tracker=None):
        if not route_edges:
            return 0.0

        total_time = 0.0
        for edge_id in route_edges:
            try:
                if veh_id and battery_tracker:
                    total_time += self._scout_score(
                        edge_id, veh_id, battery_tracker
                    )
                else:
                    total_time += traci.edge.getTraveltime(edge_id)
            except traci.exceptions.TraCIException:
                found = False
                for u, v, data in self.graph.edges(data=True):
                    if data.get("edge_id") == edge_id:
                        total_time += data.get("weight", 10.0)
                        found = True
                        break
                if not found:
                    total_time += 10.0

        overlap_penalty = 1.0
        for existing_routes in self.scout_reports.values():
            for ext_route in existing_routes:
                shared = len(set(route_edges).intersection(ext_route))
                if shared > 2:
                    overlap_penalty += 0.5

        return 1.0 / (total_time * overlap_penalty)

    # ------------------------------------------------------------------ #
    # Scout exploration                                                    #
    # ------------------------------------------------------------------ #

    def trigger_scout_exploration(self, blocked_edge, current_step):
        print(
            f"[BCO SWARM] Deploying Scouts (15% fleet) to bypass '{blocked_edge}'..."
        )

        u, v = self._edge_to_nodes(blocked_edge)
        if not u or not v:
            return

        edge_data = self.graph.get_edge_data(u, v)
        if edge_data:
            self.graph.remove_edge(u, v)

        bypass_targets = find_dynamic_bypass_targets(self.net_file, blocked_edge)
        discovered_routes = []
        active_evs = [
            v_id for v_id in traci.vehicle.getIDList()
            if traci.vehicle.getTypeID(v_id) == "ev_swarm"
        ]
        scout_count = max(1, int(len(active_evs) * self.ratio_scouts))
        scouts = random.sample(active_evs, min(scout_count, len(active_evs)))

        for scout_id in scouts:
            try:
                curr_edge = traci.vehicle.getRoadID(scout_id)
                if curr_edge == blocked_edge:
                    continue
                try:
                    own_dest = traci.vehicle.getRoute(scout_id)[-1]
                except Exception:
                    own_dest = None
                candidate_targets = []
                for bypass_target in bypass_targets:
                    if bypass_target not in (blocked_edge, curr_edge):
                        candidate_targets.append(bypass_target)
                if own_dest and own_dest != blocked_edge:
                    candidate_targets.append(own_dest)
                candidate_targets = list(dict.fromkeys(candidate_targets))

                for dest_edge in candidate_targets:
                    start_node, _ = self._edge_to_nodes(curr_edge)
                    _, end_node   = self._edge_to_nodes(dest_edge)

                    if not (start_node and end_node and start_node != end_node):
                        continue

                    node_path = nx.shortest_path(
                        self.graph,
                        source=start_node,
                        target=end_node,
                        weight="weight",
                    )
                    route_edges = []
                    for i in range(len(node_path) - 1):
                        ed = self.graph.get_edge_data(node_path[i], node_path[i + 1])
                        if ed and "edge_id" in ed:
                            route_edges.append(ed["edge_id"])

                    if route_edges and blocked_edge not in route_edges:
                        fitness = self.compute_fitness(route_edges, current_step)
                        discovered_routes.append(
                            {"route": route_edges, "fitness": fitness}
                        )
                        break

            except (nx.NetworkXNoPath, nx.NodeNotFound,
                    traci.exceptions.TraCIException):
                continue

        if edge_data:
            self.graph.add_edge(u, v, **edge_data)

        discovered_routes.sort(key=lambda x: x["fitness"], reverse=True)

        if discovered_routes:
            self.scout_reports[blocked_edge] = [
                r["route"] for r in discovered_routes[:3]
            ]
            self.best_detours[blocked_edge] = discovered_routes[0]["route"]
            print(
                f"[BCO SWARM] Scouts cached "
                f"{len(self.scout_reports[blocked_edge])} unique detours."
            )
        else:
            print(f"[BCO SWARM] No detours found for '{blocked_edge}'.")

    # ------------------------------------------------------------------ #
    # Deferred re-assertion                                               #
    # ------------------------------------------------------------------ #

    def tick_pending_reassertions(self, current_step):
        completed = []
        for v_id, state in self._pending_reassertion.items():
            try:
                curr_road = traci.vehicle.getRoadID(v_id)
            except traci.exceptions.TraCIException:
                completed.append(v_id)
                continue

            intermediate = state["intermediate_edge"]
            final_edge   = state["final_edge"]

            if curr_road == intermediate or curr_road == final_edge:
                try:
                    traci.vehicle.setRoutingMode(
                        v_id, traci.constants.ROUTING_MODE_AGGREGATED
                    )
                    traci.vehicle.changeTarget(v_id, final_edge)
                    traci.vehicle.rerouteTraveltime(v_id)
                    print(
                        f"[BCO RE-ASSERT] {v_id} reached intermediate "
                        f"'{intermediate}' at step {current_step}. "
                        f"Swarm re-asserted '{final_edge}'."
                    )
                except traci.exceptions.TraCIException:
                    pass
                completed.append(v_id)

            elif current_step - state["since_step"] > 150:
                print(
                    f"[BCO RE-ASSERT] Timeout for {v_id} after 150 steps. "
                    f"Releasing to SUMO default."
                )
                completed.append(v_id)

        for v_id in completed:
            self._pending_reassertion.pop(v_id, None)

    # ------------------------------------------------------------------ #
    # 3-stage injection                                                    #
    # ------------------------------------------------------------------ #

    def _inject_route_to_vehicle(self, v_id, clean_detour, current_step,
                              blocked_edges=None):
        """
        Stage 1 — setRoute with stitched edge sequence.
        Stage 2 — Validated changeTarget (turn-restriction check passes).
        Stage 3 — Safe intermediate passthrough + deferred re-assertion.
        """
        curr_edge  = traci.vehicle.getRoadID(v_id)
        final_edge = clean_detour[-1]

        if str(curr_edge) in clean_detour:
            stitch_idx = clean_detour.index(str(curr_edge))
            new_route  = clean_detour[stitch_idx:]
        else:
            new_route = [str(curr_edge)] + clean_detour

        # Stage 1
        # Stage 1 — ask SUMO's own router first (guaranteed connectivity)
        try:
            sumo_route = traci.simulation.findRoute(curr_edge, final_edge)
            if sumo_route.edges and len(sumo_route.edges) >= 2:
                if not any(be in sumo_route.edges for be in (blocked_edges or [])):
                    traci.vehicle.setRoute(v_id, list(sumo_route.edges))
                    return True
            # Fallback: try our computed route directly
            traci.vehicle.setRoute(v_id, new_route)
            return True
        except traci.exceptions.TraCIException:
            pass

        # Stage 2
        can_enter, _ = validate_detour_entry(v_id, clean_detour)
        if can_enter is None:
            return False
        if can_enter:
            try:
                traci.vehicle.setRoutingMode(
                    v_id, traci.constants.ROUTING_MODE_AGGREGATED
                )
                traci.vehicle.changeTarget(v_id, final_edge)
                traci.vehicle.rerouteTraveltime(v_id)
                return True
            except traci.exceptions.TraCIException:
                return False

        # Stage 3
        intermediate = find_safe_intermediate_target(v_id, clean_detour)
        if intermediate:
            try:
                traci.vehicle.setRoutingMode(
                    v_id, traci.constants.ROUTING_MODE_DEFAULT
                )
                traci.vehicle.changeTarget(v_id, intermediate)
                traci.vehicle.rerouteTraveltime(v_id)
                self._pending_reassertion[v_id] = {
                    "final_edge":        final_edge,
                    "intermediate_edge": intermediate,
                    "since_step":        current_step,
                }
                print(
                    f"[BCO PASSTHROUGH] {v_id} → intermediate "
                    f"'{intermediate}' (final: '{final_edge}')."
                )
                return True
            except traci.exceptions.TraCIException:
                return False

        print(f"[BCO FALLBACK] No reachable entry for {v_id}. SUMO default.")
        return False

    # ------------------------------------------------------------------ #
    # Onlooker processing                                                  #
    # ------------------------------------------------------------------ #

    def process_swarm_onlookers(
        self, blocked_edge, logger, env_constraints, current_step
    ):
        """
        FIX (scout guard): If scouts haven't run yet for this edge,
        trigger them now before attempting onlooker processing.
        This closes the race condition where continuous re-evaluation
        fires at step 100 before the explicit scout call at step 100.
        """
        if blocked_edge not in self.scout_reports:
            print(
                f"[BCO ONLOOKER] No scout data for '{blocked_edge}' — "
                f"triggering scout exploration first."
            )
            self.trigger_scout_exploration(blocked_edge, current_step)

        detour_pool = self.scout_reports.get(blocked_edge, [])
        if not detour_pool:
            print(f"[BCO ONLOOKER] Scout found no valid detours. Skipping.")
            return

        # Only target EVs whose future route still passes through blocked edge
        candidates = []
        for v_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(v_id) != "ev_swarm":
                    continue
                remaining = traci.vehicle.getRoute(v_id)
                idx       = traci.vehicle.getRouteIndex(v_id)
                if blocked_edge in remaining[idx:]:
                    candidates.append(v_id)
            except traci.exceptions.TraCIException:
                continue

        if not candidates:
            print(
                f"[*] Step {current_step}: No EVs heading toward "
                f"blocked edge. Skipping."
            )
            return

        max_reroute = max(1, int(len(candidates) * self.ratio_onlookers))

        onlookers = random.sample(candidates, min(max_reroute, len(candidates)))

        rerouted = 0
        for v_id in onlookers:
            try:
                if env_constraints.check_driver_compliance(v_id):
                    chosen  = random.choice(detour_pool)
                    clean   = [str(e) for e in chosen]
                    success = self._inject_route_to_vehicle(v_id, clean, current_step)
                    if success:
                        logger.record_reroute(v_id)
                        rerouted += 1
            except Exception as e:
                print(f"   [SWARM ERROR] {v_id}: {e}")
                continue

        print(
            f"[*] Step {current_step}: {rerouted}/{len(candidates)} EVs "
            f"rerouted (heading toward blocked edge)."
        )
