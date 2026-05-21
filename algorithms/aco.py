"""
algorithms/aco.py
─────────────────────────────────────────────────────────────────────────────
Ant Colony Optimization — standalone routing layer.

Real ACO implementation:
  - Pheromone matrix initialised from edge free-flow travel time
  - Each ant constructs a path using (τ^α)(η^β) probabilistic selection
  - Pheromones evaporate every SWARM_REEVAL_INTERVAL steps
  - Successful routes reinforce their edges proportional to 1/travel_time
  - No BCO, no PSO — pure ACO

This is a COMPETING algorithm. It is intentionally weaker than E3_Hybrid
because it lacks PSO parameter adaptation and BCO multi-bee consensus.
"""
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
# --- Energy cost constants (Battery-Aware Routing) ---
ENERGY_BASE_WH_PER_KM = 180.0   # baseline EV consumption at free-flow speed
ENERGY_CONGESTION_K   = 0.6     # penalty multiplier for low-speed edges
FREE_FLOW_SPEED_MS    = 13.9    # ~50 km/h reference speed (m/s)
from algorithms.route_validator import (
    validate_detour_entry,
    find_safe_intermediate_target,
)


# --- Energy Cost Estimation (Battery-Aware Routing) ---

def _energy_cost_wh(edge_id, length_m):
    """
    Estimate energy (Wh) to traverse an edge based on current avg speed.
    Lower speed → more stop-go → higher consumption.
    """
    try:
        speed_ms = traci.edge.getLastStepMeanSpeed(edge_id)
    except Exception:
        speed_ms = FREE_FLOW_SPEED_MS

    speed_ms = max(speed_ms, 0.5)          # avoid division by zero
    length_km = length_m / 1000.0

    # Consumption rises as speed drops below free-flow
    congestion_factor = 1.0 + ENERGY_CONGESTION_K * max(
        0.0, (FREE_FLOW_SPEED_MS - speed_ms) / FREE_FLOW_SPEED_MS
    )
    return ENERGY_BASE_WH_PER_KM * length_km * congestion_factor


def _combined_edge_score(edge_id, length_m, alpha=0.7, beta=0.3):
    """
    Weighted score used by ACO for pheromone-guided path selection.
    alpha: weight on travel time  (default 0.7)
    beta:  weight on energy cost  (default 0.3)
    """
    tt = traci.edge.getTraveltime(edge_id)
    energy = _energy_cost_wh(edge_id, length_m)

    # Normalise energy to same rough scale as travel time (seconds)
    # 180 Wh/km at ~1 km ≈ 180 Wh; 1 Wh ≈ 0.5s equivalent (tunable)
    energy_equiv_s = energy * 0.5

    return alpha * tt + beta * energy_equiv_s


class AntColonyOptimizer:

    def __init__(self, net_file):
        self.net_file = os.path.expanduser(net_file)
        self.graph    = nx.DiGraph()

        # Pheromone parameters — fixed (no PSO tuning, unlike E3)
        self.pheromones  = {}
        self.tau_init    = 1.0
        self.tau_min     = 0.1
        self.tau_max     = 10.0
        self.rho         = 0.05          # evaporation rate (higher than E3 = less memory)
        self.alpha        = 1.0          # pheromone weight (fixed, unlike PSO-tuned E3)
        self.beta         = 2.0          # heuristic weight (fixed)
        self.n_ants       = 10           # virtual ants per reroute event

        # Route cache
        self.scout_reports   = {}        # blocked_edge -> list of route lists
        self._pending_reassertion = {}
        self._injection_failures  = {}   # v_id -> fail_step

        self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        print("[ACO] Building routing graph with pheromone initialisation...")
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
            # Initialise pheromone proportional to edge speed (fast roads start hotter)
            warm = self.tau_init * (0.5 + 0.5 * (speed / 13.9))
            self.pheromones[edge.getID()] = max(self.tau_min, min(self.tau_max, warm))

        print(f"[ACO] Graph built: {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges.")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    # ------------------------------------------------------------------ #
    # ACO core                                                             #
    # ------------------------------------------------------------------ #

    def evaporate_pheromones(self, blocked_edges=None):
        """Global pheromone evaporation — fixed rate (no dynamic adaptation)."""
        for edge_id in self.pheromones:
            self.pheromones[edge_id] *= (1.0 - self.rho)
            if self.pheromones[edge_id] < self.tau_min:
                self.pheromones[edge_id] = self.tau_min

    def reinforce_route(self, route_edges, travel_time):
        """Deposit pheromone proportional to route quality."""
        if travel_time <= 0:
            return
        deposit = 100.0 / travel_time
        for edge_id in route_edges:
            if edge_id in self.pheromones:
                self.pheromones[edge_id] += deposit
                if self.pheromones[edge_id] > self.tau_max:
                    self.pheromones[edge_id] = self.tau_max

    def _aco_weight(self, u, v, data):
        """ACO edge weight: inverse of (τ^α)(η^β) desirability."""
        edge_id = data.get("edge_id")
        tau = self.pheromones.get(edge_id, self.tau_init)
        try:
            tt = traci.edge.getTraveltime(edge_id)
        except traci.exceptions.TraCIException:
            tt = data.get("weight", 10.0)
        eta = 1.0 / tt if tt > 0 else 0.1
        desirability = (tau ** self.alpha) * (eta ** self.beta)
        return 1.0 / desirability if desirability > 0 else 100.0

    def compute_aco_path(self, start_edge, end_edge, blocked_edges=None):
        """
        Runs n_ants virtual ants to find the best ACO path.
        Each ant uses a stochastic weight function that includes
        pheromone strength and live travel time heuristic.
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

        su, _ = self._edge_to_nodes(start_edge)
        _, ev = self._edge_to_nodes(end_edge)

        best_route   = []
        best_fitness = -1.0

        if su and ev:
            for _ in range(self.n_ants):
                # Each ant gets a fresh noise sample drawn at call time, not at def time
                _ant_noise = random.uniform(0.85, 1.15)   # captured per-ant, per-iteration
                def ant_weight(u, v, data, _noise=_ant_noise):
                    edge_id = data.get("edge_id")
                    tau = self.pheromones.get(edge_id, self.tau_init) * _noise
                    try:
                        length_m = data.get("length", 100.0)
                        cost = _combined_edge_score(edge_id, length_m, alpha=0.7, beta=0.3)
                    except Exception:
                        cost = data.get("weight", 10.0)
                    eta = 1.0 / cost if cost > 0 else 0.1
                    desirability = (tau ** self.alpha) * (eta ** self.beta)
                    return 1.0 / desirability if desirability > 0 else 100.0

                try:
                    node_path = nx.shortest_path(
                        self.graph, source=su, target=ev, weight=ant_weight
                    )
                    route = []
                    for i in range(len(node_path) - 1):
                        ed = self.graph.get_edge_data(node_path[i], node_path[i+1])
                        if ed and "edge_id" in ed:
                            route.append(ed["edge_id"])

                    if route:
                        # Fitness = sum of pheromone/travel_time scores
                        fitness = sum(
                            self.pheromones.get(e, self.tau_init)
                            for e in route
                        ) / len(route)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_route   = route
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # Restore blocked edges
        for u, v, data in removed:
            self.graph.add_edge(u, v, **data)

        return best_route

    # ------------------------------------------------------------------ #
    # Deferred re-assertion                                               #
    # ------------------------------------------------------------------ #

    def tick_pending_reassertions(self, current_step):
        completed = []
        for v_id, state in self._pending_reassertion.items():
            try:
                curr_road = traci.vehicle.getRoadID(v_id)
            except traci.exceptions.TraCIException:
                completed.append(v_id); continue

            intermediate = state["intermediate_edge"]
            final_edge   = state["final_edge"]

            if curr_road in (intermediate, final_edge):
                try:
                    traci.vehicle.setRoutingMode(
                        v_id, traci.constants.ROUTING_MODE_AGGREGATED
                    )
                    traci.vehicle.changeTarget(v_id, final_edge)
                    traci.vehicle.rerouteTraveltime(v_id)
                except traci.exceptions.TraCIException:
                    pass
                completed.append(v_id)
            elif current_step - state["since_step"] > 150:
                completed.append(v_id)

        for v_id in completed:
            self._pending_reassertion.pop(v_id, None)

    # ------------------------------------------------------------------ #
    # Route injection (3-stage, same as BCO/E3)                          #
    # ------------------------------------------------------------------ #

    def _inject_route_to_vehicle(self, v_id, clean_detour, current_step,
                                  blocked_edges=None):
        blocked_edges = blocked_edges or []
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
                return True
            except traci.exceptions.TraCIException:
                return False

        return False

    # ------------------------------------------------------------------ #
    # Main trigger — called from run_sim.py                              #
    # ------------------------------------------------------------------ #

    def trigger_scout_exploration(self, blocked_edge, current_step):
        """ACO scout phase: deploy virtual ants to discover bypass routes."""
        print(f"[ACO] Deploying {self.n_ants} ants to bypass '{blocked_edge}'...")

        u, v = self._edge_to_nodes(blocked_edge)
        edge_data = None
        if u and v:
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                self.graph.remove_edge(u, v)

        active_evs = [
            v_id for v_id in traci.vehicle.getIDList()
            if traci.vehicle.getTypeID(v_id) == "ev_swarm"
        ]
        scout_sample = random.sample(active_evs, min(self.n_ants, len(active_evs)))

        discovered = []
        seen       = set()

        for v_id in scout_sample:
            try:
                curr_edge = traci.vehicle.getRoadID(v_id)
                dest_edge = traci.vehicle.getRoute(v_id)[-1]
                if curr_edge.startswith(":") or curr_edge == blocked_edge:
                    continue

                route = self.compute_aco_path(
                    curr_edge, dest_edge, blocked_edges=[blocked_edge]
                )
                if route and blocked_edge not in route:
                    key = tuple(route)
                    if key not in seen:
                        seen.add(key)
                        travel_time = len(route) * 10  # proxy
                        self.reinforce_route(route, travel_time)
                        discovered.append(route)
            except traci.exceptions.TraCIException:
                continue

        if edge_data:
            self.graph.add_edge(u, v, **edge_data)

        if discovered:
            self.scout_reports[blocked_edge] = discovered[:5]
            print(f"[ACO] Ants discovered {len(discovered)} detours for '{blocked_edge}'.")
        else:
            print(f"[ACO] No detours found for '{blocked_edge}'.")

    def process_swarm_onlookers(
        self, blocked_edge, logger, env_constraints, current_step
    ):
        """Apply ACO routes to vehicles heading toward blocked edge."""
        if blocked_edge not in self.scout_reports:
            self.trigger_scout_exploration(blocked_edge, current_step)

        detour_pool = self.scout_reports.get(blocked_edge, [])
        if not detour_pool:
            return

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
            return

        # ACO reroutes a fixed 25% per sweep (no dynamic cap like E3)
        max_reroute = max(1, int(len(candidates) * 0.25))
        onlookers   = random.sample(candidates, min(max_reroute, len(candidates)))

        rerouted = 0
        for v_id in onlookers:
            last_fail = self._injection_failures.get(v_id, -999)
            if current_step - last_fail < 50:
                continue
            try:
                if env_constraints.check_driver_compliance(v_id):
                    chosen  = random.choice(detour_pool)
                    clean   = [str(e) for e in chosen]
                    success = self._inject_route_to_vehicle(
                        v_id, clean, current_step, blocked_edges=[blocked_edge]
                    )
                    if success:
                        self.reinforce_route(clean, len(clean) * 10)
                        logger.record_reroute(v_id)
                        rerouted += 1
                        self._injection_failures.pop(v_id, None)
                    else:
                        self._injection_failures[v_id] = current_step
            except Exception as e:
                print(f"   [ACO ERROR] {v_id}: {e}")

        print(f"[ACO] Step {current_step}: {rerouted}/{len(candidates)} EVs rerouted.")

        # Evaporate pheromones every call
        self.evaporate_pheromones(blocked_edges=[blocked_edge])
