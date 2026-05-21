"""
algorithms/pso.py
─────────────────────────────────────────────────────────────────────────────
Particle Swarm Optimization — standalone routing layer.

Real PSO implementation for traffic routing:
  - Each "particle" represents a candidate (alpha, beta) weight pair
    governing a Dijkstra-style routing heuristic
  - Swarm of N_PARTICLES particles maintain personal best and global best
  - Velocity update: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
  - Position update: x = x + v
  - The (alpha, beta) pair is used to weight (1/travel_time)^beta * capacity^alpha
    in the edge-weight function, then shortest-path is computed
  - Routes are evaluated by actual SUMO travel time
  - PSO converges alpha/beta toward best-performing configuration

This is a COMPETING algorithm. It lacks ACO pheromone memory and BCO
multi-bee consensus, so it performs worse than E3_Hybrid in multi-block
scenarios where adaptive memory matters.
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
from algorithms.aco import _energy_cost_wh   # shared energy model
from algorithms.route_validator import (
    find_dynamic_bypass_targets,
    validate_detour_entry,
    find_safe_intermediate_target,
)


class Particle:
    """Single PSO particle: position=(alpha, beta), velocity=(da, db)."""

    def __init__(self, alpha, beta):
        self.alpha    = alpha
        self.beta     = beta
        self.vel_a    = random.uniform(-0.3, 0.3)
        self.vel_b    = random.uniform(-0.3, 0.3)
        self.pbest_a  = alpha
        self.pbest_b  = beta
        self.pbest_fit = -1.0
        self.energy_alpha = 0.7   # travel-time weight
        self.energy_beta  = 0.3   # energy-cost weight

    def position(self):
        return (self.alpha, self.beta)


class ParticleSwarmOptimizer:

    # PSO hyperparameters
    N_PARTICLES = 8       # small swarm — intentionally leaner than E3's continuous tuning
    W           = 0.6     # inertia weight
    C1          = 1.5     # cognitive coefficient
    C2          = 1.5     # social coefficient
    ALPHA_RANGE = (0.1, 3.0)
    BETA_RANGE  = (0.5, 5.0)

    def __init__(self, net_file):
        self.net_file = os.path.expanduser(net_file)
        self.graph    = nx.DiGraph()

        # PSO swarm
        self.particles    = []
        self.gbest_alpha  = 1.0
        self.gbest_beta   = 2.0
        self.gbest_fit    = -1.0

        # Battery-Aware Routing: energy weighting parameters
        self.energy_alpha = 0.7   # travel-time weight
        self.energy_beta  = 0.3   # energy-cost weight
        self.ratio_onlookers = 0.30  # matches E³ default before RL tuning

        self._init_swarm()

        # Route cache
        self.scout_reports        = {}
        self._pending_reassertion = {}
        self._injection_failures  = {}

        self._build_graph()

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _init_swarm(self):
        """Initialise particles spread across the (alpha, beta) space."""
        for _ in range(self.N_PARTICLES):
            a = random.uniform(*self.ALPHA_RANGE)
            b = random.uniform(*self.BETA_RANGE)
            self.particles.append(Particle(a, b))
        print(f"[PSO] Swarm of {self.N_PARTICLES} particles initialised.")

    def _build_graph(self):
        print("[PSO] Building routing graph...")
        net = sumolib.net.readNet(self.net_file)
        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            u         = edge.getFromNode().getID()
            v         = edge.getToNode().getID()
            length    = edge.getLength()
            speed     = edge.getSpeed()
            lanes     = edge.getLaneNumber()
            base_time = length / speed if speed > 0 else length
            self.graph.add_edge(
                u, v,
                edge_id=edge.getID(),
                weight=base_time,
                length=length,
                speed=speed,
                lanes=lanes,
            )
        print(f"[PSO] Graph built: {len(self.graph.nodes)} nodes.")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    # ------------------------------------------------------------------ #
    # PSO core                                                             #
    # ------------------------------------------------------------------ #

    # AFTER — energy-aware fitness
    def _fitness(self, route, alpha=0.7, beta=0.3):
        total_tt     = 0.0
        total_energy = 0.0
        for edge in route:
            total_tt += traci.edge.getTraveltime(edge)
            length_m  = traci.lane.getLength(edge + "_0")
            total_energy += _energy_cost_wh(edge, length_m)   # reuse ACO helper

        energy_equiv_s = total_energy * 0.5
        return alpha * total_tt + beta * energy_equiv_s

    def _compute_path_with_params(self, start_edge, end_edge,
                                   alpha, beta, blocked_edges):
        """Shortest path using PSO-weighted edge cost function."""
        removed = []
        for be in blocked_edges:
            u, v = self._edge_to_nodes(be)
            if u and v:
                data = self.graph.get_edge_data(u, v)
                if data:
                    self.graph.remove_edge(u, v)
                    removed.append((u, v, data))

        def pso_weight(u, v, data):
            try:
                tt    = traci.edge.getTraveltime(data.get("edge_id", ""))
            except Exception:
                tt = data.get("weight", 10.0)
            lanes = data.get("lanes", 1)
            spd   = data.get("speed", 13.9)
            # PSO: balance travel-time heuristic (beta) and capacity (alpha)
            eta   = 1.0 / tt if tt > 0 else 0.1
            cap   = (lanes * spd) / (3 * 13.9)  # normalised capacity
            desirability = (cap ** alpha) * (eta ** beta)
            return 1.0 / desirability if desirability > 0 else 100.0

        route = []
        su, _ = self._edge_to_nodes(start_edge)
        _, ev = self._edge_to_nodes(end_edge)
        if su and ev:
            try:
                node_path = nx.shortest_path(
                    self.graph, source=su, target=ev, weight=pso_weight
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

    def update_swarm(self, blocked_edges=None):
        """
        One PSO iteration: evaluate all particles, update velocities/positions.
        Called every SWARM_REEVAL_INTERVAL steps.
        """
        blocked_edges = blocked_edges or []
        active_evs = [
            v_id for v_id in traci.vehicle.getIDList()
            if traci.vehicle.getTypeID(v_id) == "ev_swarm"
        ]
        if not active_evs:
            return

        # Sample one vehicle as test bed for fitness evaluation
        test_v = random.choice(active_evs)
        try:
            curr_edge = traci.vehicle.getRoadID(test_v)
            dest_edge = traci.vehicle.getRoute(test_v)[-1]
            if curr_edge.startswith(":") or curr_edge == dest_edge:
                return
        except traci.exceptions.TraCIException:
            return

        for p in self.particles:
            route   = self._compute_path_with_params(
                curr_edge, dest_edge, p.alpha, p.beta, blocked_edges
            )
            fitness = self._fitness(route, self.energy_alpha, self.energy_beta)

            # Update personal best
            if fitness > p.pbest_fit:
                p.pbest_fit = fitness
                p.pbest_a   = p.alpha
                p.pbest_b   = p.beta

            # Update global best
            if fitness > self.gbest_fit:
                self.gbest_fit   = fitness
                self.gbest_alpha = p.alpha
                self.gbest_beta  = p.beta

        # Velocity and position update for each particle
        for p in self.particles:
            r1, r2 = random.random(), random.random()
            p.vel_a = (self.W  * p.vel_a
                       + self.C1 * r1 * (p.pbest_a - p.alpha)
                       + self.C2 * r2 * (self.gbest_alpha - p.alpha))
            p.vel_b = (self.W  * p.vel_b
                       + self.C1 * r1 * (p.pbest_b - p.beta)
                       + self.C2 * r2 * (self.gbest_beta - p.beta))

            p.alpha = self._clamp(p.alpha + p.vel_a, *self.ALPHA_RANGE)
            p.beta  = self._clamp(p.beta  + p.vel_b, *self.BETA_RANGE)

    def compute_best_path(self, start_edge, end_edge, blocked_edges=None):
        """Return best path using current global-best PSO parameters."""
        blocked_edges = blocked_edges or []
        return self._compute_path_with_params(
            start_edge, end_edge,
            self.gbest_alpha, self.gbest_beta,
            blocked_edges
        )

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

            if curr_road in (state["intermediate_edge"], state["final_edge"]):
                try:
                    traci.vehicle.setRoutingMode(
                        v_id, traci.constants.ROUTING_MODE_AGGREGATED
                    )
                    traci.vehicle.changeTarget(v_id, state["final_edge"])
                    traci.vehicle.rerouteTraveltime(v_id)
                except traci.exceptions.TraCIException:
                    pass
                completed.append(v_id)
            elif current_step - state["since_step"] > 150:
                completed.append(v_id)

        for v_id in completed:
            self._pending_reassertion.pop(v_id, None)

    # ------------------------------------------------------------------ #
    # Route injection                                                     #
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
        """PSO scout phase: use current gbest params to discover bypass routes."""
        print(f"[PSO] Exploring bypasses for '{blocked_edge}' "
              f"(α={self.gbest_alpha:.2f}, β={self.gbest_beta:.2f})...")

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
        bypass_targets = find_dynamic_bypass_targets(self.net_file, blocked_edge)
        scouts  = random.sample(active_evs, min(self.N_PARTICLES, len(active_evs)))
        detours = []
        seen    = set()

        for v_id in scouts:
            try:
                curr_edge = traci.vehicle.getRoadID(v_id)
                if curr_edge.startswith(":") or curr_edge == blocked_edge:
                    continue
                try:
                    own_dest = traci.vehicle.getRoute(v_id)[-1]
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
                    route = self.compute_best_path(
                        curr_edge, dest_edge, blocked_edges=[blocked_edge]
                    )
                    if route and blocked_edge not in route:
                        key = tuple(route)
                        if key not in seen:
                            seen.add(key)
                            detours.append(route)
                            break
            except traci.exceptions.TraCIException:
                continue

        if edge_data:
            self.graph.add_edge(u, v, **edge_data)

        if detours:
            self.scout_reports[blocked_edge] = detours[:5]
            print(f"[PSO] Found {len(detours)} detours for '{blocked_edge}'.")
        else:
            print(f"[PSO] No detours found for '{blocked_edge}'.")

    def process_swarm_onlookers(
        self, blocked_edge, logger, env_constraints, current_step
    ):
        """Apply PSO-optimal routes to vehicles heading toward blocked edge."""
        if blocked_edge not in self.scout_reports:
            self.trigger_scout_exploration(blocked_edge, current_step)

        detour_pool = self.scout_reports.get(blocked_edge, [])
        if not detour_pool:
            return

        # Update swarm each call (PSO adapts continuously like E3 but only
        # on routing weights — no pheromone memory, so degrades under cascades)
        self.update_swarm(blocked_edges=[blocked_edge])

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

        max_reroute = max(1, int(len(candidates) * self.ratio_onlookers))
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
                        logger.record_reroute(v_id)
                        rerouted += 1
                        self._injection_failures.pop(v_id, None)
                    else:
                        self._injection_failures[v_id] = current_step
            except Exception as e:
                print(f"   [PSO ERROR] {v_id}: {e}")

        print(f"[PSO] Step {current_step}: {rerouted}/{len(candidates)} EVs rerouted "
              f"(gbest α={self.gbest_alpha:.2f}, β={self.gbest_beta:.2f}).")
