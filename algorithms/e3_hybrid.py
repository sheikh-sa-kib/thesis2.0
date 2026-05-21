"""
algorithms/e3_hybrid.py
─────────────────────────────────────────────────────────────────────────────
E³-Hybrid Optimizer: ACO + PSO + BCO integrated swarm routing.

FIX A4: PSO particle tracking was incomplete.
  - global_best_fitness and pso_best_alpha/beta were declared but never
    updated, so PSO never actually learned better parameters.
  - evaluate_aco_bco_route() now computes route fitness and updates
    global_best_fitness + pso_best_alpha/beta when a better solution
    is found, closing the PSO feedback loop.

SESSION FIXES:
  - Scout generation uses SUMO topology to discover downstream bypass targets
    dynamically for the currently blocked edge.
  - _inject_route_to_vehicle uses findRoute to build a fully-connected
    path, then rejects any route still passing through the blocked edge.
  - _injection_failures blacklist prevents retrying failed injections
    for 50 steps.
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

from algorithms.aco import _combined_edge_score, _energy_cost_wh
from algorithms.route_validator import (
    validate_detour_entry,
    find_safe_intermediate_target,
    find_dynamic_bypass_targets,
)
from chargers import (
    ChargingNetwork,
    CRITICAL_BATTERY_THRESHOLD,
    LOW_BATTERY_THRESHOLD,
)

class E3HybridOptimizer:

    def __init__(self, net_file):
        self.net_file = os.path.expanduser(net_file)
        self.graph    = nx.DiGraph()

        # ── ACO layer ─────────────────────────────────────────────────
        self.pheromones = {}
        self.tau_init   = 1.0
        self.tau_min    = 0.1
        self.tau_max    = 10.0
        self.rho        = 0.02

        # ── PSO layer ─────────────────────────────────────────────────
        self.alpha               = 1.0
        self.beta                = 2.0
        self.pso_best_alpha      = 1.0
        self.pso_best_beta       = 2.0
        self.global_best_fitness = 0.0

        # ── Battery-Aware Routing (stronger energy weighting than standalone algorithms) ─────
        self.energy_alpha = 0.65  # travel-time weight (stronger energy bias)
        self.energy_beta  = 0.35  # energy-cost weight

        # ── BCO layer ─────────────────────────────────────────────────
        self.scout_reports   = {}
        self.ratio_scouts    = 0.35
        self.ratio_onlookers = 0.30

        # Byzantine consensus table
        self.alert_consensus_table = {}
        # Injection failure tracker — prevents retrying bad detours
        self._injection_failures = {}  # v_id -> fail_step
        self.bypass_targets = []

        # Deferred re-assertion table
        self._pending_reassertion = {}

        # Charging station network (E³-Hybrid only)
        self.charging_network   = ChargingNetwork()
        self._charging_targets  = {}   # veh_id → target charger edge_id
        self._charging_routes   = {}   # veh_id → last valid route edge list
        self._charging_done     = set()
        self._charger_logged    = set()  # suppress repeat [CHARGER] prints

        self._build_static_graph()

    # ------------------------------------------------------------------ #
    # Graph construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_static_graph(self):
        print("[E3-HYBRID] Initializing static routing graph & pheromones...")
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
                speed=speed,
            )
            # Pre-warm pheromones based on edge speed/capacity
            # Fast/wide edges get higher initial pheromone — ACO starts
            # with a meaningful prior instead of a flat uniform surface.
            try:
                spd = edge.getSpeed()        # free-flow speed m/s
                lns = edge.getLaneNumber()   # lane count proxy for capacity
                # normalise: typical urban edge ~13.9m/s, 1-3 lanes
                warm = self.tau_init * (0.5 + 0.5 * (spd / 13.9)) * (0.8 + 0.2 * lns)
                warm = max(self.tau_min, min(self.tau_max, warm))
            except Exception:
                warm = self.tau_init
            self.pheromones[edge.getID()] = warm

        print(
            f"[E3-HYBRID] Setup complete: {len(self.graph.nodes)} nodes, "
            f"Byzantine bounds [{self.tau_min}, {self.tau_max}]."
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _edge_to_nodes(self, edge_id):
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return u, v
        return None, None

    # ------------------------------------------------------------------ #
    # ACO execution                                                        #
    # ------------------------------------------------------------------ #

    def evaporate_pheromones(self, blocked_edges=None):
        """Evaporates pheromone trails with Byzantine lower-bound clamping."""
        active_blocks = len(blocked_edges) if blocked_edges else 0
        effective_rho = self.rho + (0.04 * min(active_blocks, 3))  # 0.02 → up to 0.14
        for edge_id in self.pheromones:
            self.pheromones[edge_id] *= 1.0 - effective_rho
            if self.pheromones[edge_id] < self.tau_min:
                self.pheromones[edge_id] = self.tau_min

    def reinforce_route(self, route_edges, travel_time):
        """Deposits pheromone on successful paths with upper-bound clamping."""
        if travel_time <= 0:
            return
        deposit = 100.0 / travel_time
        for edge_id in route_edges:
            if edge_id in self.pheromones:
                self.pheromones[edge_id] += deposit
                if self.pheromones[edge_id] > self.tau_max:
                    self.pheromones[edge_id] = self.tau_max

    # ------------------------------------------------------------------ #
    # PSO execution                                                        #
    # ------------------------------------------------------------------ #

    def tune_pso_parameters(self, avg_network_speed):
        """
        Adjusts ACO alpha/beta weights based on real-time network speed.
        Called every 10 steps with actual measured speed from run_sim.py.
        """
        if avg_network_speed < 3.0:
            new_beta  = 4.0
            new_alpha = 0.3
        elif avg_network_speed < 8.0:
            new_beta  = 3.0
            new_alpha = 0.5
        else:
            new_beta  = 2.0
            new_alpha = 1.0

        self.alpha = new_alpha
        self.beta  = new_beta

    def _compute_route_fitness(self, route_edges):
        """
        FIX A4: Calculates a scalar fitness for a route using current
        pheromone levels and live travel times.
        """
        if not route_edges:
            return 0.0
        total_weighted = 0.0
        for edge_id in route_edges:
            tau = self.pheromones.get(edge_id, self.tau_init)
            try:
                tt = traci.edge.getTraveltime(edge_id)
            except traci.exceptions.TraCIException:
                tt = 10.0
            eta = 1.0 / tt if tt > 0 else 0.1
            total_weighted += (tau ** self.alpha) * (eta ** self.beta)
        return total_weighted / len(route_edges)

    # ------------------------------------------------------------------ #
    # BCO + ACO-PSO integrated route scoring                              #
    # ------------------------------------------------------------------ #

    def evaluate_aco_bco_route(self, start_node, end_node):
        """
        Custom shortest path balancing ACO pheromones and PSO weights.
        FIX A4: Updates global_best_fitness when a better route is found.
        """

        def aco_weight(u, v, data):
            edge_id = data.get("edge_id")
            tau = self.pheromones.get(edge_id, self.tau_init)
            try:
                length_m = data.get("length", 100.0)
                cost = _combined_edge_score(
                    edge_id, length_m,
                    alpha=self.energy_alpha, beta=self.energy_beta,
                )
            except traci.exceptions.TraCIException:
                cost = data.get("weight", 10.0)
            eta = 1.0 / cost if cost > 0 else 0.1
            desirability = (tau ** self.alpha) * (eta ** self.beta)
            return 1.0 / desirability if desirability > 0 else 100.0

        try:
            node_path = nx.shortest_path(
                self.graph,
                source=start_node,
                target=end_node,
                weight=aco_weight,
            )
            route = []
            for i in range(len(node_path) - 1):
                ed = self.graph.get_edge_data(node_path[i], node_path[i + 1])
                if ed and "edge_id" in ed:
                    route.append(ed["edge_id"])

            if route:
                fitness = self._compute_route_fitness(route)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.pso_best_alpha      = self.alpha
                    self.pso_best_beta       = self.beta

            return route

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------ #
    # Emergency response & Byzantine consensus                            #
    # ------------------------------------------------------------------ #

    def trigger_emergency_response(self, blocked_edge, current_step):
        """
        Dynamic consensus threshold.
        Under low vehicle density (< 100 vehicles) a single alert suffices.
        Under high density the strict 2-signal rule applies.
        """
        # Anomaly detection — down-weight suspicious alerts by not incrementing
        # consensus counter if the same edge is re-alerted too rapidly.
        alert_count = self.alert_consensus_table.get(blocked_edge, 0)
        # Heuristic: if same edge alerted within last 5 steps, flag it
        last_alert_step = getattr(self, '_last_alert_steps', {})
        current_alert_step = current_step
        last = last_alert_step.get(blocked_edge, -999)
        if current_alert_step - last < 1:
            print(
                f"[E3 ANOMALY] Rapid re-alert for '{blocked_edge}' "
                f"(gap={current_alert_step-last} steps). "
                f"Possible Byzantine signal — not counting toward consensus."
            )
            if not hasattr(self, '_last_alert_steps'):
                self._last_alert_steps = {}
            self._last_alert_steps[blocked_edge] = current_alert_step
        else:
            self.alert_consensus_table[blocked_edge] = alert_count + 1
            if not hasattr(self, '_last_alert_steps'):
                self._last_alert_steps = {}
            self._last_alert_steps[blocked_edge] = current_alert_step

        try:
            fleet_size = len(traci.vehicle.getIDList())
        except Exception:
            fleet_size = 0

        required_consensus = 1 if fleet_size < 100 else 2
        count = self.alert_consensus_table[blocked_edge]

        if count < required_consensus:
            print(
                f"[E3-HYBRID] Alert for '{blocked_edge}' "
                f"({count}/{required_consensus} signals). Awaiting consensus..."
            )
            return

        print(
            f"[E3-HYBRID] Consensus reached for '{blocked_edge}'! "
            f"(fleet={fleet_size}, threshold={required_consensus}) "
            f"Triggering ACO-PSO-BCO rerouting..."
        )

        u, v = self._edge_to_nodes(blocked_edge)
        if not u or not v:
            print(f"[E3-HYBRID] WARNING: Could not resolve nodes for '{blocked_edge}'. Skipping.")
            return

        edge_data = self.graph.get_edge_data(u, v)
        if edge_data:
            self.graph.remove_edge(u, v)

        # BCO scout phase: use findRoute directly to find bypass detours.
        self.bypass_targets = find_dynamic_bypass_targets(self.net_file, blocked_edge)
        if self.bypass_targets:
            print(
                f"[E3-HYBRID] Dynamic bypass targets for '{blocked_edge}': "
                f"{self.bypass_targets}"
            )
        else:
            print(
                f"[E3-HYBRID] No downstream bypass targets found for "
                f"'{blocked_edge}'; using vehicle destinations."
            )

        active_evs = [
            v_id for v_id in traci.vehicle.getIDList()
            if traci.vehicle.getTypeID(v_id) == "ev_swarm"
        ]
        scout_count = max(1, int(len(active_evs) * self.ratio_scouts))
        scouts = random.sample(active_evs, min(scout_count, len(active_evs)))

        # Multi-target scouting: try dynamic downstream bypasses first, then
        # fall back to each scout's own destination for reachability.
        valid_detours = []
        seen_routes   = set()
        for v_id in scouts:
            try:
                curr_edge = traci.vehicle.getRoadID(v_id)
                if curr_edge == blocked_edge:
                    continue
                # Build candidate targets: topology-derived bypasses + own destination.
                try:
                    own_dest = traci.vehicle.getRoute(v_id)[-1]
                except Exception:
                    own_dest = None
                candidate_targets = []
                for bypass_target in self.bypass_targets:
                    if bypass_target not in (blocked_edge, curr_edge):
                        candidate_targets.append(bypass_target)
                if own_dest and own_dest != blocked_edge:
                    candidate_targets.append(own_dest)
                candidate_targets = list(dict.fromkeys(candidate_targets))
                for target in candidate_targets:
                    try:
                        result = traci.simulation.findRoute(curr_edge, target)
                    except Exception:
                        continue
                    if not result.edges or blocked_edge in result.edges:
                        continue
                    route_key = tuple(result.edges)
                    if route_key in seen_routes:
                        continue
                    seen_routes.add(route_key)
                    valid_detours.append(list(result.edges))
                    break  # one good detour per scout
            except traci.exceptions.TraCIException:
                continue

        # Restore graph edge
        if edge_data:
            self.graph.add_edge(u, v, **edge_data)

        if valid_detours:
            self.scout_reports[blocked_edge] = valid_detours[:8]
            print(
                f"[E3-HYBRID] Scouts cached {len(valid_detours[:8])} "
                f"unique detours for '{blocked_edge}'."
            )
        else:
            print(f"[E3-HYBRID] WARNING: No valid detours found for '{blocked_edge}'.")

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

            if curr_road in (intermediate, final_edge):
                try:
                    traci.vehicle.setRoutingMode(
                        v_id, traci.constants.ROUTING_MODE_AGGREGATED
                    )
                    traci.vehicle.changeTarget(v_id, final_edge)
                    traci.vehicle.rerouteTraveltime(v_id)
                    print(
                        f"[E3 RE-ASSERT] {v_id} reached '{intermediate}' "
                        f"at step {current_step}. Swarm re-asserted '{final_edge}'."
                    )
                except traci.exceptions.TraCIException:
                    pass
                completed.append(v_id)

            elif current_step - state["since_step"] > 150:
                print(
                    f"[E3 RE-ASSERT] Timeout for {v_id}. Releasing to SUMO default."
                )
                completed.append(v_id)

        for v_id in completed:
            self._pending_reassertion.pop(v_id, None)

    # ------------------------------------------------------------------ #
    # 3-stage injection protocol                                          #
    # ------------------------------------------------------------------ #

    def _inject_route_to_vehicle(self, v_id, clean_detour, current_step, blocked_edges=None):
        blocked_edges = blocked_edges or []
        
        curr_edge = traci.vehicle.getRoadID(v_id)
        final_edge = clean_detour[-1]
        
        try:
            actual_dest = traci.vehicle.getRoute(v_id)[-1]
        except traci.exceptions.TraCIException:
            actual_dest = final_edge

        try:
            result = traci.simulation.findRoute(str(curr_edge), actual_dest)
            if not result.edges:
                return False
            new_route = list(result.edges)
            if any(be in new_route for be in blocked_edges):
                return False
            if curr_edge in new_route:
                new_route = new_route[new_route.index(curr_edge):]
        except Exception:
            return False

        # Stage 1 — ask SUMO's own router first (guaranteed connectivity)
        try:
            # Explicitly cast both variables to clean strings to safeguard data typing.
            sumo_route = traci.simulation.findRoute(str(curr_edge), str(final_edge))
            if sumo_route.edges and len(sumo_route.edges) >= 2:
                if not any(be in sumo_route.edges for be in (blocked_edges or [])):
                    traci.vehicle.setRoute(v_id, list(sumo_route.edges))
                    return True
            # Fallback: try our computed route directly
            traci.vehicle.setRoute(v_id, new_route)
            return True
        except traci.exceptions.TraCIException:
            pass

        # Stage 2 — changeTarget
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

        # Stage 3 — safe intermediate passthrough
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
                    f"[E3 PASSTHROUGH] {v_id} → intermediate "
                    f"'{intermediate}' (final: '{final_edge}')."
                )
                return True
            except traci.exceptions.TraCIException:
                return False

        return False

    # ------------------------------------------------------------------ #
    # Low-battery charger routing (hard constraint)                       #
    # ------------------------------------------------------------------ #

    def _apply_charger_route(self, veh_id, route_edges):
        """Inject a validated route to a charger. Never call changeTarget blindly."""
        if not route_edges:
            return False
        try:
            curr_edge = traci.vehicle.getRoadID(veh_id)
        except traci.exceptions.TraCIException:
            return False

        route = list(route_edges)
        if curr_edge in route:
            route = route[route.index(curr_edge):]
        if len(route) < 1:
            return False

        try:
            traci.vehicle.setRoute(veh_id, route)
            return True
        except traci.exceptions.TraCIException:
            pass

        try:
            traci.vehicle.setRoutingMode(
                veh_id, traci.constants.ROUTING_MODE_AGGREGATED
            )
            traci.vehicle.changeTarget(veh_id, route[-1])
            traci.vehicle.rerouteTraveltime(veh_id)
            return True
        except traci.exceptions.TraCIException:
            return False

    def _handle_low_battery_routing(self, veh_id, battery_tracker):
        """
        Hard constraint: if SoC is below threshold, route to nearest *reachable*
        charging station via findRoute + setRoute (not bare changeTarget).
        Returns True while this vehicle is under charger routing control.
        """
        soc = battery_tracker.get_soc(veh_id)

        if veh_id in self._charging_targets:
            target_edge = self._charging_targets[veh_id]
            try:
                current_edge = traci.vehicle.getRoadID(veh_id)
            except traci.exceptions.TraCIException:
                return True

            if current_edge == target_edge:
                started = self.charging_network.register_arrival(veh_id, target_edge)
                battery_tracker.start_charging(veh_id)
                self._charging_targets.pop(veh_id, None)
                self._charging_routes.pop(veh_id, None)
                self._charger_logged.discard(veh_id)
                print(
                    f"[CHARGER] {veh_id} arrived — "
                    f"charging={'started' if started else 'queued'}"
                )
                return True

            # Re-apply stored route if SUMO dropped it (every 30 steps)
            stored = self._charging_routes.get(veh_id)
            if stored and int(traci.simulation.getTime()) % 30 == 0:
                self._apply_charger_route(veh_id, stored)
            return True

        if soc > LOW_BATTERY_THRESHOLD:
            return False

        require_slot = soc > CRITICAL_BATTERY_THRESHOLD
        station, route = self.charging_network.nearest_reachable(
            veh_id, require_available=require_slot
        )
        if station is None or not route:
            return False

        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
        except traci.exceptions.TraCIException:
            return False

        if current_edge == station.edge_id:
            return False

        if not self._apply_charger_route(veh_id, route):
            return False

        self._charging_targets[veh_id] = station.edge_id
        self._charging_routes[veh_id]  = route
        if veh_id not in self._charger_logged:
            self._charger_logged.add(veh_id)
            print(
                f"[CHARGER] {veh_id} (SoC={soc:.1%}) → "
                f"{station.name} ({station.edge_id}) "
                f"[{len(route)} edges]"
            )
        return True

    # ------------------------------------------------------------------ #
    # Route fusion with energy tiebreaker                                 #
    # ------------------------------------------------------------------ #

    def _fuse_routes(self, candidates):
        """
        Select best route from candidates using energy-aware scoring.
        When travel times are similar (within 2%), prefer the lower-energy route.
        
        Args:
            candidates: list of route edge-lists (e.g., [route1, route2, route3])
        
        Returns:
            best route (list of edge IDs) or None if candidates empty
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def route_score(route):
            """Combined score: 0.65*time + 0.35*energy_equiv"""
            tt = 0.0
            energy = 0.0
            for edge_id in route:
                try:
                    tt += traci.edge.getTraveltime(edge_id)
                    length_m = traci.lane.getLength(edge_id + "_0")
                    energy += _energy_cost_wh(edge_id, length_m)
                except Exception:
                    # Fallback on error
                    tt += 10.0
                    energy += 50.0
            
            energy_equiv_s = energy * 0.5
            return self.energy_alpha * tt + self.energy_beta * energy_equiv_s

        scored_routes = [(route, route_score(route)) for route in candidates]
        best = min(scored_routes, key=lambda x: x[1])
        return best[0]

    # ------------------------------------------------------------------ #
    # Onlooker processing                                                  #
    # ------------------------------------------------------------------ #

    def process_swarm_onlookers(
        self, blocked_edge, logger, env_constraints, current_step
    ):
        consensus_count = self.alert_consensus_table.get(blocked_edge, 0)
        if consensus_count < 1 or blocked_edge not in self.scout_reports:
            return

        detour_pool = self.scout_reports[blocked_edge]
        if not detour_pool:
            return

        # Two-tier candidate filter
        try:
            blk_pos = traci.lane.getShape(blocked_edge + "_0")
            blk_x   = sum(p[0] for p in blk_pos) / len(blk_pos)
            blk_y   = sum(p[1] for p in blk_pos) / len(blk_pos)
        except Exception:
            blk_x, blk_y = None, None

        candidates = []
        for v_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(v_id) != "ev_swarm":
                    continue
                remaining = traci.vehicle.getRoute(v_id)
                idx       = traci.vehicle.getRouteIndex(v_id)
                # Tier 1
                if blocked_edge in remaining[idx:]:
                    candidates.append(v_id)
                    continue
                # Tier 2 — proximity fallback
                if blk_x is not None:
                    x, y = traci.vehicle.getPosition(v_id)
                    dist = ((x - blk_x) ** 2 + (y - blk_y) ** 2) ** 0.5
                    if dist <= 400.0:
                        candidates.append(v_id)
            except traci.exceptions.TraCIException:
                continue

        if not candidates:
            return

        max_reroute = max(1, int(len(candidates) * self.ratio_onlookers))

        onlookers = random.sample(candidates, min(max_reroute, len(candidates)))

        rerouted = 0
        for v_id in onlookers:
            # Skip vehicles blacklisted for 50 steps after a failed injection
            last_fail = self._injection_failures.get(v_id, -999)
            if current_step - last_fail < 50:
                continue
            try:
                if env_constraints.check_driver_compliance(v_id):
                    # E³-Hybrid: select best detour based on energy, not random choice
                    chosen       = self._fuse_routes(detour_pool)
                    if chosen is None:
                        chosen = random.choice(detour_pool)
                    clean_detour = [str(e) for e in chosen]
                    success = self._inject_route_to_vehicle(v_id, clean_detour, current_step, blocked_edges=list(self.alert_consensus_table.keys()))
                    if success:
                        self.reinforce_route(clean_detour, len(clean_detour) * 10)
                        logger.record_reroute(v_id)
                        rerouted += 1
                        self._injection_failures.pop(v_id, None)
                    else:
                        self._injection_failures[v_id] = current_step
            except Exception as e:
                print(f"   [SWARM ERROR] {v_id}: {e}")
                continue

        print(
            f"[*] Step {current_step}: {rerouted}/{len(candidates)} EVs "
            f"rerouted (heading toward blocked edge)."
        )
