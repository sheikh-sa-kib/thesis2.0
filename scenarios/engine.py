import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import traci


class ScenarioEngine:

    def __init__(self, scenario_id=0):
        self.scenario_id          = int(scenario_id)
        self.blocked_edges        = []
        self.emergency_edges_pool = []
        self.blackout_active      = False
        self.first_block_step     = None

        # Tracking structural details for clearing loops
        self._blockage_registry   = {} # mapping: primary_edge -> [all_blocked_sub_edges]
        self._blockage_timestamps = {} # mapping: primary_edge -> step_injected
        self.load_emergency_edges()
        self.seed = None  # set externally before first process_step
        self._edges_shuffled = False

    def load_emergency_edges(self):
        edges_file = os.path.expanduser("~/thesis/scenarios/important_edges.txt")
        if os.path.exists(edges_file):
            with open(edges_file, "r") as f:
                self.emergency_edges_pool = [
                    line.strip() for line in f if line.strip()
                ]
        else:
            print(f"[SCENARIO ENGINE] Warning: {edges_file} not found.")

    def _block_edge(self, edge_id, current_step, v2x_layer, logger, severity=3):
        """Core mechanism that shuts down an edge and handles cascading severities."""
        if edge_id in self.blocked_edges:
            return

        # Target list of edges to disable based on severity
        edges_to_disable = [edge_id]

        # Multi-edge adjacent road cascading pile-up implementation
        if severity >= 4:
            try:
                # Find downstream edge lines natively from SUMO's network logic topology
                links = traci.edge.getLinks(edge_id)
                if links:
                    # Select the first viable neighboring edge connecting to this road segment
                    neighbor_edge = links[0][0]
                    if neighbor_edge and not neighbor_edge.startswith(":") and neighbor_edge not in self.blocked_edges:
                        edges_to_disable.append(neighbor_edge)
            except Exception:
                pass

        # Track the cluster registry for clean unblocking events later
        self._blockage_registry[edge_id] = edges_to_disable
        self._blockage_timestamps[edge_id] = current_step

        for target in edges_to_disable:
            if target in self.blocked_edges:
                continue
            try:
                # Pre-clearance reroute for vehicles already on this segment
                vehicles_on_edge = [
                    v for v in traci.vehicle.getIDList()
                    if traci.vehicle.getRoadID(v) == target
                ]
                for v_id in vehicles_on_edge:
                    try:
                        traci.vehicle.setRoutingMode(v_id, traci.constants.ROUTING_MODE_AGGREGATED)
                        traci.vehicle.rerouteTraveltime(v_id)
                    except traci.exceptions.TraCIException:
                        pass

                traci.edge.adaptTraveltime(target, 9999.0)
                traci.edge.setDisallowed(target, ["passenger"])
                self.blocked_edges.append(target)
            except traci.exceptions.TraCIException:
                pass

        if self.first_block_step is None:
            self.first_block_step = current_step
        logger.trigger_emergency(current_step)

        # Locate localized vehicle for V2X broadcast alert origin mapping
        broadcaster_id = None
        best_dist      = float("inf")
        try:
            blocked_pos = traci.lane.getShape(f"{edge_id}_0")[0]
            for v in traci.vehicle.getIDList():
                try:
                    vx, vy = traci.vehicle.getPosition(v)
                    dist = ((vx - blocked_pos[0]) ** 2 + (vy - blocked_pos[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist      = dist
                        broadcaster_id = v
                except traci.exceptions.TraCIException:
                    continue
        except Exception:
            pass

        if broadcaster_id is None:
            active = traci.vehicle.getIDList()
            if active:
                broadcaster_id = active[0]

        if broadcaster_id:
            notified = v2x_layer.broadcast_alert(
                sender_id=broadcaster_id,
                current_step=current_step,
                edge_id=edge_id,
                custom_packet_loss=(0.95 if self.blackout_active else v2x_layer.base_packet_loss),
            )
            print(
                f"[SCENARIO ENGINE] t={current_step}s: Edge Cluster '{edge_id}' blocked (Sev: {severity}). "
                f"V2X from '{broadcaster_id}' → {notified} EVs notified. Jamming Status: {self.blackout_active}"
            )

    def _unblock_cluster(self, primary_edge, current_step, v2x_layer):
        """Clears all cascading roads associated with a primary blockage cluster."""
        sub_edges = self._blockage_registry.pop(primary_edge, [primary_edge])
        self._blockage_timestamps.pop(primary_edge, None)

        for edge_id in sub_edges:
            if edge_id not in self.blocked_edges:
                continue
            try:
                traci.edge.adaptTraveltime(edge_id, -1.0)  # reset link cost to network default
                traci.edge.setAllowed(edge_id, ["passenger"])
                self.blocked_edges.remove(edge_id)
                print(f"[SCENARIO ENGINE] t={current_step}s: Sub-segment link '{edge_id}' re-opened.")
            except traci.exceptions.TraCIException:
                pass

    def process_step(self, current_step, v2x_layer, logger):
        if not self._edges_shuffled and self.emergency_edges_pool:
            import random
            self._rng = random.Random(self.seed if self.seed is not None else 42)
            self._rng.shuffle(self.emergency_edges_pool)
            self._edges_shuffled = True
        if not self.emergency_edges_pool or self.scenario_id == 0:
            return

        # ── Step 1: Evaluate Stochastic Time-Of-Day Shifts ──
        # Scales dynamically to 2400, 3600, or higher.
        # Splitting progress windows proportionally:
        # First 33% = Morning Peak Hour | Middle 33% = Off-Peak | Final 33% = Evening Peak Hour
        
        # Determine ticker check interval, probability boundaries, and structural intensities
        if current_step <= 1200:
            # Morning Peak: High frequency evaluation ticker every 120 seconds
            is_check_interval = (current_step % 120 == 0)
            disruption_probability = 0.65
            severity = 4  # Triggers neighbor cascaded edge blockages
            lifetime = 450
        elif 1200 < current_step <= 2400:
            # Off-Peak Midday: Lower vulnerability window every 240 steps
            is_check_interval = (current_step % 240 == 0)
            disruption_probability = 0.15
            severity = 2  # Standard isolated edge stall
            lifetime = 300
        else:
            # Evening Peak: Maximum complexity layout every 100 seconds
            is_check_interval = (current_step % 100 == 0)
            disruption_probability = 0.75
            severity = 5  # Complex cascade cluster block
            lifetime = 500

        # ── Step 2: Random V2X Communications Jamming Control (Scenario 4) ──
        # Activates blackout status organically with a low probability (e.g., 2% chance every 10 steps during gridlock)
        if self.scenario_id == 4 and current_step % 10 == 0:
            if not self.blackout_active and self.blocked_edges:
                if self._rng.random() < 0.02:  # 2% low probability entry trigger
                    self.blackout_active = True
                    print(f"[SCENARIO ENGINE] t={current_step}s: Dynamic Network Jamming Active! Drop rate: 95%.")
            elif self.blackout_active:
                if self._rng.random() < 0.08:  # 8% recovery check loop
                    self.blackout_active = False
                    print(f"[SCENARIO ENGINE] t={current_step}s: V2X Communication connectivity restored.")

        # ── Step 3: Inject Dynamic Disruption ──
        if is_check_interval and hasattr(self, "_rng"):
            if self._rng.random() < disruption_probability:
                # Select the next unallocated index from our seed-shuffled edge target array
                allocated_count = len(self._blockage_registry)
                if allocated_count < len(self.emergency_edges_pool):
                    next_target = self.emergency_edges_pool[allocated_count]
                    
                    # Enforce strict scenario constraints if specific configurations are called
                    active_severity = severity
                    if self.scenario_id == 1:
                        active_severity = 2 # Scenario 1 limits structural density to minor disruptions
                    
                    self._block_edge(next_target, current_step, v2x_layer, logger, severity=active_severity)

        # ── Step 4: Empirical Real-Time Clearing Lifecycle Engine ──
        if current_step % 20 == 0 and self._blockage_timestamps:
            for primary_edge, spawn_time in list(self._blockage_timestamps.items()):
                if current_step - spawn_time >= lifetime:
                    self._unblock_cluster(primary_edge, current_step, v2x_layer)