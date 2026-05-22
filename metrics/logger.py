"""
metrics/logger.py
─────────────────────────────────────────────────────────────────────────────
FIX: trigger_emergency() guard restored — only records the FIRST trigger.
     Without this, multi-block scenarios (2/3) overwrite emergency_start_time
     on every block event, making ERT measured from the last block instead of
     the first — artificially short and scientifically incorrect.
"""
import csv
import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import traci
from algorithms.aco import _energy_cost_wh


class SimulationLogger:

    def __init__(self, output_file, algo_name="Unknown", scenario_id=0):
        self.output_file = os.path.expanduser(output_file)
        self.algo_name   = algo_name
        self.scenario_id = scenario_id

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        self.travel_times      = {}
        self.waiting_times     = {}
        self.distances         = {}
        self.reroute_counts    = {}
        self.energy_wh         = {}  # Battery-Aware Routing: energy per vehicle (Wh)
        self.stranded_vehicles = set()

        self.emergency_active      = False
        self.emergency_start_time  = None
        self.emergency_clear_time  = None

        # Teleportation tracking
        self._sumo_teleport_total = 0
        self._guard_removal_total = 0

    def log_step(self, step):
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.exceptions.TraCIException:
            return

        for v_id in vehicle_ids:
            try:
                if traci.vehicle.getTypeID(v_id) != "ev_swarm":
                    continue
            except traci.exceptions.TraCIException:
                continue

            if v_id not in self.travel_times:
                self.travel_times[v_id]   = 0.0
                self.waiting_times[v_id]  = 0.0
                self.distances[v_id]      = 0.0
                self.reroute_counts[v_id] = 0

            self.travel_times[v_id] += 1.0

            try:
                speed = traci.vehicle.getSpeed(v_id)
            except traci.exceptions.TraCIException:
                continue

            if speed < 0.1:
                self.waiting_times[v_id] += 1.0

            self.distances[v_id] += speed

        # Drop metrics for vehicles that have left the simulation
        active = set(vehicle_ids)
        for stale_id in list(self.travel_times):
            if stale_id not in active:
                self.travel_times.pop(stale_id, None)
                self.waiting_times.pop(stale_id, None)
                self.distances.pop(stale_id, None)
                self.reroute_counts.pop(stale_id, None)

    def record_reroute(self, vehicle_id):
        if vehicle_id in self.reroute_counts:
            self.reroute_counts[vehicle_id] += 1

    def record_stranded(self, vehicle_id):
        self.stranded_vehicles.add(vehicle_id)

    def record_sumo_teleports(self, count):
        """Called from run_sim.py each step with
        traci.simulation.getStartingTeleportNumber()."""
        self._sumo_teleport_total += int(count)

    def trigger_emergency(self, step):
        self.emergency_active = True
        # CRITICAL: only record the FIRST trigger — never overwrite.
        # Scenarios 2/3 fire multiple block events; ERT must be measured
        # from when the FIRST block appeared, not the last.
        if self.emergency_start_time is None:
            self.emergency_start_time = step

    def clear_emergency(self, step):
        if (self.emergency_active
                and self.emergency_clear_time is None
                and self.emergency_start_time is not None
                and step >= self.emergency_start_time):
            self.emergency_clear_time = step
            self.emergency_active     = False

    def save_results(self, teleport_removals=0, battery_tracker=None, hybrid_algo=None):
        """Write one header + one data row (write mode, no accumulation)."""
        self._guard_removal_total = int(teleport_removals)
        total_teleport_events     = (self._sumo_teleport_total
                                     + self._guard_removal_total)

        total_vehicles = len(self.travel_times)
        if total_vehicles == 0:
            print("[LOGGER] No vehicles tracked — skipping save.")
            return

        avg_travel_time  = sum(self.travel_times.values())   / total_vehicles
        avg_waiting_time = sum(self.waiting_times.values())  / total_vehicles
        avg_distance     = sum(self.distances.values())      / total_vehicles
        avg_reroutes     = sum(self.reroute_counts.values()) / total_vehicles

        # Battery-Aware Routing: accumulate energy per completed vehicle
        total_energy_wh = 0.0
        for v_id in self.travel_times:
            try:
                route_edges = traci.vehicle.getRoute(v_id)
                trip_energy = sum(
                    _energy_cost_wh(e, traci.lane.getLength(e + "_0"))
                    for e in route_edges
                )
                total_energy_wh += trip_energy
            except Exception:
                # Vehicle may have been removed, skip
                pass
        avg_energy_wh = total_energy_wh / total_vehicles

        variance = (
            sum((t - avg_travel_time) ** 2 for t in self.travel_times.values())
            / total_vehicles
        )
        std_dev_travel_time = variance ** 0.5

        response_time = "N/A"
        if (self.emergency_start_time is not None
                and self.emergency_clear_time is not None):
            response_time = self.emergency_clear_time - self.emergency_start_time

        print("\n--- Simulation Complete: Final Metrics ---")
        print(f"Algorithm              : {self.algo_name}")
        print(f"Scenario               : {self.scenario_id}")
        print(f"Total Vehicles Tracked : {total_vehicles}")
        print(f"Average Travel Time    : {avg_travel_time:.2f} s")
        print(f"Travel Time Std Dev    : {std_dev_travel_time:.2f} s")
        print(f"Average Waiting Time   : {avg_waiting_time:.2f} s")
        print(f"Average Distance       : {avg_distance:.2f} m")
        print(f"Average Reroutes/Veh   : {avg_reroutes:.2f}")
        print(f"Average Energy/Veh     : {avg_energy_wh:.2f} Wh")
        stranded_battery = (
            battery_tracker.stranded_count
            if battery_tracker is not None
            else len(self.stranded_vehicles)
        )
        vehicles_charged = (
            len(hybrid_algo._charging_done)
            if hybrid_algo is not None and hasattr(hybrid_algo, "_charging_done")
            else 0
        )
        print(f"Stranded EVs (Battery) : {stranded_battery}")
        print(f"Vehicles Charged       : {vehicles_charged}")
        print(f"Emergency Response Time: {response_time} s")
        print(f"SUMO Teleports         : {self._sumo_teleport_total}")
        print(f"Guard Removals         : {self._guard_removal_total}")
        print(f"Total Teleport Events  : {total_teleport_events}")

        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm", "Scenario", "Total_Vehicles",
                "Avg_Travel_Time", "Std_Dev_Travel_Time",
                "Avg_Waiting_Time", "Avg_Distance", "Avg_Reroutes",
                "Avg_Energy_Wh", "Stranded_EVs", "Stranded_Due_To_Battery",
                "Vehicles_Charged", "Emergency_Response_Time",
                "SUMO_Teleports", "Guard_Removals", "Total_Teleport_Events",
            ])
            writer.writerow([
                self.algo_name, self.scenario_id, total_vehicles,
                f"{avg_travel_time:.2f}", f"{std_dev_travel_time:.2f}",
                f"{avg_waiting_time:.2f}", f"{avg_distance:.2f}",
                f"{avg_reroutes:.2f}", f"{avg_energy_wh:.2f}",
                len(self.stranded_vehicles),
                stranded_battery,
                vehicles_charged,
                response_time,
                self._sumo_teleport_total,
                self._guard_removal_total,
                total_teleport_events,
            ])

        print(f"[LOGGER] Results saved to: {self.output_file}")
