"""
Charging Station Network for E³-Hybrid Battery-Aware Routing.
Defines 10 stations across Midtown Manhattan with queuing,
availability tracking, and nearest-charger lookup.
"""

import math

import traci

# ---------------------------------------------------------------------------
# Station definitions — approximate Midtown Manhattan locations
# Edge IDs are resolved at runtime via nearest-edge lookup (see resolve_edges)
# ---------------------------------------------------------------------------
CHARGING_STATIONS_LATLON = [
    {"id": "CS_01", "name": "Penn Station",      "lat": 40.7506, "lon": -73.9971, "slots": 4},
    {"id": "CS_02", "name": "Grand Central",      "lat": 40.7527, "lon": -73.9772, "slots": 4},
    {"id": "CS_03", "name": "Hudson Yards",       "lat": 40.7535, "lon": -74.0022, "slots": 6},
    {"id": "CS_04", "name": "Times Square",       "lat": 40.7580, "lon": -73.9855, "slots": 3},
    {"id": "CS_05", "name": "Columbus Circle",    "lat": 40.7681, "lon": -73.9819, "slots": 4},
    {"id": "CS_06", "name": "Bryant Park",        "lat": 40.7536, "lon": -73.9832, "slots": 3},
    {"id": "CS_07", "name": "Lincoln Tunnel",     "lat": 40.7607, "lon": -74.0021, "slots": 5},
    {"id": "CS_08", "name": "FDR at 42nd",        "lat": 40.7510, "lon": -73.9696, "slots": 4},
    {"id": "CS_09", "name": "Port Authority",     "lat": 40.7572, "lon": -74.0008, "slots": 5},
    {"id": "CS_10", "name": "Lexington at 59th",  "lat": 40.7625, "lon": -73.9678, "slots": 3},
]

LOW_BATTERY_THRESHOLD = 0.20       # 20% SoC
CRITICAL_BATTERY_THRESHOLD = 0.10  # 10% SoC — emergency reroute regardless of route cost

CHARGE_RATE_PER_STEP = 0.005       # 0.5% per step → ~200 steps to full from 0%


class ChargingStation:
    def __init__(self, station_def):
        self.id       = station_def["id"]
        self.name     = station_def["name"]
        self.lat      = station_def["lat"]
        self.lon      = station_def["lon"]
        self.slots    = station_def["slots"]
        self.edge_id  = station_def.get("edge_id_override")
        self.occupied = set()
        self.queue    = []

    @property
    def available_slots(self):
        return self.slots - len(self.occupied)

    @property
    def is_available(self):
        return self.available_slots > 0

    def arrive(self, veh_id):
        """Vehicle arrives at station. Starts charging or joins queue."""
        if self.is_available:
            self.occupied.add(veh_id)
            return True
        if veh_id not in self.queue:
            self.queue.append(veh_id)
        return False

    def depart(self, veh_id):
        """Vehicle leaves station (charged enough to continue)."""
        self.occupied.discard(veh_id)
        if self.queue and self.is_available:
            next_veh = self.queue.pop(0)
            self.occupied.add(next_veh)

    def step(self, battery_tracker):
        """
        Called every simulation step.
        Increments SoC for all vehicles currently charging.
        Returns list of vehicle IDs that are done charging (SoC >= 80%).
        """
        done = []
        for veh_id in list(self.occupied):
            current_soc = battery_tracker.get_soc(veh_id)
            new_soc     = min(1.0, current_soc + CHARGE_RATE_PER_STEP)
            battery_tracker.set_soc(veh_id, new_soc)
            if new_soc >= 0.80:
                done.append(veh_id)
        for veh_id in done:
            self.depart(veh_id)
        return done


class ChargingNetwork:
    """
    Manages all charging stations.
    Call resolve_edges() once after traci.start() to bind lat/lon → edge IDs.
    """

    def __init__(self):
        self.stations  = [ChargingStation(s) for s in CHARGING_STATIONS_LATLON]
        self._edge_map = {}
        self._resolved = False

    def resolve_edges(self):
        """Maps each station's lat/lon to the nearest SUMO edge via TraCI."""
        for station in self.stations:
            if station.edge_id:
                self._edge_map[station.edge_id] = station
                print(f"[CHARGER] {station.id} ({station.name}) → edge {station.edge_id} (override)")
                continue
            try:
                x, y = traci.simulation.convertGeo(
                    station.lon, station.lat, fromGeo=True
                )
                results = traci.simulation.convertRoad(x, y, isGeo=False)
                edge_id = results[0]
                station.edge_id = edge_id
                self._edge_map[edge_id] = station
                print(f"[CHARGER] {station.id} ({station.name}) → edge {edge_id}")
            except Exception as e:
                print(f"[CHARGER WARNING] Could not resolve {station.id}: {e}")

        self._resolved = True

    @staticmethod
    def find_route_to_edge(from_edge, to_edge):
        """
        Return a valid edge list from from_edge to to_edge, or None if unreachable.
        SUMO's changeTarget() fails silently when no graph path exists — always
        validate with findRoute first.
        """
        if not from_edge or not to_edge:
            return None
        if str(from_edge).startswith(":") or str(to_edge).startswith(":"):
            return None
        try:
            result = traci.simulation.findRoute(str(from_edge), str(to_edge))
            if result.edges and len(result.edges) >= 1:
                return list(result.edges)
        except Exception:
            pass
        return None

    def _stations_by_distance(self, veh_id, require_available=False):
        """Return [(station, distance_m), ...] sorted nearest-first."""
        try:
            veh_x, veh_y = traci.vehicle.getPosition(veh_id)
        except Exception:
            return []

        ranked = []
        for station in self.stations:
            if station.edge_id is None:
                continue
            if require_available and not station.is_available:
                continue
            try:
                sx, sy = traci.simulation.convert2D(station.edge_id, 0)
                dist = math.hypot(veh_x - sx, veh_y - sy)
                ranked.append((station, dist))
            except Exception:
                continue

        ranked.sort(key=lambda item: item[1])
        return ranked

    def nearest_reachable(self, veh_id, require_available=False):
        """
        Nearest charging station with a valid SUMO route from the vehicle.
        Returns (station, route_edges) or (None, None).
        """
        try:
            curr_edge = traci.vehicle.getRoadID(veh_id)
        except Exception:
            return None, None

        for station, _dist in self._stations_by_distance(veh_id, require_available):
            route = self.find_route_to_edge(curr_edge, station.edge_id)
            if route:
                return station, route
        return None, None

    def nearest_available(self, veh_id):
        """Returns nearest available ChargingStation (may be unreachable)."""
        ranked = self._stations_by_distance(veh_id, require_available=True)
        return ranked[0][0] if ranked else None

    def nearest_any(self, veh_id):
        """Like nearest_available but ignores slot availability."""
        ranked = self._stations_by_distance(veh_id, require_available=False)
        return ranked[0][0] if ranked else None

    def step(self, battery_tracker, active_vehicle_ids=None):
        """Advance all stations by one simulation step."""
        if active_vehicle_ids is not None:
            active = set(active_vehicle_ids)
            for station in self.stations:
                station.occupied = {v for v in station.occupied if v in active}
                station.queue = [v for v in station.queue if v in active]

        released = []
        for station in self.stations:
            done = station.step(battery_tracker)
            released.extend(done)
        return released

    def station_at_edge(self, edge_id):
        """Returns ChargingStation if this edge has one, else None."""
        return self._edge_map.get(edge_id)

    def register_arrival(self, veh_id, edge_id):
        """Call when a vehicle reaches its charger edge."""
        station = self._edge_map.get(edge_id)
        if station:
            return station.arrive(veh_id)
        return False

    def summary(self):
        """Returns dict of station utilisation for logging."""
        return {
            s.id: {
                "occupied": len(s.occupied),
                "queued":   len(s.queue),
                "slots":    s.slots,
            }
            for s in self.stations
        }
