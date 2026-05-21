import random

import traci


class BatteryModel:
    """EV battery drain model with charging-station pause and stranding tracking."""

    def __init__(self):
        self._soc           = {}   # veh_id → float (0.0–1.0)
        self._stranded      = set()
        self._at_charger    = set()
        self.stranded_count = 0

    def get_soc(self, veh_id):
        if veh_id not in self._soc:
            self._soc[veh_id] = random.uniform(0.4, 1.0)
        return self._soc[veh_id]

    def set_soc(self, veh_id, value):
        """Used by ChargingStation to increment SoC during charging."""
        self._soc[veh_id] = max(0.0, min(1.0, value))

    def start_charging(self, veh_id):
        """Pause battery drain while vehicle is at a charging station."""
        self._at_charger.add(veh_id)

    def stop_charging(self, veh_id):
        """Resume battery drain after vehicle leaves charging station."""
        self._at_charger.discard(veh_id)

    def is_charging(self, veh_id):
        return veh_id in self._at_charger

    def update(self, veh_id, speed_ms, step_length_s=1.0):
        """
        Update SoC for one vehicle for one simulation step.
        Skips drain if vehicle is currently at a charging station.
        Records stranding event if SoC reaches 0.
        """
        if veh_id not in self._soc:
            self._soc[veh_id] = random.uniform(0.4, 1.0)

        if veh_id in self._at_charger:
            return

        if speed_ms < 0.1:
            drain = 0.0008
        else:
            drain = 0.0012 * (speed_ms / 20.0)

        drain *= step_length_s
        self._soc[veh_id] = max(0.0, self._soc[veh_id] - drain)

        if self._soc[veh_id] <= 0.0 and veh_id not in self._stranded:
            self._stranded.add(veh_id)
            self.stranded_count += 1
            print(
                f"[BATTERY] Vehicle {veh_id} STRANDED (SoC=0) at step "
                f"{traci.simulation.getTime():.0f}"
            )

    def process_step(self, logger):
        """Called every TraCI step to drain batteries and catch dead EVs."""
        try:
            active_vehicles = traci.vehicle.getIDList()
        except traci.exceptions.TraCIException:
            return

        for v_id in list(self._soc.keys()):
            if v_id not in active_vehicles:
                del self._soc[v_id]
                self._at_charger.discard(v_id)

        for v_id in active_vehicles:
            try:
                if traci.vehicle.getTypeID(v_id) != "ev_swarm":
                    continue
            except traci.exceptions.TraCIException:
                continue

            try:
                speed = traci.vehicle.getSpeed(v_id)
            except traci.exceptions.TraCIException:
                continue

            self.update(v_id, speed)

            if self._soc.get(v_id, 1.0) <= 0.0:
                logger.record_stranded(v_id)
                try:
                    traci.vehicle.setSpeed(v_id, 0.0)
                    traci.vehicle.setColor(v_id, (100, 0, 0, 255))
                except traci.exceptions.TraCIException:
                    pass
