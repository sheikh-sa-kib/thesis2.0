import json
import math
import random
import traci


def _active_ev_swarm_ids(vehicle_ids=None):
    """
    Return ev_swarm IDs that still exist in SUMO.
    Vehicles can depart between getIDList() and a follow-up TraCI call.
    """
    if vehicle_ids is None:
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.exceptions.TraCIException:
            return []
    ev_ids = []
    for v_id in vehicle_ids:
        try:
            if traci.vehicle.getTypeID(v_id) == "ev_swarm":
                ev_ids.append(v_id)
        except traci.exceptions.TraCIException:
            continue
    return ev_ids


class V2XCommunicationLayer:
    def __init__(self):
        self._omnet_log = []
        self.broadcast_range_m = 300.0   # Standard DSRC/C-V2X range
        self.base_packet_loss  = 0.12    # 12% chance a message drops
        self.message_buffer    = []      # Holds delayed messages

    def _get_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def generate_json_alert(self, sender_id, timestamp, edge_id, severity=2):
        """Generates the universal cross-brand emergency JSON payload."""
        payload = {
            "message_type":        "EMERGENCY_ALERT",
            "sender_id":           sender_id,
            "timestamp":           timestamp,
            "emergency_type":      "BLOCKAGE",
            "location":            {"edge_id": edge_id, "position": 0.5},
            "severity":            severity,
            "affected_radius_m":   self.broadcast_range_m,
            "suggested_avoid_edges": [edge_id],
        }
        return json.dumps(payload)

    def broadcast_alert(
        self, sender_id, current_step, edge_id, custom_packet_loss=None
    ):
        """Broadcasts emergency JSON to all nearby smart EVs."""
        loss_rate = (
            custom_packet_loss
            if custom_packet_loss is not None
            else self.base_packet_loss
        )

        try:
            sender_pos = traci.vehicle.getPosition(sender_id)
        except traci.exceptions.TraCIException:
            return 0

        active_vehicles    = traci.vehicle.getIDList()
        receivers_notified = 0

        json_payload   = self.generate_json_alert(sender_id, current_step, edge_id)
        parsed_message = json.loads(json_payload)

        for v_id in active_vehicles:
            if v_id == sender_id:
                continue
            try:
                if traci.vehicle.getTypeID(v_id) != "ev_swarm":
                    continue
                v_pos = traci.vehicle.getPosition(v_id)
            except traci.exceptions.TraCIException:
                continue

            if self._get_distance(sender_pos, v_pos) > self.broadcast_range_m:
                continue

            if random.random() < loss_rate:
                continue   # Packet dropped

            delay_steps = random.choice([0, 1])
            self.message_buffer.append({
                "deliver_at_step": current_step + delay_steps,
                "receiver_id":     v_id,
                "alert_data":      parsed_message,
            })
            receivers_notified += 1
        # Log for OMNeT++ analysis
        self._omnet_log.append({
            "step":        current_step,
            "sender":      sender_id,
            "edge":        edge_id,
            "loss_rate":   loss_rate,
            "notified":    receivers_notified,
            "dropped":     len([v for v in active_vehicles
                               if v != sender_id]) - receivers_notified,
        })
        return receivers_notified
    def detect_anomalous_message(self, edge_id, current_step):
        """
        Anomaly detection: flags V2X alerts that deviate >2σ from recent
        broadcast history for the same edge. Returns True if message is
        suspicious (should be down-weighted in Byzantine consensus).
        Implements the 2-standard-deviation flagging from P2 feedback.
        """
        if not hasattr(self, '_alert_history'):
            self._alert_history = {}  # edge_id -> list of broadcast steps

        history = self._alert_history.get(edge_id, [])

        if len(history) >= 3:
            mean_interval = sum(
                history[i+1] - history[i]
                for i in range(len(history)-1)
            ) / (len(history) - 1)
            intervals = [
                history[i+1] - history[i]
                for i in range(len(history)-1)
            ]
            std_interval = (
                sum((x - mean_interval)**2 for x in intervals)
                / len(intervals)
            ) ** 0.5

            if history:
                last_seen = history[-1]
                current_interval = current_step - last_seen
                # Flag if interval deviates >2σ from normal pattern
                if std_interval > 0:
                    z_score = abs(current_interval - mean_interval) / std_interval
                    if z_score > 2.0:
                        print(
                            f"[V2X ANOMALY] Edge '{edge_id}' alert at step "
                            f"{current_step} flagged (z={z_score:.2f}). "
                            f"Down-weighting in Byzantine consensus."
                        )
                        history.append(current_step)
                        self._alert_history[edge_id] = history[-10:]
                        return True  # anomalous

        history.append(current_step)
        self._alert_history[edge_id] = history[-10:]
        return False  # normal
    def process_message_queue(self, current_step, routing_controller=None):
        """Delivers buffered messages that have completed their latency delay."""
        remaining_buffer = []
        delivered_count  = 0

        for msg in self.message_buffer:
            if current_step >= msg["deliver_at_step"]:
                receiver_id = msg["receiver_id"]
                try:
                    still_present = traci.vehicle.getIDCount(receiver_id) > 0
                except traci.exceptions.TraCIException:
                    still_present = False
                if still_present and routing_controller:
                    try:
                        avoid_edge = msg["alert_data"]["location"]["edge_id"]
                        routing_controller.handle_v2x_alert(
                            receiver_id, avoid_edge
                        )
                    except Exception:
                        pass
                if still_present:
                    delivered_count += 1
            else:
                remaining_buffer.append(msg)

        self.message_buffer = remaining_buffer
        return delivered_count

    def log_periodic_broadcast(self, current_step, packet_loss_override=None):
        """
        Logs ambient V2X activity (sampled every 10 sim steps to limit TraCI load).
        Called from run_sim.py to capture ongoing fleet communication
        (pheromone updates, congestion signals) beyond one-time block alerts.
        """
        if current_step % 10 != 0:
            return

        loss_rate = (
            packet_loss_override
            if packet_loss_override is not None
            else self.base_packet_loss
        )
        try:
            ev_vehicles = _active_ev_swarm_ids()
        except Exception:
            return

        if not ev_vehicles:
            return

        # Sample one sender per step (representative beacon)
        sender_id = random.choice(ev_vehicles)
        try:
            sender_pos = traci.vehicle.getPosition(sender_id)
            sender_edge = traci.vehicle.getRoadID(sender_id)
        except Exception:
            return

        in_range = 0
        notified = 0
        for v_id in ev_vehicles:
            if v_id == sender_id:
                continue
            try:
                v_pos = traci.vehicle.getPosition(v_id)
            except Exception:
                continue
            if self._get_distance(sender_pos, v_pos) <= self.broadcast_range_m:
                in_range += 1
                if random.random() >= loss_rate:
                    notified += 1

        dropped = in_range - notified

        self._omnet_log.append({
            "step":      current_step,
            "sender":    sender_id,
            "edge":      sender_edge,
            "loss_rate": round(loss_rate, 4),
            "notified":  notified,
            "dropped":   dropped,
        })

    def export_omnet_log(self, filepath):
        """Exports V2X message log in OMNeT++ compatible CSV format."""
        import csv, os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not self._omnet_log:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'step','sender','edge','loss_rate','notified','dropped'
            ])
            writer.writeheader()
            writer.writerows(self._omnet_log)
        print(f"[V2X] OMNeT++ log exported: {filepath}")
