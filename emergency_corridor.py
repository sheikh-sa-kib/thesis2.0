"""
communication/emergency_corridor.py
─────────────────────────────────────────────────────────────────────────────
Emergency Vehicle Corridor — Uniform V2X EV Yielding System.

NOTE: The primary corridor logic is implemented INLINE in run_sim.py as
_apply_emergency_corridor() for tight integration with per-step state.
This module documents the design and provides the standalone class for
potential future use in other simulation frameworks.

Two-tier yielding:
  Tier 1 — Multi-lane edge:
    traci.vehicle.changeLane(v_id, rightmost_lane, duration)
  Tier 2 — Single-lane edge OR lane change rejected:
    traci.vehicle.setSpeed(v_id, 0.5)  ← near-stop pull-over

Fair evaluation default:
  - Uniform V2X pre-warning zone: 550m for every algorithm
  - Pre-warned EVs reduce speed to 55% before ambulance arrives
  - Routing differences are isolated from communication-range differences

Restoration:
  - traci.vehicle.setSpeed(v_id, -1.0) restores SUMO speed control
  - Triggered when emergency vehicle > 400m away
  - prewarned_evs set is cleared when emergency vehicles leave simulation

Thesis relevance:
  The corridor feature is active for ALL algorithms (emergency spawning is
  universal), and the default uniform V2X mode prevents E3 from receiving a
  hidden communication-range advantage over baselines.
"""

UNIFORM_V2X_CORRIDOR = True
V2X_WARNING_RANGE_M = 550.0
LEGACY_PROXIMITY_RANGE_M = 300.0


def calculate_corridor_warning_range(algo_name, uniform_v2x=UNIFORM_V2X_CORRIDOR):
    """
    Return the emergency pre-warning range used by the corridor controller.

    When uniform_v2x is True, all algorithms receive the same communication
    infrastructure so the experiment isolates routing behavior. Setting it to
    False enables legacy system-archetype comparison.
    """
    if uniform_v2x:
        return V2X_WARNING_RANGE_M
    if algo_name in ("E3_Hybrid_Complete", "E3_NoRL"):
        return V2X_WARNING_RANGE_M
    return LEGACY_PROXIMITY_RANGE_M
