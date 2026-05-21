"""
algorithms/route_validator.py
─────────────────────────────────────────────────────────────────────────────
Shared utility for lane-level turn validation before swarm route injection.

Fixes Lane-Level Deadlock & Teleportation Penalty by answering two questions
before any changeTarget / setRoute call:

  1. Can this vehicle physically enter the first edge of the proposed detour
     from its current lane, given SUMO's micro-level turn restrictions?

  2. If not, what is the first edge in the detour that IS reachable, so we
     can assign it as a safe intermediate target?

All functions are pure TraCI queries — no simulation state is mutated here.
"""

import traci


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_parent_edge(lane_id: str) -> str:
    """
    Strips the lane index suffix to recover the parent edge ID.
    SUMO lane IDs: '<edge_id>_<lane_index>'.
    Returns raw lane_id if no underscore (handles internal junction lanes).
    """
    if "_" not in lane_id:
        return lane_id
    parts = lane_id.rsplit("_", 1)
    return parts[0]


def _get_reachable_edges_from_lane(lane_id: str) -> set:
    """
    Returns the set of edge IDs reachable from lane_id via SUMO's connection
    (turn-restriction) table.

    traci.lane.getLinks returns tuples; index 0 is the approached lane ID.
    """
    reachable = set()
    try:
        links = traci.lane.getLinks(lane_id)
        for link in links:
            next_lane = link[0]
            if next_lane:
                reachable.add(_get_parent_edge(next_lane))
    except traci.exceptions.TraCIException:
        pass
    return reachable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vehicle_reachable_edges(v_id: str) -> set:
    """
    Returns edge IDs the vehicle can legally move to from its current lane.
    Returns empty set if vehicle is on an internal junction lane (starts ':').
    """
    try:
        lane_id = traci.vehicle.getLaneID(v_id)
    except traci.exceptions.TraCIException:
        return set()

    if lane_id.startswith(":"):
        return set()

    return _get_reachable_edges_from_lane(lane_id)


def validate_detour_entry(v_id: str, detour_edge_list: list) -> tuple:
    """
    Checks whether vehicle can legally enter the first edge of detour_edge_list.

    Returns:
        (True,  first_edge)  — safe for direct injection
        (False, None)        — lane-turn mismatch; use safe passthrough
        (None,  None)        — vehicle is on junction lane; defer injection
    """
    if not detour_edge_list:
        return False, None

    first_edge = detour_edge_list[0]

    try:
        curr_lane = traci.vehicle.getLaneID(v_id)
    except traci.exceptions.TraCIException:
        return False, None

    if curr_lane.startswith(":"):
        return None, None

    reachable = _get_reachable_edges_from_lane(curr_lane)
    curr_edge = _get_parent_edge(curr_lane)

    if curr_edge == first_edge or first_edge in reachable:
        return True, first_edge

    return False, None


def find_safe_intermediate_target(v_id: str, detour_edge_list: list) -> str | None:
    """
    Walks forward through detour_edge_list to find the first edge vehicle
    can physically reach — used as a stepping-stone when direct entry fails.

    Returns edge ID of safe intermediate, or None if none found (leave to
    SUMO default routing).
    """
    if not detour_edge_list:
        return None

    try:
        curr_lane = traci.vehicle.getLaneID(v_id)
    except traci.exceptions.TraCIException:
        return None

    if curr_lane.startswith(":"):
        return None

    direct_reachable = _get_reachable_edges_from_lane(curr_lane)

    for edge in detour_edge_list:
        if edge in direct_reachable:
            return edge

    # 2-hop look-ahead
    for intermediate_edge in direct_reachable:
        try:
            lane_count = traci.edge.getLaneNumber(intermediate_edge)
        except traci.exceptions.TraCIException:
            continue
        for lane_idx in range(lane_count):
            intermediate_lane = f"{intermediate_edge}_{lane_idx}"
            second_hop = _get_reachable_edges_from_lane(intermediate_lane)
            for edge in detour_edge_list:
                if edge in second_hop:
                    return intermediate_edge

    return None
