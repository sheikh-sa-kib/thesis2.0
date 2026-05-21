"""
run_sim.py  —  Master simulation entry point
─────────────────────────────────────────────────────────────────────────────
UPDATES:
  - __init__.py required in all package dirs (run setup_project.py first)
  - RL meta-controller (dqn.py) integrated for E3_Hybrid_Complete only.
    Every 20 steps the RLMetaController observes network state and adjusts
    E3's scout ratio, onlooker ratio, and pheromone evaporation — giving E3
    an adaptive advantage competitors lack (addresses P2 RL feedback).
  - Emergency corridor uses uniform V2X by default:
      All algorithms receive the same 550m alert range and slowdown mechanism.
  - --time-to-teleport 400 → gives rerouter more reaction time
  - Teleport count tracked and logged per run
  - FIX A2: BCO call site corrected.
  - FIX A3: _spawn_emergency_vehicles() uses sumolib geometry.
  - FIX A1: SimulationLogger receives algo_name and scenario_id.
"""
import argparse
import os
import random
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

import sumolib
import traci

from algorithms.aco       import AntColonyOptimizer
from algorithms.pso       import ParticleSwarmOptimizer
from algorithms.bco       import BeeColonyOptimizer
from algorithms.e3_hybrid import E3HybridOptimizer
from algorithms.dijkstra  import DijkstraRouter
from algorithms.astar     import AStarRouter
from algorithms.dqn       import RLMetaController
from communication.v2x_model        import V2XCommunicationLayer
from constraints.battery            import BatteryModel
from constraints.driver_environment import EnvironmentConstraints
from metrics.logger                 import SimulationLogger
from scenarios.engine               import ScenarioEngine
from emergency_corridor             import (
    UNIFORM_V2X_CORRIDOR,
    calculate_corridor_warning_range,
)
# ── Clean terminal output — all verbose output goes to log file ───────────────
class _SimLogger:
    """
    Redirects ALL stdout (including prints from algorithm files) to a log file.
    Only lines explicitly passed to tprint() appear in the terminal.
    """
    def __init__(self, log_path, real_stdout):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log    = open(log_path, "w", buffering=1, encoding="utf-8")
        self._real   = real_stdout

    def write(self, msg):
        self._log.write(msg)          # everything → log file

    def flush(self):
        self._log.flush()

    def tprint(self, msg):
        """Print to BOTH terminal and log file (for important events only)."""
        self._real.write(msg + "\n")
        self._real.flush()
        self._log.write(msg + "\n")
        self._log.flush()

    def close(self):
        sys.stdout = self._real       # restore terminal
        self._log.close()
# ── Constants ────────────────────────────────────────────────────────────────
TELEPORT_WARNING_THRESHOLD = 550
_teleport_attempts         = {}
TELEPORT_MAX_ATTEMPTS      = 5
SWARM_REEVAL_INTERVAL      = 10
EMERGENCY_SPAWN_DELAY      = 5
EMERGENCY_COUNT            = 2

# Emergency corridor constants
CORRIDOR_RANGE_M        = 300.0   # close-range pull-over/lane-change zone
PULL_OVER_SPEED         = 0.5     # m/s — near-stop
RESUME_RANGE_M          = 400.0   # resume normal speed after this distance

# RL meta-controller action parameters (applied to E3 only)
_RL_ACTIONS = {
    0: {"ratio_scouts": 0.50, "ratio_onlookers": 0.30, "rho": 0.02},  # aggressive scout
    1: {"ratio_scouts": 0.35, "ratio_onlookers": 0.45, "rho": 0.02},  # wider onlooker
    2: {"ratio_scouts": 0.35, "ratio_onlookers": 0.30, "rho": 0.01},  # slow evaporation
    3: {"ratio_scouts": 0.35, "ratio_onlookers": 0.30, "rho": 0.02},  # defaults
}
E3_NORL_DEFAULT_ACTION = 3


# ── Emergency corridor ────────────────────────────────────────────────────────

def _apply_emergency_corridor(emergency_vehicle_ids, all_vehicles,
                               algo_name="", v2x_layer=None,
                               current_step=0, prewarned_evs=None):
    """Emergency corridor with uniform V2X pre-warning by default."""
    if not emergency_vehicle_ids:
        return
    if prewarned_evs is None:
        prewarned_evs = set()
    warning_range = calculate_corridor_warning_range(
        algo_name, uniform_v2x=UNIFORM_V2X_CORRIDOR
    )

    for emerg_id in [v for v in emergency_vehicle_ids if v in all_vehicles]:
        try:
            emerg_pos = traci.vehicle.getPosition(emerg_id)
            emerg_edge = traci.vehicle.getRoadID(emerg_id)
            emerg_lane_idx = traci.vehicle.getLaneIndex(emerg_id)
        except:
            continue

        for v_id in all_vehicles:
            if v_id == emerg_id or v_id.startswith("emerg_"):
                continue
            try:
                v_pos = traci.vehicle.getPosition(v_id)
                dist = ((v_pos[0] - emerg_pos[0])**2 + (v_pos[1] - emerg_pos[1])**2)**0.5

                if v2x_layer and dist <= warning_range and v_id not in prewarned_evs:
                    try:
                        traci.vehicle.setSpeed(v_id, traci.vehicle.getAllowedSpeed(v_id) * 0.55)
                        prewarned_evs.add(v_id)
                    except:
                        pass

                if dist > RESUME_RANGE_M:
                    try:
                        if traci.vehicle.getSpeed(v_id) < 1.0:
                            traci.vehicle.setSpeed(v_id, -1.0)
                        prewarned_evs.discard(v_id)
                    except:
                        pass
                    continue

                if dist <= CORRIDOR_RANGE_M:
                    v_edge = traci.vehicle.getRoadID(v_id)
                    v_lane = traci.vehicle.getLaneIndex(v_id)

                    if v_edge == emerg_edge:
                        # Same edge: try lane change first
                        try:
                            n_lanes = traci.edge.getLaneNumber(v_edge)
                            if n_lanes > 1:
                                target = 0 if emerg_lane_idx > 0 else n_lanes-1
                                if v_lane != target:
                                    traci.vehicle.changeLane(v_id, target, 4.0)
                            else:
                                traci.vehicle.setSpeed(v_id, PULL_OVER_SPEED)
                        except:
                            traci.vehicle.setSpeed(v_id, PULL_OVER_SPEED)
                    else:
                        traci.vehicle.setSpeed(v_id, PULL_OVER_SPEED)
            except:
                continue


# ── Teleport guard ────────────────────────────────────────────────────────────
def _apply_teleport_guard(all_vehicles, algo_name="", teleport_counter=None):
    global _teleport_attempts
    active_set = set(all_vehicles)
    _teleport_attempts = {k: v for k, v in _teleport_attempts.items() if k in active_set}

    for v_id in list(all_vehicles):
        try:
            waiting = traci.vehicle.getWaitingTime(v_id)
            if waiting < TELEPORT_WARNING_THRESHOLD:
                _teleport_attempts.pop(v_id, None)
                continue

            attempts = _teleport_attempts.get(v_id, 0)
            if attempts >= TELEPORT_MAX_ATTEMPTS:
                try:
                    traci.vehicle.remove(v_id)
                    if teleport_counter is not None:
                        teleport_counter[0] += 1
                    print(f"[TELEPORT GUARD] {v_id} removed after {attempts} failed attempts.")
                except:
                    pass
                _teleport_attempts.pop(v_id, None)
                continue

            # Stronger recovery: try direct findRoute with current destination
            try:
                curr = traci.vehicle.getRoadID(v_id)
                dest = traci.vehicle.getRoute(v_id)[-1]
                result = traci.simulation.findRoute(curr, dest, departPos="best")
                if result.edges and len(result.edges) > 1:
                    traci.vehicle.setRoute(v_id, list(result.edges))
                else:
                    traci.vehicle.rerouteTraveltime(v_id)
            except:
                try:
                    traci.vehicle.rerouteTraveltime(v_id)
                except:
                    pass

            _teleport_attempts[v_id] = attempts + 1
        except:
            continue
# ── Emergency vehicle spawner ─────────────────────────────────────────────────

def _spawn_emergency_vehicles(blocked_edge, seed_counter, sumo_net=None):
    spawned    = []
    spawn_edge = None
    dest_edge  = None

    if sumo_net:
        try:
            net_edge = sumo_net.getEdge(blocked_edge)
            shape    = net_edge.getLane(0).getShape()
            blk_x    = sum(p[0] for p in shape) / len(shape)
            blk_y    = sum(p[1] for p in shape) / len(shape)

            best_dist    = float("inf")
            nearest_edge = None
            for v in traci.vehicle.getIDList():
                try:
                    vx, vy = traci.vehicle.getPosition(v)
                    dist   = ((vx - blk_x)**2 + (vy - blk_y)**2)**0.5
                    road   = traci.vehicle.getRoadID(v)
                    if (50.0 < dist < 800.0 and not road.startswith(":")
                            and road != blocked_edge):
                        if dist < best_dist:
                            best_dist    = dist
                            nearest_edge = road
                except traci.exceptions.TraCIException:
                    continue

            if nearest_edge:
                spawn_edge = nearest_edge
                try:
                    result = traci.simulation.findRoute(spawn_edge, blocked_edge)
                    if result.edges and len(result.edges) >= 2:
                        dest_edge = result.edges[-2]
                    elif result.edges:
                        dest_edge = result.edges[-1]
                except Exception:
                    pass

        except Exception as e:
            print(f"[EMERGENCY SPAWN] sumolib geometry failed: {e}")

    if not spawn_edge or not dest_edge:
        print("[EMERGENCY SPAWN] Falling back to static spawn/dest pair.")
        spawn_edge = "5670106#0"
        dest_edge  = "1198594026#0"

    for i in range(EMERGENCY_COUNT):
        v_id     = f"emerg_{seed_counter}_{i}"
        route_id = f"route_emerg_{seed_counter}_{i}"
        try:
            traci.route.add(route_id, [spawn_edge])
            traci.vehicle.add(
                vehID=v_id,
                routeID=route_id,
                typeID="emergency",
                depart="now",
                departLane="best",
                departSpeed="max",
            )
        except traci.exceptions.TraCIException as e:
            print(f"[EMERGENCY SPAWN] Failed for {v_id}: {e}")
            continue
        try:
            traci.vehicle.changeTarget(v_id, dest_edge)
            traci.vehicle.setRoutingMode(
                v_id, traci.constants.ROUTING_MODE_AGGREGATED
            )
            traci.vehicle.rerouteTraveltime(v_id)
            spawned.append(v_id)
            print(f"[EMERGENCY SPAWN] '{v_id}' deployed: "
                  f"{spawn_edge} -> {dest_edge} (incident: {blocked_edge})")
        except traci.exceptions.TraCIException as e:
            print(f"[EMERGENCY SPAWN] Routing failed for {v_id}: {e}")

    return spawned


# ── RL state builder ──────────────────────────────────────────────────────────

def _build_rl_state(blocked_edges, fleet_size_count, avg_speed, step, max_steps):
    """4-dim state vector for RLMetaController with safe data typing."""
    return [
        min(len(blocked_edges) / 5.0, 1.0),   # normalised block count
        min(avg_speed / 13.9, 1.0),            # normalised network speed
        min(fleet_size_count / 200.0, 1.0),    # normalised fleet size integer count
        min(step / max_steps, 1.0),            # SAFE: Enforces hard clip at 1.0 maximum progress
    ]


def _apply_rl_action(action, hybrid_swarm):
    """Apply RL-selected action to E3 swarm parameters."""
    params = _RL_ACTIONS.get(action, _RL_ACTIONS[3])
    hybrid_swarm.ratio_scouts    = params["ratio_scouts"]
    hybrid_swarm.ratio_onlookers = params["ratio_onlookers"]
    hybrid_swarm.rho             = params["rho"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",     type=str, required=True)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--nogui",    action="store_true")
    args = parser.parse_args()

    algo_name    = args.algo
    scen_id      = args.scenario
    current_seed = args.seed
    MAX_STEPS    = 3600

    random.seed(current_seed)

    sumo_binary = sumolib.checkBinary("sumo" if args.nogui else "sumo-gui")
    sumo_config = os.path.expanduser("~/thesis/simulation.sumocfg")
    net_file    = os.path.expanduser("~/thesis/network/midtown.net.xml")
    sumo_net    = sumolib.net.readNet(net_file)

    output_csv = os.path.expanduser(
        f"~/thesis/results/{algo_name}_scen{scen_id}_seed{current_seed}.csv"
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ── Redirect all stdout to log file; terminal stays clean ────────────
    _run_log_path = os.path.expanduser(
        f"~/thesis/results/logs/run_{algo_name}_s{scen_id}_seed{current_seed}.log"
    )
    _sim_logger = _SimLogger(_run_log_path, sys.stdout)
    sys.stdout  = _sim_logger
    tprint      = _sim_logger.tprint   # shortcut for terminal-visible prints

    tprint(f"\n{'='*55}")
    tprint(f"  {algo_name} | Scenario {scen_id} | Seed {current_seed}")
    tprint(f"  Log -> {_run_log_path}")  # SAFE: Replaced \u2192 with standard ASCII ->
    tprint(f"{'='*55}")

    logger          = SimulationLogger(output_csv, algo_name=algo_name, scenario_id=scen_id)
    battery_model   = BatteryModel()
    env_constraints = EnvironmentConstraints()
    v2x_layer       = V2XCommunicationLayer()
    v2x_layer.broadcast_range_m = calculate_corridor_warning_range(
        algo_name, uniform_v2x=UNIFORM_V2X_CORRIDOR
    )
    scenario_engine = ScenarioEngine(scenario_id=scen_id)
    scenario_engine.seed = current_seed

    dijkstra_router = None
    astar_router    = None
    bco_engine      = None
    aco_engine      = None
    pso_engine      = None
    hybrid_swarm    = None
    rl_controller   = None   # Only for E3

    if algo_name == "Dijkstra":
        dijkstra_router = DijkstraRouter(net_file)
    elif algo_name == "A_Star":
        astar_router = AStarRouter(net_file)
    elif algo_name == "BCO_Standalone":
        bco_engine = BeeColonyOptimizer(net_file)
    elif algo_name == "ACO_Standalone":
        aco_engine = AntColonyOptimizer(net_file)
    elif algo_name == "PSO_Standalone":
        pso_engine = ParticleSwarmOptimizer(net_file)
    elif algo_name == "E3_Hybrid_Complete":
        hybrid_swarm  = E3HybridOptimizer(net_file)
        rl_controller = RLMetaController(state_dim=4, action_dim=4)
        hybrid_swarm._triggered_blocks = set()
    elif algo_name == "E3_NoRL":
        # Ablation control: identical to E3_Hybrid_Complete but DQN is disabled.
        # rl_controller is set to None — swarm parameters stay at defaults.
        hybrid_swarm  = E3HybridOptimizer(net_file)
        rl_controller = None
        
        hybrid_swarm._triggered_blocks = set()


    sumo_msg_log = os.path.expanduser(
        f"~/thesis/results/logs/sumo_msg_{algo_name}_s{scen_id}_seed{current_seed}.log"
    )
    os.makedirs(os.path.dirname(sumo_msg_log), exist_ok=True)

    traci.start([
        sumo_binary,
        "-c", sumo_config,
        "--ignore-route-errors",          "true",
        "--time-to-teleport",             "600",
        "--time-to-teleport.highways",    "-1",
        "--collision.action",             "warn",
        "--no-step-log",                  "true",
        "--message-log",                  sumo_msg_log,
        "--seed",                         str(current_seed),
    ])

    # Initialise charging network (resolves lat/lon → edge IDs)
    if algo_name in ("E3_Hybrid_Complete", "E3_NoRL") and hybrid_swarm is not None:
        hybrid_swarm.charging_network.resolve_edges()
        mapped = sum(1 for s in hybrid_swarm.charging_network.stations if s.edge_id)
        print(
            f"[CHARGER] Network ready — {mapped} / "
            f"{len(hybrid_swarm.charging_network.stations)} stations mapped"
        )

    step                  = 0
    emergency_spawned     = False
    emergency_vehicle_ids = []
    prewarned_evs         = set()      # EVs pre-warned by the V2X corridor
    teleport_counter      = [0]        # mutable counter for removal events
    dijkstra_handled_blocks = set()   # tracks which blocks Dijkstra has already processed
    astar_handled_blocks    = set()   # same for A*
    prev_avg_speed         = None       # tracks previous avg speed for RL reward
    avg_speed             = 12.0

    # --- RL ablation: apply fixed default action for E3_NoRL at start ---
    if algo_name == "E3_NoRL":
        _apply_rl_action(E3_NORL_DEFAULT_ACTION, hybrid_swarm)
        print(f"[E3-NORL] Fixed default action {E3_NORL_DEFAULT_ACTION} applied. "
              f"RL disabled for entire run.")

    while step < MAX_STEPS:
        traci.simulationStep()
        logger.log_step(step)
        battery_model.process_step(logger)
        v2x_layer.process_message_queue(step)

        # Step charging network (E³-Hybrid / ablation)
        if algo_name in ("E3_Hybrid_Complete", "E3_NoRL") and hybrid_swarm is not None:
            released = hybrid_swarm.charging_network.step(battery_model)
            for veh_id in released:
                battery_model.stop_charging(veh_id)
                hybrid_swarm._charging_done.add(veh_id)
                try:
                    traci.vehicle.rerouteTraveltime(veh_id)
                except traci.exceptions.TraCIException:
                    pass

        # Track SUMO-initiated teleports (separate from our guard's removals)
        try:
            sumo_teleports = traci.simulation.getStartingTeleportNumber()
            if sumo_teleports > 0:
                logger.record_sumo_teleports(sumo_teleports)
        except Exception:
            pass

        _blackout_loss = (0.95 if (scen_id == 4
                                  and getattr(scenario_engine, "blackout_active", False))
                                  else None)
        v2x_layer.log_periodic_broadcast(step, packet_loss_override=_blackout_loss)

        try:
            all_vehicles = list(traci.vehicle.getIDList())
        except traci.exceptions.TraCIException:
            step += 1
            continue

        active_evs = []
        for v in all_vehicles:
            try:
                if traci.vehicle.getTypeID(v) == "ev_swarm":
                    active_evs.append(v)
            except traci.exceptions.TraCIException:
                continue

        _apply_teleport_guard(all_vehicles, algo_name=algo_name,
                              teleport_counter=teleport_counter)

        # Low-battery → charger (every step; must not use bare changeTarget)
        if algo_name in ("E3_Hybrid_Complete", "E3_NoRL") and hybrid_swarm is not None:
            for v_id in active_evs:
                hybrid_swarm._handle_low_battery_routing(v_id, battery_model)

        # ── Emergency corridor ────────────────────────────────────────────
        _apply_emergency_corridor(
            emergency_vehicle_ids, all_vehicles,
            algo_name=algo_name,
            v2x_layer=v2x_layer,
            current_step=step,
            prewarned_evs=prewarned_evs,
        )

        # ── Emergency vehicle clearance tracking ──────────────────────────
        if emergency_vehicle_ids:
            still_active = [v for v in emergency_vehicle_ids if v in all_vehicles]
            if not still_active:
                logger.clear_emergency(step)
                emergency_vehicle_ids = []
                prewarned_evs.clear()

        # ── Deferred re-assertion tickers ─────────────────────────────────
        if hybrid_swarm is not None:
            hybrid_swarm.tick_pending_reassertions(step)
        if bco_engine is not None:
            bco_engine.tick_pending_reassertions(step)
        if aco_engine is not None:
            aco_engine.tick_pending_reassertions(step)
        if pso_engine is not None:
            pso_engine.tick_pending_reassertions(step)

        # ── E3: PSO tuning + ACO evaporation every 10 steps ──────────────
        if algo_name in ("E3_Hybrid_Complete", "E3_NoRL") and step % 10 == 0:
            hybrid_swarm.evaporate_pheromones(
                blocked_edges=scenario_engine.blocked_edges or None
            )
            try:
                speeds = [
                    traci.edge.getLastStepMeanSpeed(e)
                    for e in traci.edge.getIDList()
                    if not e.startswith(":")
                ]
                avg_speed = sum(speeds) / len(speeds) if speeds else 12.0
            except Exception:
                avg_speed = 12.0
            hybrid_swarm.tune_pso_parameters(avg_network_speed=avg_speed)

        # ── E3: RL meta-controller every 20 steps (emergency active) ─────
        if (algo_name == "E3_Hybrid_Complete"
                and rl_controller is not None
                and scenario_engine.blocked_edges
                and step % 20 == 0):
            # Force strict integer counting for the fleet size array
            current_fleet_size = len(active_evs)
            state = _build_rl_state(
                scenario_engine.blocked_edges, current_fleet_size, avg_speed, step, MAX_STEPS
            )
            # Seeds 42, 123 = training (explore). Seeds 456, 789, 1337 = evaluation (exploit).
            freeze = current_seed in (456, 789, 1337)
            action = rl_controller.select_action(state, freeze_policy=freeze)
            _apply_rl_action(action, hybrid_swarm)

            reward = rl_controller._compute_reward(traci)
            prev_avg_speed = avg_speed

            # Log RL decision for thesis evidence
            print(
                f"[RL] Step {step} | State={[round(s,2) for s in state]} "
                f"| Action={action} ({list(_RL_ACTIONS.keys())[action]}) "
                f"| Reward={reward:+.4f} | e={rl_controller.epsilon:.3f}"
            )

            # Build a simple next-state for replay memory
            next_state = _build_rl_state(
                scenario_engine.blocked_edges, current_fleet_size, avg_speed,
                step + 20, MAX_STEPS
            )
            done = (step + 20) >= MAX_STEPS
            rl_controller.memory.push(state, action, reward, next_state, done)
            rl_controller.optimize_model()
            if step % 200 == 0:
                rl_controller.update_target_network()
                rl_controller.decay_epsilon()

        # ── ACO: evaporate pheromones every 10 steps ─────────────────────
        if algo_name == "ACO_Standalone" and step % 10 == 0:
            aco_engine.evaporate_pheromones(
                blocked_edges=scenario_engine.blocked_edges or None
            )

        # ── PSO: update swarm every 10 steps ─────────────────────────────
        if algo_name == "PSO_Standalone" and step % 10 == 0:
            pso_engine.update_swarm(
                blocked_edges=scenario_engine.blocked_edges or None
            )

        # ── Scenario engine ───────────────────────────────────────────────
        scenario_engine.process_step(step, v2x_layer, logger)

        if scenario_engine.blocked_edges:
            current_block    = scenario_engine.blocked_edges[-1]
            first_block_step = getattr(scenario_engine, "first_block_step", 100)

            # Spawn emergency vehicles (once per run)
            tprint(f"  [BLOCK]     Step {step}: '{current_block}' blocked "
                       f"({len(scenario_engine.blocked_edges)} total)")
            if (not emergency_spawned
                    and step >= first_block_step + EMERGENCY_SPAWN_DELAY):
                first_blocked = scenario_engine.blocked_edges[0]
                emergency_vehicle_ids = _spawn_emergency_vehicles(
                    first_blocked, current_seed, sumo_net=sumo_net
                )
                emergency_spawned = True
                if emergency_vehicle_ids:
                    logger.trigger_emergency(step)
                    tprint(f"  [EMERGENCY] Step {step}: "
                           f"{len(emergency_vehicle_ids)} vehicle(s) spawned")

                    # Uniform V2X corridor alert: same infrastructure for every algorithm.
                    if emergency_vehicle_ids:
                        for emerg_id in emergency_vehicle_ids:
                            try:
                                notified = v2x_layer.broadcast_alert(
                                    sender_id=emerg_id,
                                    current_step=step,
                                    edge_id=first_blocked,
                                    custom_packet_loss=0.05,
                                )
                                print(
                                    f"[V2X CORRIDOR] Ambulance alert broadcast from "
                                    f"'{emerg_id}' -> {notified} EVs pre-warned."
                                )
                            except Exception:
                                pass

            # ── Per-algorithm routing logic ───────────────────────────────

            if algo_name == "Dijkstra":
                new_blocks = [be for be in scenario_engine.blocked_edges
                            if be not in dijkstra_handled_blocks]
                if new_blocks:
                    packet_loss = 0.95 if scenario_engine.blackout_active else 0.22
                    for v_id in active_evs:
                        try:
                            if random.random() < packet_loss:
                                continue
                            if env_constraints.check_driver_compliance(v_id):
                                dijkstra_router.reroute_vehicle(
                                    v_id,
                                    blocked_edges=list(scenario_engine.blocked_edges),
                                    logger=logger
                                )
                        except traci.exceptions.TraCIException:
                            continue
                    dijkstra_handled_blocks.update(new_blocks)

            elif algo_name == "A_Star":
                new_blocks = [be for be in scenario_engine.blocked_edges
                            if be not in astar_handled_blocks]
                if new_blocks:
                    packet_loss = 0.95 if scenario_engine.blackout_active else 0.22
                    for v_id in active_evs:
                        try:
                            if random.random() < packet_loss:
                                continue
                            if env_constraints.check_driver_compliance(v_id):
                                astar_router.reroute_vehicle(
                                    v_id,
                                    blocked_edges=list(scenario_engine.blocked_edges),
                                    logger=logger
                                )
                        except traci.exceptions.TraCIException:
                            continue
                    astar_handled_blocks.update(new_blocks)
            elif algo_name == "BCO_Standalone":
                if step == first_block_step:
                    bco_engine.trigger_scout_exploration(
                        scenario_engine.blocked_edges[0], step
                    )
                if step >= first_block_step and step % SWARM_REEVAL_INTERVAL == 0:
                    for be in scenario_engine.blocked_edges:
                        bco_engine.process_swarm_onlookers(
                            be, logger, env_constraints, current_step=step
                        )

            elif algo_name == "ACO_Standalone":
                if step == first_block_step:
                    aco_engine.trigger_scout_exploration(
                        scenario_engine.blocked_edges[0], step
                    )
                if step >= first_block_step and step % SWARM_REEVAL_INTERVAL == 0:
                    for be in scenario_engine.blocked_edges:
                        aco_engine.process_swarm_onlookers(
                            be, logger, env_constraints, current_step=step
                        )

            elif algo_name == "PSO_Standalone":
                if step == first_block_step:
                    pso_engine.trigger_scout_exploration(
                        scenario_engine.blocked_edges[0], step
                    )
                if step >= first_block_step and step % SWARM_REEVAL_INTERVAL == 0:
                    for be in scenario_engine.blocked_edges:
                        pso_engine.process_swarm_onlookers(
                            be, logger, env_constraints, current_step=step
                        )

            elif algo_name in ("E3_Hybrid_Complete", "E3_NoRL"):
                for be in scenario_engine.blocked_edges:
                    if be not in hybrid_swarm._triggered_blocks:
                        hybrid_swarm.trigger_emergency_response(be, step)
                        hybrid_swarm.trigger_emergency_response(be, step + 2)
                        hybrid_swarm._triggered_blocks.add(be)

                if step >= 100 and step % SWARM_REEVAL_INTERVAL == 0:
                    for be in scenario_engine.blocked_edges:
                        hybrid_swarm.process_swarm_onlookers(
                            be, logger, env_constraints, step
                        )

        step += 1
        if step % 100 == 0:
            n_blocks   = len(scenario_engine.blocked_edges)
            n_evs      = len(active_evs)
            n_reroutes = getattr(logger, "total_reroutes", 0)
            n_tele     = teleport_counter[0]
            soc_str    = ""
            if algo_name in ("E3_Hybrid_Complete", "E3_NoRL"):
                charging = len(getattr(hybrid_swarm, "_charging_targets", {}))
                soc_str  = f" | Charging: {charging}"
            tprint(
                f"  Step {step:4d}/{MAX_STEPS}"
                f" | EVs: {n_evs:3d}"
                f" | Blocks: {n_blocks}"
                f" | Reroutes: {n_reroutes:4d}"
                f" | Stuck: {n_tele}{soc_str}"
            )

    traci.close()
    tprint(f"\n  [DONE] Steps: {step} | "
           f"Reroutes: {getattr(logger, 'total_reroutes', 0)} | "
           f"Stuck removed: {teleport_counter[0]}")
    tprint(f"  [SAVED] {output_csv}")
    tprint(f"{'='*55}\n")
    _sim_logger.close()   # restore sys.stdout

    # Save RL checkpoint for E3
    if rl_controller is not None:
        rl_controller.save_checkpoint()

    v2x_log_path = os.path.expanduser(
        f"~/thesis/omnet_logs/{algo_name}_scen{scen_id}_seed{current_seed}.csv"
    )
    v2x_layer.export_omnet_log(v2x_log_path)
    logger.save_results(
        teleport_removals=teleport_counter[0],
        battery_tracker=battery_model,
        hybrid_algo=hybrid_swarm,
    )


if __name__ == "__main__":
    main()