"""Run the multi-robot warehouse fleet in CoppeliaSim (or offline with ``--mock``).

Supports choosing the number of robots, start locations, and either pick->drop tasks or
direct goals via ``config/fleet_config.json`` and/or CLI flags.

Examples
--------
    python scripts/run_fleet.py
    python scripts/run_fleet.py --num-robots 4
    python scripts/run_fleet.py --num-robots 2 --start Robot_0=-4,-4,0 --start Robot_1=3.5,-3.5,3.14
    python scripts/run_fleet.py --num-robots 3 --goal Robot_0=P6 --goal Robot_1=P4 --goal Robot_2=D1
    python scripts/run_fleet.py --mock --num-robots 3 --save output/fleet_run.png --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_fleet.config import GRID, ROBOT, SIM, SimParams  # noqa: E402
from warehouse_fleet.fleet_config import (  # noqa: E402
    ConfigError, add_fleet_cli_arguments, fleet_config_from_args)
from warehouse_fleet.fleet_manager import FleetManager  # noqa: E402
from warehouse_fleet.grid_map import GridMap  # noqa: E402
from warehouse_fleet.scene_extractor import load_or_extract_scene  # noqa: E402
from warehouse_fleet.sim_interface import (  # noqa: E402
    MockSim, SimConnectionError, ZMQSimInterface)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_fleet_cli_arguments(parser)
    parser.add_argument("--scene", type=Path, default=None,
                        help="optional scene JSON override (otherwise extracted from CoppeliaSim)")
    parser.add_argument("--no-space-time", action="store_true",
                        help="use standard A* + runtime yielding instead of Space-Time A*")
    parser.add_argument("--max-steps", type=int, default=3500,
                        help="maximum simulation steps (default: 3500)")
    parser.add_argument("--mock", action="store_true",
                        help="run against offline kinematic MockSim (no CoppeliaSim required)")
    parser.add_argument("--save", type=Path, default=None, help="save summary plot to PNG")
    parser.add_argument("--no-show", action="store_true", help="do not open plot window")
    parser.add_argument("--host", default=SIM.host)
    parser.add_argument("--port", type=int, default=SIM.port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = fleet_config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    sim = MockSim() if args.mock else ZMQSimInterface(SimParams(host=args.host, port=args.port))
    try:
        sim.connect()
        scene = load_or_extract_scene(None if args.mock else sim, scene_file=args.scene)
        grid = GridMap.from_scene(scene, GRID.resolution, GRID.inflation_radius(ROBOT),
                                  GRID.border_is_obstacle)

        manager = FleetManager(
            sim=sim,
            grid=grid,
            cfg=cfg,
            use_space_time=not args.no_space_time,
        )

        print(f"Starting fleet with {len(manager.agents)} robot(s) "
              f"[mode={cfg.goals_mode}, space_time_astar={not args.no_space_time}]")
        metrics = manager.run(max_steps=args.max_steps)

        print("\n=== Fleet Run Summary ===")
        print(f"  Completed     : {metrics.tasks_completed}/{metrics.total_tasks} "
              f"({'ALL DONE' if metrics.all_done else 'TIMEOUT'})")
        print(f"  Makespan      : {metrics.sim_time:.2f} s ({metrics.steps} steps)")
        print(f"  Total distance: {metrics.total_distance:.2f} m")
        print(f"  Yield steps   : {metrics.total_yield_steps} | Replans: {metrics.total_replans}")
        if metrics.min_inter_robot_distance < float("inf"):
            print(f"  Min separation: {metrics.min_inter_robot_distance:.2f} m")
        for name, agent in manager.agents.items():
            print(f"  - {name:<8}: state={agent.state.value:<8} tasks={agent.tasks_completed} "
                  f"dist={agent.distance_travelled:5.2f} m yields={agent.yield_steps} replans={agent.replans}")

        if args.save is not None or not args.no_show:
            from warehouse_fleet.visualizer import plot_grid_map

            plot_grid_map(
                grid,
                paths={name: a.all_planned_points() for name, a in manager.agents.items()},
                trajectories={name: a.trajectory for name, a in manager.agents.items()},
                stations=manager.stations,
                starts={name: a.home_pose for name, a in manager.agents.items()},
                title=f"Fleet Run ({len(manager.agents)} robots, "
                      f"{metrics.tasks_completed}/{metrics.total_tasks} completed)",
                save_path=args.save,
                show=not args.no_show,
            )
        return 0
    except (SimConnectionError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            sim.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
