"""Drive a single Pioneer P3-DX robot from its current pose (or ``--start``) to a goal
using A* path planning and the P/PID differential-drive controller.

Works both against a live CoppeliaSim instance and against the offline kinematic MockSim
(``--mock``).

Examples
--------
    python scripts/run_single_robot.py --goal 2.25,3.8
    python scripts/run_single_robot.py --start -4,-4,0 --goal P6
    python scripts/run_single_robot.py --mock --goal P4 --save output/single_robot.png --no-show
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_fleet.astar import plan_path  # noqa: E402
from warehouse_fleet.config import (  # noqa: E402
    DEFAULT_FLEET_CONFIG_PATH, GAINS, GRID, ROBOT, SIM, SimParams)
from warehouse_fleet.controller import WaypointFollower  # noqa: E402
from warehouse_fleet.fleet_config import load_fleet_config, resolve_stations  # noqa: E402
from warehouse_fleet.geometry import Pose2D  # noqa: E402
from warehouse_fleet.grid_map import GridMap  # noqa: E402
from warehouse_fleet.scene_extractor import load_or_extract_scene  # noqa: E402
from warehouse_fleet.sim_interface import (  # noqa: E402
    MockSim, SimConnectionError, ZMQSimInterface)


def _parse_pose(text: str) -> Pose2D:
    parts = [float(x) for x in text.split(",") if x.strip()]
    if len(parts) not in (2, 3):
        raise ValueError(f"Expected x,y or x,y,yaw, got '{text}'")
    return Pose2D(parts[0], parts[1], parts[2] if len(parts) == 3 else 0.0)


def _parse_goal(text: str, stations: dict) -> Tuple[float, float]:
    if text in stations:
        return stations[text]
    parts = [float(x) for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Goal must be a station name ({', '.join(stations)}) or x,y; got '{text}'")
    return (parts[0], parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="Robot_0", help="robot alias (default: Robot_0)")
    parser.add_argument("--start", default=None, metavar="X,Y[,YAW]",
                        help="optional start pose override (teleports robot before starting)")
    parser.add_argument("--goal", default="P6", metavar="X,Y|STATION",
                        help="goal coordinates or station name from fleet_config.json (default: P6)")
    parser.add_argument("--scene", type=Path, default=None,
                        help="optional scene JSON override (otherwise extracted from CoppeliaSim)")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="maximum simulation steps before timeout (default: 1500)")
    parser.add_argument("--mock", action="store_true",
                        help="run with offline kinematic MockSim (no CoppeliaSim needed)")
    parser.add_argument("--save", type=Path, default=None, help="save trajectory plot to PNG")
    parser.add_argument("--no-show", action="store_true", help="do not open plot window")
    parser.add_argument("--host", default=SIM.host)
    parser.add_argument("--port", type=int, default=SIM.port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sim = MockSim() if args.mock else ZMQSimInterface(SimParams(host=args.host, port=args.port))
    try:
        sim.connect()
        scene = load_or_extract_scene(None if args.mock else sim, scene_file=args.scene)
        grid = GridMap.from_scene(scene, GRID.resolution, GRID.inflation_radius(ROBOT),
                                  GRID.border_is_obstacle)

        cfg = load_fleet_config(DEFAULT_FLEET_CONFIG_PATH)
        stations = resolve_stations(cfg, grid)
        goal_xy = _parse_goal(args.goal, stations)

        if args.start is not None:
            start_pose = _parse_pose(args.start)
            sim.set_pose(args.robot, start_pose)
        else:
            start_pose = sim.get_pose(args.robot)

        planned = plan_path(grid, start_pose.xy(), goal_xy, max_snap_distance=GRID.max_snap_distance)
        if planned is None:
            print(f"ERROR: A* found no collision-free path from {start_pose.xy()} to {goal_xy}",
                  file=sys.stderr)
            return 2

        print(f"Planned path for {args.robot}: {len(planned.waypoints)} waypoints, "
              f"length={planned.length:.2f} m (expanded {planned.expanded} nodes)")

        sim.draw_path(planned.waypoints)
        controller = WaypointFollower(GAINS, ROBOT)
        controller.set_path(planned.waypoints, current_pose=start_pose)

        sim.start(stepping=True)
        trajectory: List[Tuple[float, float]] = [start_pose.xy()]

        for step_idx in range(1, args.max_steps + 1):
            pose = sim.get_pose(args.robot)
            trajectory.append(pose.xy())
            out = controller.step(pose, SIM.dt)
            sim.set_wheel_speeds(args.robot, out.omega_left, out.omega_right)
            if out.goal_reached:
                print(f"Goal reached in {step_idx} steps ({step_idx * SIM.dt:.2f} s)! "
                      f"Final error: {pose.distance_to(planned.waypoints[-1]):.3f} m")
                break
            sim.step()
        else:
            pose = sim.get_pose(args.robot)
            print(f"WARNING: Timeout after {args.max_steps} steps. "
                  f"Remaining distance to goal: {pose.distance_to(planned.waypoints[-1]):.3f} m")

        sim.stop()

        if args.save is not None or not args.no_show:
            from warehouse_fleet.visualizer import plot_grid_map

            plot_grid_map(
                grid,
                paths={args.robot: planned.waypoints},
                trajectories={args.robot: trajectory},
                stations=stations,
                starts={args.robot: start_pose},
                title=f"Single Robot A* & P/PID Tracking ({args.robot})",
                save_path=args.save,
                show=not args.no_show,
            )
        return 0
    except (SimConnectionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            sim.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
