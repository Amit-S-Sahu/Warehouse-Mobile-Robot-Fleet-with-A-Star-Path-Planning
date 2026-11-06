"""Offline A* demo (no CoppeliaSim needed).

Builds the occupancy grid from a scene JSON (reference layout or the cache written by
scene extraction), resolves robot start poses from the fleet config, plans one path per
robot and plots the result.

Examples
--------
    python scripts/plan_offline.py
    python scripts/plan_offline.py --num-robots 5 --save output/plan.png --no-show
    python scripts/plan_offline.py --start Robot_0=-4,-4,0 --goal Robot_0=P6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_fleet.astar import plan_path  # noqa: E402
from warehouse_fleet.config import GRID, REFERENCE_LAYOUT_PATH, ROBOT  # noqa: E402
from warehouse_fleet.fleet_config import (  # noqa: E402
    ConfigError, add_fleet_cli_arguments, fleet_config_from_args, resolve_goals,
    resolve_start_poses, resolve_stations)
from warehouse_fleet.grid_map import GridMap  # noqa: E402
from warehouse_fleet.scene_model import SceneDescription  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=REFERENCE_LAYOUT_PATH,
                        help="scene JSON (default: config/reference_layout.json)")
    parser.add_argument("--resolution", type=float, default=GRID.resolution)
    parser.add_argument("--save", type=Path, default=None, help="save the plot to this PNG")
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    add_fleet_cli_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scene = SceneDescription.load(args.scene)
    grid = GridMap.from_scene(scene, args.resolution, GRID.inflation_radius(ROBOT),
                              GRID.border_is_obstacle)
    print(grid)

    try:
        cfg = fleet_config_from_args(args)
        starts = resolve_start_poses(cfg, grid)
        stations = resolve_stations(cfg, grid)
        if cfg.goals_mode == "goals":
            goals = resolve_goals(cfg, grid)
        else:
            # Demo: robot i goes to the pick station of task i (round robin).
            goals = {}
            for i, name in enumerate(cfg.robot_names):
                if cfg.tasks:
                    goals[name] = stations[cfg.tasks[i % len(cfg.tasks)].pick]
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    paths = {}
    for name, pose in starts.items():
        if name not in goals:
            print(f"{name}: no goal, skipping")
            continue
        result = plan_path(grid, pose.xy(), goals[name], max_snap_distance=GRID.max_snap_distance)
        if result is None:
            print(f"{name}: NO PATH from {pose.xy()} to {goals[name]}")
            continue
        paths[name] = result.waypoints
        print(f"{name}: {len(result.cells)} cells -> {len(result.waypoints)} waypoints, "
              f"length {result.length:.2f} m, expanded {result.expanded} nodes")

    from warehouse_fleet.visualizer import plot_grid_map

    plot_grid_map(grid, paths=paths, stations=stations, starts=starts,
                  save_path=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
