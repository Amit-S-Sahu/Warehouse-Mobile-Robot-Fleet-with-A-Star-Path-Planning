"""Step 1 verification script: inspect the CoppeliaSim warehouse scene and plot the grid.

Connects to CoppeliaSim (or reads a scene JSON with ``--offline``), prints all detected
``Wall_*`` / ``Shelf_*`` obstacles and ``Robot_*`` models, warns about any unnamed
cuboids, builds the inflated 2D occupancy grid, and displays/saves a plot.

Examples
--------
    python scripts/inspect_scene.py
    python scripts/inspect_scene.py --save output/scene_grid.png
    python scripts/inspect_scene.py --offline --no-show --save output/ref_grid.png
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_fleet.config import (  # noqa: E402
    GRID, REFERENCE_LAYOUT_PATH, ROBOT, SCENE_CACHE_PATH, SIM, SimParams)
from warehouse_fleet.grid_map import GridMap  # noqa: E402
from warehouse_fleet.scene_extractor import extract_scene  # noqa: E402
from warehouse_fleet.scene_model import SceneDescription  # noqa: E402
from warehouse_fleet.sim_interface import SimConnectionError, ZMQSimInterface  # noqa: E402
from warehouse_fleet.visualizer import plot_grid_map  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=SIM.host)
    parser.add_argument("--port", type=int, default=SIM.port)
    parser.add_argument("--offline", action="store_true",
                        help="use a saved scene JSON instead of connecting to CoppeliaSim")
    parser.add_argument("--scene", type=Path, default=None,
                        help="scene JSON to load in --offline mode (default: cache or reference_layout.json)")
    parser.add_argument("--resolution", type=float, default=GRID.resolution)
    parser.add_argument("--save", type=Path, default=None, help="save grid plot to this PNG")
    parser.add_argument("--no-show", action="store_true", help="do not open an interactive plot window")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    starts = {}
    if args.offline:
        src = args.scene or (SCENE_CACHE_PATH if SCENE_CACHE_PATH.is_file() else REFERENCE_LAYOUT_PATH)
        print(f"Loading offline scene from {src}")
        scene = SceneDescription.load(src)
    else:
        sim = ZMQSimInterface(SimParams(host=args.host, port=args.port))
        try:
            sim.connect()
            report = extract_scene(sim)
            scene = report.scene
            print(f"Robots found ({len(report.robot_names)}): {', '.join(report.robot_names) or '(none)'}")
            for r_name in report.robot_names:
                starts[r_name] = sim.get_pose(r_name)
            if report.unrecognized_shapes:
                print(f"WARNING: Ignored unnamed cuboids: {', '.join(report.unrecognized_shapes)}")
        except SimConnectionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    xmin, xmax, ymin, ymax = scene.bounds
    print(f"Floor bounds: x in [{xmin:.2f}, {xmax:.2f}] m, y in [{ymin:.2f}, {ymax:.2f}] m")
    print(f"Obstacles ({len(scene.obstacles)}):")
    for ob in scene.obstacles:
        print(f"  - {ob.name:<12} pos=({ob.x:6.2f}, {ob.y:6.2f}) "
              f"size=({ob.size_x:5.2f} x {ob.size_y:5.2f}) yaw={math.degrees(ob.yaw):5.1f} deg")

    grid = GridMap.from_scene(scene, args.resolution, GRID.inflation_radius(ROBOT),
                              GRID.border_is_obstacle)
    free_pct = 100.0 * (1.0 - grid.inflated.mean())
    print(f"Grid: {grid.rows}x{grid.cols} cells at {grid.resolution:.2f} m/cell | "
          f"inflated free space: {free_pct:.1f}%")

    plot_grid_map(grid, starts=starts, save_path=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
