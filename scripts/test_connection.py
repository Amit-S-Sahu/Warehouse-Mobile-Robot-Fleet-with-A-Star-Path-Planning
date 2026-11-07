"""Smoke test & calibration script for CoppeliaSim ZMQ Remote API (or MockSim).

Connects to CoppeliaSim, finds Robot_0, prints its pose and motor handles, runs a short
forward motion + spin-in-place test, and estimates the effective wheel radius and wheel
separation from the simulated motion so you can verify ``config.py``.

Examples
--------
    python scripts/test_connection.py
    python scripts/test_connection.py --mock
    python scripts/test_connection.py --robot Robot_0 --port 23000
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_fleet.config import ROBOT, SIM, SimParams  # noqa: E402
from warehouse_fleet.geometry import wrap_angle  # noqa: E402
from warehouse_fleet.sim_interface import (  # noqa: E402
    MockSim, SimConnectionError, ZMQSimInterface)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=SIM.host, help="CoppeliaSim host (default: localhost)")
    parser.add_argument("--port", type=int, default=SIM.port, help="ZMQ port (default: 23000)")
    parser.add_argument("--robot", default="Robot_0", help="robot alias to test (default: Robot_0)")
    parser.add_argument("--steps", type=int, default=20, help="steps per motion test (default: 20)")
    parser.add_argument("--mock", action="store_true", help="run against offline MockSim instead of CoppeliaSim")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.mock:
        sim = MockSim()
    else:
        sim = ZMQSimInterface(SimParams(host=args.host, port=args.port, dt=SIM.dt, stepping=True))

    try:
        sim.connect()
        robots = sim.list_scene_robots()
        print(f"Detected robots in scene: {robots}")
        handles = sim.get_robot(args.robot)
        print(f"Handles for {args.robot}: body={handles.body}, "
              f"left_motor={handles.left_motor}, right_motor={handles.right_motor}")

        sim.start(stepping=True)
        p0 = sim.get_pose(args.robot)
        t0 = sim.sim_time()
        print(f"Initial pose (t={t0:.2f}s): x={p0.x:.3f} m, y={p0.y:.3f} m, yaw={math.degrees(p0.yaw):.1f} deg")

        # 1. Straight forward test (omega_L = omega_R = 2.0 rad/s)
        omega_test = 2.0
        sim.set_wheel_speeds(args.robot, omega_test, omega_test)
        for _ in range(args.steps):
            sim.step()
        p1 = sim.get_pose(args.robot)
        t1 = sim.sim_time()
        dt_fwd = max(t1 - t0, 1e-6)
        dist = p0.distance_to(p1)
        est_r = dist / (omega_test * dt_fwd)
        print(f"After forward test (dt={dt_fwd:.2f}s): x={p1.x:.3f} m, y={p1.y:.3f} m, "
              f"yaw={math.degrees(p1.yaw):.1f} deg | dist={dist:.3f} m "
              f"(est. wheel_radius ~ {est_r:.4f} m, config={ROBOT.wheel_radius:.4f} m)")

        # 2. Spin-in-place test (omega_L = -2.0, omega_R = +2.0 rad/s)
        sim.set_wheel_speeds(args.robot, -omega_test, omega_test)
        for _ in range(args.steps):
            sim.step()
        p2 = sim.get_pose(args.robot)
        t2 = sim.sim_time()
        dt_spin = max(t2 - t1, 1e-6)
        dyaw = abs(wrap_angle(p2.yaw - p1.yaw))
        est_L = (2.0 * ROBOT.wheel_radius * omega_test * dt_spin) / max(dyaw, 1e-6)
        print(f"After spin test    (dt={dt_spin:.2f}s): x={p2.x:.3f} m, y={p2.y:.3f} m, "
              f"yaw={math.degrees(p2.yaw):.1f} deg | dyaw={math.degrees(dyaw):.1f} deg "
              f"(est. wheel_separation ~ {est_L:.4f} m, config={ROBOT.wheel_separation:.4f} m)")

        sim.set_wheel_speeds(args.robot, 0.0, 0.0)
        sim.stop()
        print("Connection and motor test PASSED.")
        return 0
    except SimConnectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            sim.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
