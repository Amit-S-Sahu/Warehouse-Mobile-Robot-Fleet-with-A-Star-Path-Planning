"""Prepare the robot fleet in CoppeliaSim (or MockSim).

Given a :class:`~warehouse_fleet.fleet_config.FleetConfig` and a validated mapping of robot
start poses:
1. Inspects existing ``Robot_*`` models in the scene.
2. Clones ``Robot_0`` (or the first available robot) to create any missing robots up to ``N``.
3. Teleports each active robot to its validated start pose and resets its dynamics.
4. Stops any extra robots present in the scene without deleting them.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Union

from .fleet_config import FleetConfig, resolve_start_poses
from .geometry import Pose2D
from .grid_map import GridMap
from .sim_interface import MockSim, RobotHandles, SimConnectionError, ZMQSimInterface

log = logging.getLogger(__name__)

SimLike = Union[ZMQSimInterface, MockSim]


def setup_fleet_in_scene(sim: SimLike, cfg: FleetConfig,
                         grid: GridMap) -> Dict[str, Pose2D]:
    """Ensure all ``cfg.robot_names`` exist in ``sim`` and place them at validated start poses.

    Returns the dictionary ``{robot_name: start_pose}`` in ``cfg.robot_names`` order.
    """
    existing_names = sim.list_scene_robots()
    if not existing_names:
        raise SimConnectionError(
            "No 'Robot_*' models found in the scene. Place at least 'Robot_0' (Pioneer P3-DX) "
            "in CoppeliaSim before running."
        )

    # Read current poses of robots that already exist in the scene
    existing_poses: Dict[str, Pose2D] = {}
    for name in existing_names:
        try:
            existing_poses[name] = sim.get_pose(name)
        except Exception as exc:
            log.debug("Could not read initial pose of %s: %s", name, exc)

    # Validate & resolve start poses for all active robots before mutating the scene
    start_poses = resolve_start_poses(cfg, grid, existing_poses=existing_poses)

    template_name = existing_names[0]
    for name, pose in start_poses.items():
        if name not in existing_names:
            log.info("Cloning %s -> %s", template_name, name)
            sim.clone_robot(template_name, name)
            existing_names.append(name)
        sim.set_pose(name, pose)
        sim.set_wheel_speeds(name, 0.0, 0.0)
        log.info("Placed %s at (%.2f, %.2f, yaw=%.2f rad)", name, pose.x, pose.y, pose.yaw)

    # Park any extra robots already in the scene that aren't part of the active fleet
    active_set = set(start_poses)
    for extra in existing_names:
        if extra not in active_set:
            try:
                sim.set_wheel_speeds(extra, 0.0, 0.0)
            except Exception:
                pass

    return start_poses
