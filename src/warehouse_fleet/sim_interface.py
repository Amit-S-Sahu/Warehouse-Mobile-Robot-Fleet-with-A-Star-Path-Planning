"""ZeroMQ Remote API wrapper for CoppeliaSim and an offline kinematic MockSim.

Provides a unified interface for robot handles, poses, motor velocities, simulation
stepping, cloning robots, and reading scene geometry, while abstracting away API
differences between CoppeliaSim 4.3-4.5 and >= 4.6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (LEFT_MOTOR_ALIAS, RIGHT_MOTOR_ALIAS, ROBOT, ROBOT_PREFIX, SIM,
                     RobotParams, SimParams)
from .geometry import Pose2D, transform_point, wrap_angle, yaw_from_matrix

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobotHandles:
    """Simulation handles for a single differential-drive robot."""

    name: str
    body: int
    left_motor: int
    right_motor: int
    z_height: float = 0.138  # preserved when teleporting in 2D


class SimConnectionError(RuntimeError):
    """Raised when connecting to or querying CoppeliaSim fails."""


# ======================================================================================
# CoppeliaSim ZMQ Interface
# ======================================================================================
class ZMQSimInterface:
    """Wrapper around ``coppeliasim_zmqremoteapi_client.RemoteAPIClient``."""

    def __init__(self, params: SimParams = SIM, robot_params: RobotParams = ROBOT) -> None:
        self.params = params
        self.robot_params = robot_params
        self.client: Any = None
        self.sim: Any = None
        self._robots: Dict[str, RobotHandles] = {}
        self._drawings: List[int] = []

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as exc:
            raise SimConnectionError(
                "coppeliasim-zmqremoteapi-client is not installed. "
                "Run: pip install coppeliasim-zmqremoteapi-client"
            ) from exc

        log.info("Connecting to CoppeliaSim ZMQ Remote API at %s:%d ...",
                 self.params.host, self.params.port)
        try:
            self.client = RemoteAPIClient(host=self.params.host, port=self.params.port)
            # CoppeliaSim >= 4.6 uses client.require('sim'); 4.3-4.5 uses client.getObject('sim')
            if hasattr(self.client, "require"):
                self.sim = self.client.require("sim")
            else:
                self.sim = self.client.getObject("sim")
        except Exception as exc:
            raise SimConnectionError(
                f"Failed to connect to CoppeliaSim on {self.params.host}:{self.params.port}. "
                f"Ensure CoppeliaSim is open and the ZMQ remote API is running."
            ) from exc

    def start(self, stepping: Optional[bool] = None) -> None:
        self._ensure_connected()
        use_stepping = self.params.stepping if stepping is None else stepping
        self._set_stepping(use_stepping)
        state = self.sim.getSimulationState()
        if state == self.sim.simulation_stopped:
            self.sim.startSimulation()
            log.info("Simulation started (stepping=%s)", use_stepping)

    def step(self) -> None:
        self._ensure_connected()
        if hasattr(self.sim, "step"):
            self.sim.step()
        else:
            self.client.step()

    def stop(self) -> None:
        if self.sim is None:
            return
        for name in list(self._robots):
            try:
                self.set_wheel_speeds(name, 0.0, 0.0)
            except Exception:
                pass
        self.clear_drawings()
        try:
            if self.sim.getSimulationState() != self.sim.simulation_stopped:
                self.sim.stopSimulation()
                # Wait briefly until simulation fully stops
                for _ in range(50):
                    if self.sim.getSimulationState() == self.sim.simulation_stopped:
                        break
                    self.step()
        except Exception as exc:
            log.debug("Ignoring error while stopping simulation: %s", exc)

    def sim_time(self) -> float:
        self._ensure_connected()
        return float(self.sim.getSimulationTime())

    def _set_stepping(self, enabled: bool) -> None:
        if hasattr(self.sim, "setStepping"):
            self.sim.setStepping(bool(enabled))
        elif hasattr(self.client, "setStepping"):
            self.client.setStepping(bool(enabled))

    def _ensure_connected(self) -> None:
        if self.sim is None:
            self.connect()

    def _const(self, *names: str) -> int:
        """Return the first sim constant that exists (handles API renames across versions)."""
        for n in names:
            try:
                return getattr(self.sim, n)
            except AttributeError:
                continue
        raise SimConnectionError(f"None of these sim constants exist: {names}")

    @property
    def _ALL(self) -> int:
        return self._const("handle_all", "object_all")

    @property
    def _JOINT(self) -> int:
        return self._const("sceneobject_joint", "object_joint_type")

    # ------------------------------------------------------------------ robot discovery & control
    def get_robot(self, name: str) -> RobotHandles:
        """Look up handles for ``name`` (e.g. ``Robot_0``) and cache them."""
        if name in self._robots:
            return self._robots[name]
        self._ensure_connected()

        body = self._find_object(f"/{name}")
        if body is None:
            body = self._find_object_by_alias(name)
        if body is None:
            raise SimConnectionError(
                f"Robot '{name}' not found in the CoppeliaSim scene. "
                f"Make sure a Pioneer P3-DX named '{name}' (or '{ROBOT_PREFIX}0') exists."
            )

        left = self._find_child_joint(body, LEFT_MOTOR_ALIAS, fallback_Hint="left")
        right = self._find_child_joint(body, RIGHT_MOTOR_ALIAS, fallback_Hint="right")
        if left is None or right is None:
            raise SimConnectionError(
                f"Could not find left/right motor joints under '{name}'. "
                f"Expected child joints named '{LEFT_MOTOR_ALIAS}' and '{RIGHT_MOTOR_ALIAS}'."
            )

        self.disable_child_scripts(body)
        pos = self.sim.getObjectPosition(body, self.sim.handle_world)
        handles = RobotHandles(name=name, body=int(body), left_motor=int(left),
                               right_motor=int(right), z_height=float(pos[2]))
        self._robots[name] = handles
        return handles

    def list_scene_robots(self, prefix: str = ROBOT_PREFIX) -> List[str]:
        """Return sorted aliases of all top-level robot models matching ``prefix``."""
        self._ensure_connected()
        objs = self.sim.getObjectsInTree(self.sim.handle_scene, self._ALL, 0)
        found: List[str] = []
        for h in objs:
            alias = self.sim.getObjectAlias(h, 0)
            if alias.startswith(prefix):
                parent = self.sim.getObjectParent(h)
                if parent == -1 or not self.sim.getObjectAlias(parent, 0).startswith(prefix):
                    found.append(alias)
        return sorted(set(found), key=_natural_robot_sort_key)

    def clone_robot(self, source_name: str, new_name: str) -> RobotHandles:
        """Duplicate ``source_name`` model in the scene and rename the clone to ``new_name``."""
        self._ensure_connected()
        src = self.get_robot(source_name)
        pasted = self.sim.copyPasteObjects([src.body], 1)  # 1 = copy full model tree
        if not pasted:
            raise SimConnectionError(f"copyPasteObjects failed for '{source_name}'")
        new_body = int(pasted[0])
        self.sim.setObjectAlias(new_body, new_name)
        self.disable_child_scripts(new_body)
        self._robots.pop(new_name, None)
        return self.get_robot(new_name)

    def disable_child_scripts(self, root_handle: int) -> None:
        """Defensively disable any built-in Lua child scripts (e.g. Braitenberg demo) under robot."""
        self._ensure_connected()
        # Method 1: CoppeliaSim >= 4.5 script objects in tree
        script_type = None
        for _n in ("sceneobject_script", "object_script_type"):
            try:
                script_type = getattr(self.sim, _n)
                break
            except AttributeError:
                continue
        if script_type is not None:
            try:
                scripts = self.sim.getObjectsInTree(root_handle, script_type, 0)
                for s in scripts:
                    if hasattr(self.sim, "setScriptInt32Param") and hasattr(self.sim, "scriptintparam_enabled"):
                        self.sim.setScriptInt32Param(s, self.sim.scriptintparam_enabled, 0)
                    else:
                        self.sim.removeObjects([s])
                if scripts:
                    log.info("Disabled %d child script(s) under handle %d", len(scripts), root_handle)
            except Exception as exc:
                log.debug("Script object disable check: %s", exc)

        # Method 2: Legacy associated child script
        if hasattr(self.sim, "getScript") and hasattr(self.sim, "scripttype_childscript"):
            try:
                s = self.sim.getScript(self.sim.scripttype_childscript, root_handle)
                if s != -1 and hasattr(self.sim, "setScriptInt32Param"):
                    self.sim.setScriptInt32Param(s, self.sim.scriptintparam_enabled, 0)
            except Exception:
                pass

    def get_pose(self, name: str) -> Pose2D:
        h = self.get_robot(name)
        m = self.sim.getObjectMatrix(h.body, self.sim.handle_world)
        return Pose2D(x=float(m[3]), y=float(m[7]), yaw=yaw_from_matrix(m))

    def set_pose(self, name: str, pose: Pose2D) -> None:
        """Teleport ``name`` to ``pose`` and reset dynamic state."""
        h = self.get_robot(name)
        # Reset wheels first
        self.set_wheel_speeds(name, 0.0, 0.0)
        # Collect all objects in the model tree to reset dynamics cleanly
        tree = self.sim.getObjectsInTree(h.body, self._ALL, 0)
        self.sim.setObjectPosition(h.body, self.sim.handle_world, [pose.x, pose.y, h.z_height])
        self.sim.setObjectOrientation(h.body, self.sim.handle_world, [0.0, 0.0, pose.yaw])
        for obj in tree:
            try:
                self.sim.resetDynamicObject(obj)
            except Exception:
                pass

    def set_wheel_speeds(self, name: str, omega_left: float, omega_right: float) -> None:
        h = self.get_robot(name)
        lim = self.robot_params.max_wheel_speed
        wl = max(-lim, min(lim, float(omega_left)))
        wr = max(-lim, min(lim, float(omega_right)))
        self.sim.setJointTargetVelocity(h.left_motor, wl)
        self.sim.setJointTargetVelocity(h.right_motor, wr)

    # ------------------------------------------------------------------ visualization in sim
    def draw_path(self, waypoints: Sequence[Tuple[float, float]],
                  color: Tuple[float, float, float] = (0.1, 0.7, 0.2),
                  z_offset: float = 0.03) -> Optional[int]:
        """Draw a 3D line strip in the CoppeliaSim scene for visual debugging."""
        self._ensure_connected()
        if len(waypoints) < 2 or not hasattr(self.sim, "addDrawingObject"):
            return None
        try:
            mode = self.sim.drawing_lines
            obj = self.sim.addDrawingObject(mode, 3.0, 0.0, -1, len(waypoints) * 2, list(color))
            for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
                self.sim.addDrawingObjectItem(obj, [x0, y0, z_offset, x1, y1, z_offset])
            self._drawings.append(int(obj))
            return int(obj)
        except Exception as exc:
            log.debug("Could not draw path in CoppeliaSim: %s", exc)
            return None

    def clear_drawings(self) -> None:
        if self.sim is None:
            return
        for obj in self._drawings:
            try:
                self.sim.removeDrawingObject(obj)
            except Exception:
                pass
        self._drawings.clear()

    # ------------------------------------------------------------------ internal lookup helpers
    def _find_object(self, path: str) -> Optional[int]:
        try:
            return int(self.sim.getObject(path, {"noError": True}))
        except Exception:
            try:
                h = int(self.sim.getObject(path))
                return h if h >= 0 else None
            except Exception:
                return None

    def _find_object_by_alias(self, alias: str) -> Optional[int]:
        objs = self.sim.getObjectsInTree(self.sim.handle_scene, self._ALL, 0)
        for h in objs:
            if self.sim.getObjectAlias(h, 0) == alias:
                return int(h)
        return None

    def _find_child_joint(self, parent: int, expected_alias: str,
                          fallback_Hint: str) -> Optional[int]:
        joints = self.sim.getObjectsInTree(parent, self._JOINT, 0)
        for j in joints:
            alias = self.sim.getObjectAlias(j, 0)
            if alias == expected_alias or alias.endswith("_" + expected_alias):
                return int(j)
        for j in joints:
            alias = self.sim.getObjectAlias(j, 0).lower()
            if fallback_Hint.lower() in alias and "motor" in alias:
                return int(j)
        return None

    # ------------------------------------------------------------------ shape bounding box helper
    def get_shape_footprint(self, handle: int) -> Tuple[float, float, float, float, float, float]:
        """Return ``(cx, cy, size_x, size_y, yaw, height)`` for a shape handle in world frame.

        Handles shapes whose geometric center is offset from their object frame origin
        by transforming the local bounding-box center with the object's world matrix.
        """
        self._ensure_connected()
        m = self.sim.getObjectMatrix(handle, self.sim.handle_world)
        sx, sy, sz, local_center = self._read_local_bb(handle)
        wx, wy, _ = transform_point(m, local_center)
        yaw = yaw_from_matrix(m)
        return float(wx), float(wy), float(sx), float(sy), float(yaw), float(sz)

    def _read_local_bb(self, handle: int) -> Tuple[float, float, float, Tuple[float, float, float]]:
        # CoppeliaSim >= 4.5 has sim.getShapeBB(handle) -> [sx, sy, sz]
        if hasattr(self.sim, "getShapeBB"):
            try:
                res = self.sim.getShapeBB(handle)
                if isinstance(res, tuple) and len(res) == 2:
                    size, pose = res
                    return float(size[0]), float(size[1]), float(size[2]), (
                        float(pose[0]), float(pose[1]), float(pose[2]))
                return float(res[0]), float(res[1]), float(res[2]), (0.0, 0.0, 0.0)
            except Exception:
                pass
        # Fallback: objfloatparam_objbbox_min/max (local frame)
        xmin = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_x)
        xmax = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_x)
        ymin = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_y)
        ymax = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_y)
        zmin = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_z)
        zmax = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_z)
        return (xmax - xmin, ymax - ymin, zmax - zmin,
                (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)))


def _natural_robot_sort_key(name: str) -> Tuple[str, int]:
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (name, -1)


# ======================================================================================
# Offline Kinematic Mock Simulator (for unit tests & dry runs without CoppeliaSim)
# ======================================================================================
class MockSim:
    """Pure-Python differential-drive kinematic simulator with the same API as ZMQSimInterface."""

    def __init__(self, dt: float = SIM.dt, robot_params: RobotParams = ROBOT,
                 initial_robots: Optional[Dict[str, Pose2D]] = None) -> None:
        self.dt = float(dt)
        self.robot_params = robot_params
        self._poses: Dict[str, Pose2D] = dict(initial_robots or {f"{ROBOT_PREFIX}0": Pose2D(-4.0, -4.0, 0.0)})
        self._wheel_speeds: Dict[str, Tuple[float, float]] = {k: (0.0, 0.0) for k in self._poses}
        self._time: float = 0.0
        self.running: bool = False

    def connect(self) -> None:
        pass

    def start(self, stepping: Optional[bool] = None) -> None:
        self.running = True

    def stop(self) -> None:
        for k in list(self._wheel_speeds):
            self._wheel_speeds[k] = (0.0, 0.0)
        self.running = False

    def sim_time(self) -> float:
        return self._time

    def get_robot(self, name: str) -> RobotHandles:
        if name not in self._poses:
            raise SimConnectionError(f"Robot '{name}' does not exist in MockSim")
        idx = list(self._poses).index(name) + 1
        return RobotHandles(name=name, body=idx * 10, left_motor=idx * 10 + 1, right_motor=idx * 10 + 2)

    def list_scene_robots(self, prefix: str = ROBOT_PREFIX) -> List[str]:
        return sorted([k for k in self._poses if k.startswith(prefix)], key=_natural_robot_sort_key)

    def clone_robot(self, source_name: str, new_name: str) -> RobotHandles:
        if source_name not in self._poses:
            raise SimConnectionError(f"Source robot '{source_name}' not in MockSim")
        self._poses[new_name] = self._poses[source_name]
        self._wheel_speeds[new_name] = (0.0, 0.0)
        return self.get_robot(new_name)

    def get_pose(self, name: str) -> Pose2D:
        if name not in self._poses:
            raise SimConnectionError(f"Robot '{name}' not in MockSim")
        return self._poses[name]

    def set_pose(self, name: str, pose: Pose2D) -> None:
        self._poses[name] = Pose2D(float(pose.x), float(pose.y), wrap_angle(float(pose.yaw)))
        self._wheel_speeds[name] = (0.0, 0.0)

    def set_wheel_speeds(self, name: str, omega_left: float, omega_right: float) -> None:
        if name not in self._poses:
            raise SimConnectionError(f"Robot '{name}' not in MockSim")
        lim = self.robot_params.max_wheel_speed
        wl = max(-lim, min(lim, float(omega_left)))
        wr = max(-lim, min(lim, float(omega_right)))
        self._wheel_speeds[name] = (wl, wr)

    def step(self) -> None:
        r = self.robot_params.wheel_radius
        L = self.robot_params.wheel_separation
        for name, pose in list(self._poses.items()):
            wl, wr = self._wheel_speeds.get(name, (0.0, 0.0))
            v = 0.5 * r * (wr + wl)
            omega = (r / L) * (wr - wl)
            mid_yaw = pose.yaw + 0.5 * omega * self.dt
            nx = pose.x + v * math.cos(mid_yaw) * self.dt
            ny = pose.y + v * math.sin(mid_yaw) * self.dt
            nyaw = wrap_angle(pose.yaw + omega * self.dt)
            self._poses[name] = Pose2D(nx, ny, nyaw)
        self._time += self.dt

    def draw_path(self, waypoints: Sequence[Tuple[float, float]],
                  color: Tuple[float, float, float] = (0.1, 0.7, 0.2),
                  z_offset: float = 0.03) -> Optional[int]:
        return None

    def clear_drawings(self) -> None:
        pass
