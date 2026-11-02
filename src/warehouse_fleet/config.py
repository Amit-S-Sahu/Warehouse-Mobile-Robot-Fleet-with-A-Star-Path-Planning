"""Static project parameters: robot geometry, grid, controller gains, simulation, fleet.

Everything that is *physical* or *tuning* related lives here. Things the user chooses per
run (number of robots, start locations, stations, tasks) live in ``config/fleet_config.json``
(see :mod:`warehouse_fleet.fleet_config`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DEFAULT_FLEET_CONFIG_PATH: Path = CONFIG_DIR / "fleet_config.json"
REFERENCE_LAYOUT_PATH: Path = CONFIG_DIR / "reference_layout.json"
SCENE_CACHE_PATH: Path = CONFIG_DIR / "scene_cache.json"

# --------------------------------------------------------------------------------------
# Scene naming conventions (objects built in the CoppeliaSim GUI)
# --------------------------------------------------------------------------------------
OBSTACLE_PREFIXES: Tuple[str, ...] = ("Wall_", "Shelf_")
ROBOT_PREFIX: str = "Robot_"
FLOOR_ALIAS: str = "Floor"
LEFT_MOTOR_ALIAS: str = "leftMotor"
RIGHT_MOTOR_ALIAS: str = "rightMotor"


@dataclass(frozen=True)
class RobotParams:
    """Pioneer P3-DX geometry and limits (verify with scripts/test_connection.py)."""

    wheel_radius: float = 0.0975  # [m]
    wheel_separation: float = 0.331  # [m] distance between wheel contact points
    footprint_radius: float = 0.28  # [m] radius of a circle enclosing the robot (around axle centre)
    max_wheel_speed: float = 8.0  # [rad/s]
    max_linear_speed: float = 0.4  # [m/s]
    max_angular_speed: float = 1.5  # [rad/s]


@dataclass(frozen=True)
class GridParams:
    """Occupancy grid settings."""

    resolution: float = 0.10  # [m/cell] for single-robot A*
    safety_margin: float = 0.08  # [m] extra clearance added to the robot radius
    border_is_obstacle: bool = True  # treat everything outside the floor as occupied
    default_floor_bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)
    max_snap_distance: float = 0.6  # [m] how far a goal may be moved to reach a free cell

    def inflation_radius(self, robot: "RobotParams") -> float:
        return robot.footprint_radius + self.safety_margin


@dataclass(frozen=True)
class ControllerGains:
    """Go-to-waypoint controller gains (tune in README 'Tuning' section)."""

    kp_angular: float = 2.0
    ki_angular: float = 0.0
    kd_angular: float = 0.1
    integral_limit: float = 0.5  # anti-windup clamp on the integral term
    kp_linear: float = 0.8
    rotate_in_place_angle: float = 0.785  # [rad] (~45 deg) above this, v = 0
    waypoint_tolerance: float = 0.15  # [m] switch to next intermediate waypoint
    goal_tolerance: float = 0.05  # [m] final goal reached


@dataclass(frozen=True)
class SimParams:
    """CoppeliaSim connection & stepping."""

    host: str = "localhost"
    port: int = 23000
    dt: float = 0.05  # [s] simulation time step (must match the scene's dt)
    stepping: bool = True  # synchronous stepping mode


@dataclass(frozen=True)
class FleetParams:
    """Multi-robot coordination settings."""

    min_robot_spacing: float = 0.8  # [m] minimum distance between start poses
    safety_distance: float = 0.7  # [m] runtime: yield if another robot is closer than this ahead
    yield_cone_deg: float = 60.0  # [deg] half-angle of the 'ahead' cone
    deadlock_timeout: float = 5.0  # [s] blocked longer than this -> replan
    st_resolution: float = 0.25  # [m/cell] coarse grid for space-time A*
    st_max_time_steps: int = 600  # horizon for space-time A*
    dwell_time: float = 2.0  # [s] time spent at pick/drop stations


ROBOT = RobotParams()
GRID = GridParams()
GAINS = ControllerGains()
SIM = SimParams()
FLEET = FleetParams()
