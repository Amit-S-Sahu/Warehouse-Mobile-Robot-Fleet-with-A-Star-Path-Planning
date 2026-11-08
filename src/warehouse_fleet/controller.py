"""Differential-drive P/PID waypoint-following controller for Pioneer P3-DX.

Given the current robot pose ``(x, y, yaw)`` and a sequence of 2D waypoints, computes left
and right wheel angular velocities ``(omega_left, omega_right)`` [rad/s] using:
* PID control on the heading error ``alpha``
* Proportional linear velocity scaled by ``max(cos(alpha), 0)`` so the robot slows down
  on sharp turns and never reverses into an unseen obstacle
* Rotate-in-place mode when ``|alpha| > rotate_in_place_angle``
* Ratio-preserving wheel velocity saturation at ``max_wheel_speed``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import GAINS, ROBOT, ControllerGains, RobotParams
from .geometry import Pose2D, Point, wrap_angle


@dataclass(frozen=True)
class ControlOutput:
    """Result of one controller step."""

    omega_left: float  # [rad/s]
    omega_right: float  # [rad/s]
    v: float  # linear speed command [m/s]
    omega: float  # angular speed command [rad/s]
    distance_error: float  # distance to active waypoint [m]
    heading_error: float  # wrapped angle to active waypoint [rad]
    waypoint_index: int
    goal_reached: bool


def diff_drive_ik(v: float, omega: float,
                  robot: RobotParams = ROBOT) -> Tuple[float, float]:
    """Convert body ``(v [m/s], omega [rad/s])`` to ``(omega_left, omega_right) [rad/s]``
    and saturate while preserving the turning radius ratio."""
    v = max(-robot.max_linear_speed, min(robot.max_linear_speed, float(v)))
    omega = max(-robot.max_angular_speed, min(robot.max_angular_speed, float(omega)))

    r = robot.wheel_radius
    L = robot.wheel_separation
    wl = (v - 0.5 * omega * L) / r
    wr = (v + 0.5 * omega * L) / r

    peak = max(abs(wl), abs(wr))
    if peak > robot.max_wheel_speed:
        scale = robot.max_wheel_speed / peak
        wl *= scale
        wr *= scale
    return wl, wr


class WaypointFollower:
    """Stateful P/PID controller that drives a differential-drive robot along a waypoint list."""

    def __init__(self, gains: ControllerGains = GAINS,
                 robot: RobotParams = ROBOT) -> None:
        self.gains = gains
        self.robot = robot
        self._waypoints: List[Point] = []
        self._index: int = 0
        self._integral_alpha: float = 0.0
        self._prev_alpha: Optional[float] = None
        self._goal_reached: bool = True

    # ------------------------------------------------------------------ path management
    @property
    def waypoints(self) -> List[Point]:
        return list(self._waypoints)

    @property
    def waypoint_index(self) -> int:
        return self._index

    @property
    def current_target(self) -> Optional[Point]:
        if not self._waypoints or self._goal_reached:
            return None
        return self._waypoints[min(self._index, len(self._waypoints) - 1)]

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    def set_path(self, waypoints: Sequence[Point],
                 current_pose: Optional[Pose2D] = None) -> None:
        """Load a new waypoint list and reset PID state.

        If the first waypoint coincides with ``current_pose`` (within ``waypoint_tolerance``)
        and there are further waypoints, advances immediately to index 1.
        """
        self._waypoints = [(float(p[0]), float(p[1])) for p in waypoints]
        self._integral_alpha = 0.0
        self._prev_alpha = None
        if not self._waypoints:
            self._index = 0
            self._goal_reached = True
            return

        self._index = 0
        self._goal_reached = False
        if current_pose is not None and len(self._waypoints) > 1:
            if current_pose.distance_to(self._waypoints[0]) <= self.gains.waypoint_tolerance:
                self._index = 1

    def reset_pid(self) -> None:
        self._integral_alpha = 0.0
        self._prev_alpha = None

    # ------------------------------------------------------------------ control step
    def step(self, pose: Pose2D, dt: float) -> ControlOutput:
        """Compute wheel velocities for ``pose`` over time step ``dt`` [s]."""
        if not self._waypoints or self._goal_reached:
            return ControlOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._index, True)

        # Advance through any intermediate waypoints already within tolerance
        while self._index < len(self._waypoints) - 1:
            tx, ty = self._waypoints[self._index]
            if math.hypot(tx - pose.x, ty - pose.y) <= self.gains.waypoint_tolerance:
                self._index += 1
                self.reset_pid()
            else:
                break

        tx, ty = self._waypoints[self._index]
        rho = math.hypot(tx - pose.x, ty - pose.y)
        is_final = (self._index == len(self._waypoints) - 1)
        tol = self.gains.goal_tolerance if is_final else self.gains.waypoint_tolerance

        if rho <= tol:
            if is_final:
                self._goal_reached = True
                self.reset_pid()
                return ControlOutput(0.0, 0.0, 0.0, 0.0, rho, 0.0, self._index, True)

        desired_yaw = math.atan2(ty - pose.y, tx - pose.x)
        alpha = wrap_angle(desired_yaw - pose.yaw)

        # PID on heading error alpha
        if dt > 1e-6:
            self._integral_alpha = max(
                -self.gains.integral_limit,
                min(self.gains.integral_limit, self._integral_alpha + alpha * dt),
            )
            d_alpha = 0.0 if self._prev_alpha is None else wrap_angle(alpha - self._prev_alpha) / dt
        else:
            d_alpha = 0.0
        self._prev_alpha = alpha

        omega = (self.gains.kp_angular * alpha
                 + self.gains.ki_angular * self._integral_alpha
                 + self.gains.kd_angular * d_alpha)

        if abs(alpha) > self.gains.rotate_in_place_angle:
            v = 0.0
        else:
            v_raw = min(self.gains.kp_linear * rho, self.robot.max_linear_speed)
            v = v_raw * max(math.cos(alpha), 0.0)

        wl, wr = diff_drive_ik(v, omega, self.robot)
        return ControlOutput(
            omega_left=wl,
            omega_right=wr,
            v=v,
            omega=omega,
            distance_error=rho,
            heading_error=alpha,
            waypoint_index=self._index,
            goal_reached=False,
        )
