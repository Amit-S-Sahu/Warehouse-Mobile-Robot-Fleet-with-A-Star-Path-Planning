"""Per-robot state machine and telemetry for the warehouse fleet."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .config import GAINS, ROBOT, ControllerGains, RobotParams
from .controller import ControlOutput, WaypointFollower
from .fleet_config import Task
from .geometry import Pose2D, Point


class RobotState(str, Enum):
    IDLE = "IDLE"
    TO_PICK = "TO_PICK"
    PICKING = "PICKING"
    TO_DROP = "TO_DROP"
    DROPPING = "DROPPING"
    TO_GOAL = "TO_GOAL"
    TO_DOCK = "TO_DOCK"
    DONE = "DONE"


@dataclass
class RobotAgent:
    """State, controller, active task, and recorded trajectory of one robot."""

    name: str
    priority: int  # lower number = higher priority (0 is highest)
    home_pose: Pose2D
    pose: Pose2D
    gains: ControllerGains = GAINS
    robot_params: RobotParams = ROBOT

    state: RobotState = RobotState.IDLE
    active_task: Optional[Task] = None
    goal_xy: Optional[Point] = None
    dwell_remaining: float = 0.0
    blocked_time: float = 0.0
    yielding: bool = False

    controller: WaypointFollower = field(init=False)
    planned_history: List[List[Point]] = field(default_factory=list)
    trajectory: List[Point] = field(default_factory=list)
    distance_travelled: float = 0.0
    tasks_completed: int = 0
    yield_steps: int = 0
    replans: int = 0

    def __post_init__(self) -> None:
        self.controller = WaypointFollower(self.gains, self.robot_params)
        self.trajectory.append(self.pose.xy())

    def update_pose(self, new_pose: Pose2D) -> None:
        step_dist = self.pose.distance_to(new_pose)
        self.distance_travelled += step_dist
        self.pose = new_pose
        self.trajectory.append(new_pose.xy())

    def assign_leg(self, state: RobotState, goal_xy: Point,
                   waypoints: Sequence[Point]) -> None:
        self.state = state
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.blocked_time = 0.0
        self.controller.set_path(waypoints, current_pose=self.pose)
        self.planned_history.append(list(self.controller.waypoints))

    def start_dwell(self, next_state: RobotState, duration: float) -> None:
        self.state = next_state
        self.dwell_remaining = max(0.0, float(duration))
        self.blocked_time = 0.0

    @property
    def is_moving_state(self) -> bool:
        return self.state in (RobotState.TO_PICK, RobotState.TO_DROP,
                              RobotState.TO_GOAL, RobotState.TO_DOCK)

    def all_planned_points(self) -> List[Point]:
        """Concatenate all planned waypoint segments for visualization."""
        out: List[Point] = []
        for seg in self.planned_history:
            for pt in seg:
                if not out or out[-1] != pt:
                    out.append(pt)
        return out
