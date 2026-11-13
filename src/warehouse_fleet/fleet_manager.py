"""Fleet coordinator: task allocation, prioritized planning, runtime safety, and replanning.

Responsibilities
----------------
1. **Task allocation**: assigns pending pick->drop tasks to idle robots using lowest A* path
   cost to the pick station (or direct goals in ``goals`` mode).
2. **Path planning**: supports both classic A* (with LOS smoothing) and Prioritized
   Space-Time A* on a coarse grid so robots coordinate through aisles.
3. **Runtime safety layer**: on every control step, checks if another robot is within
   ``safety_distance`` inside the forward ``+-yield_cone_deg`` cone. The lower-priority
   robot yields (stops) until clear.
4. **Deadlock recovery**: if a robot yields or makes no progress for longer than
   ``deadlock_timeout``, replans its path around the current positions of all other robots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .astar import densify, plan_path, prune_path
from .config import FLEET, GAINS, GRID, ROBOT, SIM, FleetParams, RobotParams
from .fleet_config import FleetConfig, Task, resolve_goals, resolve_stations
from .fleet_setup import SimLike, setup_fleet_in_scene
from .geometry import Pose2D, Point, wrap_angle
from .grid_map import GridMap
from .robot_agent import RobotAgent, RobotState
from .space_time_astar import ReservationTable, space_time_astar

log = logging.getLogger(__name__)


@dataclass
class FleetMetrics:
    """Summary statistics after a fleet run."""

    steps: int
    sim_time: float
    tasks_completed: int
    total_tasks: int
    all_done: bool
    total_distance: float
    total_yield_steps: int
    total_replans: int
    min_inter_robot_distance: float


class FleetManager:
    """Coordinates N Pioneer P3-DX robots executing warehouse tasks or direct goals."""

    def __init__(
        self,
        sim: SimLike,
        grid: GridMap,
        cfg: FleetConfig,
        *,
        use_space_time: bool = True,
        fleet_params: FleetParams = FLEET,
        robot_params: RobotParams = ROBOT,
        dt: float = SIM.dt,
    ) -> None:
        self.sim = sim
        self.grid = grid
        self.cfg = cfg
        self.use_space_time = bool(use_space_time)
        self.fleet_params = fleet_params
        self.robot_params = robot_params
        self.dt = float(dt)

        self.stations: Dict[str, Point] = resolve_stations(cfg, grid)
        self.direct_goals: Dict[str, Point] = (
            resolve_goals(cfg, grid) if cfg.goals_mode == "goals" else {}
        )
        self.pending_tasks: List[Task] = list(cfg.tasks) if cfg.goals_mode == "tasks" else []
        self.total_tasks: int = (
            len(self.pending_tasks) if cfg.goals_mode == "tasks" else len(self.direct_goals)
        )

        # Coarse grid + reservation table for Space-Time A*
        self._st_grid: Optional[GridMap] = None
        self._reservations: Optional[ReservationTable] = None
        self._step_count: int = 0
        self.min_inter_robot_distance: float = math.inf

        start_poses = setup_fleet_in_scene(sim, cfg, grid)
        self.agents: Dict[str, RobotAgent] = {}
        for prio, (name, pose) in enumerate(start_poses.items()):
            self.agents[name] = RobotAgent(
                name=name,
                priority=prio,
                home_pose=pose,
                pose=pose,
                gains=GAINS,
                robot_params=robot_params,
            )

        if self.use_space_time:
            self._init_space_time_planner()

        # Initial assignment
        self._allocate_work()

    def _init_space_time_planner(self) -> None:
        res = max(self.grid.resolution, self.fleet_params.st_resolution)
        # Re-sample the inflated grid at coarse resolution if needed
        if abs(res - self.grid.resolution) < 1e-6:
            self._st_grid = self.grid
        else:
            factor = max(1, int(round(res / self.grid.resolution)))
            rows = self.grid.rows // factor
            cols = self.grid.cols // factor
            trimmed = self.grid.raw[: rows * factor, : cols * factor]
            coarse_raw = trimmed.reshape(rows, factor, cols, factor).any(axis=(1, 3))
            self._st_grid = GridMap(
                coarse_raw,
                resolution=self.grid.resolution * factor,
                origin=(self.grid.origin_x, self.grid.origin_y),
                inflation_radius=self.grid.inflation_radius,
                border_is_obstacle=self.grid.border_is_obstacle,
            )
        conflict_cells = (1.5 * self.robot_params.footprint_radius) / self._st_grid.resolution
        self._reservations = ReservationTable(conflict_radius_cells=conflict_cells)

    # ------------------------------------------------------------------ planning
    def _plan_route(self, agent: RobotAgent, goal_xy: Point,
                    avoid_others: bool = False) -> Optional[List[Point]]:
        """Plan waypoints for ``agent`` to ``goal_xy``."""
        other_positions = [
            other.pose.xy() for other in self.agents.values() if other.name != agent.name
        ]
        active_grid = (
            self.grid.with_temporary_obstacles(
                other_positions, radius=2.0 * self.robot_params.footprint_radius
            )
            if avoid_others and other_positions
            else self.grid
        )

        # 1. Try Space-Time A* on the coarse grid (unless doing an emergency local detour)
        if self.use_space_time and not avoid_others and self._st_grid is not None and self._reservations is not None:
            s_cell = self._st_grid.world_to_grid(*agent.pose.xy())
            g_cell = self._st_grid.world_to_grid(*goal_xy)
            occ = self._st_grid.inflated
            if not self._st_grid.is_free(s_cell, occ):
                s_cell = self._st_grid.nearest_free(s_cell, occupancy=occ) or s_cell
            if not self._st_grid.is_free(g_cell, occ):
                g_cell = self._st_grid.nearest_free(g_cell, occupancy=occ) or g_cell

            st_res = space_time_astar(
                occ,
                s_cell,
                g_cell,
                self._reservations,
                start_t=self._step_count,
                max_time_steps=self.fleet_params.st_max_time_steps,
                diagonal=True,
            )
            if st_res.success and st_res.timed_cells:
                self._reservations.reserve_path(
                    st_res.timed_cells, start_t=self._step_count, hold_goal_steps=8
                )
                pruned = prune_path(self._st_grid, st_res.timed_cells, occ)
                wps = [self._st_grid.grid_to_world(*c) for c in pruned]
                if len(wps) == 1:
                    wps = [wps[0], wps[0]]
                wps[0] = agent.pose.xy()
                wps[-1] = (float(goal_xy[0]), float(goal_xy[1]))
                return wps

        # 2. Classic A* on the fine grid (with optional temporary obstacles for other robots)
        planned = plan_path(
            active_grid,
            agent.pose.xy(),
            goal_xy,
            max_snap_distance=GRID.max_snap_distance,
        )
        if planned is None and avoid_others:
            # Fallback to static grid if temporary obstacles completely boxed the robot in
            planned = plan_path(
                self.grid,
                agent.pose.xy(),
                goal_xy,
                max_snap_distance=GRID.max_snap_distance,
            )
        return planned.waypoints if planned is not None else None

    # ------------------------------------------------------------------ task allocation
    def _allocate_work(self) -> None:
        """Assign tasks or direct goals to any IDLE robots."""
        if self.cfg.goals_mode == "goals":
            for agent in self.agents.values():
                if agent.state == RobotState.IDLE:
                    target = self.direct_goals.get(agent.name)
                    if target is None:
                        agent.state = RobotState.DONE
                        continue
                    wps = self._plan_route(agent, target)
                    if wps is None:
                        log.warning("%s: no path to goal %s", agent.name, target)
                        agent.state = RobotState.DONE
                        continue
                    agent.assign_leg(RobotState.TO_GOAL, target, wps)
                    self.sim.draw_path(wps, color=_robot_color(agent.priority))
            return

        # Tasks mode: greedy assignment by lowest A* path length to pick station
        for agent in self.agents.values():
            if agent.state != RobotState.IDLE:
                continue

            if self.pending_tasks:
                best_idx = 0
                best_cost = math.inf
                for idx, task in enumerate(self.pending_tasks):
                    pick_xy = self.stations[task.pick]
                    res = plan_path(self.grid, agent.pose.xy(), pick_xy)
                    cost = res.cost if res is not None else math.inf
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx

                task = self.pending_tasks.pop(best_idx)
                pick_xy = self.stations[task.pick]
                wps = self._plan_route(agent, pick_xy)
                if wps is not None:
                    agent.active_task = task
                    agent.assign_leg(RobotState.TO_PICK, pick_xy, wps)
                    self.sim.draw_path(wps, color=_robot_color(agent.priority))
                    log.info("Assigned task %s (%s -> %s) to %s",
                             task.id, task.pick, task.drop, agent.name)
                    continue

            # No tasks left for this idle robot -> return to home dock or finish
            if self.cfg.return_to_start and agent.pose.distance_to(agent.home_pose) > GAINS.waypoint_tolerance:
                wps = self._plan_route(agent, agent.home_pose.xy())
                if wps is not None:
                    agent.assign_leg(RobotState.TO_DOCK, agent.home_pose.xy(), wps)
                    continue
            agent.state = RobotState.DONE

    # ------------------------------------------------------------------ runtime safety
    def _should_yield(self, agent: RobotAgent) -> bool:
        """True if ``agent`` should yield to another nearby robot."""
        if not agent.is_moving_state:
            return False

        cone_rad = math.radians(self.fleet_params.yield_cone_deg)
        safe_d = self.fleet_params.safety_distance
        hard_d = 2.0 * self.robot_params.footprint_radius + 0.08

        for other in self.agents.values():
            if other.name == agent.name:
                continue
            d = agent.pose.distance_to(other.pose)
            if d > safe_d:
                continue

            bearing = math.atan2(other.pose.y - agent.pose.y, other.pose.x - agent.pose.x)
            rel_angle = abs(wrap_angle(bearing - agent.pose.yaw))

            # Stationary/dwelling robot directly ahead -> always yield so we don't ram it
            if not other.is_moving_state and rel_angle <= cone_rad:
                return True

            # Both moving -> lower priority (higher priority index) yields when other is ahead
            # or dangerously close
            if agent.priority > other.priority:
                if rel_angle <= cone_rad or d <= hard_d:
                    return True
        return False

    def _update_min_distance(self) -> None:
        poses = [a.pose for a in self.agents.values()]
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                d = poses[i].distance_to(poses[j])
                if d < self.min_inter_robot_distance:
                    self.min_inter_robot_distance = d

    # ------------------------------------------------------------------ stepping
    def step(self) -> bool:
        """Execute one fleet control step and advance the simulation. Returns True when all DONE."""
        self._step_count += 1

        # 1. Read poses
        for agent in self.agents.values():
            agent.update_pose(self.sim.get_pose(agent.name))
        self._update_min_distance()

        # 2. Advance state machines & compute wheel speeds
        for agent in self.agents.values():
            if agent.state == RobotState.DONE:
                self.sim.set_wheel_speeds(agent.name, 0.0, 0.0)
                continue

            if agent.state in (RobotState.PICKING, RobotState.DROPPING):
                self.sim.set_wheel_speeds(agent.name, 0.0, 0.0)
                agent.dwell_remaining -= self.dt
                if agent.dwell_remaining <= 1e-6:
                    if agent.state == RobotState.PICKING and agent.active_task is not None:
                        drop_xy = self.stations[agent.active_task.drop]
                        wps = self._plan_route(agent, drop_xy)
                        if wps is not None:
                            agent.assign_leg(RobotState.TO_DROP, drop_xy, wps)
                            self.sim.draw_path(wps, color=_robot_color(agent.priority))
                            log.info("%s picked at %s, heading to %s",
                                     agent.name, agent.active_task.pick, agent.active_task.drop)
                        else:
                            agent.state = RobotState.IDLE
                    else:
                        if agent.active_task is not None:
                            log.info("%s completed task %s at %s",
                                     agent.name, agent.active_task.id, agent.active_task.drop)
                        agent.tasks_completed += 1
                        agent.active_task = None
                        agent.state = RobotState.IDLE
                continue

            if agent.state == RobotState.IDLE:
                self.sim.set_wheel_speeds(agent.name, 0.0, 0.0)
                continue

            # Moving states: check runtime safety layer
            if self._should_yield(agent):
                agent.yielding = True
                agent.yield_steps += 1
                agent.blocked_time += self.dt
                self.sim.set_wheel_speeds(agent.name, 0.0, 0.0)

                if agent.blocked_time >= self.fleet_params.deadlock_timeout and agent.goal_xy is not None:
                    log.info("%s blocked for %.1f s -> replanning around fleet",
                             agent.name, agent.blocked_time)
                    wps = self._plan_route(agent, agent.goal_xy, avoid_others=True)
                    agent.blocked_time = 0.0
                    if wps is not None:
                        agent.replans += 1
                        agent.controller.set_path(wps, current_pose=agent.pose)
                        agent.planned_history.append(list(wps))
                continue

            agent.yielding = False
            agent.blocked_time = max(0.0, agent.blocked_time - 0.5 * self.dt)
            out = agent.controller.step(agent.pose, self.dt)
            self.sim.set_wheel_speeds(agent.name, out.omega_left, out.omega_right)

            if out.goal_reached:
                self.sim.set_wheel_speeds(agent.name, 0.0, 0.0)
                if agent.state == RobotState.TO_PICK:
                    agent.start_dwell(RobotState.PICKING, self.cfg.dwell_time)
                elif agent.state == RobotState.TO_DROP:
                    agent.start_dwell(RobotState.DROPPING, self.cfg.dwell_time)
                elif agent.state == RobotState.TO_GOAL:
                    agent.tasks_completed += 1
                    log.info("%s reached direct goal %s", agent.name, agent.goal_xy)
                    agent.state = RobotState.DONE
                elif agent.state == RobotState.TO_DOCK:
                    log.info("%s returned to dock", agent.name)
                    agent.state = RobotState.DONE

        # 3. Allocate tasks to any newly IDLE robots
        self._allocate_work()

        # 4. Step physics
        self.sim.step()
        return self.all_done

    @property
    def all_done(self) -> bool:
        return all(a.state == RobotState.DONE for a in self.agents.values())

    def run(self, max_steps: int = 3000) -> FleetMetrics:
        """Start the simulation and step until all robots finish or ``max_steps`` is reached."""
        self.sim.start(stepping=True)
        for _ in range(max_steps):
            if self.step():
                break
        self.sim.stop()
        return self.metrics()

    def metrics(self) -> FleetMetrics:
        return FleetMetrics(
            steps=self._step_count,
            sim_time=self._step_count * self.dt,
            tasks_completed=sum(a.tasks_completed for a in self.agents.values()),
            total_tasks=self.total_tasks,
            all_done=self.all_done,
            total_distance=sum(a.distance_travelled for a in self.agents.values()),
            total_yield_steps=sum(a.yield_steps for a in self.agents.values()),
            total_replans=sum(a.replans for a in self.agents.values()),
            min_inter_robot_distance=self.min_inter_robot_distance,
        )


def _robot_color(index: int) -> Tuple[float, float, float]:
    palette = (
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
    )
    return palette[index % len(palette)]
