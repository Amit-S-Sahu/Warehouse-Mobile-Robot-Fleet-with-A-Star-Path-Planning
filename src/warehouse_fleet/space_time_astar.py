"""Prioritized Space-Time A* planner in (row, col, t) for multi-robot conflict avoidance.

Features
--------
* Searches over 4-connected or 8-connected grid moves **plus a 'wait' action** ``(0, 0)``.
* Maintains a :class:`ReservationTable` of cells occupied at time step ``t`` and directed
  edges traversed from ``t -> t + 1`` by higher-priority robots.
* Prevents **vertex conflicts** (two robots within ``conflict_radius_cells`` at the same ``t``)
  and **edge / swap conflicts** (two robots swapping cells between ``t`` and ``t + 1``).
* Supports permanent goal reservations (a robot that finishes at ``t_end`` blocks its goal
  cell for ``t >= t_end`` unless given a new task).
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .astar import SQRT2, manhattan_distance, octile_distance
from .grid_map import Cell, GridMap, disk_offsets

TimeStep = int
STNode = Tuple[int, int, TimeStep]  # (row, col, t)

# Moves: (dr, dc, step_cost). Wait action costs 1.0 time step (slightly >1 tie-break vs arriving early)
MOVES_8_WAIT: Tuple[Tuple[int, int, float], ...] = (
    (0, 0, 1.05),  # wait in place (slight penalty so robots prefer arriving early over waiting at start)
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)
MOVES_4_WAIT: Tuple[Tuple[int, int, float], ...] = MOVES_8_WAIT[:5]


class ReservationTable:
    """Space-time reservations populated by higher-priority robots."""

    def __init__(self, conflict_radius_cells: float = 0.0) -> None:
        self.conflict_radius_cells = float(conflict_radius_cells)
        self._offsets: List[Cell] = disk_offsets(self.conflict_radius_cells)
        self._vertex: Dict[TimeStep, Set[Cell]] = {}
        self._edge: Dict[TimeStep, Set[Tuple[Cell, Cell]]] = {}
        self._goal_holds: Dict[Cell, TimeStep] = {}  # cell -> earliest t from which it's held forever

    def clear(self) -> None:
        self._vertex.clear()
        self._edge.clear()
        self._goal_holds.clear()

    def reserve_vertex(self, cell: Cell, t: TimeStep) -> None:
        bucket = self._vertex.setdefault(int(t), set())
        r, c = cell
        for dr, dc in self._offsets:
            bucket.add((r + dr, c + dc))

    def reserve_edge(self, u: Cell, v: Cell, t: TimeStep) -> None:
        """Reserve transition from ``u`` at time ``t`` to ``v`` at time ``t + 1``."""
        if u == v:
            return
        self._edge.setdefault(int(t), set()).add((u, v))

    def reserve_path(self, timed_cells: Sequence[Cell], start_t: TimeStep = 0,
                     hold_goal_steps: int = 0) -> None:
        """Reserve every vertex and transition along ``timed_cells`` starting at ``start_t``."""
        if not timed_cells:
            return
        for idx, cell in enumerate(timed_cells):
            t = start_t + idx
            self.reserve_vertex(cell, t)
            if idx + 1 < len(timed_cells):
                nxt = timed_cells[idx + 1]
                self.reserve_edge(cell, nxt, t)
                # Also reserve target vertex at t to prevent close-following collisions
                self.reserve_vertex(nxt, t)
        end_t = start_t + len(timed_cells) - 1
        goal = timed_cells[-1]
        for extra in range(1, hold_goal_steps + 1):
            self.reserve_vertex(goal, end_t + extra)

    def reserve_permanent_from(self, cell: Cell, from_t: TimeStep) -> None:
        r, c = cell
        for dr, dc in self._offsets:
            nb = (r + dr, c + dc)
            prev = self._goal_holds.get(nb)
            if prev is None or from_t < prev:
                self._goal_holds[nb] = int(from_t)

    def is_vertex_free(self, cell: Cell, t: TimeStep) -> bool:
        if cell in self._vertex.get(t, ()):
            return False
        hold_t = self._goal_holds.get(cell)
        if hold_t is not None and t >= hold_t:
            return False
        return True

    def is_transition_free(self, u: Cell, v: Cell, t: TimeStep) -> bool:
        """Check if moving from ``u`` at time ``t`` to ``v`` at time ``t + 1`` is conflict-free."""
        if not self.is_vertex_free(v, t + 1):
            return False
        # Forbid head-on swap (another robot moving v -> u at time t)
        if (v, u) in self._edge.get(t, ()):
            return False
        return True


@dataclass
class SpaceTimeResult:
    timed_cells: Optional[List[Cell]]  # index k corresponds to time start_t + k
    cost: float
    expanded: int
    start_t: int = 0

    @property
    def success(self) -> bool:
        return self.timed_cells is not None


def space_time_astar(occupancy: np.ndarray, start: Cell, goal: Cell,
                     reservations: Optional[ReservationTable] = None, *,
                     start_t: int = 0, max_time_steps: int = 400,
                     diagonal: bool = False) -> SpaceTimeResult:
    """Find a conflict-free path in ``(row, col, t)`` from ``(start, start_t)`` to ``goal``."""
    occ = np.asarray(occupancy, dtype=bool)
    rows, cols = occ.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    res = reservations if reservations is not None else ReservationTable()

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return SpaceTimeResult(None, math.inf, 0, start_t)
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return SpaceTimeResult(None, math.inf, 0, start_t)
    if occ[start] or occ[goal]:
        return SpaceTimeResult(None, math.inf, 0, start_t)

    heuristic = octile_distance if diagonal else manhattan_distance
    moves = MOVES_8_WAIT if diagonal else MOVES_4_WAIT

    start_node: STNode = (start[0], start[1], int(start_t))
    g_score: Dict[STNode, float] = {start_node: 0.0}
    came_from: Dict[STNode, STNode] = {}
    closed: Set[STNode] = set()

    tie = itertools.count()
    open_heap: List[Tuple[float, int, int, STNode]] = [
        (heuristic(start, goal), start_t, next(tie), start_node)
    ]
    expanded = 0
    limit_t = start_t + max_time_steps

    while open_heap:
        _, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1

        r, c, t = cur
        if (r, c) == goal and res.is_vertex_free(goal, t + 1):
            return SpaceTimeResult(_reconstruct_st(came_from, cur), g_score[cur], expanded, start_t)

        if t >= limit_t:
            continue

        g_cur = g_score[cur]
        for dr, dc, step_cost in moves:
            nr, nc, nt = r + dr, c + dc, t + 1
            if not (0 <= nr < rows and 0 <= nc < cols) or occ[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (occ[r, nc] or occ[nr, c]):
                continue  # no corner cutting
            if not res.is_transition_free((r, c), (nr, nc), t):
                continue

            nxt: STNode = (nr, nc, nt)
            if nxt in closed:
                continue
            ng = g_cur + step_cost
            if ng < g_score.get(nxt, math.inf):
                g_score[nxt] = ng
                came_from[nxt] = cur
                f = ng + heuristic((nr, nc), goal)
                heapq.heappush(open_heap, (f, nt, next(tie), nxt))

    return SpaceTimeResult(None, math.inf, expanded, start_t)


def _reconstruct_st(came_from: Dict[STNode, STNode], cur: STNode) -> List[Cell]:
    cells: List[Cell] = [(cur[0], cur[1])]
    while cur in came_from:
        cur = came_from[cur]
        cells.append((cur[0], cur[1]))
    cells.reverse()
    return cells


def plan_prioritized_space_time(
    grid_map: GridMap,
    starts_goals: Sequence[Tuple[str, Tuple[float, float], Tuple[float, float]]],
    *,
    conflict_radius_m: float = 0.0,
    max_time_steps: int = 400,
    hold_goal_steps: int = 10,
    diagonal: bool = False,
) -> Dict[str, Optional[List[Tuple[float, float]]]]:
    """Plan conflict-free paths for an ordered list of ``(robot_name, start_xy, goal_xy)``."""
    radius_cells = conflict_radius_m / grid_map.resolution if conflict_radius_m > 0 else 0.0
    table = ReservationTable(conflict_radius_cells=radius_cells)
    occ = grid_map.inflated
    plans: Dict[str, Optional[List[Tuple[float, float]]]] = {}

    for name, s_xy, g_xy in starts_goals:
        s_cell = grid_map.world_to_grid(*s_xy)
        g_cell = grid_map.world_to_grid(*g_xy)
        if occ[s_cell]:
            s_cell = grid_map.nearest_free(s_cell, occupancy=occ) or s_cell
        if occ[g_cell]:
            g_cell = grid_map.nearest_free(g_cell, occupancy=occ) or g_cell

        res = space_time_astar(
            occ, s_cell, g_cell, table, max_time_steps=max_time_steps, diagonal=diagonal
        )
        if not res.success or res.timed_cells is None:
            plans[name] = None
            continue

        table.reserve_path(res.timed_cells, start_t=0, hold_goal_steps=hold_goal_steps)
        # Convert cell sequence to world waypoints (preserving wait steps as repeated points
        # or compressed waypoints for the follower)
        pts = [grid_map.grid_to_world(*c) for c in res.timed_cells]
        if pts:
            pts[0] = (float(s_xy[0]), float(s_xy[1]))
            pts[-1] = (float(g_xy[0]), float(g_xy[1]))
        plans[name] = pts
    return plans
