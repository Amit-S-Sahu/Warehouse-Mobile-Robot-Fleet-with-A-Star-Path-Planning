"""A* (A-Star) grid search plus path post-processing.

* 8-connected (default) or 4-connected moves; straight cost 1, diagonal cost sqrt(2).
* Diagonal moves are not allowed to cut obstacle corners.
* Octile heuristic for 8-connectivity (Manhattan for 4) -> admissible & consistent, so the
  returned path is optimal on the grid.
* :func:`plan_path` wraps everything for world coordinates: snapping start/goal out of the
  inflated zone, line-of-sight pruning and optional densification.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .grid_map import Cell, GridMap

log = logging.getLogger(__name__)

SQRT2 = math.sqrt(2.0)
Point = Tuple[float, float]

# (d_row, d_col, cost); first four are the 4-connected moves.
NEIGHBORS_8: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)
NEIGHBORS_4 = NEIGHBORS_8[:4]


# ======================================================================================
# Heuristics
# ======================================================================================
def octile_distance(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


def manhattan_distance(a: Cell, b: Cell) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


# ======================================================================================
# Core search
# ======================================================================================
@dataclass
class AStarResult:
    path: Optional[List[Cell]]  # start..goal inclusive, None if no path
    cost: float  # in cells (multiply by resolution for metres)
    expanded: int  # number of expanded nodes (for diagnostics)

    @property
    def success(self) -> bool:
        return self.path is not None


def astar_search(occupancy: np.ndarray, start: Cell, goal: Cell, *, diagonal: bool = True,
                 heuristic_weight: float = 1.0) -> AStarResult:
    """A* on a boolean occupancy grid (True = blocked).

    ``heuristic_weight`` > 1 gives weighted A* (faster, no longer optimal).
    """
    occ = np.asarray(occupancy, dtype=bool)
    rows, cols = occ.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    def inside(cell: Cell) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    if not inside(start) or not inside(goal):
        log.debug("A*: start %s or goal %s outside grid %s", start, goal, occ.shape)
        return AStarResult(None, math.inf, 0)
    if occ[start] or occ[goal]:
        log.debug("A*: start %s or goal %s is occupied", start, goal)
        return AStarResult(None, math.inf, 0)
    if start == goal:
        return AStarResult([start], 0.0, 0)

    heuristic = octile_distance if diagonal else manhattan_distance
    neighbors = NEIGHBORS_8 if diagonal else NEIGHBORS_4
    w = float(heuristic_weight)

    g = np.full((rows, cols), np.inf)
    parent = np.full((rows, cols), -1, dtype=np.int64)
    closed = np.zeros((rows, cols), dtype=bool)
    g[start] = 0.0

    tie = itertools.count()
    # Heap entries: (f, -g, tiebreak, cell). Preferring larger g on equal f expands
    # nodes closer to the goal first, which reduces the number of expansions.
    open_heap: List[Tuple[float, float, int, Cell]] = [
        (w * heuristic(start, goal), -0.0, next(tie), start)]
    expanded = 0

    while open_heap:
        _, neg_g, _, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue  # stale entry
        closed[cur] = True
        expanded += 1
        if cur == goal:
            return AStarResult(_reconstruct(parent, cur, cols), float(g[cur]), expanded)

        g_cur = -neg_g
        r, c = cur
        for dr, dc, step in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if occ[nr, nc] or closed[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (occ[r, nc] or occ[nr, c]):
                continue  # no corner cutting
            ng = g_cur + step
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = r * cols + c
                nb = (nr, nc)
                heapq.heappush(open_heap, (ng + w * heuristic(nb, goal), -ng, next(tie), nb))

    return AStarResult(None, math.inf, expanded)


def _reconstruct(parent: np.ndarray, cur: Cell, cols: int) -> List[Cell]:
    path = [cur]
    idx = int(parent[cur])
    while idx >= 0:
        cell = divmod(idx, cols)
        path.append(cell)
        idx = int(parent[cell])
    path.reverse()
    return path


# ======================================================================================
# Post-processing
# ======================================================================================
def prune_path(grid_map: GridMap, cells: Sequence[Cell],
               occupancy: Optional[np.ndarray] = None) -> List[Cell]:
    """Greedy line-of-sight pruning: keep only the cells where the path must turn."""
    cells = list(cells)
    n = len(cells)
    if n <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not grid_map.line_of_sight(cells[i], cells[j], occupancy):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def densify(waypoints: Sequence[Point], max_segment: float) -> List[Point]:
    """Insert evenly spaced points so that no segment is longer than ``max_segment``."""
    if max_segment <= 0:
        raise ValueError("max_segment must be > 0")
    pts = list(waypoints)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / max_segment - 1e-9)))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def path_length(waypoints: Sequence[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(waypoints[:-1], waypoints[1:]))


# ======================================================================================
# World-coordinate convenience wrapper
# ======================================================================================
@dataclass
class PlannedPath:
    waypoints: List[Point]  # world coordinates, first = start, last = goal
    cells: List[Cell] = field(repr=False)  # raw A* cell sequence
    cost: float = 0.0  # A* cost in metres
    expanded: int = 0
    start_snapped: bool = False  # start was inside the inflated zone and moved
    goal_snapped: bool = False  # goal was inside the inflated zone and moved

    @property
    def length(self) -> float:
        return path_length(self.waypoints)


def plan_path(grid_map: GridMap, start_xy: Point, goal_xy: Point, *, smooth: bool = True,
              max_segment: Optional[float] = None, snap: bool = True,
              max_snap_distance: float = 0.6, diagonal: bool = True,
              occupancy: Optional[np.ndarray] = None) -> Optional[PlannedPath]:
    """Plan a collision-free path between two world points.

    Returns ``None`` if start/goal are outside the map, cannot be snapped to free space,
    or no path exists.
    """
    occ = grid_map.occupancy(occupancy)
    s = grid_map.world_to_grid(*start_xy)
    g = grid_map.world_to_grid(*goal_xy)
    if not grid_map.in_bounds(s) or not grid_map.in_bounds(g):
        log.warning("plan_path: start %s or goal %s outside the map", start_xy, goal_xy)
        return None

    max_snap_cells = max_snap_distance / grid_map.resolution
    start_snapped = goal_snapped = False
    if occ[s]:
        ns = grid_map.nearest_free(s, max_snap_cells, occ) if snap else None
        if ns is None:
            log.warning("plan_path: start %s is in an occupied/inflated cell", start_xy)
            return None
        s, start_snapped = ns, True
    if occ[g]:
        ng = grid_map.nearest_free(g, max_snap_cells, occ) if snap else None
        if ng is None:
            log.warning("plan_path: goal %s is in an occupied/inflated cell", goal_xy)
            return None
        g, goal_snapped = ng, True

    result = astar_search(occ, s, g, diagonal=diagonal)
    if not result.success:
        log.warning("plan_path: no path from %s to %s (expanded %d nodes)",
                    start_xy, goal_xy, result.expanded)
        return None

    assert result.path is not None
    key_cells = prune_path(grid_map, result.path, occ) if smooth else list(result.path)
    waypoints: List[Point] = [grid_map.grid_to_world(*cell) for cell in key_cells]

    # Use the exact robot position as first point and the exact goal (if not snapped).
    # If the start was snapped, keep the snapped free cell so the robot first backs out of
    # the inflated zone along the shortest route before following the path.
    start_pt = (float(start_xy[0]), float(start_xy[1]))
    if start_snapped:
        waypoints.insert(0, start_pt)
    else:
        if len(waypoints) == 1:
            waypoints = [waypoints[0], waypoints[0]]
        waypoints[0] = start_pt
    if not goal_snapped:
        waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))

    if max_segment is not None:
        waypoints = densify(waypoints, max_segment)

    return PlannedPath(waypoints=waypoints, cells=list(result.path),
                       cost=result.cost * grid_map.resolution, expanded=result.expanded,
                       start_snapped=start_snapped, goal_snapped=goal_snapped)
