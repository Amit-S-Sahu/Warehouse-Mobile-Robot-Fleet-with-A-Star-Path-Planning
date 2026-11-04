"""2D occupancy grid of the warehouse.

Conventions
-----------
* ``grid[row, col]``; ``row`` grows with world **y**, ``col`` grows with world **x**.
* Cell ``(0, 0)`` has its lower-left corner at ``(origin_x, origin_y)``.
* ``raw``      – True where an obstacle overlaps the cell (0 = free, 1 = obstacle).
* ``inflated`` – ``raw`` dilated by the robot radius + safety margin (configuration space).
  Planners treat the robot as a point on ``inflated``.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .scene_model import Bounds, Obstacle, SceneDescription

Cell = Tuple[int, int]  # (row, col)


# ======================================================================================
# Rasterization & inflation helpers
# ======================================================================================
def rasterize_obstacle(raw: np.ndarray, obstacle: Obstacle, resolution: float,
                       origin: Tuple[float, float]) -> None:
    """Mark (in place) every cell of ``raw`` that overlaps ``obstacle``.

    Axis-aligned boxes are rasterized exactly (a cell is marked iff its square overlaps
    the box interior). Rotated boxes use a slightly conservative test, so thin walls are
    never missed.
    """
    rows, cols = raw.shape
    ox, oy = origin
    xs = [p[0] for p in obstacle.corners()]
    ys = [p[1] for p in obstacle.corners()]

    c0 = max(0, int(math.floor((min(xs) - ox) / resolution)))
    c1 = min(cols - 1, int(math.floor((max(xs) - ox) / resolution)))
    r0 = max(0, int(math.floor((min(ys) - oy) / resolution)))
    r1 = min(rows - 1, int(math.floor((max(ys) - oy) / resolution)))
    if c0 > c1 or r0 > r1:
        return  # completely outside the grid

    cx = ox + (np.arange(c0, c1 + 1) + 0.5) * resolution
    cy = oy + (np.arange(r0, r1 + 1) + 0.5) * resolution
    xx, yy = np.meshgrid(cx, cy)  # shape (n_rows, n_cols)

    dx, dy = xx - obstacle.x, yy - obstacle.y
    c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    u = c * dx + s * dy  # coordinates in the box frame
    v = -s * dx + c * dy

    axis_aligned = abs(math.sin(2.0 * obstacle.yaw)) < 1e-6
    pad = resolution / 2.0 if axis_aligned else resolution * math.sqrt(2.0) / 2.0
    eps = 1e-9
    mask = (np.abs(u) < obstacle.size_x / 2.0 + pad - eps) & \
           (np.abs(v) < obstacle.size_y / 2.0 + pad - eps)
    raw[r0:r1 + 1, c0:c1 + 1] |= mask


def disk_offsets(radius_cells: float) -> List[Cell]:
    """Integer offsets (dr, dc) whose centre distance is <= ``radius_cells``."""
    if radius_cells < 0:
        raise ValueError("radius_cells must be >= 0")
    r = int(math.ceil(radius_cells))
    lim = radius_cells * radius_cells + 1e-9
    return [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
            if dr * dr + dc * dc <= lim]


def inflate(occupancy: np.ndarray, radius_cells: float,
            border_is_obstacle: bool = True) -> np.ndarray:
    """Binary dilation with a disk kernel (pure NumPy).

    If ``border_is_obstacle`` the area outside the grid counts as occupied, so cells close
    to the grid edge are also blocked.
    """
    occ = np.asarray(occupancy, dtype=bool)
    r = int(math.ceil(radius_cells))
    if r == 0:
        return occ.copy()
    rows, cols = occ.shape
    padded = np.pad(occ, r, mode="constant", constant_values=bool(border_is_obstacle))
    out = np.zeros_like(occ)
    for dr, dc in disk_offsets(radius_cells):
        out |= padded[r + dr:r + dr + rows, r + dc:r + dc + cols]
    return out


# ======================================================================================
# GridMap
# ======================================================================================
class GridMap:
    """Occupancy grid with world <-> grid transforms and query helpers."""

    def __init__(self, raw: np.ndarray, resolution: float, origin: Tuple[float, float],
                 inflation_radius: float = 0.0, border_is_obstacle: bool = True) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        self.raw: np.ndarray = np.asarray(raw, dtype=bool).copy()
        if self.raw.ndim != 2:
            raise ValueError("raw occupancy must be a 2D array")
        self.resolution: float = float(resolution)
        self.origin_x, self.origin_y = float(origin[0]), float(origin[1])
        self.inflation_radius: float = float(inflation_radius)
        self.border_is_obstacle: bool = bool(border_is_obstacle)
        self.inflated: np.ndarray = inflate(self.raw, self.inflation_radius / self.resolution,
                                            self.border_is_obstacle)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_scene(cls, scene: SceneDescription, resolution: float,
                   inflation_radius: float = 0.0,
                   border_is_obstacle: bool = True) -> "GridMap":
        xmin, xmax, ymin, ymax = scene.bounds
        cols = int(math.ceil((xmax - xmin) / resolution - 1e-9))
        rows = int(math.ceil((ymax - ymin) / resolution - 1e-9))
        raw = np.zeros((rows, cols), dtype=bool)
        for obstacle in scene.obstacles:
            rasterize_obstacle(raw, obstacle, resolution, (xmin, ymin))
        return cls(raw, resolution, (xmin, ymin), inflation_radius, border_is_obstacle)

    @classmethod
    def from_obstacles(cls, obstacles: Iterable[Obstacle], bounds: Bounds, resolution: float,
                       inflation_radius: float = 0.0,
                       border_is_obstacle: bool = True) -> "GridMap":
        return cls.from_scene(SceneDescription(bounds, list(obstacles)), resolution,
                              inflation_radius, border_is_obstacle)

    # ---------------------------------------------------------------- properties
    @property
    def shape(self) -> Tuple[int, int]:
        return self.raw.shape  # type: ignore[return-value]

    @property
    def rows(self) -> int:
        return self.raw.shape[0]

    @property
    def cols(self) -> int:
        return self.raw.shape[1]

    @property
    def bounds(self) -> Bounds:
        return (self.origin_x, self.origin_x + self.cols * self.resolution,
                self.origin_y, self.origin_y + self.rows * self.resolution)

    def occupancy(self, occupancy: Optional[np.ndarray] = None) -> np.ndarray:
        """Return ``occupancy`` if given, else the inflated grid (planning default)."""
        return self.inflated if occupancy is None else occupancy

    # ---------------------------------------------------------------- transforms
    def world_to_grid(self, x: float, y: float) -> Cell:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """World coordinates of the cell centre."""
        return (self.origin_x + (col + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)

    # ---------------------------------------------------------------- queries
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def contains_world(self, x: float, y: float) -> bool:
        return self.in_bounds(self.world_to_grid(x, y))

    def is_free(self, cell: Cell, occupancy: Optional[np.ndarray] = None) -> bool:
        return self.in_bounds(cell) and not bool(self.occupancy(occupancy)[cell])

    def is_free_world(self, x: float, y: float, occupancy: Optional[np.ndarray] = None) -> bool:
        return self.is_free(self.world_to_grid(x, y), occupancy)

    def nearest_free(self, cell: Cell, max_radius_cells: Optional[float] = None,
                     occupancy: Optional[np.ndarray] = None) -> Optional[Cell]:
        """Closest free cell (BFS, 8-connected) to ``cell``; ``None`` if none in range."""
        occ = self.occupancy(occupancy)
        r0, c0 = cell
        # Clamp the seed into the grid so out-of-bounds queries still work.
        start = (min(max(r0, 0), self.rows - 1), min(max(c0, 0), self.cols - 1))
        lim = None if max_radius_cells is None else max_radius_cells ** 2 + 1e-9
        seen = np.zeros_like(occ, dtype=bool)
        seen[start] = True
        queue = deque([start])
        best: Optional[Cell] = None
        best_d2 = math.inf
        while queue:
            cur = queue.popleft()
            d2 = (cur[0] - r0) ** 2 + (cur[1] - c0) ** 2
            if lim is not None and d2 > lim:
                continue
            if not occ[cur]:
                if d2 < best_d2:
                    best, best_d2 = cur, d2
                continue  # don't expand beyond the first free ring through free cells
            if best is not None and d2 > best_d2:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (cur[0] + dr, cur[1] + dc)
                    if self.in_bounds(nb) and not seen[nb]:
                        seen[nb] = True
                        queue.append(nb)
        return best

    def line_of_sight(self, a: Cell, b: Cell, occupancy: Optional[np.ndarray] = None) -> bool:
        """True if the straight segment between cell centres ``a`` and ``b`` is free.

        Bresenham traversal; on diagonal steps both side cells must be free too, so the
        segment can never squeeze between two diagonally touching obstacle cells.
        """
        occ = self.occupancy(occupancy)

        def free(r: int, c: int) -> bool:
            return 0 <= r < self.rows and 0 <= c < self.cols and not occ[r, c]

        (r, c), (r1, c1) = a, b
        if not free(r, c) or not free(r1, c1):
            return False
        dr, dc = abs(r1 - r), abs(c1 - c)
        sr = 1 if r1 > r else -1
        sc = 1 if c1 > c else -1
        err = dc - dr
        while (r, c) != (r1, c1):
            e2 = 2 * err
            step_c = e2 > -dr
            step_r = e2 < dc
            if step_c and step_r and (not free(r, c + sc) or not free(r + sr, c)):
                return False
            if step_c:
                err -= dr
                c += sc
            if step_r:
                err += dc
                r += sr
            if not free(r, c):
                return False
        return True

    # ---------------------------------------------------------------- dynamic obstacles
    def with_temporary_obstacles(self, points: Sequence[Tuple[float, float]],
                                 radius: float) -> "GridMap":
        """Copy of this map whose *inflated* layer also blocks disks around ``points``.

        Used for replanning around other robots (``radius`` should already include the
        robot's own size, e.g. 2 * footprint radius).
        """
        new = copy.copy(self)
        new.raw = self.raw  # raw static map is shared (read-only by convention)
        extra = np.zeros_like(self.inflated)
        offsets = disk_offsets(radius / self.resolution)
        for x, y in points:
            r0, c0 = self.world_to_grid(x, y)
            for dr, dc in offsets:
                cell = (r0 + dr, c0 + dc)
                if self.in_bounds(cell):
                    extra[cell] = True
        new.inflated = self.inflated | extra
        return new

    def __repr__(self) -> str:
        return (f"GridMap(shape={self.shape}, resolution={self.resolution}, "
                f"bounds={self.bounds}, inflation_radius={self.inflation_radius})")
