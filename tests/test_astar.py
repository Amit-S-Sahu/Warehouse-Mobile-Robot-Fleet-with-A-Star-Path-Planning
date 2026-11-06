import heapq
import math

import numpy as np
import pytest

from warehouse_fleet.astar import (SQRT2, astar_search, densify, octile_distance, path_length,
                                   plan_path, prune_path)
from warehouse_fleet.config import GRID, REFERENCE_LAYOUT_PATH, ROBOT
from warehouse_fleet.grid_map import GridMap
from warehouse_fleet.scene_model import Obstacle, SceneDescription


def path_cost(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))


def assert_valid_path(occ, path, start, goal, diagonal=True):
    assert path[0] == start and path[-1] == goal
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        assert (dr, dc) in ({(1, 0), (0, 1), (1, 1)} if diagonal else {(1, 0), (0, 1)})
        assert not occ[r1, c1]
        if dr and dc:  # no corner cutting
            assert not occ[r0, c1] and not occ[r1, c0]


def dijkstra_cost(occ, start, goal):
    """Reference implementation with the same move rules."""
    rows, cols = occ.shape
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == goal:
            return d
        if d > dist[(r, c)]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or occ[nr, nc]:
                    continue
                if dr and dc and (occ[r, nc] or occ[nr, c]):
                    continue
                nd = d + (SQRT2 if dr and dc else 1.0)
                if nd < dist.get((nr, nc), math.inf):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return math.inf


# ------------------------------------------------------------------ heuristic
def test_octile():
    assert octile_distance((0, 0), (3, 3)) == pytest.approx(3 * SQRT2)
    assert octile_distance((0, 0), (0, 5)) == pytest.approx(5)
    assert octile_distance((0, 0), (2, 5)) == pytest.approx(3 + 2 * SQRT2)


# ------------------------------------------------------------------ search
def test_empty_grid_diagonal():
    occ = np.zeros((10, 10), dtype=bool)
    res = astar_search(occ, (0, 0), (9, 9))
    assert res.success
    assert res.cost == pytest.approx(9 * SQRT2)
    assert len(res.path) == 10
    assert_valid_path(occ, res.path, (0, 0), (9, 9))


def test_empty_grid_straight():
    occ = np.zeros((10, 10), dtype=bool)
    res = astar_search(occ, (0, 0), (0, 9))
    assert res.cost == pytest.approx(9.0)


def test_four_connected_cost():
    occ = np.zeros((10, 10), dtype=bool)
    res = astar_search(occ, (0, 0), (4, 6), diagonal=False)
    assert res.cost == pytest.approx(10.0)
    assert_valid_path(occ, res.path, (0, 0), (4, 6), diagonal=False)


def test_start_equals_goal():
    occ = np.zeros((5, 5), dtype=bool)
    res = astar_search(occ, (2, 2), (2, 2))
    assert res.path == [(2, 2)] and res.cost == 0.0


def test_routes_through_gap():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, :9] = True  # wall with a gap at column 9
    res = astar_search(occ, (0, 0), (9, 0))
    assert res.success
    assert (5, 9) in res.path
    assert_valid_path(occ, res.path, (0, 0), (9, 0))
    assert res.cost > 9.0


def test_fully_blocked_returns_none():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, :] = True
    res = astar_search(occ, (0, 0), (9, 9))
    assert not res.success and res.path is None and math.isinf(res.cost)


def test_occupied_start_or_goal():
    occ = np.zeros((5, 5), dtype=bool)
    occ[0, 0] = True
    assert not astar_search(occ, (0, 0), (4, 4)).success
    assert not astar_search(occ, (4, 4), (0, 0)).success


def test_out_of_bounds():
    occ = np.zeros((5, 5), dtype=bool)
    assert not astar_search(occ, (0, 0), (5, 5)).success


def test_no_corner_cutting():
    occ = np.zeros((3, 3), dtype=bool)
    occ[0, 1] = True
    res = astar_search(occ, (0, 0), (1, 1))
    assert res.path == [(0, 0), (1, 0), (1, 1)]
    assert res.cost == pytest.approx(2.0)

    occ[1, 0] = True  # now (0,0) is boxed in diagonally
    assert not astar_search(occ, (0, 0), (1, 1)).success


@pytest.mark.parametrize("seed", range(8))
def test_optimal_vs_dijkstra(seed):
    rng = np.random.default_rng(seed)
    occ = rng.random((25, 25)) < 0.25
    occ[0, 0] = occ[24, 24] = False
    res = astar_search(occ, (0, 0), (24, 24))
    ref = dijkstra_cost(occ, (0, 0), (24, 24))
    if math.isinf(ref):
        assert not res.success
    else:
        assert res.success
        assert res.cost == pytest.approx(ref)
        assert path_cost(res.path) == pytest.approx(ref)
        assert_valid_path(occ, res.path, (0, 0), (24, 24))


# ------------------------------------------------------------------ post-processing
def test_prune_straight_line():
    g = GridMap(np.zeros((10, 10), dtype=bool), 1.0, (0.0, 0.0), 0.0, False)
    cells = [(0, c) for c in range(10)]
    assert prune_path(g, cells) == [(0, 0), (0, 9)]


def test_prune_keeps_corner():
    raw = np.zeros((10, 10), dtype=bool)
    raw[0:8, 5] = True  # vertical wall, gap at the top rows 8..9
    g = GridMap(raw, 1.0, (0.0, 0.0), 0.0, False)
    res = astar_search(g.raw, (0, 0), (0, 9))
    pruned = prune_path(g, res.path)
    assert len(pruned) < len(res.path)
    for a, b in zip(pruned[:-1], pruned[1:]):
        assert g.line_of_sight(a, b)


def test_densify():
    pts = densify([(0.0, 0.0), (1.0, 0.0)], 0.3)
    assert pts[0] == (0.0, 0.0) and pts[-1] == pytest.approx((1.0, 0.0))
    assert len(pts) == 5  # ceil(1/0.3) = 4 segments
    assert path_length(pts) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        densify(pts, 0.0)


# ------------------------------------------------------------------ world wrapper
@pytest.fixture(scope="module")
def warehouse_grid():
    scene = SceneDescription.load(REFERENCE_LAYOUT_PATH)
    return GridMap.from_scene(scene, GRID.resolution, GRID.inflation_radius(ROBOT))


def test_plan_path_reference_warehouse(warehouse_grid):
    g = warehouse_grid
    start, goal = (-4.0, -4.0), (2.25, 3.8)  # dock -> top-right aisle (P6)
    res = plan_path(g, start, goal)
    assert res is not None
    assert res.waypoints[0] == start and res.waypoints[-1] == goal
    assert not res.start_snapped and not res.goal_snapped
    assert len(res.waypoints) < len(res.cells)
    # every segment of the smoothed path is collision free on the inflated grid
    cells = [g.world_to_grid(*p) for p in res.waypoints]
    for a, b in zip(cells[:-1], cells[1:]):
        assert g.line_of_sight(a, b)
    # smoothed path is never longer than the raw grid path
    assert res.length <= res.cost + 2 * g.resolution


def test_plan_path_snaps_goal_near_shelf(warehouse_grid):
    g = warehouse_grid
    goal = (-2.25, -1.05)  # 0.15 m from Shelf_1's face -> inside inflation zone
    assert not g.is_free_world(*goal)
    res = plan_path(g, (-4.0, -4.0), goal)
    assert res is not None and res.goal_snapped
    assert g.is_free_world(*res.waypoints[-1])


def test_plan_path_goal_inside_shelf_fails(warehouse_grid):
    assert plan_path(warehouse_grid, (-4.0, -4.0), (-2.25, -1.5), max_snap_distance=0.2) is None


def test_plan_path_outside_map(warehouse_grid):
    assert plan_path(warehouse_grid, (-4.0, -4.0), (8.0, 0.0)) is None


def test_plan_path_with_max_segment(warehouse_grid):
    res = plan_path(warehouse_grid, (-4.0, -4.0), (3.5, -3.5), max_segment=0.5)
    assert res is not None
    segs = [math.dist(a, b) for a, b in zip(res.waypoints[:-1], res.waypoints[1:])]
    assert max(segs) <= 0.5 + 1e-9


def test_plan_path_respects_temporary_obstacles():
    g = GridMap.from_obstacles([Obstacle("Wall_A", 0.0, -1.5, 0.2, 7.0)], (-5, 5, -5, 5), 0.1, 0.3)
    free = plan_path(g, (-2.0, 3.0), (2.0, 3.0))
    assert free is not None and len(free.waypoints) == 2  # straight over the wall's end
    blocked = g.with_temporary_obstacles([(0.0, 3.0)], radius=0.8)
    detour = plan_path(blocked, (-2.0, 3.0), (2.0, 3.0))
    assert detour is not None and detour.length > free.length
