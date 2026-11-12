import numpy as np

from warehouse_fleet.grid_map import GridMap
from warehouse_fleet.space_time_astar import (
    ReservationTable, plan_prioritized_space_time, space_time_astar)


def assert_no_vertex_or_edge_conflicts(path_a, path_b):
    T = max(len(path_a), len(path_b))
    # Extend shorter path by holding its goal cell
    a = list(path_a) + [path_a[-1]] * (T - len(path_a))
    b = list(path_b) + [path_b[-1]] * (T - len(path_b))
    for t in range(T):
        assert a[t] != b[t], f"Vertex conflict at t={t}: {a[t]}"
        if t + 1 < T:
            assert not (a[t] == b[t + 1] and a[t + 1] == b[t]), (
                f"Edge swap conflict between t={t} and t={t + 1}: {a[t]} <-> {b[t]}"
            )


def test_two_robots_crossing_Wait_avoids_collision():
    occ = np.zeros((7, 7), dtype=bool)
    table = ReservationTable()

    res_a = space_time_astar(occ, (3, 0), (3, 6), table)
    assert res_a.success
    table.reserve_path(res_a.timed_cells, start_t=0, hold_goal_steps=10)

    res_b = space_time_astar(occ, (0, 3), (6, 3), table)
    assert res_b.success
    assert_no_vertex_or_edge_conflicts(res_a.timed_cells, res_b.timed_cells)


def test_corridor_swap_uses_passing_bay():
    # 3x7 map: top & bottom rows are walls except a passing bay at (0, 3)
    occ = np.ones((3, 7), dtype=bool)
    occ[1, :] = False  # narrow single-cell corridor along row 1
    occ[0, 3] = False  # passing bay in the middle

    table = ReservationTable()
    # Robot A goes left -> right along the corridor
    res_a = space_time_astar(occ, (1, 0), (1, 6), table)
    assert res_a.success
    table.reserve_path(res_a.timed_cells, start_t=0, hold_goal_steps=10)

    # Robot B starts near the bay, steps into (0, 3) to let A pass, then goes to (1, 0)
    res_b = space_time_astar(occ, (1, 4), (1, 0), table)
    assert res_b.success
    assert (0, 3) in res_b.timed_cells
    assert_no_vertex_or_edge_conflicts(res_a.timed_cells, res_b.timed_cells)


def test_plan_prioritized_space_time_world_wrapper():
    g = GridMap(np.zeros((20, 20), dtype=bool), 0.25, (-2.5, -2.5), 0.0, False)
    requests = [
        ("Robot_0", (-2.0, 0.0), (2.0, 0.0)),
        ("Robot_1", (0.0, -2.0), (0.0, 2.0)),
    ]
    plans = plan_prioritized_space_time(g, requests, conflict_radius_m=0.25)
    assert plans["Robot_0"] is not None
    assert plans["Robot_1"] is not None
    assert plans["Robot_0"][0] == (-2.0, 0.0) and plans["Robot_0"][-1] == (2.0, 0.0)
