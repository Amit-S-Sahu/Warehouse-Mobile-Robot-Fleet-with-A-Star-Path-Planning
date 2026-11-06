import math

import numpy as np
import pytest

from warehouse_fleet.config import REFERENCE_LAYOUT_PATH
from warehouse_fleet.grid_map import GridMap, disk_offsets, inflate
from warehouse_fleet.scene_model import Obstacle, SceneDescription

BOUNDS = (-5.0, 5.0, -5.0, 5.0)


def empty_map(res=0.1, inflation=0.0, border=False):
    return GridMap.from_obstacles([], BOUNDS, res, inflation, border)


# ------------------------------------------------------------------ transforms
def test_grid_shape_and_bounds():
    g = empty_map()
    assert g.shape == (100, 100)
    assert g.bounds == pytest.approx(BOUNDS)


def test_world_grid_round_trip():
    g = empty_map()
    for x, y in [(-4.95, -4.95), (0.0, 0.0), (1.23, -3.21), (4.99, 4.99)]:
        r, c = g.world_to_grid(x, y)
        cx, cy = g.grid_to_world(r, c)
        assert abs(cx - x) <= g.resolution / 2 + 1e-9
        assert abs(cy - y) <= g.resolution / 2 + 1e-9


def test_row_is_y_col_is_x():
    g = empty_map()
    r, c = g.world_to_grid(4.95, -4.95)  # far +x, far -y (cell centres, no float edge cases)
    assert c == 99 and r == 0


def test_out_of_bounds():
    g = empty_map()
    assert not g.contains_world(5.01, 0.0)
    assert not g.is_free_world(-6.0, 0.0)


# ------------------------------------------------------------------ rasterization
def test_axis_aligned_box_exact():
    g = GridMap.from_obstacles([Obstacle("Shelf_1", 0.0, 0.0, 1.0, 1.0)], BOUNDS, 0.1)
    assert g.raw.sum() == 100  # 10 x 10 cells
    assert g.raw[g.world_to_grid(0.0, 0.0)]
    assert not g.raw[g.world_to_grid(0.6, 0.0)]


def test_thin_wall_never_missed():
    g = GridMap.from_obstacles([Obstacle("Wall_A", 0.03, 0.0, 0.02, 4.0)], BOUNDS, 0.1)
    rows_with_wall = g.raw.any(axis=1).sum()
    assert rows_with_wall == 40
    assert (g.raw.sum(axis=1)[g.raw.any(axis=1)] >= 1).all()


def test_rotated_box():
    ob = Obstacle("Shelf_R", 0.0, 0.0, 2.0, 0.2, yaw=math.pi / 4)
    g = GridMap.from_obstacles([ob], BOUNDS, 0.1)
    assert g.raw[g.world_to_grid(0.0, 0.0)]
    assert g.raw[g.world_to_grid(0.5, 0.5)]  # on the diagonal
    assert not g.raw[g.world_to_grid(0.5, -0.5)]  # off the diagonal
    assert not g.raw[g.world_to_grid(1.0, 0.0)]


def test_obstacle_outside_grid_ignored():
    g = GridMap.from_obstacles([Obstacle("Shelf_X", 20.0, 20.0, 1.0, 1.0)], BOUNDS, 0.1)
    assert g.raw.sum() == 0


def test_obstacle_requires_positive_size():
    with pytest.raises(ValueError):
        Obstacle("bad", 0, 0, 0.0, 1.0)


# ------------------------------------------------------------------ inflation
def test_disk_offsets_count():
    assert len(disk_offsets(0)) == 1
    assert len(disk_offsets(1)) == 5
    assert len(disk_offsets(2)) == 13


def test_inflate_single_cell():
    occ = np.zeros((21, 21), dtype=bool)
    occ[10, 10] = True
    out = inflate(occ, 2.0, border_is_obstacle=False)
    assert out.sum() == 13
    assert out[10, 12] and out[12, 10] and not out[12, 12]


def test_inflate_border():
    occ = np.zeros((10, 10), dtype=bool)
    out = inflate(occ, 1.0, border_is_obstacle=True)
    assert out[0, :].all() and out[-1, :].all() and out[:, 0].all() and out[:, -1].all()
    assert not out[1:-1, 1:-1].any()


def test_inflated_superset_of_raw():
    scene = SceneDescription.load(REFERENCE_LAYOUT_PATH)
    g = GridMap.from_scene(scene, 0.1, 0.36)
    assert (g.inflated | ~g.raw).all()  # raw implies inflated
    assert g.inflated.sum() > g.raw.sum()


# ------------------------------------------------------------------ queries
def test_line_of_sight():
    g = GridMap.from_obstacles([Obstacle("Wall_M", 0.0, 0.0, 0.2, 4.0)], BOUNDS, 0.1)
    a = g.world_to_grid(-2.0, 0.0)
    b = g.world_to_grid(2.0, 0.0)
    assert not g.line_of_sight(a, b, g.raw)
    c = g.world_to_grid(-2.0, 3.0)
    d = g.world_to_grid(2.0, 3.0)
    assert g.line_of_sight(c, d, g.raw)


def test_line_of_sight_no_diagonal_squeeze():
    raw = np.zeros((5, 5), dtype=bool)
    raw[1, 2] = True
    raw[2, 1] = True
    g = GridMap(raw, 1.0, (0.0, 0.0), 0.0, False)
    assert not g.line_of_sight((1, 1), (2, 2))


def test_nearest_free():
    g = GridMap.from_obstacles([Obstacle("Shelf_1", 0.0, 0.0, 1.0, 1.0)], BOUNDS, 0.1)
    centre = g.world_to_grid(0.0, 0.0)
    nf = g.nearest_free(centre, occupancy=g.raw)
    assert nf is not None and not g.raw[nf]
    # nearest free cell is just outside the 1 m box: <= ~0.6 m away
    d = math.dist(g.grid_to_world(*nf), g.grid_to_world(*centre))
    assert d <= 0.6 + 1e-9
    assert g.nearest_free(centre, max_radius_cells=2, occupancy=g.raw) is None


def test_with_temporary_obstacles():
    g = empty_map()
    g2 = g.with_temporary_obstacles([(1.0, 1.0)], radius=0.3)
    assert g2.inflated[g.world_to_grid(1.0, 1.0)]
    assert not g.inflated[g.world_to_grid(1.0, 1.0)]  # original untouched
    assert not g2.inflated[g.world_to_grid(2.0, 2.0)]


# ------------------------------------------------------------------ scene io
def test_scene_round_trip(tmp_path):
    scene = SceneDescription(BOUNDS, [Obstacle("Shelf_1", 1.0, 2.0, 3.0, 0.6, 0.1, 1.0)])
    p = tmp_path / "scene.json"
    scene.save(p)
    loaded = SceneDescription.load(p)
    assert loaded.bounds == scene.bounds
    assert loaded.obstacles == scene.obstacles


def test_reference_layout_loads():
    scene = SceneDescription.load(REFERENCE_LAYOUT_PATH)
    names = {o.name for o in scene.obstacles}
    assert {"Wall_North", "Shelf_1", "Shelf_6"} <= names
