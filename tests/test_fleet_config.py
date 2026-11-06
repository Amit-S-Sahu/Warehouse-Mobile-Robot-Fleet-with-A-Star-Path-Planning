import math

import pytest

from warehouse_fleet.config import DEFAULT_FLEET_CONFIG_PATH, GRID, REFERENCE_LAYOUT_PATH, ROBOT
from warehouse_fleet.fleet_config import (ConfigError, Pose2D, StartArea, apply_overrides,
                                          fleet_config_from_dict, load_fleet_config,
                                          parse_goal_arg, parse_start_arg, resolve_goals,
                                          resolve_start_poses, resolve_stations)
from warehouse_fleet.grid_map import GridMap
from warehouse_fleet.scene_model import SceneDescription


@pytest.fixture(scope="module")
def grid():
    scene = SceneDescription.load(REFERENCE_LAYOUT_PATH)
    return GridMap.from_scene(scene, GRID.resolution, GRID.inflation_radius(ROBOT))


def base_dict(**overrides):
    d = {
        "num_robots": 2,
        "robots": [{"name": "Robot_0", "start": [-4.0, -4.0, 0.0]},
                   {"name": "Robot_1", "start": [-3.0, -4.0]}],
        "auto_start_area": {"x": [-4.2, -0.6], "y": [-4.3, -2.8], "spacing": 0.9},
        "stations": {"P1": [-2.25, -0.5], "D1": [3.5, -3.5]},
        "tasks": [{"id": "T1", "pick": "P1", "drop": "D1"}],
    }
    d.update(overrides)
    return d


# ------------------------------------------------------------------ loading
def test_default_config_file_loads():
    cfg = load_fleet_config(DEFAULT_FLEET_CONFIG_PATH)
    assert cfg.num_robots == len(cfg.robots) == 3
    assert cfg.robot_names == ["Robot_0", "Robot_1", "Robot_2"]
    assert all(t.pick in cfg.stations and t.drop in cfg.stations for t in cfg.tasks)
    assert "_comment_" not in cfg.goals


def test_missing_file():
    with pytest.raises(ConfigError):
        load_fleet_config("does/not/exist.json")


def test_num_robots_extends_list():
    cfg = fleet_config_from_dict(base_dict(num_robots=4))
    assert cfg.robot_names == ["Robot_0", "Robot_1", "Robot_2", "Robot_3"]
    assert cfg.robots[2].start is None


def test_num_robots_truncates_list():
    cfg = fleet_config_from_dict(base_dict(num_robots=1))
    assert cfg.robot_names == ["Robot_0"]


def test_num_robots_range():
    with pytest.raises(ConfigError):
        fleet_config_from_dict(base_dict(num_robots=0))


def test_duplicate_names_rejected():
    d = base_dict(robots=[{"name": "Robot_0"}, {"name": "Robot_0"}])
    with pytest.raises(ConfigError):
        fleet_config_from_dict(d)


def test_unknown_task_station_rejected():
    with pytest.raises(ConfigError):
        fleet_config_from_dict(base_dict(tasks=[{"pick": "P9", "drop": "D1"}]))


def test_yaw_defaults_to_zero():
    cfg = fleet_config_from_dict(base_dict())
    assert cfg.robots[1].start == Pose2D(-3.0, -4.0, 0.0)


def test_start_area_candidates():
    area = StartArea((0.0, 1.0), (0.0, 0.5), spacing=0.5)
    assert area.candidates() == [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]


# ------------------------------------------------------------------ CLI parsing
def test_parse_start_arg():
    assert parse_start_arg("Robot_1=1.0,-4.0,1.57") == ("Robot_1", Pose2D(1.0, -4.0, 1.57))
    assert parse_start_arg("Robot_1=1,2") == ("Robot_1", Pose2D(1.0, 2.0, 0.0))
    for bad in ("Robot_1", "=1,2", "Robot_1=1", "Robot_1=a,b"):
        with pytest.raises(ConfigError):
            parse_start_arg(bad)


def test_parse_goal_arg():
    assert parse_goal_arg("Robot_0=P3") == ("Robot_0", "P3")
    assert parse_goal_arg("Robot_0=1.5,2") == ("Robot_0", (1.5, 2.0))
    with pytest.raises(ConfigError):
        parse_goal_arg("Robot_0=")


def test_apply_overrides():
    cfg = fleet_config_from_dict(base_dict())
    new = apply_overrides(cfg, num_robots=3, starts=["Robot_2=-1.0,-4.0,0"],
                          goals=["Robot_0=P1", "Robot_2=0.0,3.8"])
    assert new.num_robots == 3
    assert new.robot("Robot_2").start == Pose2D(-1.0, -4.0, 0.0)
    assert new.goals_mode == "goals"
    assert new.goals == {"Robot_0": (-2.25, -0.5), "Robot_2": (0.0, 3.8)}
    assert cfg.num_robots == 2  # original untouched


def test_apply_overrides_unknown_robot():
    cfg = fleet_config_from_dict(base_dict())
    with pytest.raises(ConfigError):
        apply_overrides(cfg, starts=["Robot_7=0,0"])
    with pytest.raises(ConfigError):
        apply_overrides(cfg, goals=["Robot_0=NOPE"])


# ------------------------------------------------------------------ grid validation
def test_resolve_starts_explicit(grid):
    cfg = fleet_config_from_dict(base_dict())
    poses = resolve_start_poses(cfg, grid)
    assert list(poses) == ["Robot_0", "Robot_1"]
    assert poses["Robot_0"] == Pose2D(-4.0, -4.0, 0.0)


def test_resolve_starts_auto_placement(grid):
    cfg = fleet_config_from_dict(base_dict(num_robots=6))
    poses = resolve_start_poses(cfg, grid, min_spacing=0.8)
    assert len(poses) == 6
    pts = list(poses.values())
    for i in range(len(pts)):
        assert grid.is_free_world(pts[i].x, pts[i].y)
        for j in range(i + 1, len(pts)):
            assert math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y) >= 0.8 - 1e-9


def test_resolve_starts_prefers_scene_pose(grid):
    cfg = fleet_config_from_dict(base_dict(num_robots=3))
    poses = resolve_start_poses(cfg, grid, existing_poses={"Robot_2": Pose2D(0.0, -4.0, 1.0)})
    assert poses["Robot_2"] == Pose2D(0.0, -4.0, 1.0)


def test_start_inside_shelf_rejected(grid):
    cfg = apply_overrides(fleet_config_from_dict(base_dict()), starts=["Robot_1=-2.25,-1.5"])
    with pytest.raises(ConfigError, match="obstacle"):
        resolve_start_poses(cfg, grid)


def test_start_outside_floor_rejected(grid):
    cfg = apply_overrides(fleet_config_from_dict(base_dict()), starts=["Robot_1=7,0"])
    with pytest.raises(ConfigError, match="outside"):
        resolve_start_poses(cfg, grid)


def test_starts_too_close_rejected(grid):
    cfg = apply_overrides(fleet_config_from_dict(base_dict()), starts=["Robot_1=-3.6,-4.0"])
    with pytest.raises(ConfigError, match="closer"):
        resolve_start_poses(cfg, grid)


def test_auto_area_exhausted(grid):
    d = base_dict(num_robots=5, auto_start_area={"x": [-4.0, -4.0], "y": [-3.0, -3.0]})
    cfg = fleet_config_from_dict(d)
    with pytest.raises(ConfigError, match="no free spot"):
        resolve_start_poses(cfg, grid)


def test_resolve_stations_and_goals(grid):
    d = base_dict(stations={"P1": [-2.25, -0.5], "NEAR": [-2.25, -1.05]},
                  tasks=[], goals_mode="goals", goals={"Robot_0": "NEAR"})
    cfg = fleet_config_from_dict(d)
    st = resolve_stations(cfg, grid)
    assert st["P1"] == (-2.25, -0.5)
    assert st["NEAR"] != (-2.25, -1.05) and grid.is_free_world(*st["NEAR"])  # snapped
    goals = resolve_goals(cfg, grid)
    assert grid.is_free_world(*goals["Robot_0"])


def test_default_config_valid_on_reference_layout(grid):
    cfg = load_fleet_config(DEFAULT_FLEET_CONFIG_PATH)
    resolve_start_poses(cfg, grid)
    stations = resolve_stations(cfg, grid)
    assert stations == cfg.stations  # no snapping needed for the shipped config
