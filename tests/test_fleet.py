import pytest

from warehouse_fleet.config import GRID, REFERENCE_LAYOUT_PATH, ROBOT
from warehouse_fleet.fleet_config import fleet_config_from_dict
from warehouse_fleet.fleet_manager import FleetManager
from warehouse_fleet.grid_map import GridMap
from warehouse_fleet.scene_model import SceneDescription
from warehouse_fleet.sim_interface import MockSim


@pytest.fixture(scope="module")
def warehouse_grid():
    scene = SceneDescription.load(REFERENCE_LAYOUT_PATH)
    return GridMap.from_scene(scene, GRID.resolution, GRID.inflation_radius(ROBOT))


def test_fleet_clones_robots_and_completes_direct_goals(warehouse_grid):
    cfg = fleet_config_from_dict({
        "num_robots": 2,
        "robots": [
            {"name": "Robot_0", "start": [-4.0, -4.0, 0.0]},
            {"name": "Robot_1", "start": [-2.5, -4.0, 0.0]},
        ],
        "stations": {"D1": [3.5, -3.5], "D2": [1.5, -3.5]},
        "goals_mode": "goals",
        "goals": {"Robot_0": [-3.5, -2.5], "Robot_1": "D2"},
    })
    sim = MockSim(dt=0.05)  # starts with only Robot_0 -> FleetManager clones Robot_1
    manager = FleetManager(sim, warehouse_grid, cfg, use_space_time=True, dt=0.05)
    assert set(sim.list_scene_robots()) == {"Robot_0", "Robot_1"}

    metrics = manager.run(max_steps=800)
    assert metrics.all_done
    assert metrics.tasks_completed == 2
    assert metrics.min_inter_robot_distance > 0.4


def test_fleet_executes_pick_and_drop_tasks(warehouse_grid):
    cfg = fleet_config_from_dict({
        "num_robots": 2,
        "robots": [
            {"name": "Robot_0", "start": [-4.0, -4.0, 0.0]},
            {"name": "Robot_1", "start": [0.0, -4.0, 0.0]},
        ],
        "stations": {
            "P1": [-2.25, -0.5],
            "P2": [2.25, -0.5],
            "D1": [3.5, -3.5],
            "D2": [1.5, -3.5],
        },
        "tasks": [
            {"id": "T1", "pick": "P1", "drop": "D2"},
            {"id": "T2", "pick": "P2", "drop": "D1"},
        ],
        "goals_mode": "tasks",
        "return_to_start": False,
        "dwell_time": 0.1,
    })
    sim = MockSim(dt=0.05)
    manager = FleetManager(sim, warehouse_grid, cfg, use_space_time=True, dt=0.05)
    metrics = manager.run(max_steps=1200)
    assert metrics.all_done
    assert metrics.tasks_completed == 2
