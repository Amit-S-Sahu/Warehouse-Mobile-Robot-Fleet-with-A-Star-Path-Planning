# Warehouse Mobile Robot Fleet with A\* Path Planning

A multi-robot warehouse navigation and task-execution system built for **CoppeliaSim** (Pioneer P3-DX) and **Python** using the **ZeroMQ Remote API**.

- **GUI Scene Extraction**: Build the warehouse in the CoppeliaSim GUI (`Wall_*`, `Shelf_*`, `Robot_0`); Python extracts the exact obstacle geometry over ZMQ and rasterizes a C-space inflated 2D occupancy grid.
- **A\* & Prioritized Space-Time A\***: 8-connected octile A\* with line-of-sight smoothing for single robots, plus Prioritized Space-Time A\* `(row, col, t)` with a reservation table for multi-robot fleets.
- **Selectable Robot Count & Locations**: Choose the number of robots (`--num-robots N`), custom start poses (`--start`), and either pick-and-drop tasks or direct goals (`--goal`) via `config/fleet_config.json` or CLI flags. Python automatically clones `Robot_0` up to `N` robots in the scene.
- **P/PID Differential-Drive Control & Runtime Safety**: Heading PID + `cos(alpha)`-scaled linear speed, rotate-in-place threshold, forward safety-cone yielding, and automatic deadlock replanning.
- **Offline Kinematic Mock Mode**: Every script and test can also run without CoppeliaSim (`--mock` / `--offline`).

> Full design & architecture: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
> Step-by-step CoppeliaSim GUI setup: [docs/SCENE_SETUP.md](docs/SCENE_SETUP.md)

---

## Status

| Phase | Content                                                                                              | State |
| ----- | ---------------------------------------------------------------------------------------------------- | ----- |
| P0    | Scaffolding, `config.py`, `fleet_config.py` (selectable robot count & locations)                     | ✅    |
| P1    | Occupancy grid, C-space inflation, A\*, LOS smoothing, visualizer, unit tests                        | ✅    |
| P2    | ZeroMQ `sim_interface.py` (4.3–4.5 & ≥ 4.6 compatible) + `MockSim` + `test_connection.py`            | ✅    |
| P3    | CoppeliaSim GUI guide (`docs/SCENE_SETUP.md`), `scene_extractor.py`, `inspect_scene.py`              | ✅    |
| P4    | Single-robot P/PID controller (`controller.py`) + `run_single_robot.py`                              | ✅    |
| P5    | Fleet setup (`fleet_setup.py`), state machine (`robot_agent.py`), `fleet_manager.py`, `run_fleet.py` | ✅    |
| P6    | Prioritized Space-Time A\* (`space_time_astar.py`) integrated with fleet manager                     | ✅    |
| P7    | Documentation, CLI reference, controller & planner tuning guide                                      | ✅    |

---

## 1. Installation

Requires **Python ≥ 3.9** and **CoppeliaSim ≥ 4.6** (4.3–4.5 also supported).

```bash
pip install -r requirements.txt
```

---

## 2. Step-by-Step Workflow

### Step 1 — Build the Warehouse Scene in CoppeliaSim GUI

Follow **[docs/SCENE_SETUP.md](docs/SCENE_SETUP.md)**:

1. Open CoppeliaSim, resize the floor to **10 m × 10 m**.
2. Drag-and-drop **one** Pioneer P3-DX, rename it **`Robot_0`**, and **delete its built-in Lua child script**.
3. Add cuboid walls and shelves (`Add → Primitive shape → Cuboid`), mark them **Static + Respondable**, and name them **`Wall_*`** or **`Shelf_*`** (table of coordinates in `docs/SCENE_SETUP.md`).
4. Save as `scenes/warehouse.ttt` and leave CoppeliaSim open (ZeroMQ Remote API listens on port `23000` by default).

Verify the ZMQ connection, motor handles, and wheel kinematics:

```bash
python scripts/test_connection.py
```

Verify obstacle extraction and inspect the 2D occupancy grid:

```bash
python scripts/inspect_scene.py --save output/scene_grid.png
```

---

### Step 2 — Offline A\* Path Planning Demo (No Simulator Needed)

```bash
python scripts/plan_offline.py
python scripts/plan_offline.py --num-robots 5 --save output/plan.png --no-show
```

---

### Step 3 — Single Robot Navigation (`run_single_robot.py`)

```bash
# Drive Robot_0 in CoppeliaSim to station P6
python scripts/run_single_robot.py --goal P6

# Drive from a custom start pose to arbitrary (x, y) coordinates
python scripts/run_single_robot.py --start -4.0,-4.0,0.0 --goal 2.25,3.8 --save output/single.png

# Run offline using the kinematic MockSim
python scripts/run_single_robot.py --mock --goal P4 --save output/single_mock.png --no-show
```

---

### Step 4 — Multi-Robot Fleet (`run_fleet.py`)

```bash
# Run default 3-robot fleet executing pick->drop tasks from config/fleet_config.json
python scripts/run_fleet.py

# Choose 4 robots (Robot_3 is auto-cloned from Robot_0 and placed in auto_start_area)
python scripts/run_fleet.py --num-robots 4

# Custom start poses (x, y [m], yaw [rad])
python scripts/run_fleet.py --num-robots 3 \
    --start Robot_0=-4.0,-4.0,0.0 \
    --start Robot_1=-2.5,-4.0,0.0 \
    --start Robot_2=3.5,-3.5,3.14

# Direct goals mode instead of pick/drop tasks
python scripts/run_fleet.py --num-robots 3 \
    --goal Robot_0=P6 --goal Robot_1=P4 --goal Robot_2=0.0,3.8

# Run offline in MockSim and save trajectory plot
python scripts/run_fleet.py --mock --num-robots 3 --save output/fleet_mock.png --no-show
```

---

## 3. Project Structure

```
Warehouse-Mobile-Robot-Fleet-with-A-Star-Path-Planning/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── fleet_config.json        # num_robots, robot start poses, stations, tasks, goals
│   └── reference_layout.json    # reference 10x10 m warehouse layout
├── docs/
│   ├── IMPLEMENTATION_PLAN.md   # architecture & design specification
│   └── SCENE_SETUP.md           # click-by-click CoppeliaSim GUI setup guide
├── src/warehouse_fleet/
│   ├── config.py                # robot geometry, grid params, PID gains, fleet settings
│   ├── geometry.py              # Pose2D, angle wrapping, 3x4 matrix helpers
│   ├── scene_model.py           # Obstacle & SceneDescription data models
│   ├── grid_map.py              # 2D occupancy grid, disk inflation, Bresenham LOS
│   ├── astar.py                 # 8-connected A*, LOS pruning, densification
│   ├── space_time_astar.py      # Prioritized Space-Time A* in (row, col, t)
│   ├── fleet_config.py          # JSON + CLI config loader & grid validation
│   ├── sim_interface.py         # CoppeliaSim ZMQ wrapper + offline MockSim
│   ├── scene_extractor.py       # live obstacle & floor extraction over ZMQ
│   ├── controller.py            # P/PID waypoint follower & diff-drive inverse kinematics
│   ├── fleet_setup.py           # automatic robot cloning & placement in scene
│   ├── robot_agent.py           # per-robot state machine & telemetry
│   ├── fleet_manager.py         # task allocation, coordination, safety cone & replanning
│   └── visualizer.py            # matplotlib grid, path & trajectory plotting
├── scripts/
│   ├── plan_offline.py          # offline A* path planning & plotting
│   ├── test_connection.py       # ZMQ connection & wheel calibration test
│   ├── inspect_scene.py         # extract CoppeliaSim scene & plot occupancy grid
│   ├── run_single_robot.py      # single Pioneer P3-DX A* + P/PID navigation
│   └── run_fleet.py             # multi-robot fleet coordinator
└── tests/
    ├── test_grid_map.py
    ├── test_astar.py
    ├── test_fleet_config.py
    ├── test_controller.py
    ├── test_space_time_astar.py
    └── test_fleet.py
```

---

## 4. Tuning Guide (`src/warehouse_fleet/config.py`)

- **Robot oscillates / fishtails along straight lines**:
  - Increase `kd_angular` (e.g. `0.15` → `0.25`) or lower `kp_angular` (e.g. `2.0` → `1.5`).
  - Increase `waypoint_tolerance` slightly (e.g. `0.15` → `0.20` m).
- **Robot clips shelf corners**:
  - Increase `GridParams.safety_margin` (e.g. `0.08` → `0.14` m) so the C-space inflation zone is wider.
  - Decrease `rotate_in_place_angle` (e.g. `0.785` rad / 45° → `0.52` rad / 30°) so the robot finishes turning before accelerating into an aisle.
- **Robots stop too close / too far from each other**:
  - Adjust `FleetParams.safety_distance` (default `0.70` m) and `yield_cone_deg` (default `60°`).
  - Adjust `FleetParams.deadlock_timeout` (default `5.0` s) to control how quickly a yielding robot replans around stationary peers.

---

## 5. Running Unit Tests

```bash
pytest -q
```
