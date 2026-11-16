# CoppeliaSim Warehouse Scene Setup Guide (Step 1)

This guide walks through creating `scenes/warehouse.ttt` in the **CoppeliaSim GUI** so that Python's `scene_extractor.py` and fleet scripts can read the environment and control the Pioneer P3-DX fleet over the ZeroMQ Remote API.

---

## 1. Create a New Scene & Resize the Floor

1. Open **CoppeliaSim** (version ≥ 4.6 recommended; 4.3–4.5 also supported).
2. Start with a new empty scene (`File → New scene`).
3. Resize the floor to **10 m × 10 m**:
   - In the **Scene Hierarchy** (left panel), select `ResizableFloor_5_25` (or `Floor`).
   - Click the **User Parameters** / script parameters icon (or in CoppeliaSim ≥ 4.6 double-click the customization config icon next to `Floor`) and set both X and Y size to **10 m** (or scale range to create a `[-5, +5] × [-5, +5]` m floor centered at the origin).
   - *(Note: Even if you leave the visual floor at 5×5 m, Python defaults to `[-5, 5] × [-5, 5]` m unless it detects a larger floor object).*

---

## 2. Add the Pioneer P3-DX Robot (`Robot_0`)

You only need to place **one** Pioneer P3-DX in the GUI. When you run a multi-robot fleet (`--num-robots N`), Python automatically clones `Robot_0` into `Robot_1 … Robot_{N-1}` and places each robot at its configured start pose.

1. In the **Model browser** (left sidebar), navigate to:
   `robots → mobile → pioneer p3dx.ttm`
2. Drag and drop **Pioneer P3-DX** onto the floor.
3. **Rename the robot to `Robot_0`:**
   - Double-click its alias (`Pioneer_p3dx` or `PioneerP3DX`) in the Scene Hierarchy and rename it to **`Robot_0`**.
4. **CRITICAL — Remove or disable the built-in child script:**
   - The stock Pioneer P3-DX model includes a demo Braitenberg obstacle-avoidance Lua script that constantly sets wheel velocities and will fight our Python controller.
   - In the Scene Hierarchy, expand `Robot_0` (or look next to `Robot_0`), click the small **script icon** (or `script` child object in CoppeliaSim ≥ 4.6), and press **Delete** (or right-click → `Remove → Associated child script`).
   - *(Note: Python also attempts to disable any leftover child scripts on `Robot_*` automatically at startup, but deleting it in the scene before saving is cleanest).*
5. Move `Robot_0` to the default dock position:
   - Double-click `Robot_0` to open **Scene Object Properties → Position** (or use the Position dialog) and set World position `X = -4.0`, `Y = -4.0`, `Z = 0.138` (keep default Z so wheels rest on the floor).

---

## 3. Build Warehouse Walls & Shelves (Primitive Cuboids)

Python automatically scans the scene for shapes whose alias starts with **`Wall_`** or **`Shelf_`** and rasterizes their footprints into the 2D occupancy grid.

### Rules for every obstacle:
1. Create via **`Add → Primitive shape → Cuboid`**.
2. In the **Primitive shape** dialog, enter `X-size`, `Y-size`, `Z-size` (values in the table below).
3. In **Scene Object Properties → Dynamic properties** (if not already set):
   - Uncheck **Body is dynamic** (so the wall/shelf is **static** and won't get pushed around).
   - Keep **Body is respondable** checked (so robots physically collide with it).
4. Rename the object in the Scene Hierarchy to match **`Wall_*`** or **`Shelf_*`**.

### Reference Warehouse Table (matches `config/reference_layout.json` & `config/fleet_config.json`)

| Alias | Size `(size_x, size_y, size_z)` [m] | World Position `(X, Y, Z)` [m] | Description |
| :--- | :--- | :--- | :--- |
| `Wall_North` | `10.0, 0.1, 0.5` | `0.00,  4.95, 0.25` | North perimeter wall |
| `Wall_South` | `10.0, 0.1, 0.5` | `0.00, -4.95, 0.25` | South perimeter wall |
| `Wall_East` | `0.1, 10.0, 0.5` | `4.95,  0.00, 0.25` | East perimeter wall |
| `Wall_West` | `0.1, 10.0, 0.5` | `-4.95, 0.00, 0.25` | West perimeter wall |
| `Shelf_1` | `3.0, 0.6, 1.0` | `-2.25, -1.50, 0.50` | Row 1 — West shelf |
| `Shelf_2` | `3.0, 0.6, 1.0` | ` 2.25, -1.50, 0.50` | Row 1 — East shelf |
| `Shelf_3` | `3.0, 0.6, 1.0` | `-2.25,  0.50, 0.50` | Row 2 — West shelf |
| `Shelf_4` | `3.0, 0.6, 1.0` | ` 2.25,  0.50, 0.50` | Row 2 — East shelf |
| `Shelf_5` | `3.0, 0.6, 1.0` | `-2.25,  2.50, 0.50` | Row 3 — West shelf |
| `Shelf_6` | `3.0, 0.6, 1.0` | ` 2.25,  2.50, 0.50` | Row 3 — East shelf |

> **Tip:** You can also design your own custom shelf arrangement! Because `scene_extractor.py` reads the poses and bounding boxes directly from CoppeliaSim at runtime, the occupancy grid will automatically adapt to whatever `Wall_*` and `Shelf_*` cuboids you place. Just ensure aisles are at least **1.2 m** wide so two inflated Pioneer footprints can pass.

---

## 4. Save & Verify the Scene

1. Save the scene as **`scenes/warehouse.ttt`** (`File → Save scene as...`).
2. Keep CoppeliaSim open (the ZeroMQ Remote API server runs automatically on port `23000`).
3. Verify the connection and motors:
   ```bash
   python scripts/test_connection.py
   ```
4. Verify obstacle extraction and inspect the generated occupancy grid:
   ```bash
   python scripts/inspect_scene.py
   ```
   A matplotlib window will open showing the dark obstacle footprints, grey C-space inflation zone, free aisles, and `Robot_0`'s start pose.
