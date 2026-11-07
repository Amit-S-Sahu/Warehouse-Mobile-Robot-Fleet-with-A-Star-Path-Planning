"""Extract warehouse floor bounds and box obstacles from a live CoppeliaSim scene.

Finds all shape objects whose alias starts with ``Wall_`` or ``Shelf_`` (configured in
``config.OBSTACLE_PREFIXES``), reads their world-frame pose and bounding box, and returns
a :class:`~warehouse_fleet.scene_model.SceneDescription`. Also warns about any suspicious
unnamed cuboids in the scene that might have been added by hand without renaming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .config import (FLOOR_ALIAS, GRID, OBSTACLE_PREFIXES, REFERENCE_LAYOUT_PATH,
                     ROBOT_PREFIX, SCENE_CACHE_PATH)
from .scene_model import Bounds, Obstacle, SceneDescription
from .sim_interface import ZMQSimInterface

log = logging.getLogger(__name__)


@dataclass
class ExtractionReport:
    """Result of scanning the CoppeliaSim scene."""

    scene: SceneDescription
    robot_names: List[str] = field(default_factory=list)
    unrecognized_shapes: List[str] = field(default_factory=list)


def extract_scene(sim_iface: ZMQSimInterface, *,
                  obstacle_prefixes: Sequence[str] = OBSTACLE_PREFIXES,
                  floor_alias: str = FLOOR_ALIAS,
                  default_bounds: Bounds = GRID.default_floor_bounds,
                  cache_path: Optional[Union[str, Path]] = SCENE_CACHE_PATH) -> ExtractionReport:
    """Read obstacles and floor bounds from a connected :class:`ZMQSimInterface`."""
    sim_iface._ensure_connected()
    sim = sim_iface.sim

    shapes = sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape, 0)
    obstacles: List[Obstacle] = []
    unrecognized: List[str] = []
    floor_bounds: Optional[Bounds] = None

    # Identify handles that belong to any Robot_* model tree so we don't flag robot body parts
    robot_handles = set()
    for r_name in sim_iface.list_scene_robots(ROBOT_PREFIX):
        try:
            r_body = sim_iface.get_robot(r_name).body
            robot_handles.update(sim.getObjectsInTree(r_body, sim.handle_all, 0))
        except Exception:
            pass

    # Also skip children of floor objects (e.g. Floor/element)
    floor_handles = set()
    for h in shapes:
        alias = sim.getObjectAlias(h, 0)
        if alias == floor_alias or alias.lower().startswith("resizablefloor"):
            floor_handles.update(sim.getObjectsInTree(h, sim.handle_all, 0))
            if floor_bounds is None:
                cx, cy, sx, sy, _, _ = sim_iface.get_shape_footprint(h)
                if sx >= 2.0 and sy >= 2.0:
                    floor_bounds = (cx - 0.5 * sx, cx + 0.5 * sx, cy - 0.5 * sy, cy + 0.5 * sy)

    for h in shapes:
        if h in robot_handles:
            continue
        alias = sim.getObjectAlias(h, 0)
        if any(alias.startswith(pfx) for pfx in obstacle_prefixes):
            cx, cy, sx, sy, yaw, sz = sim_iface.get_shape_footprint(h)
            obstacles.append(Obstacle(name=alias, x=cx, y=cy, size_x=sx, size_y=sy,
                                      yaw=yaw, height=sz))
        elif h not in floor_handles and alias.lower().startswith("cuboid"):
            unrecognized.append(alias)

    bounds = floor_bounds if floor_bounds is not None else default_bounds
    if not obstacles:
        log.warning("No obstacles with prefixes %s found in the CoppeliaSim scene!", obstacle_prefixes)
    if unrecognized:
        log.warning("Found %d shape(s) named 'Cuboid*' that were ignored: %s. "
                    "Rename them to 'Wall_*' or 'Shelf_*' if they are obstacles.",
                    len(unrecognized), ", ".join(unrecognized))

    scene = SceneDescription(bounds=bounds, obstacles=obstacles)
    if cache_path is not None and obstacles:
        try:
            scene.save(cache_path)
            log.info("Cached extracted scene (%d obstacles) to %s", len(obstacles), cache_path)
        except Exception as exc:
            log.debug("Could not write scene cache: %s", exc)

    return ExtractionReport(
        scene=scene,
        robot_names=sim_iface.list_scene_robots(ROBOT_PREFIX),
        unrecognized_shapes=unrecognized,
    )


def load_or_extract_scene(sim_iface: Optional[ZMQSimInterface] = None, *,
                          scene_file: Optional[Union[str, Path]] = None) -> SceneDescription:
    """Load scene from ``scene_file`` if given, else extract from live ``sim_iface``,
    falling back to ``config/scene_cache.json`` or ``config/reference_layout.json``."""
    if scene_file is not None:
        return SceneDescription.load(scene_file)
    if sim_iface is not None and isinstance(sim_iface, ZMQSimInterface):
        report = extract_scene(sim_iface)
        if report.scene.obstacles:
            return report.scene
        log.warning("Live scene had 0 obstacles; falling back to reference layout %s",
                    REFERENCE_LAYOUT_PATH)
    if SCENE_CACHE_PATH.is_file():
        return SceneDescription.load(SCENE_CACHE_PATH)
    return SceneDescription.load(REFERENCE_LAYOUT_PATH)
