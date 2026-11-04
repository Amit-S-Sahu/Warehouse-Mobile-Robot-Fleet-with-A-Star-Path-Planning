"""User-selectable fleet configuration: number of robots, start locations, stations, tasks.

Sources (later overrides earlier):
1. ``config/fleet_config.json`` (or ``--config``)
2. CLI flags: ``--num-robots``, ``--start NAME=X,Y[,YAW]``, ``--goal NAME=X,Y|STATION``

Validation against the occupancy grid (:func:`resolve_start_poses`,
:func:`resolve_stations`, :func:`resolve_goals`) happens once the grid is known, so that
no robot is ever started inside a shelf or too close to another robot.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .config import DEFAULT_FLEET_CONFIG_PATH, FLEET, GRID, ROBOT_PREFIX
from .geometry import Pose2D  # re-exported for convenience
from .grid_map import GridMap

log = logging.getLogger(__name__)

Point = Tuple[float, float]
MAX_ROBOTS = 20
GOAL_MODES = ("tasks", "goals")


class ConfigError(ValueError):
    """Raised for invalid fleet configuration (message is user-facing)."""


# ======================================================================================
# Data classes
# ======================================================================================
@dataclass
class RobotSpec:
    name: str
    start: Optional[Pose2D] = None  # None -> keep scene pose or auto-place


@dataclass(frozen=True)
class StartArea:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    spacing: float = 1.0
    yaw: float = 0.0

    def candidates(self) -> List[Point]:
        """Lattice of candidate positions, row by row (y outer, x inner)."""
        (x0, x1), (y0, y1) = sorted(self.x_range), sorted(self.y_range)
        nx = int(math.floor((x1 - x0) / self.spacing + 1e-9)) + 1
        ny = int(math.floor((y1 - y0) / self.spacing + 1e-9)) + 1
        return [(x0 + i * self.spacing, y0 + j * self.spacing)
                for j in range(ny) for i in range(nx)]


@dataclass(frozen=True)
class Task:
    id: str
    pick: str  # station name
    drop: str  # station name


@dataclass
class FleetConfig:
    num_robots: int
    robots: List[RobotSpec]
    stations: Dict[str, Point] = field(default_factory=dict)
    tasks: List[Task] = field(default_factory=list)
    goals_mode: str = "tasks"
    goals: Dict[str, Point] = field(default_factory=dict)  # robot name -> goal (goals mode)
    auto_start_area: Optional[StartArea] = None
    return_to_start: bool = True
    dwell_time: float = FLEET.dwell_time

    @property
    def robot_names(self) -> List[str]:
        return [r.name for r in self.robots]

    def robot(self, name: str) -> RobotSpec:
        for r in self.robots:
            if r.name == name:
                return r
        raise ConfigError(f"Unknown robot '{name}'. Valid names: {', '.join(self.robot_names)}")


# ======================================================================================
# Parsing helpers
# ======================================================================================
def _point(value: Any, what: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{what} must be [x, y], got {value!r}")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{what} must contain numbers, got {value!r}") from exc


def _pose(value: Any, what: str) -> Pose2D:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ConfigError(f"{what} must be [x, y] or [x, y, yaw], got {value!r}")
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{what} must contain numbers, got {value!r}") from exc
    return Pose2D(nums[0], nums[1], nums[2] if len(nums) == 3 else 0.0)


def _range(value: Any, what: str) -> Tuple[float, float]:
    lo, hi = _point(value, what)
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def default_robot_name(index: int) -> str:
    return f"{ROBOT_PREFIX}{index}"


def _normalize_robots(num_robots: Optional[int], robots: List[RobotSpec]) -> Tuple[int, List[RobotSpec]]:
    """Make the robot list exactly ``num_robots`` long with unique names."""
    names = [r.name for r in robots]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ConfigError(f"Duplicate robot names: {', '.join(sorted(dupes))}")

    n = len(robots) if num_robots is None else int(num_robots)
    if n == 0 and num_robots is None:
        n = 1
    if not 1 <= n <= MAX_ROBOTS:
        raise ConfigError(f"num_robots must be between 1 and {MAX_ROBOTS}, got {n}")

    if n < len(robots):
        log.info("num_robots=%d: using the first %d of %d listed robots", n, n, len(robots))
        return n, [copy.copy(r) for r in robots[:n]]

    out = [copy.copy(r) for r in robots]
    used = set(names)
    idx = 0
    while len(out) < n:
        name = default_robot_name(idx)
        idx += 1
        if name not in used:
            out.append(RobotSpec(name=name, start=None))
            used.add(name)
    return n, out


def _resolve_target(value: Any, stations: Mapping[str, Point], what: str) -> Point:
    """A goal may be a station name or [x, y]."""
    if isinstance(value, str):
        if value not in stations:
            raise ConfigError(f"{what}: unknown station '{value}'. "
                              f"Known stations: {', '.join(sorted(stations)) or '(none)'}")
        return stations[value]
    return _point(value, what)


# ======================================================================================
# Loading
# ======================================================================================
def fleet_config_from_dict(d: Mapping[str, Any]) -> FleetConfig:
    if not isinstance(d, Mapping):
        raise ConfigError("Fleet config root must be a JSON object")

    robots: List[RobotSpec] = []
    for i, entry in enumerate(d.get("robots", []) or []):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"robots[{i}] must be an object")
        name = str(entry.get("name") or default_robot_name(i))
        start = entry.get("start")
        robots.append(RobotSpec(name=name,
                                start=None if start is None else _pose(start, f"robots[{i}].start")))

    num_robots, robots = _normalize_robots(d.get("num_robots"), robots)

    stations = {str(k): _point(v, f"stations.{k}")
                for k, v in (d.get("stations") or {}).items() if not str(k).startswith("_")}

    tasks: List[Task] = []
    for i, t in enumerate(d.get("tasks", []) or []):
        try:
            task = Task(id=str(t.get("id", f"T{i + 1}")), pick=str(t["pick"]), drop=str(t["drop"]))
        except (KeyError, AttributeError) as exc:
            raise ConfigError(f"tasks[{i}] needs 'pick' and 'drop' station names") from exc
        for role, st in (("pick", task.pick), ("drop", task.drop)):
            if st not in stations:
                raise ConfigError(f"Task '{task.id}' {role} station '{st}' is not defined in 'stations'")
        tasks.append(task)

    goals_mode = str(d.get("goals_mode", "tasks"))
    if goals_mode not in GOAL_MODES:
        raise ConfigError(f"goals_mode must be one of {GOAL_MODES}, got '{goals_mode}'")

    goals: Dict[str, Point] = {}
    valid_names = {r.name for r in robots}
    for name, target in (d.get("goals") or {}).items():
        if str(name).startswith("_"):
            continue  # comment keys
        if name not in valid_names:
            log.info("Ignoring goal for '%s' (not among the %d active robots)", name, num_robots)
            continue
        goals[str(name)] = _resolve_target(target, stations, f"goals.{name}")

    area = None
    a = d.get("auto_start_area")
    if a:
        spacing = float(a.get("spacing", 1.0))
        if spacing <= 0:
            raise ConfigError("auto_start_area.spacing must be > 0")
        area = StartArea(x_range=_range(a.get("x"), "auto_start_area.x"),
                         y_range=_range(a.get("y"), "auto_start_area.y"),
                         spacing=spacing, yaw=float(a.get("yaw", 0.0)))

    dwell = float(d.get("dwell_time", FLEET.dwell_time))
    if dwell < 0:
        raise ConfigError("dwell_time must be >= 0")

    return FleetConfig(num_robots=num_robots, robots=robots, stations=stations, tasks=tasks,
                       goals_mode=goals_mode, goals=goals, auto_start_area=area,
                       return_to_start=bool(d.get("return_to_start", True)), dwell_time=dwell)


def load_fleet_config(path: Union[str, Path] = DEFAULT_FLEET_CONFIG_PATH) -> FleetConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Fleet config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    return fleet_config_from_dict(data)


# ======================================================================================
# CLI overrides
# ======================================================================================
def parse_start_arg(text: str) -> Tuple[str, Pose2D]:
    """Parse ``NAME=X,Y[,YAW]`` (yaw in radians)."""
    name, sep, rest = text.partition("=")
    if not sep or not name.strip():
        raise ConfigError(f"--start expects NAME=X,Y[,YAW], got '{text}'")
    parts = [p for p in rest.split(",") if p.strip()]
    return name.strip(), _pose(parts, f"--start {name.strip()}")


def parse_goal_arg(text: str) -> Tuple[str, Union[str, Point]]:
    """Parse ``NAME=X,Y`` or ``NAME=STATION``."""
    name, sep, rest = text.partition("=")
    if not sep or not name.strip() or not rest.strip():
        raise ConfigError(f"--goal expects NAME=X,Y or NAME=STATION, got '{text}'")
    rest = rest.strip()
    if "," in rest:
        return name.strip(), _point([p for p in rest.split(",") if p.strip()], f"--goal {name.strip()}")
    return name.strip(), rest


def apply_overrides(cfg: FleetConfig, *, num_robots: Optional[int] = None,
                    starts: Iterable[str] = (), goals: Iterable[str] = ()) -> FleetConfig:
    """Return a new config with CLI-style overrides applied."""
    new = copy.deepcopy(cfg)
    if num_robots is not None:
        new.num_robots, new.robots = _normalize_robots(num_robots, new.robots)
        active = set(new.robot_names)
        new.goals = {k: v for k, v in new.goals.items() if k in active}

    for text in starts:
        name, pose = parse_start_arg(text)
        new.robot(name).start = pose  # raises ConfigError for unknown names

    goal_list = list(goals)
    if goal_list:
        new.goals_mode = "goals"
        for text in goal_list:
            name, target = parse_goal_arg(text)
            new.robot(name)  # validate name
            new.goals[name] = _resolve_target(target, new.stations, f"--goal {name}")
    return new


def add_fleet_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("fleet configuration")
    g.add_argument("--config", type=Path, default=DEFAULT_FLEET_CONFIG_PATH,
                   help="fleet config JSON (default: config/fleet_config.json)")
    g.add_argument("--num-robots", type=int, default=None,
                   help="number of robots to use (overrides the config file)")
    g.add_argument("--start", action="append", default=[], metavar="NAME=X,Y[,YAW]",
                   help="start pose for a robot, yaw in radians (repeatable)")
    g.add_argument("--goal", action="append", default=[], metavar="NAME=X,Y|STATION",
                   help="direct goal for a robot; switches to 'goals' mode (repeatable)")
    return parser


def fleet_config_from_args(args: argparse.Namespace) -> FleetConfig:
    cfg = load_fleet_config(args.config)
    return apply_overrides(cfg, num_robots=args.num_robots, starts=args.start, goals=args.goal)


# ======================================================================================
# Validation against the occupancy grid
# ======================================================================================
def _check_point_free(grid: GridMap, xy: Point, what: str) -> Optional[str]:
    x, y = xy
    if not grid.contains_world(x, y):
        xmin, xmax, ymin, ymax = grid.bounds
        return (f"{what} ({x:.2f}, {y:.2f}) is outside the floor "
                f"[{xmin:.2f}..{xmax:.2f}] x [{ymin:.2f}..{ymax:.2f}]")
    if not grid.is_free_world(x, y):
        return (f"{what} ({x:.2f}, {y:.2f}) is inside or too close to an obstacle "
                f"(needs {grid.inflation_radius:.2f} m clearance)")
    return None


def resolve_start_poses(cfg: FleetConfig, grid: GridMap, *,
                        existing_poses: Optional[Mapping[str, Pose2D]] = None,
                        min_spacing: float = FLEET.min_robot_spacing) -> Dict[str, Pose2D]:
    """Final start pose for every active robot, in ``cfg.robots`` order.

    Priority per robot: explicit start (config/CLI) > pose already in the scene
    (``existing_poses``) > auto-placement in ``cfg.auto_start_area``.
    Raises :class:`ConfigError` listing *all* problems found.
    """
    existing_poses = existing_poses or {}
    chosen: Dict[str, Pose2D] = {}
    errors: List[str] = []

    def too_close(p: Point) -> Optional[str]:
        for other, q in chosen.items():
            if math.hypot(p[0] - q.x, p[1] - q.y) < min_spacing - 1e-9:
                return other
        return None

    pending: List[str] = []
    for spec in cfg.robots:
        pose = spec.start or existing_poses.get(spec.name)
        if pose is None:
            pending.append(spec.name)
            continue
        source = "start" if spec.start is not None else "scene pose"
        err = _check_point_free(grid, pose.xy(), f"{spec.name} {source}")
        if err:
            errors.append(err)
        other = too_close(pose.xy())
        if other:
            errors.append(f"{spec.name} {source} ({pose.x:.2f}, {pose.y:.2f}) is closer than "
                          f"{min_spacing:.2f} m to {other}")
        chosen[spec.name] = pose
    if errors:
        raise ConfigError("Invalid start poses:\n  - " + "\n  - ".join(errors))

    if pending:
        area = cfg.auto_start_area
        if area is None:
            raise ConfigError(f"No start pose for {', '.join(pending)} and no "
                              f"'auto_start_area' defined in the fleet config")
        candidates = area.candidates()
        for name in pending:
            for p in candidates:
                if grid.is_free_world(*p) and too_close(p) is None:
                    chosen[name] = Pose2D(p[0], p[1], area.yaw)
                    log.info("Auto-placed %s at (%.2f, %.2f)", name, p[0], p[1])
                    break
            else:
                raise ConfigError(f"auto_start_area has no free spot left for {name}; enlarge "
                                  f"the area, reduce 'spacing' or lower num_robots")

    return {spec.name: chosen[spec.name] for spec in cfg.robots}


def _resolve_point(grid: GridMap, xy: Point, what: str, max_snap: float) -> Point:
    err = _check_point_free(grid, xy, what)
    if err is None:
        return xy
    if not grid.contains_world(*xy):
        raise ConfigError(err)
    cell = grid.nearest_free(grid.world_to_grid(*xy), max_snap / grid.resolution)
    if cell is None:
        raise ConfigError(err + f" and no free cell within {max_snap:.2f} m")
    snapped = grid.grid_to_world(*cell)
    log.warning("%s moved from (%.2f, %.2f) to nearest free point (%.2f, %.2f)",
                what, xy[0], xy[1], snapped[0], snapped[1])
    return snapped


def resolve_stations(cfg: FleetConfig, grid: GridMap,
                     max_snap: float = GRID.max_snap_distance) -> Dict[str, Point]:
    """Validate stations; snap ones slightly inside the inflated zone to free space."""
    return {name: _resolve_point(grid, xy, f"Station {name}", max_snap)
            for name, xy in cfg.stations.items()}


def resolve_goals(cfg: FleetConfig, grid: GridMap,
                  max_snap: float = GRID.max_snap_distance) -> Dict[str, Point]:
    """Validate direct goals (goals mode)."""
    return {name: _resolve_point(grid, xy, f"Goal of {name}", max_snap)
            for name, xy in cfg.goals.items()}
