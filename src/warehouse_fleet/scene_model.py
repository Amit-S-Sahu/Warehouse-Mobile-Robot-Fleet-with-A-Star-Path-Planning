"""Plain data description of the warehouse scene (floor bounds + box obstacles).

A :class:`SceneDescription` is produced either by reading the live CoppeliaSim scene
(``scene_extractor``, phase P3) or by loading a JSON file (reference layout / cache), so
grid building and planning work identically online and offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

Bounds = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) in metres, world frame
PathLike = Union[str, Path]


@dataclass(frozen=True)
class Obstacle:
    """A box obstacle, rotated only about the world Z axis (yaw)."""

    name: str
    x: float  # centre, world frame [m]
    y: float
    size_x: float  # full extent along the box's local x axis [m]
    size_y: float  # full extent along the box's local y axis [m]
    yaw: float = 0.0  # [rad]
    height: float = 0.0  # informational only [m]

    def __post_init__(self) -> None:
        if self.size_x <= 0 or self.size_y <= 0:
            raise ValueError(f"Obstacle '{self.name}' must have positive size, got "
                             f"({self.size_x}, {self.size_y})")

    def corners(self) -> List[Tuple[float, float]]:
        """World-frame corners (counter-clockwise)."""
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
        return [(self.x + c * dx - s * dy, self.y + s * dx + c * dy) for dx, dy in local]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Obstacle":
        try:
            return cls(
                name=str(d["name"]),
                x=float(d["x"]),
                y=float(d["y"]),
                size_x=float(d["size_x"]),
                size_y=float(d["size_y"]),
                yaw=float(d.get("yaw", 0.0)),
                height=float(d.get("height", 0.0)),
            )
        except KeyError as exc:
            raise ValueError(f"Obstacle entry {d!r} is missing key {exc}") from exc


@dataclass
class SceneDescription:
    """Floor bounds plus the list of obstacles."""

    bounds: Bounds
    obstacles: List[Obstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        xmin, xmax, ymin, ymax = (float(v) for v in self.bounds)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"Invalid bounds {self.bounds}: need xmin < xmax and ymin < ymax")
        self.bounds = (xmin, xmax, ymin, ymax)

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        xmin, xmax, ymin, ymax = self.bounds
        return {
            "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneDescription":
        b = d.get("bounds")
        if b is None:
            raise ValueError("Scene description is missing 'bounds'")
        if isinstance(b, dict):
            bounds = (b["xmin"], b["xmax"], b["ymin"], b["ymax"])
        else:
            bounds = tuple(b)  # type: ignore[assignment]
            if len(bounds) != 4:
                raise ValueError("'bounds' list must be [xmin, xmax, ymin, ymax]")
        obstacles = [Obstacle.from_dict(o) for o in d.get("obstacles", [])]
        return cls(bounds=bounds, obstacles=obstacles)  # type: ignore[arg-type]

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: PathLike) -> "SceneDescription":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
