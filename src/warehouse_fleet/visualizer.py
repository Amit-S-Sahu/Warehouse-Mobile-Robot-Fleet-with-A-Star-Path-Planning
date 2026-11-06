"""Matplotlib plots of the occupancy grid, planned paths and recorded trajectories."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .grid_map import GridMap

Point = Tuple[float, float]

FREE_RGB = (1.0, 1.0, 1.0)
INFLATED_RGB = (0.82, 0.82, 0.82)
OBSTACLE_RGB = (0.15, 0.15, 0.15)


def grid_image(grid_map: GridMap) -> np.ndarray:
    """RGB image (rows, cols, 3): obstacles dark, inflation zone grey, free white."""
    img = np.empty(grid_map.shape + (3,), dtype=float)
    img[:] = FREE_RGB
    img[grid_map.inflated] = INFLATED_RGB
    img[grid_map.raw] = OBSTACLE_RGB
    return img


def plot_grid_map(grid_map: GridMap, *,
                  paths: Optional[Mapping[str, Sequence[Point]]] = None,
                  trajectories: Optional[Mapping[str, Sequence[Point]]] = None,
                  stations: Optional[Mapping[str, Point]] = None,
                  starts: Optional[Mapping[str, object]] = None,
                  title: Optional[str] = None,
                  save_path: Optional[Union[str, Path]] = None,
                  show: bool = True, ax=None):
    """Plot the map with optional overlays. Returns ``(fig, ax)``.

    ``starts`` values may be ``Pose2D`` (x, y, yaw attributes) or ``(x, y[, yaw])`` tuples.
    """
    import matplotlib

    if not show:
        try:
            matplotlib.use("Agg")
        except Exception:  # backend already fixed by an earlier pyplot import
            pass
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    xmin, xmax, ymin, ymax = grid_map.bounds
    ax.imshow(grid_image(grid_map), origin="lower", extent=(xmin, xmax, ymin, ymax),
              interpolation="nearest")

    names = sorted(set(paths or {}) | set(trajectories or {}) | set(starts or {}))
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(names)}

    for name, pts in (paths or {}).items():
        if len(pts) == 0:
            continue
        arr = np.asarray(pts, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], "o--", color=colors[name], lw=1.5, ms=4,
                label=f"{name} plan")

    for name, pts in (trajectories or {}).items():
        if len(pts) == 0:
            continue
        arr = np.asarray(pts, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], "-", color=colors[name], lw=2.2, alpha=0.8,
                label=f"{name} actual")

    for name, pose in (starts or {}).items():
        x, y, yaw = _pose_tuple(pose)
        ax.plot(x, y, "s", color=colors[name], ms=9, mec="k")
        ax.arrow(x, y, 0.35 * math.cos(yaw), 0.35 * math.sin(yaw), head_width=0.12,
                 color=colors[name], length_includes_head=True)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8)

    for name, (x, y) in (stations or {}).items():
        is_drop = name.upper().startswith("D")
        ax.plot(x, y, "v" if is_drop else "*", color="tab:red" if is_drop else "tab:green",
                ms=12, mec="k")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9,
                    fontweight="bold")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"Occupancy grid ({grid_map.resolution:.2f} m/cell, "
                          f"inflation {grid_map.inflation_radius:.2f} m)")
    ax.grid(True, ls=":", lw=0.5)
    if paths or trajectories:
        ax.legend(loc="upper right", fontsize=8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _pose_tuple(pose: object) -> Tuple[float, float, float]:
    if hasattr(pose, "x") and hasattr(pose, "y"):
        return float(pose.x), float(pose.y), float(getattr(pose, "yaw", 0.0))  # type: ignore[attr-defined]
    seq = list(pose)  # type: ignore[call-overload]
    return float(seq[0]), float(seq[1]), float(seq[2]) if len(seq) > 2 else 0.0
