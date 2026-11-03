"""Small shared geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Pose2D:
    """Planar pose in the CoppeliaSim world frame."""

    x: float
    y: float
    yaw: float = 0.0  # [rad], counter-clockwise from world +x

    def xy(self) -> Point:
        return (self.x, self.y)

    def distance_to(self, other: "Pose2D | Point") -> float:
        ox, oy = (other.x, other.y) if isinstance(other, Pose2D) else other
        return math.hypot(ox - self.x, oy - self.y)


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_matrix(m: Sequence[float]) -> float:
    """Yaw of an object from a CoppeliaSim 12-element (3x4, row-major) pose matrix.

    Uses the projection of the object's x axis onto the world XY plane, so it does not
    depend on the Euler-angle convention.
    """
    return math.atan2(m[4], m[0])


def transform_point(m: Sequence[float], p: Sequence[float]) -> Tuple[float, float, float]:
    """Apply a CoppeliaSim 3x4 row-major matrix to a local point (x, y, z)."""
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])
