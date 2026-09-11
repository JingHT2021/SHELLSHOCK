"""Geometry for the projectile origin at the end of the tank barrel."""

from __future__ import annotations

from math import cos, isfinite, radians, sin
from numbers import Real

Point = tuple[float, float]

BARREL_LENGTH_AT_REFERENCE = 35.0
REFERENCE_IMAGE_WIDTH = 2560.0


def muzzle_position(
    center: Point,
    direction: str,
    angle_degrees: float,
    image_width: float,
    *, barrel_length: float = BARREL_LENGTH_AT_REFERENCE,
) -> Point:
    """Return the barrel-tip projectile origin for a UI firing angle."""
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    if not _is_finite_point(center):
        raise ValueError("center must contain finite coordinates")
    if not isinstance(angle_degrees, Real) or isinstance(angle_degrees, bool) or not isfinite(angle_degrees) or not 0 <= angle_degrees <= 90:
        raise ValueError("angle_degrees must be between 0 and 90")
    if not isinstance(image_width, Real) or isinstance(image_width, bool) or not isfinite(image_width) or image_width <= 0:
        raise ValueError("image_width must be positive")

    length = barrel_length * float(image_width) / REFERENCE_IMAGE_WIDTH
    angle = radians(float(angle_degrees))
    horizontal_sign = 1.0 if direction == "right" else -1.0
    return (
        float(center[0]) + horizontal_sign * length * cos(angle),
        float(center[1]) - length * sin(angle),
    )


def _is_finite_point(point: object) -> bool:
    return (
        isinstance(point, (tuple, list))
        and len(point) == 2
        and all(isinstance(value, Real) and not isinstance(value, bool) and isfinite(value) for value in point)
    )
