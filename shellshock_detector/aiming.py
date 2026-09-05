"""Aim-disc click geometry."""

from math import cos, isfinite, radians, sin
from numbers import Real


AIM_DISC_RADIUS_AT_REFERENCE = 325.0
"""Full-power aim-disc radius at 1920-pixel client width (empirically calibrated)."""

AIM_DISC_CENTER_OFFSET_AT_REFERENCE = (2.5, 3.5)
"""Aim-disc center minus green-tank detection center at 1920-pixel client width."""
_REFERENCE_IMAGE_WIDTH = 1920.0


def disc_click_point(
    center_x: float,
    center_y: float,
    direction: str,
    angle_degrees: float,
    power: float,
    image_width: float,
) -> tuple[int, int]:
    """Return the screen point on the aim disc for a shot setting."""
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    if not _is_valid_number(angle_degrees) or not 0 <= angle_degrees <= 90:
        raise ValueError("angle_degrees must be between 0 and 90")
    if not _is_valid_number(power) or not 0 <= power <= 100:
        raise ValueError("power must be between 0 and 100")
    if not _is_valid_number(image_width) or image_width <= 0:
        raise ValueError("image_width must be positive")

    scale = image_width / _REFERENCE_IMAGE_WIDTH
    radius = AIM_DISC_RADIUS_AT_REFERENCE * scale
    displacement = radius * power / 100
    angle = radians(angle_degrees)
    horizontal = displacement * cos(angle)
    vertical = displacement * sin(angle)
    if direction == "left":
        horizontal = -horizontal

    center_x += AIM_DISC_CENTER_OFFSET_AT_REFERENCE[0] * scale
    center_y += AIM_DISC_CENTER_OFFSET_AT_REFERENCE[1] * scale
    return (round(center_x + horizontal), round(center_y - vertical))


def _is_valid_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
