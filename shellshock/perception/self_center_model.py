"""ROI utilities for the optional second-stage self-center pose model."""
from __future__ import annotations

from typing import TypeAlias

import numpy as np

from shellshock.domain.world import DetectionBox, Point

Origin: TypeAlias = tuple[int, int]
Size: TypeAlias = tuple[int, int]


def crop_self_roi(
    image: np.ndarray,
    box: DetectionBox,
    *,
    padding: float = 0.15,
    required_point: Point | None = None,
) -> tuple[np.ndarray, Origin]:
    """Crop a padded self box and return the crop origin in full-frame pixels."""
    if image.ndim < 2 or not 0.0 <= padding <= 1.0:
        raise ValueError("invalid image or padding")
    if box.width <= 0 or box.height <= 0:
        raise ValueError("self box must have positive dimensions")
    pad_x, pad_y = box.width * padding, box.height * padding
    left = int(np.floor(box.x - pad_x))
    top = int(np.floor(box.y - pad_y))
    right = int(np.ceil(box.x + box.width + pad_x))
    bottom = int(np.ceil(box.y + box.height + pad_y))
    if required_point is not None:
        left = min(left, int(np.floor(required_point[0])))
        top = min(top, int(np.floor(required_point[1])))
        right = max(right, int(np.floor(required_point[0])) + 1)
        bottom = max(bottom, int(np.floor(required_point[1])) + 1)
    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)
    if right <= left or bottom <= top:
        raise ValueError("self box lies outside image")
    return image[top:bottom, left:right].copy(), (left, top)


def map_roi_point_to_image(point: tuple[float, float], *, origin: Origin, size: Size) -> Point:
    """Map a normalized ROI point back to full-frame pixel coordinates."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("ROI size must be positive")
    return origin[0] + float(point[0]) * width, origin[1] + float(point[1]) * height


def valid_roi_keypoint(point: tuple[float, float, float | int], confidence: float, *, threshold: float = 0.5) -> bool:
    """Return whether a normalized model keypoint is usable."""
    x, y, visible = (float(value) for value in point)
    return (
        np.isfinite(x) and np.isfinite(y) and np.isfinite(visible)
        and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        and visible > 0.0 and float(confidence) >= threshold
    )


def make_self_center_label(
    *, center: Point, origin: Origin, size: Size, box: DetectionBox | None = None
) -> str:
    """Build one-class YOLO Pose label for a cropped self image."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("ROI size must be positive")
    keypoint_x = (float(center[0]) - origin[0]) / width
    keypoint_y = (float(center[1]) - origin[1]) / height
    if box is None:
        box_x, box_y, box_width, box_height = 0.5, 0.5, 1.0, 1.0
    else:
        box_x = (box.x + box.width / 2 - origin[0]) / width
        box_y = (box.y + box.height / 2 - origin[1]) / height
        box_width, box_height = box.width / width, box.height / height
    values = (box_x, box_y, box_width, box_height, keypoint_x, keypoint_y)
    return "0 " + " ".join(f"{value:.6f}" for value in values) + " 2"
