"""Robust extraction of the faint dashed white aiming guide from a screenshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, radians, sin

import cv2
import numpy as np


@dataclass(frozen=True)
class GuideDetection:
    status: str
    points: tuple[tuple[float, float], ...] = ()
    confidence: float = 0.0
    reason: str | None = None
    fit_parameters: dict[str, float] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


def _angle_distance(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def detect_game_guide(image, muzzle, *, direction="right", angle_degrees=0.0, exclude_regions=(), max_distance=None):
    if image is None or getattr(image, "size", 0) == 0:
        return GuideDetection("uncertain", reason="empty_image")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (0, 0), 7)
    mask = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 145) & (gray.astype(np.int16) > blurred.astype(np.int16) + 22)).astype(np.uint8) * 255
    for x, y, w, h in exclude_regions:
        cv2.rectangle(mask, (max(0, int(x)), max(0, int(y))), (min(width - 1, int(x + w)), min(height - 1, int(y + h))), 0, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    mx, my = float(muzzle[0]), float(muzzle[1])
    expected = radians(float(angle_degrees)) * (-1.0 if direction == "left" else 1.0)
    expected = -expected
    candidates = []
    for index in range(1, components):
        x, y, w, h, area = stats[index]
        if area < 2 or area > max(500, width * height * 0.01):
            continue
        cx, cy = (float(value) for value in centroids[index])
        dx, dy = cx - mx, cy - my
        distance = hypot(dx, dy)
        if distance < 8 or (max_distance is not None and distance > max_distance):
            continue
        angle = atan2(dy, dx)
        if _angle_distance(degrees(angle), degrees(expected)) > 70:
            continue
        candidates.append((distance, cx, cy, int(area), angle))
    if len(candidates) < 3:
        return GuideDetection("uncertain", reason="insufficient_segments", diagnostics=(f"candidates={len(candidates)}",))
    candidates.sort()
    selected = []
    last_distance = -1.0
    for candidate in candidates:
        if candidate[0] - last_distance < 6:
            continue
        selected.append(candidate)
        last_distance = candidate[0]
    if len(selected) < 3:
        return GuideDetection("uncertain", reason="non_continuous_segments")
    points = tuple((mx, my) for _ in [0]) + tuple((item[1], item[2]) for item in selected)
    distances = [hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)]
    confidence = min(1.0, 0.25 + 0.08 * len(selected) + 0.25 * min(1.0, np.mean(distances) / 30.0))
    return GuideDetection("matched", points, float(confidence), fit_parameters={"segment_count": float(len(selected))})
