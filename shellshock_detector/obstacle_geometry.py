"""Geometry-first pink obstacle detection and annotation storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import pi
from pathlib import Path

import cv2
import numpy as np

from shellshock_detector.training_data import PinkObstacleConfig


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[int, int]
    radius: int


@dataclass(frozen=True)
class LineObstacle:
    start: tuple[int, int]
    end: tuple[int, int]


@dataclass(frozen=True)
class ObstacleGeometry:
    circles: list[CircleObstacle]
    lines: list[LineObstacle]


def _scale(value: float, image_width: int, config: PinkObstacleConfig) -> float:
    return value * image_width / config.reference_width


def _circle_candidates(mask: np.ndarray, config: PinkObstacleConfig) -> list[CircleObstacle]:
    image_height, image_width = mask.shape
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    minimum_radius = max(
        _scale(config.circle_min_radius, image_width, config),
        _scale(config.hough_circle_min_radius, image_width, config),
    )
    candidates = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(round(_scale(80, image_width, config)))),
        param1=50,
        param2=12,
        minRadius=max(1, int(round(minimum_radius))),
        maxRadius=max(1, int(round(_scale(config.circle_max_radius, image_width, config)))),
    )
    if candidates is None:
        return []

    support_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    accepted: list[tuple[float, CircleObstacle]] = []
    for center_x, center_y, radius in np.round(candidates[0]).astype(int):
        angles = np.linspace(0, 2 * pi, 720, endpoint=False)
        xs = np.rint(center_x + radius * np.cos(angles)).astype(int)
        ys = np.rint(center_y + radius * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
        if not valid.any():
            continue
        support = float(np.mean(support_mask[ys[valid], xs[valid]] > 0))
        if support < config.hough_circle_min_support:
            continue
        circle = CircleObstacle((int(center_x), int(center_y)), int(radius))
        if any(
            np.hypot(circle.center[0] - old.center[0], circle.center[1] - old.center[1])
            < max(circle.radius, old.radius) * 0.35
            and abs(circle.radius - old.radius) < max(circle.radius, old.radius) * 0.35
            for _, old in accepted
        ):
            continue
        accepted.append((support, circle))
    return [circle for _, circle in sorted(accepted, key=lambda item: (-item[0], item[1].center))]


def _exclude_circle_rings(mask: np.ndarray, circles: list[CircleObstacle], config: PinkObstacleConfig) -> np.ndarray:
    result = mask.copy()
    padding = max(4, int(round(_scale(config.line_max_thickness, mask.shape[1], config))))
    for circle in circles:
        cv2.circle(result, circle.center, circle.radius, 0, thickness=padding * 2 + 1)
    return result


def _fit_line(contour: np.ndarray) -> LineObstacle | None:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy])
    origin = np.array([x0, y0])
    projection = (points - origin) @ direction
    start = tuple(map(int, np.rint(origin + projection.min() * direction)))
    end = tuple(map(int, np.rint(origin + projection.max() * direction)))
    return LineObstacle(*sorted((start, end)))


def _line_candidates(mask: np.ndarray, config: PinkObstacleConfig) -> list[LineObstacle]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_length = _scale(config.line_min_length, mask.shape[1], config)
    minimum_thickness = _scale(config.line_min_thickness, mask.shape[1], config)
    maximum_thickness = _scale(config.line_max_thickness, mask.shape[1], config)
    lines: list[LineObstacle] = []
    for contour in contours:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        length, thickness = sorted((width, height), reverse=True)
        if (
            thickness < minimum_thickness
            or thickness > maximum_thickness
            or length < minimum_length
            or length / thickness < config.line_min_aspect_ratio
        ):
            continue
        line = _fit_line(contour)
        if line is not None:
            lines.append(line)
    return sorted(lines, key=lambda item: (item.start, item.end))


def detect_pink_obstacle_geometry(
    image: np.ndarray, config: PinkObstacleConfig = PinkObstacleConfig()
) -> ObstacleGeometry:
    """Detect circles first, then fit lines only from pixels outside circle rings."""
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    circles = _circle_candidates(mask, config)
    line_mask = _exclude_circle_rings(mask, circles, config)
    return ObstacleGeometry(circles=circles, lines=_line_candidates(line_mask, config))


def geometry_to_dict(geometry: ObstacleGeometry) -> dict[str, object]:
    return {
        "circles": [asdict(circle) for circle in geometry.circles],
        "lines": [asdict(line) for line in geometry.lines],
    }


def save_geometry(path: Path, geometry: ObstacleGeometry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry_to_dict(geometry), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_obstacle_yolo_labels(existing_text: str) -> str:
    """Remove only old rectangular obstacle rows, retaining all unrelated bytes."""
    return "".join(
        line
        for line in existing_text.splitlines(keepends=True)
        if not (line.split() and line.split()[0] in {"3", "4"})
    )


def draw_geometry_preview(image: np.ndarray, geometry: ObstacleGeometry) -> np.ndarray:
    preview = image.copy()
    for circle in geometry.circles:
        cv2.circle(preview, circle.center, circle.radius, (0, 255, 255), 3)
        cv2.circle(preview, circle.center, 5, (0, 255, 255), -1)
        cv2.putText(preview, f"circle r={circle.radius}", (circle.center[0], max(18, circle.center[1] - circle.radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for line in geometry.lines:
        cv2.line(preview, line.start, line.end, (255, 255, 0), 3)
        cv2.circle(preview, line.start, 5, (255, 255, 0), -1)
        cv2.circle(preview, line.end, 5, (255, 255, 0), -1)
    return preview
