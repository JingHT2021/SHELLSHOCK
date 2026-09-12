"""Geometry-first bright-white obstacle detection and annotation storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import pi
from pathlib import Path

import cv2
import numpy as np

from shellshock.datasets.export import (
    CLASS_NAMES,
    ObstacleDetection,
    PinkObstacleConfig,
    _parse_yolo_boxes,
)


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[int, int]
    radius: int


@dataclass(frozen=True)
class LineObstacle:
    start: tuple[int, int]
    end: tuple[int, int]


@dataclass(frozen=True)
class PortalConfig:
    """HSV and circle-fitting thresholds for blue and orange portal rings."""

    orange_lower: tuple[int, int, int] = (14, 230, 150)
    orange_upper: tuple[int, int, int] = (16, 255, 255)
    blue_lower: tuple[int, int, int] = (99, 180, 150)
    blue_upper: tuple[int, int, int] = (102, 255, 255)
    reference_width: int = 1920
    min_radius: float = 25.0
    max_radius: float = 350.0
    min_support: float = 0.45
    max_relative_radius_error: float = 0.15


@dataclass(frozen=True)
class PortalGeometry:
    class_id: int
    name: str
    center: tuple[int, int]
    radius: int
    pair_id: int | None


@dataclass(frozen=True)
class ObstacleGeometry:
    circles: list[CircleObstacle]
    lines: list[LineObstacle]
    portals: list[PortalGeometry] = field(default_factory=list)


def _scale(value: float, image_width: int, config: PinkObstacleConfig) -> float:
    return value * image_width / config.reference_width


def _portal_scale(value: float, image_width: int, config: PortalConfig) -> float:
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
    """Detect bright-white circles first, then fit lines outside their rings."""
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    circles = _circle_candidates(mask, config)
    line_mask = _exclude_circle_rings(mask, circles, config)
    return ObstacleGeometry(circles=circles, lines=_line_candidates(line_mask, config))


def _portal_circle_candidates(mask: np.ndarray, config: PortalConfig) -> list[CircleObstacle]:
    """Fit supported ring circles from one already-colour-filtered portal mask."""
    height, width = mask.shape
    close_size = max(3, int(round(_portal_scale(3, width, config))) | 1)
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    min_radius = max(1, int(round(_portal_scale(config.min_radius, width, config))))
    max_radius = max(min_radius, int(round(_portal_scale(config.max_radius, width, config))))
    candidates = cv2.HoughCircles(
        cv2.GaussianBlur(cleaned, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(round(_portal_scale(80, width, config)))),
        param1=50,
        param2=12,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if candidates is None:
        return []
    support_mask = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    accepted: list[tuple[float, CircleObstacle]] = []
    for center_x, center_y, radius in np.round(candidates[0]).astype(int):
        angles = np.linspace(0, 2 * pi, 720, endpoint=False)
        xs = np.rint(center_x + radius * np.cos(angles)).astype(int)
        ys = np.rint(center_y + radius * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not valid.any():
            continue
        support = float(np.mean(support_mask[ys[valid], xs[valid]] > 0))
        circle = CircleObstacle((int(center_x), int(center_y)), int(radius))
        if support < config.min_support or any(
            np.hypot(circle.center[0] - old.center[0], circle.center[1] - old.center[1])
            < max(circle.radius, old.radius) * 0.35
            and abs(circle.radius - old.radius) < max(circle.radius, old.radius) * 0.35
            for _, old in accepted
        ):
            continue
        accepted.append((support, circle))
    return [circle for _, circle in sorted(accepted, key=lambda item: (-item[0], item[1].center))]


def detect_portal_geometry(
    image: np.ndarray, config: PortalConfig = PortalConfig()
) -> list[PortalGeometry]:
    """Detect blue/orange portal rings and assign deterministic, radius-based pairs."""
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates: list[tuple[int, str, CircleObstacle]] = []
    for class_id, name, lower, upper in (
        (5, "portal_orange", config.orange_lower, config.orange_upper),
        (6, "portal_blue", config.blue_lower, config.blue_upper),
    ):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        candidates.extend((class_id, name, circle) for circle in _portal_circle_candidates(mask, config))

    oranges = [item for item in candidates if item[0] == 5]
    blues = [item for item in candidates if item[0] == 6]
    edges: list[tuple[float, tuple[int, int], tuple[int, int], int, int]] = []
    for orange_index, (_, _, orange) in enumerate(oranges):
        for blue_index, (_, _, blue) in enumerate(blues):
            error = abs(orange.radius - blue.radius) / max(orange.radius, blue.radius)
            if error <= config.max_relative_radius_error:
                edges.append((error, orange.center, blue.center, orange_index, blue_index))
    used_orange: set[int] = set()
    used_blue: set[int] = set()
    pair_ids: dict[tuple[int, int], int] = {}
    for _, _, _, orange_index, blue_index in sorted(edges):
        if orange_index in used_orange or blue_index in used_blue:
            continue
        pair_id = len(pair_ids) + 1
        pair_ids[(5, orange_index)] = pair_id
        pair_ids[(6, blue_index)] = pair_id
        used_orange.add(orange_index)
        used_blue.add(blue_index)

    portals: list[PortalGeometry] = []
    for class_id, name, grouped in ((5, "portal_orange", oranges), (6, "portal_blue", blues)):
        for index, (_, _, circle) in enumerate(grouped):
            portals.append(PortalGeometry(class_id, name, circle.center, circle.radius, pair_ids.get((class_id, index))))
    return sorted(portals, key=lambda item: (item.class_id, item.center, item.radius))


def obstacle_yolo_detections(
    geometry: ObstacleGeometry, image_width: int, config: PinkObstacleConfig = PinkObstacleConfig()
) -> list[ObstacleDetection]:
    """Create training boxes solely from reflection-mode obstacle geometry."""
    padding = max(2, int(round(_scale(config.line_min_thickness, image_width, config))))
    detections = [
        ObstacleDetection(3, (circle.center[0] - circle.radius, circle.center[1] - circle.radius, circle.radius * 2, circle.radius * 2))
        for circle in geometry.circles
    ]
    for line in geometry.lines:
        left, right = sorted((line.start[0], line.end[0]))
        top, bottom = sorted((line.start[1], line.end[1]))
        detections.append(ObstacleDetection(4, (left - padding, top - padding, right - left + padding * 2, bottom - top + padding * 2)))
    return detections


def portal_yolo_detections(portals: list[PortalGeometry]) -> list[ObstacleDetection]:
    """Represent each precise portal circle as the square YOLO training box."""
    return [
        ObstacleDetection(portal.class_id, (portal.center[0] - portal.radius, portal.center[1] - portal.radius, portal.radius * 2, portal.radius * 2))
        for portal in portals
    ]


def yolo_objects(label_text: str, image_width: int, image_height: int) -> list[dict[str, object]]:
    """Return all valid label rows as named normalized YOLO objects."""
    return [
        {"class_id": class_id, "name": CLASS_NAMES.get(class_id, str(class_id)), "yolo_box": [center_x / image_width, center_y / image_height, width / image_width, height / image_height]}
        for class_id, (center_x, center_y, width, height) in _parse_yolo_boxes(label_text, image_width, image_height)
    ]


def geometry_to_dict(
    geometry: ObstacleGeometry, objects: list[dict[str, object]] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "circles": [asdict(circle) for circle in geometry.circles],
        "lines": [asdict(line) for line in geometry.lines],
        "portals": [asdict(portal) for portal in geometry.portals],
    }
    if objects is not None:
        payload["objects"] = objects
    return payload


def save_geometry(path: Path, geometry: ObstacleGeometry, objects: list[dict[str, object]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry_to_dict(geometry, objects), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    for portal in geometry.portals:
        color = (0, 140, 255) if portal.class_id == 5 else (255, 180, 0)
        cv2.circle(preview, portal.center, portal.radius, color, 3)
        cv2.circle(preview, portal.center, 5, color, -1)
        pair = "unpaired" if portal.pair_id is None else f"pair={portal.pair_id}"
        cv2.putText(preview, f"{portal.name} r={portal.radius} {pair}", (portal.center[0], max(18, portal.center[1] - portal.radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return preview
