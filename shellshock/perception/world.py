"""Convert YOLO boxes into the immutable geometry used by shot simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Literal

import cv2
import numpy as np
from shellshock.perception.circle_fit import fit_standard_circle
from shellshock.perception.self_center import fit_self_center
from shellshock.perception.digits import recognize_trained_digits

from shellshock.perception.color_geometry import detect_pink_obstacle_geometry

MAX_PORTAL_RADIUS_RELATIVE_ERROR = 0.15

from shellshock.domain.world import Point, PoseKeypoint, DetectionBox, CircleObstacle, LineObstacle, Portal, PortalPair, World, RewardZone


def _clip_box(box: DetectionBox, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    """Return the intersected box as left, top, width, height."""
    left = max(0.0, min(float(image_width), box.x))
    top = max(0.0, min(float(image_height), box.y))
    right = max(0.0, min(float(image_width), box.x + box.width))
    bottom = max(0.0, min(float(image_height), box.y + box.height))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _center(left: float, top: float, width: float, height: float) -> Point:
    return left + width / 2, top + height / 2


def _portal_candidates(boxes: list[DetectionBox], image_width: int, image_height: int, image: np.ndarray | None = None) -> tuple[list[Portal], list[Portal]]:
    oranges: list[Portal] = []
    blues: list[Portal] = []
    for box in boxes:
        color: Literal["orange", "blue"] | None = {
            "portal_orange": "orange",
            "portal_blue": "blue",
        }.get(box.name)
        if color is None:
            continue
        clipped = _clip_box(box, image_width, image_height)
        if clipped is None:
            continue
        left, top, width, height = clipped
        number = None
        if image is not None:
            crop = image[int(top + height * 0.20):int(top + height * 0.80), int(left + width * 0.20):int(left + width * 0.80)]
            text, confidence = recognize_trained_digits(255 - crop)
            if text and len(text) == 1 and text.isdigit() and confidence >= 0.50:
                number = int(text)
        portal = Portal(color, _center(left, top, width, height), (width + height) / 4, number)
        (oranges if color == "orange" else blues).append(portal)
    return oranges, blues


def _legacy_portal_pairs(oranges: list[Portal], blues: list[Portal]) -> tuple[PortalPair, ...]:
    """Greedily take unique best radius matches after excluding ambiguous ties."""
    candidates: dict[int, list[tuple[float, int]]] = {}
    reverse_candidates: dict[int, list[tuple[float, int]]] = {}
    for orange_index, orange in enumerate(oranges):
        for blue_index, blue in enumerate(blues):
            error = abs(orange.radius - blue.radius) / max(orange.radius, blue.radius)
            if error <= MAX_PORTAL_RADIUS_RELATIVE_ERROR:
                candidates.setdefault(orange_index, []).append((error, blue_index))
                reverse_candidates.setdefault(blue_index, []).append((error, orange_index))

    ambiguous_oranges: set[int] = set()
    ambiguous_blues: set[int] = set()
    for orange_index, options in candidates.items():
        best = min(error for error, _ in options)
        tied = [blue_index for error, blue_index in options if error == best]
        if len(tied) > 1:
            ambiguous_oranges.add(orange_index)
            ambiguous_blues.update(tied)
    for blue_index, options in reverse_candidates.items():
        best = min(error for error, _ in options)
        tied = [orange_index for error, orange_index in options if error == best]
        if len(tied) > 1:
            ambiguous_blues.add(blue_index)
            ambiguous_oranges.update(tied)

    edges = sorted(
        (error, orange.center, blue.center, orange_index, blue_index)
        for orange_index, options in candidates.items()
        for error, blue_index in options
        for orange in [oranges[orange_index]]
        for blue in [blues[blue_index]]
        if orange_index not in ambiguous_oranges and blue_index not in ambiguous_blues
    )
    used_oranges: set[int] = set()
    used_blues: set[int] = set()
    pairs: list[PortalPair] = []
    for _, _, _, orange_index, blue_index in edges:
        if orange_index in used_oranges or blue_index in used_blues:
            continue
        pairs.append(PortalPair(oranges[orange_index], blues[blue_index]))
        used_oranges.add(orange_index)
        used_blues.add(blue_index)
    return tuple(pairs)


def _portal_pairs(oranges: list[Portal], blues: list[Portal]) -> tuple[PortalPair, ...]:
    """Pair confidently recognized numbers first, then unknown portals by geometry."""
    orange_by_number: dict[int, list[int]] = {}
    blue_by_number: dict[int, list[int]] = {}
    for index, portal in enumerate(oranges):
        if portal.number is not None:
            orange_by_number.setdefault(portal.number, []).append(index)
    for index, portal in enumerate(blues):
        if portal.number is not None:
            blue_by_number.setdefault(portal.number, []).append(index)
    pairs: list[PortalPair] = []
    recognized_oranges = {index for indexes in orange_by_number.values() for index in indexes}
    recognized_blues = {index for indexes in blue_by_number.values() for index in indexes}
    for number in sorted(set(orange_by_number) & set(blue_by_number)):
        orange_indexes, blue_indexes = orange_by_number[number], blue_by_number[number]
        if len(orange_indexes) == 1 and len(blue_indexes) == 1:
            pairs.append(PortalPair(oranges[orange_indexes[0]], blues[blue_indexes[0]]))
    unknown_oranges = [portal for index, portal in enumerate(oranges) if index not in recognized_oranges]
    unknown_blues = [portal for index, portal in enumerate(blues) if index not in recognized_blues]
    return tuple(pairs) + _legacy_portal_pairs(unknown_oranges, unknown_blues)


def build_world(boxes: list[DetectionBox], image_width: int, image_height: int) -> World:
    """Build a boundary-clipped world from named YOLO detections."""
    self_candidates: list[tuple[float, Point]] = []
    circles: list[CircleObstacle] = []
    lines: list[LineObstacle] = []
    blackholes: list[CircleObstacle] = []
    rewards: list[RewardZone] = []
    for index, box in enumerate(boxes):
        clipped = _clip_box(box, image_width, image_height)
        if clipped is None:
            continue
        left, top, width, height = clipped
        if box.name == "self":
            self_candidates.append((box.confidence, _pose_point(box, 0) or _center(left, top, width, height)))
        elif box.name == "obstacle_circle":
            circles.append(CircleObstacle(_center(left, top, width, height), (width + height) / 4))
        elif box.name == "blackhole":
            blackholes.append(_pose_circle(box) or CircleObstacle(_center(left, top, width, height), (width + height) / 4))
        elif box.name in {"double_damage", "Triple_damage"}:
            shape = _pose_circle(box) or CircleObstacle(_center(left, top, width, height), (width + height) / 4)
            rewards.append(RewardZone(shape.center, shape.radius, 2 if box.name == "double_damage" else 3, f"{box.name}:{index}"))
        elif box.name == "obstacle_line":
            if width >= height:
                middle = top + height / 2
                lines.append(LineObstacle((left, middle), (left + width, middle)))
            else:
                middle = left + width / 2
                lines.append(LineObstacle((middle, top), (middle, top + height)))

    self_position: Point | None = None
    if self_candidates:
        best_confidence = max(confidence for confidence, _ in self_candidates)
        best = [position for confidence, position in self_candidates if confidence == best_confidence]
        if len(best) == 1:
            self_position = best[0]

    oranges, blues = _portal_candidates(boxes, image_width, image_height)
    pairs = _portal_pairs(oranges, blues)
    return World(
        image_width=image_width,
        image_height=image_height,
        self_position=self_position,
        circles=tuple(circles),
        blackholes=tuple(blackholes),
        rewards=tuple(rewards),
        lines=tuple(lines),
        portal_pairs=pairs,
        unpaired_portals=len(oranges) + len(blues) - 2 * len(pairs),
        unpaired_portal_regions=tuple(p for p in (*oranges,*blues) if p not in {q for pair in pairs for q in (pair.orange,pair.blue)}),
    )


def _candidate_roi(box: DetectionBox, image: np.ndarray) -> tuple[np.ndarray, int, int, int, int] | None:
    clipped = _clip_box(box, image.shape[1], image.shape[0])
    if clipped is None:
        return None
    left, top, width, height = clipped
    x0, y0 = int(left), int(top)
    x1 = int(np.ceil(left + width))
    y1 = int(np.ceil(top + height))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return roi, x0, y0, x1, y1


def _guided_obstacle_geometry(box: DetectionBox, image: np.ndarray):
    """Run the shared fitter at full-screen scale, with pixels limited to one YOLO ROI."""
    candidate = _candidate_roi(box, image)
    if candidate is None:
        return None
    roi, x0, y0, x1, y1 = candidate
    guided = np.zeros_like(image)
    guided[y0:y1, x0:x1] = roi
    return detect_pink_obstacle_geometry(guided)


def _refined_circle(box: DetectionBox, image: np.ndarray) -> CircleObstacle | None:
    geometry = _guided_obstacle_geometry(box, image)
    if geometry is None:
        return None
    circles = geometry.circles
    if not circles:
        return None
    expected_x, expected_y = box.x + box.width / 2, box.y + box.height / 2
    circle = min(circles, key=lambda item: hypot(item.center[0] - expected_x, item.center[1] - expected_y))
    return CircleObstacle(circle.center, circle.radius)


def _fit_visible_circle(box: DetectionBox, image: np.ndarray, *, color_only=False) -> CircleObstacle | None:
    """Fit the visible arc with class color, edge and validated Hough fallbacks."""
    candidate = _candidate_roi(box, image)
    if candidate is None:
        return None
    roi, x0, y0, _, _ = candidate
    fit = fit_standard_circle(
        roi, box.name, np.array((box.x + box.width / 2 - x0, box.y + box.height / 2 - y0)),
        max(4.0, min(box.width, box.height) * .2), max(box.width, box.height) * 2, color_only=color_only)
    if fit is None:
        return None
    return CircleObstacle((fit.center[0] + x0, fit.center[1] + y0), fit.radius)


def _fit_visible_line(box: DetectionBox, image: np.ndarray) -> LineObstacle | None:
    """Fit endpoints from an elongated white component inside one YOLO ROI."""
    candidate = _candidate_roi(box, image)
    if candidate is None:
        return None
    roi, x0, y0, _, _ = candidate
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 230)), np.array((0, 0, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    expected = np.array((box.x + box.width / 2.0, box.y + box.height / 2.0))
    candidates: list[tuple[float, float, LineObstacle]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 2:
            continue
        width, height = cv2.minAreaRect(contour)[1]
        length, thickness = sorted((float(width), float(height)), reverse=True)
        if length < 6.0 or length / max(thickness, 1.0) < 2.0:
            continue
        points[:, 0] += x0
        points[:, 1] += y0
        vx, vy, center_x, center_y = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        direction = np.array((float(vx), float(vy)))
        origin = np.array((float(center_x), float(center_y)))
        projections = (points - origin) @ direction
        endpoints = [origin + projections.min() * direction, origin + projections.max() * direction]
        line = LineObstacle(tuple(map(float, endpoints[0])), tuple(map(float, endpoints[1])))
        midpoint = (endpoints[0] + endpoints[1]) / 2.0
        candidates.append((float(np.linalg.norm(midpoint - expected)), -length, line))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def _refined_line(box: DetectionBox, image: np.ndarray) -> LineObstacle | None:
    geometry = _guided_obstacle_geometry(box, image)
    if geometry is None:
        return None
    lines = geometry.lines
    if not lines:
        return None
    expected_x, expected_y = box.x + box.width / 2, box.y + box.height / 2
    line = min(
        lines,
        key=lambda item: hypot(
            (item.start[0] + item.end[0]) / 2 - expected_x,
            (item.start[1] + item.end[1]) / 2 - expected_y,
        ),
    )
    return LineObstacle(line.start, line.end)


def build_world_from_image(boxes: list[DetectionBox], image: np.ndarray) -> World:
    world, _ = build_world_from_image_with_diagnostics(boxes, image)
    return world


def _pose_point(box: DetectionBox, index: int) -> Point | None:
    if index >= len(box.keypoints):
        return None
    point = box.keypoints[index]
    if point.visible <= 0 or not np.isfinite(point.confidence) or point.confidence < .5 or not np.isfinite(point.x) or not np.isfinite(point.y):
        return None
    return point.x, point.y


def _point_error(left: Point | None, right: Point | None) -> float | None:
    if left is None or right is None:
        return None
    return float(hypot(left[0] - right[0], left[1] - right[1]))


def _pose_circle(box: DetectionBox) -> CircleObstacle | None:
    center, edge = _pose_point(box, 0), _pose_point(box, 1)
    radius = hypot(center[0] - edge[0], center[1] - edge[1]) if center and edge else 0
    return CircleObstacle(center, radius) if radius > .5 else None


def build_world_from_image_with_diagnostics(boxes: list[DetectionBox], image: np.ndarray) -> tuple[World, list[dict[str, object]]]:
    """Prefer supported circle fits, then trusted Pose, then detector geometry."""
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    image_height, image_width = image.shape[:2]
    base = build_world(boxes, image_width, image_height)
    circles: list[CircleObstacle] = []
    blackholes: list[CircleObstacle] = []
    rewards: list[RewardZone] = []
    lines: list[LineObstacle] = []
    oranges: list[Portal] = []
    blues: list[Portal] = []
    diagnostics: list[dict[str, object]] = []
    self_candidates: list[tuple[float, Point]] = []
    circle_classes = {"obstacle_circle", "blackhole", "double_damage", "Triple_damage", "portal_orange", "portal_blue"}
    for index, box in enumerate(boxes):
        clipped = _clip_box(box, image_width, image_height)
        if clipped is None:
            continue
        left, top, width, height = clipped
        bbox_center = _center(left, top, width, height)
        if box.name in circle_classes:
            pose = _pose_circle(box)
            # A trusted pose must not be displaced by another object's color or
            # white digits inside a colored portal. Generic edges are fallback
            # evidence only when no trusted pose is available.
            fitted = None if box.source == "manual" else _fit_visible_circle(box, image, color_only=pose is not None)
            fallback = CircleObstacle((box.x + box.width/2, box.y + box.height/2), (box.width + box.height)/4)
            chosen = (pose or fallback) if box.source == "manual" else (fitted or pose or fallback)
            center, edge = _pose_point(box, 0), _pose_point(box, 1)
            diagnostics.append({"class": box.name, "source": "manual" if box.source == "manual" else "fit" if fitted else "pose" if pose else "bbox", "center_error_px": _point_error(center, fitted.center if fitted else None), "edge_error_px": _point_error(edge, (fitted.center[0] - fitted.radius, fitted.center[1]) if fitted else None)})
            if box.name == "obstacle_circle":
                circles.append(chosen)
            elif box.name == "blackhole":
                blackholes.append(chosen)
            elif box.name in {"double_damage", "Triple_damage"}:
                rewards.append(RewardZone(chosen.center, chosen.radius, 2 if box.name == "double_damage" else 3, f"{box.name}:{index}"))
            else:
                # Recognize each detection once; retain its number while replacing geometry.
                orange, blue = _portal_candidates([box], image_width, image_height, image)
                baseline = (orange or blue)[0]
                portal = Portal(baseline.color, chosen.center, chosen.radius, baseline.number)
                (oranges if portal.color == "orange" else blues).append(portal)
        elif box.name == "obstacle_line":
            color_line = None if box.source == "manual" else (_fit_visible_line(box, image) or _refined_line(box, image))
            start, end = _pose_point(box, 0), _pose_point(box, 1)
            pose_line = LineObstacle(start, end) if start and end else None
            fallback = (LineObstacle((left, top + height / 2), (left + width, top + height / 2))
                        if width >= height else LineObstacle((left + width / 2, top), (left + width / 2, top + height)))
            lines.append(color_line or pose_line or fallback)
            direct = _point_error(start, color_line.start) + _point_error(end, color_line.end) if start and end and color_line else None
            reverse = _point_error(start, color_line.end) + _point_error(end, color_line.start) if start and end and color_line else None
            diagnostics.append({"class": box.name, "source": "manual" if box.source == "manual" else "color" if color_line else "pose" if pose_line else "bbox", "endpoints_error_px": min(direct, reverse) if direct is not None and reverse is not None else None})
        elif box.name == "self":
            center = _pose_point(box, 0)
            fitted = None if box.source == "manual" else fit_self_center(image, box)
            chosen = fitted or center or bbox_center
            self_candidates.append((box.confidence, chosen))
            diagnostics.append({"class": box.name, "source": "manual" if box.source == "manual" else "fit" if fitted else "pose" if center else "bbox_center", "center_error_px": _point_error(center, fitted or bbox_center)})
    self_position = None
    if self_candidates:
        best_confidence = max(confidence for confidence, _ in self_candidates)
        winners = [point for confidence, point in self_candidates if confidence == best_confidence]
        if len(winners) == 1:
            self_position = winners[0]
    pairs = _portal_pairs(oranges, blues)
    return World(
        image_width=base.image_width,
        image_height=base.image_height,
        self_position=self_position,
        circles=tuple(circles),
        blackholes=tuple(blackholes),
        rewards=tuple(rewards),
        lines=tuple(lines),
        portal_pairs=pairs,
        unpaired_portals=len(oranges) + len(blues) - 2 * len(pairs),
        unpaired_portal_regions=tuple(p for p in (*oranges,*blues) if p not in {q for pair in pairs for q in (pair.orange,pair.blue)}),
    ), diagnostics
