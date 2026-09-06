"""Convert YOLO boxes into the immutable geometry used by shot simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MAX_PORTAL_RADIUS_RELATIVE_ERROR = 0.15

Point = tuple[float, float]


@dataclass(frozen=True)
class DetectionBox:
    name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class CircleObstacle:
    center: Point
    radius: float


@dataclass(frozen=True)
class LineObstacle:
    start: Point
    end: Point


@dataclass(frozen=True)
class Portal:
    color: Literal["orange", "blue"]
    center: Point
    radius: float


@dataclass(frozen=True)
class PortalPair:
    orange: Portal
    blue: Portal


@dataclass(frozen=True)
class World:
    image_width: int = 0
    image_height: int = 0
    self_position: Point | None = None
    circles: tuple[CircleObstacle, ...] = ()
    lines: tuple[LineObstacle, ...] = ()
    portal_pairs: tuple[PortalPair, ...] = ()
    unpaired_portals: int = 0

    def __post_init__(self) -> None:
        """Defend frozen-world semantics when callers supply ordinary lists."""
        object.__setattr__(self, "circles", tuple(self.circles))
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "portal_pairs", tuple(self.portal_pairs))


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


def _portal_candidates(boxes: list[DetectionBox], image_width: int, image_height: int) -> tuple[list[Portal], list[Portal]]:
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
        portal = Portal(color, _center(left, top, width, height), (width + height) / 4)
        (oranges if color == "orange" else blues).append(portal)
    return oranges, blues


def _portal_pairs(oranges: list[Portal], blues: list[Portal]) -> tuple[PortalPair, ...]:
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


def build_world(boxes: list[DetectionBox], image_width: int, image_height: int) -> World:
    """Build a boundary-clipped world from named YOLO detections."""
    self_candidates: list[tuple[float, Point]] = []
    circles: list[CircleObstacle] = []
    lines: list[LineObstacle] = []
    for box in boxes:
        clipped = _clip_box(box, image_width, image_height)
        if clipped is None:
            continue
        left, top, width, height = clipped
        if box.name == "self":
            self_candidates.append((box.confidence, _center(left, top, width, height)))
        elif box.name == "obstacle_circle":
            circles.append(CircleObstacle(_center(left, top, width, height), (width + height) / 4))
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
        lines=tuple(lines),
        portal_pairs=pairs,
        unpaired_portals=len(oranges) + len(blues) - 2 * len(pairs),
    )
