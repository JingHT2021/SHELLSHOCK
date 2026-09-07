"""Event-ordered projectile replay, including paired wormhole teleportation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Literal

from .world_geometry import CircleObstacle, LineObstacle, Portal, PortalPair, World


Vector = tuple[float, float]
MAX_EVENTS = 32
EVENT_EPSILON = 1e-5
PORTAL_EXIT_EPSILON = 1e-4
LINE_ENDPOINT_MARGIN_AT_REFERENCE = 12.0


@dataclass(frozen=True)
class TrajectoryEvent:
    kind: Literal["portal", "reflection", "obstacle", "target", "out_of_bounds", "event_limit"]
    time: float
    point: Vector
    exit_point: Vector | None = None
    velocity_after: Vector | None = None
    obstacle_kind: Literal["circle", "line"] | None = None
    obstacle_index: int | None = None


@dataclass(frozen=True)
class Replay:
    events: tuple[TrajectoryEvent, ...]
    terminal_kind: str

    @property
    def portal_count(self) -> int:
        return sum(event.kind == "portal" for event in self.events)

    @property
    def reflection_count(self) -> int:
        return sum(event.kind == "reflection" for event in self.events)


def _position(start: Vector, velocity: Vector, acceleration: Vector, time: float) -> Vector:
    return (
        start[0] + velocity[0] * time + acceleration[0] * time * time / 2,
        start[1] + velocity[1] * time + acceleration[1] * time * time / 2,
    )


def _velocity(velocity: Vector, acceleration: Vector, time: float) -> Vector:
    return velocity[0] + acceleration[0] * time, velocity[1] + acceleration[1] * time


def _first_circle_time(start: Vector, velocity: Vector, acceleration: Vector, center: Vector, radius: float, limit: float) -> float | None:
    """Use the same exact stationary-point test as analytic portal search."""
    # Imported lazily so wormhole_solver can itself use advance_through_world.
    from .wormhole_solver import WormholeTrajectory, first_circle_entry_time

    return first_circle_entry_time(WormholeTrajectory(start, velocity, acceleration), center, radius, 0.0, limit)


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    if abs(a) <= 1e-12:
        return () if abs(b) <= 1e-12 else (-c / b,)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return ()
    root = sqrt(max(0.0, discriminant))
    return ((-b - root) / (2 * a), (-b + root) / (2 * a))


def _first_line_time(start: Vector, velocity: Vector, acceleration: Vector, line: LineObstacle, limit: float, endpoint_margin: float) -> float | None:
    """Return a finite-segment crossing after shortening both endpoints."""
    start_point, end_point = line.start, line.end
    tangent = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    length = hypot(*tangent)
    if length <= 2 * endpoint_margin or length == 0:
        return None
    unit = (tangent[0] / length, tangent[1] / length)
    a = (start_point[0] + unit[0] * endpoint_margin, start_point[1] + unit[1] * endpoint_margin)
    b = (end_point[0] - unit[0] * endpoint_margin, end_point[1] - unit[1] * endpoint_margin)
    tangent = (b[0] - a[0], b[1] - a[1])
    length_squared = tangent[0] ** 2 + tangent[1] ** 2
    cross = lambda point: tangent[0] * point[1] - tangent[1] * point[0]
    roots = _quadratic_roots(
        0.5 * cross(acceleration), cross(velocity), cross((start[0] - a[0], start[1] - a[1])),
    )
    valid: list[float] = []
    for time in roots:
        if not EVENT_EPSILON < time <= limit + EVENT_EPSILON:
            continue
        point = _position(start, velocity, acceleration, time)
        fraction = ((point[0] - a[0]) * tangent[0] + (point[1] - a[1]) * tangent[1]) / length_squared
        if 0 <= fraction <= 1:
            valid.append(time)
    return min(valid) if valid else None


def _reflection_normal(velocity: Vector, line_or_circle: CircleObstacle | LineObstacle, point: Vector) -> Vector:
    if isinstance(line_or_circle, CircleObstacle):
        normal = (point[0] - line_or_circle.center[0], point[1] - line_or_circle.center[1])
    else:
        tangent = (line_or_circle.end[0] - line_or_circle.start[0], line_or_circle.end[1] - line_or_circle.start[1])
        normal = (-tangent[1], tangent[0])
    length = hypot(*normal)
    if length == 0:
        raise ValueError("obstacle normal must not be zero")
    normal = (normal[0] / length, normal[1] / length)
    return normal if velocity[0] * normal[0] + velocity[1] * normal[1] <= 0 else (-normal[0], -normal[1])


def _reflect(velocity: Vector, normal: Vector) -> Vector:
    projection = velocity[0] * normal[0] + velocity[1] * normal[1]
    return velocity[0] - 2 * projection * normal[0], velocity[1] - 2 * projection * normal[1]


def _first_event(start: Vector, velocity: Vector, acceleration: Vector, world: World, limit: float, target: Vector | None = None, target_radius: float = 0.0) -> tuple[str, float, object] | None:
    candidates: list[tuple[float, str, object]] = []
    for index, circle in enumerate(world.circles):
        time = _first_circle_time(start, velocity, acceleration, circle.center, circle.radius, limit)
        if time is not None:
            candidates.append((time, "obstacle", ("circle", index, circle)))
    for index, line in enumerate(world.lines):
        scale = (world.image_width or 1920) / 1920
        time = _first_line_time(start, velocity, acceleration, line, limit, LINE_ENDPOINT_MARGIN_AT_REFERENCE * scale)
        if time is not None:
            candidates.append((time, "obstacle", ("line", index, line)))
    for pair in world.portal_pairs:
        for portal, partner in ((pair.orange, pair.blue), (pair.blue, pair.orange)):
            time = _first_circle_time(start, velocity, acceleration, portal.center, portal.radius, limit)
            if time is not None:
                candidates.append((time, "portal", (portal, partner)))
    if target is not None:
        time = _first_circle_time(start, velocity, acceleration, target, target_radius, limit)
        if time is not None:
            candidates.append((time, "target", target))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    first = candidates[0]
    if len(candidates) > 1 and abs(candidates[1][0] - first[0]) <= EVENT_EPSILON:
        return None
    return first[1], first[0], first[2]


def advance_through_world(start: Vector, velocity: Vector, acceleration: Vector, world: World, max_time: float, target: Vector | None = None, target_radius: float = 0.0, max_reflections: int = 0) -> Replay:
    """Replay the earliest world event on each trajectory segment until termination."""
    remaining = max_time
    point, current_velocity = start, velocity
    events: list[TrajectoryEvent] = []
    while len(events) < MAX_EVENTS and remaining > EVENT_EPSILON:
        event = _first_event(point, current_velocity, acceleration, world, remaining, target, target_radius)
        if event is None:
            return Replay(tuple(events), "out_of_bounds")
        kind, time, payload = event
        contact = _position(point, current_velocity, acceleration, time)
        velocity_at_contact = _velocity(current_velocity, acceleration, time)
        remaining -= time
        if kind == "obstacle":
            obstacle_kind, obstacle_index, obstacle = payload
            assert obstacle_kind in {"circle", "line"}
            assert isinstance(obstacle_index, int)
            assert isinstance(obstacle, (CircleObstacle, LineObstacle))
            if len([event for event in events if event.kind == "reflection"]) >= max_reflections:
                events.append(TrajectoryEvent("obstacle", time, contact, obstacle_kind=obstacle_kind, obstacle_index=obstacle_index))
                return Replay(tuple(events), "obstacle")
            normal = _reflection_normal(velocity_at_contact, obstacle, contact)
            reflected = _reflect(velocity_at_contact, normal)
            events.append(TrajectoryEvent("reflection", time, contact, velocity_after=reflected, obstacle_kind=obstacle_kind, obstacle_index=obstacle_index))
            point = _position(contact, reflected, acceleration, PORTAL_EXIT_EPSILON)
            current_velocity = _velocity(reflected, acceleration, PORTAL_EXIT_EPSILON)
            remaining -= PORTAL_EXIT_EPSILON
            continue
        if kind == "target":
            events.append(TrajectoryEvent("target", time, contact))
            return Replay(tuple(events), "target")
        portal, partner = payload
        assert isinstance(portal, Portal) and isinstance(partner, Portal)
        offset = (contact[0] - portal.center[0], contact[1] - portal.center[1])
        exit_point = (partner.center[0] + offset[0], partner.center[1] + offset[1])
        events.append(TrajectoryEvent("portal", time, contact, exit_point, velocity_at_contact))
        point = _position(exit_point, velocity_at_contact, acceleration, PORTAL_EXIT_EPSILON)
        current_velocity = _velocity(velocity_at_contact, acceleration, PORTAL_EXIT_EPSILON)
        remaining -= PORTAL_EXIT_EPSILON
    if len(events) >= MAX_EVENTS:
        events.append(TrajectoryEvent("event_limit", 0.0, point))
        return Replay(tuple(events), "event_limit")
    return Replay(tuple(events), "out_of_bounds")
