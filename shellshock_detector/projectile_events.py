"""Event-ordered projectile replay, including paired wormhole teleportation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Literal

from .world_geometry import CircleObstacle, LineObstacle, Portal, PortalPair, World


Vector = tuple[float, float]
MAX_EVENTS = 32
EVENT_EPSILON = 1e-5
PORTAL_EXIT_EPSILON = 1e-4


@dataclass(frozen=True)
class TrajectoryEvent:
    kind: Literal["portal", "obstacle", "target", "out_of_bounds", "event_limit"]
    time: float
    point: Vector
    exit_point: Vector | None = None
    velocity_after: Vector | None = None


@dataclass(frozen=True)
class Replay:
    events: tuple[TrajectoryEvent, ...]
    terminal_kind: str

    @property
    def portal_count(self) -> int:
        return sum(event.kind == "portal" for event in self.events)


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


def _first_event(start: Vector, velocity: Vector, acceleration: Vector, world: World, limit: float, target: Vector | None = None, target_radius: float = 0.0) -> tuple[str, float, object] | None:
    candidates: list[tuple[float, str, object]] = []
    for circle in world.circles:
        time = _first_circle_time(start, velocity, acceleration, circle.center, circle.radius, limit)
        if time is not None:
            candidates.append((time, "obstacle", circle))
    for line in world.lines:
        # Conservative finite-segment sampling for line obstacles; exact circle contacts
        # remain the common case and both are ordered with portal contacts.
        midpoint = ((line.start[0] + line.end[0]) / 2, (line.start[1] + line.end[1]) / 2)
        radius = max(1.0, hypot(line.end[0] - line.start[0], line.end[1] - line.start[1]) / 2)
        time = _first_circle_time(start, velocity, acceleration, midpoint, radius, limit)
        if time is not None:
            candidates.append((time, "obstacle", line))
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


def advance_through_world(start: Vector, velocity: Vector, acceleration: Vector, world: World, max_time: float, target: Vector | None = None, target_radius: float = 0.0) -> Replay:
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
            events.append(TrajectoryEvent("obstacle", time, contact))
            return Replay(tuple(events), "obstacle")
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
