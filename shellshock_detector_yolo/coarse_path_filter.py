"""Conservative, replay-free Layer-A checks for waypoint and portal paths."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


Point = tuple[float, float]
VelocityDirection = tuple[float, float]


@dataclass(frozen=True)
class PortalHop:
    """One planned transmission represented by entry and matching exit centres."""

    portal_id: str
    entry: Point
    exit: Point


@dataclass(frozen=True)
class CoarsePathResult:
    valid: bool
    reason: str | None
    points: tuple[Point, ...]


def axis_maybe_reachable(target_offset: float, velocity_component: float, acceleration_component: float) -> bool:
    """Return false only when both motion and acceleration oppose this axis.

    This intentionally ignores magnitude and time.  Layer A must preserve every
    path that can plausibly be corrected by gravity or wind.
    """
    if abs(target_offset) <= 1e-9:
        return True
    return velocity_component * target_offset > 0 or acceleration_component * target_offset > 0


def segment_possible(current: Point, next_point: Point, velocity_directions: Iterable[VelocityDirection], acceleration: Point) -> bool:
    """Check whether at least one permitted velocity direction can reach a waypoint."""
    dx, dy = next_point[0] - current[0], next_point[1] - current[1]
    return any(
        axis_maybe_reachable(dx, velocity[0], acceleration[0])
        and axis_maybe_reachable(dy, velocity[1], acceleration[1])
        for velocity in velocity_directions
    )


def _directions_that_can_reach(current: Point, next_point: Point, directions: Iterable[VelocityDirection], acceleration: Point):
    return tuple(direction for direction in directions if segment_possible(current, next_point, (direction,), acceleration))


def _walk_path(source: Point, target: Point, hops: Iterable[PortalHop], velocity_directions: Iterable[VelocityDirection], acceleration: Point):
    """Return the result and the velocity directions still possible after every leg."""
    directions = tuple(velocity_directions)
    points: list[Point] = [source]
    current = source
    for hop in hops:
        directions = _directions_that_can_reach(current, hop.entry, directions, acceleration)
        if not directions:
            return CoarsePathResult(False, "A_SEGMENT_DIRECTION", tuple(points + [hop.entry])), directions
        points.extend((hop.entry, hop.exit))
        current = hop.exit
    directions = _directions_that_can_reach(current, target, directions, acceleration)
    if not directions:
        return CoarsePathResult(False, "A_SEGMENT_DIRECTION", tuple(points + [target])), directions
    return CoarsePathResult(True, None, tuple(points + [target])), directions


def path_possible(source: Point, target: Point, hops: Iterable[PortalHop], velocity_directions: Iterable[VelocityDirection], acceleration: Point) -> CoarsePathResult:
    """Check planned portal hops in order while preserving possible velocities."""
    result, _ = _walk_path(source, target, hops, velocity_directions, acceleration)
    return result


def reflection_path_possible(source: Point, contact: Point, target: Point,
                             before_hops: Iterable[PortalHop], after_hops: Iterable[PortalHop],
                             velocity_directions: Iterable[VelocityDirection], acceleration: Point,
                             tangent: Point, normal: Point) -> CoarsePathResult:
    """Apply a local mirror event between two ordered portal-path checks.

    The tangent is part of the public adapter contract: in this local frame the
    tangent component is retained and the normal component is reversed.  The
    equivalent world-space transform below only needs the normalized normal.
    """
    del tangent
    directions = tuple(velocity_directions)
    before, directions = _walk_path(source, contact, before_hops, directions, acceleration)
    if not before.valid:
        return before
    length = hypot(*normal)
    if length <= 1e-9:
        return CoarsePathResult(False, "A_REFLECTION_NORMAL", before.points)
    nx, ny = normal[0] / length, normal[1] / length
    reflected = tuple(
        (velocity[0] - 2 * (velocity[0] * nx + velocity[1] * ny) * nx,
         velocity[1] - 2 * (velocity[0] * nx + velocity[1] * ny) * ny)
        for velocity in directions
    )
    after = path_possible(contact, target, after_hops, reflected, acceleration)
    return CoarsePathResult(after.valid, after.reason, before.points + after.points[1:])
