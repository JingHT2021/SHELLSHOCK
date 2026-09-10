"""Conservative, replay-free Layer-A checks for waypoint and portal paths."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    diagnostics: dict = field(default_factory=dict)


def acceleration_aligned_frame(acceleration: Point):
    """Return axes whose negative local Y direction follows acceleration."""
    magnitude = hypot(*acceleration)
    if magnitude <= 1e-12:
        return (1.0, 0.0), (0.0, -1.0), (0.0, 0.0)
    ax, ay = acceleration
    x_axis = (ay / magnitude, -ax / magnitude)
    y_axis = (-ax / magnitude, -ay / magnitude)
    return x_axis, y_axis, (0.0, -magnitude)


def _to_local(point: Point, origin: Point, x_axis: Point, y_axis: Point) -> Point:
    delta = (point[0] - origin[0], point[1] - origin[1])
    return (delta[0] * x_axis[0] + delta[1] * x_axis[1],
            delta[0] * y_axis[0] + delta[1] * y_axis[1])


def _to_world(point: Point, origin: Point, x_axis: Point, y_axis: Point) -> Point:
    return (origin[0] + point[0] * x_axis[0] + point[1] * y_axis[0],
            origin[1] + point[0] * x_axis[1] + point[1] * y_axis[1])


def _local_vector_to_world(vector: Point, x_axis: Point, y_axis: Point) -> Point:
    return (vector[0] * x_axis[0] + vector[1] * y_axis[0],
            vector[0] * x_axis[1] + vector[1] * y_axis[1])


def _world_vector_to_local(vector: Point, x_axis: Point, y_axis: Point) -> Point:
    return (vector[0] * x_axis[0] + vector[1] * x_axis[1],
            vector[0] * y_axis[0] + vector[1] * y_axis[1])


def portal_path_possible(source: Point, target: Point, hops: Iterable[PortalHop], acceleration: Point) -> CoarsePathResult:
    """Check portal-only paths in a force-aligned frame.

    Portal transmission preserves the possible velocity signs. Four local
    quadrant directions are used because Layer A must not assume a 45-degree
    launch direction.
"""
    x_axis, y_axis, local_acceleration = acceleration_aligned_frame(acceleration)
    origin = (0.0, 0.0)
    local_hops = tuple(PortalHop(
        hop.portal_id,
        _to_local(hop.entry, origin, x_axis, y_axis),
        _to_local(hop.exit, origin, x_axis, y_axis),
    ) for hop in hops)
    segments: list[dict] = []
    result, _ = _walk_path(
        _to_local(source, origin, x_axis, y_axis),
        _to_local(target, origin, x_axis, y_axis),
        local_hops,
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
        local_acceleration,
        trace=segments,
    )
    return CoarsePathResult(
        result.valid,
        result.reason,
        tuple(_to_world(point, origin, x_axis, y_axis) for point in result.points),
        {"acceleration_frame": {"x_axis": x_axis, "y_axis": y_axis,
                                 "local_acceleration": local_acceleration},
         "segments": segments},
    )


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


def _walk_path(source: Point, target: Point, hops: Iterable[PortalHop], velocity_directions: Iterable[VelocityDirection], acceleration: Point, *, trace=None):
    """Return the result and the velocity directions still possible after every leg."""
    directions = tuple(velocity_directions)
    points: list[Point] = [source]
    current = source
    hops = tuple(hops)
    for index, hop in enumerate(hops):
        directions_before = directions
        directions = _directions_that_can_reach(current, hop.entry, directions, acceleration)
        if trace is not None:
            trace.append({"name": "current_to_portal_entry" if index == 0 else "portal_exit_to_portal_entry",
                          "from": current, "to": hop.entry, "directions_before": directions_before,
                          "directions_after": directions, "status": "PASS" if directions else "FAIL",
                          "reason": None if directions else "A_SEGMENT_DIRECTION"})
        if not directions:
            return CoarsePathResult(False, "A_SEGMENT_DIRECTION", tuple(points + [hop.entry])), directions
        points.extend((hop.entry, hop.exit))
        current = hop.exit
    directions_before = directions
    directions = _directions_that_can_reach(current, target, directions, acceleration)
    if trace is not None:
        trace.append({"name": "current_to_target" if not hops else "portal_exit_to_target",
                      "from": current, "to": target, "directions_before": directions_before,
                      "directions_after": directions, "status": "PASS" if directions else "FAIL",
                      "reason": None if directions else "A_SEGMENT_DIRECTION"})
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
    """Check portals in the acceleration frame and reflect in the surface frame.

    Portal legs are evaluated with ``-Y`` aligned to resultant acceleration.
    Only the instantaneous reflection converts velocity components to the
    reflector frame, flips the normal component, and converts them back.
    """
    del velocity_directions
    acceleration_x, acceleration_y, local_acceleration = acceleration_aligned_frame(acceleration)
    tangent_length = hypot(*tangent)
    normal_length = hypot(*normal)
    length = normal_length
    if tangent_length <= 1e-9:
        return CoarsePathResult(False, "A_REFLECTION_TANGENT", (source, contact),
                                 {"reflection_frame": "invalid_tangent"})
    if length <= 1e-9:
        return CoarsePathResult(False, "A_REFLECTION_NORMAL", (source, contact),
                                 {"reflection_frame": "invalid_normal"})
    tx, ty = tangent[0] / tangent_length, tangent[1] / tangent_length
    nx, ny = normal[0] / length, normal[1] / length
    before_hops = tuple(before_hops)
    after_hops = tuple(after_hops)
    local = lambda point: _to_local(point, (0.0, 0.0), acceleration_x, acceleration_y)
    local_hops_before = tuple(PortalHop(hop.portal_id, local(hop.entry), local(hop.exit)) for hop in before_hops)
    local_hops_after = tuple(PortalHop(hop.portal_id, local(hop.entry), local(hop.exit)) for hop in after_hops)
    local_source = local(source)
    local_contact = local(contact)
    local_target = local(target)
    reflection_source = _to_local(source, contact, (tx, ty), (nx, ny))
    reflection_target = _to_local(target, contact, (tx, ty), (nx, ny))
    directions = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    before_trace: list[dict] = []
    after_trace: list[dict] = []
    before, directions = _walk_path(local_source, local_contact, local_hops_before, directions,
                                     local_acceleration, trace=before_trace)
    diagnostics = {"acceleration_frame": {"x_axis": acceleration_x, "y_axis": acceleration_y,
                                            "local_acceleration": local_acceleration,
                                            "local_source": local_source, "local_contact": local_contact,
                                            "local_target": local_target},
                   "reflection_frame": {"origin": contact, "tangent": (tx, ty), "normal": (nx, ny)},
                   "before_segments": before_trace, "directions_before_reflection": directions}
    diagnostics["reflection_frame"].update({"local_source": reflection_source, "local_target": reflection_target})
    same_reflection_side = reflection_source[1] * reflection_target[1] >= -1e-9
    opposite_normal_sides = reflection_source[0] * reflection_target[0] <= 1e-9
    if not before_hops and not after_hops and same_reflection_side and opposite_normal_sides:
        diagnostics["quadrant_direct"] = {"same_reflection_side": True, "opposite_normal_sides": True}
        diagnostics["before_segments"] = [{"name": "current_to_reflection", "from": source, "to": contact,
                                             "status": "PASS", "reason": "A_REFLECTION_QUADRANT_DIRECT"}]
        diagnostics["after_segments"] = [{"name": "reflection_to_target", "from": contact, "to": target,
                                            "status": "PASS", "reason": "A_REFLECTION_QUADRANT_DIRECT"}]
        diagnostics["directions_before_reflection"] = ()
        diagnostics["directions_after_reflection"] = ()
        return CoarsePathResult(True, None, (source, contact, target), diagnostics)
    if not before_hops and not after_hops:
        diagnostics["quadrant_direct"] = {"same_reflection_side": same_reflection_side,
                                            "opposite_normal_sides": opposite_normal_sides}
        diagnostics["before_segments"] = [{"name": "current_to_reflection", "from": source, "to": contact,
                                             "status": "FAIL", "reason": "A_REFLECTION_QUADRANT_UNREACHABLE"}]
        diagnostics["after_segments"] = [{"name": "reflection_to_target", "from": contact, "to": target,
                                            "status": "FAIL", "reason": "A_REFLECTION_QUADRANT_UNREACHABLE"}]
        return CoarsePathResult(False, "A_REFLECTION_QUADRANT_UNREACHABLE",
                                (source, contact, target), diagnostics)
    if not before.valid:
        points = tuple(_to_world(point, (0.0, 0.0), acceleration_x, acceleration_y) for point in before.points)
        return CoarsePathResult(False, before.reason, points, diagnostics)
    reflected = []
    reflection_directions = []
    for velocity in directions:
        world_velocity = _local_vector_to_world(velocity, acceleration_x, acceleration_y)
        tangent_velocity = world_velocity[0] * tx + world_velocity[1] * ty
        normal_velocity = world_velocity[0] * nx + world_velocity[1] * ny
        reflected_world = (tangent_velocity * tx - normal_velocity * nx,
                           tangent_velocity * ty - normal_velocity * ny)
        reflected_local = _world_vector_to_local(reflected_world, acceleration_x, acceleration_y)
        reflected.append(reflected_local)
        reflection_directions.append({"before_acceleration": velocity,
                                      "before_reflection": (tangent_velocity, normal_velocity),
                                      "after_reflection": reflected_local})
    reflected = tuple(reflected)
    diagnostics["reflection_directions"] = reflection_directions
    diagnostics["directions_after_reflection"] = reflected
    after = _walk_path(local_contact, local_target, local_hops_after, reflected,
                       local_acceleration, trace=after_trace)[0]
    diagnostics["after_segments"] = after_trace
    points = tuple(_to_world(point, (0.0, 0.0), acceleration_x, acceleration_y)
                   for point in before.points + after.points[1:])
    return CoarsePathResult(after.valid, after.reason, points, diagnostics)
