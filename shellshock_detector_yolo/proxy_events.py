"""Cheap analytic event ordering for Layer-B reflection proxies.

The functions in this module never step frames.  They reuse the quadratic
segment and quartic-circle roots from :mod:`collision` and advance only at a
small, planned set of portal/reflection events.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, inf

import numpy as np

from .collision import (closest_approach_to_target, parabola_circle_roots,
                        parabola_segment_roots, trajectory_position,
                        trajectory_velocity, trajectory_aabb)
from .portal_replay import _entry_time
from .reflection_routes import portal_entries
from .solver_config import COLLISION_TIME_EPS, portal_avoid_radius, portal_trigger_radius


@dataclass(frozen=True)
class ProxyEvent:
    event_type: str
    object_id: str
    time: float
    planned: bool
    point: tuple[float, float]
    margin: float = 0.0


@dataclass(frozen=True)
class ProxyEventValidation:
    valid: bool
    events: tuple[ProxyEvent, ...] = ()
    invalid_reason: str | None = None
    planned_portal_depth: float = 0.0
    min_unplanned_clearance: float = inf


def _first_root(roots):
    return min(roots) if roots else None


def collect_segment_events(start, velocity, acceleration, duration, world, image_width,
                           *, planned_portal=None, planned_reflector=None,
                           ignored_portal=None):
    """Return all possible physical events in time order for one free leg."""
    if duration < 0:
        return ()
    events: list[ProxyEvent] = []
    start = tuple(float(x) for x in start)
    velocity = tuple(float(x) for x in velocity)
    acceleration = tuple(float(x) for x in acceleration)

    for index, circle in enumerate(world.circles):
        time = _first_root(parabola_circle_roots(start, velocity, acceleration,
                                                 circle.center, circle.radius, duration))
        if time is not None:
            planned = planned_reflector == ("circle", index)
            events.append(ProxyEvent("circle_collision", str(index), time, planned,
                                     trajectory_position(start, velocity, acceleration, time)))

    for index, line in enumerate(world.lines):
        time = _first_root(parabola_segment_roots(start, velocity, acceleration, line, duration))
        if time is not None:
            planned = planned_reflector == ("line", index)
            events.append(ProxyEvent("line_collision", str(index), time, planned,
                                     trajectory_position(start, velocity, acceleration, time)))

    scale = image_width / 1920
    for portal_id, (portal, _, _) in portal_entries(world).items():
        planned = portal_id == planned_portal
        radius = portal_trigger_radius(portal) if planned else portal_avoid_radius(portal, scale)
        time = _entry_time(start, velocity, acceleration, portal, radius, duration,
                           allow_inside=portal_id == ignored_portal)
        if time is None:
            continue
        miss, _, _ = closest_approach_to_target(start, velocity, acceleration, portal.center, duration)
        margin = 1.0 - (miss / max(radius, 1e-9)) ** 2
        events.append(ProxyEvent("portal", portal_id, time, planned,
                                 trajectory_position(start, velocity, acceleration, time), margin))

    return tuple(sorted(events, key=lambda event: (event.time, not event.planned,
                                                    event.event_type, event.object_id)))


def _failure_for(event, planned_reflector=None, *, portal_reason="B_WRONG_FIRST_PORTAL"):
    if event.event_type == "portal":
        return portal_reason
    kind = "line" if event.event_type == "line_collision" else "circle"
    if planned_reflector == (kind, int(event.object_id)):
        return f"B_WRONG_PLANNED_{kind.upper()}_COLLISION_POINT"
    return f"B_UNPLANNED_{kind.upper()}_COLLISION"


def _with_elapsed(event, elapsed):
    return replace(event, time=elapsed + event.time)


def _segment_unplanned_clearance(start, velocity, acceleration, duration, world, image_width,
                                 planned_reflector, *, planned_portal=None, ignored_portal=None):
    """Return the normalized squared-distance margin to unplanned circles."""
    bounds = trajectory_aabb(start, velocity, acceleration, duration)
    circles = []
    def add(center, radius):
        dx = max(bounds[0] - center[0], 0.0, center[0] - bounds[2])
        dy = max(bounds[1] - center[1], 0.0, center[1] - bounds[3])
        lower = (dx * dx + dy * dy - radius * radius) / max(radius * radius, 1e-9)
        circles.append((lower, center, radius))
    for index, circle in enumerate(world.circles):
        if planned_reflector == ("circle", index):
            continue
        add(circle.center, circle.radius)
    scale = image_width / 1920
    for portal_id, (portal, _, _) in portal_entries(world).items():
        if portal_id in {planned_portal, ignored_portal}:
            continue
        radius = portal_avoid_radius(portal, scale)
        add(portal.center, radius)
    best = inf
    for lower, center, radius in sorted(circles, key=lambda item: item[0]):
        if lower >= best:
            continue
        miss, _, _ = closest_approach_to_target(start, velocity, acceleration, center, duration)
        best = min(best, (miss * miss - radius * radius) / max(radius * radius, 1e-9))
    return best


def validate_proxy_event_sequence(source, target, solution, family, route, world,
                                  acceleration, image_width):
    """Validate the planned portal/reflection order without a frame replay."""
    entries = portal_entries(world)
    planned_reflector = (family.kind, family.index)
    if any(portal_id not in entries for portal_id in (*route.before, *route.after)):
        return ProxyEventValidation(False, invalid_reason="B_EVENT_ORDER")

    recorded: list[ProxyEvent] = []
    portal_depths: list[float] = []
    min_clearance = inf
    elapsed = 0.0

    def walk_portals(point, current, duration, sequence, ignored):
        nonlocal elapsed, min_clearance
        for portal_id in sequence:
            possible = collect_segment_events(point, current, acceleration, duration, world, image_width,
                                              planned_portal=portal_id, ignored_portal=ignored)
            wanted = next((event for event in possible
                           if event.event_type == "portal" and event.object_id == portal_id), None)
            if wanted is None:
                return None, None, None, None, "B_PLANNED_PORTAL_MISS"
            first = possible[0]
            if first is not wanted and first.time < wanted.time - COLLISION_TIME_EPS:
                return None, None, None, None, _failure_for(first)
            simultaneous = [event for event in possible
                            if abs(event.time - wanted.time) <= COLLISION_TIME_EPS]
            if any(event is not wanted for event in simultaneous):
                return None, None, None, None, "B_EVENT_ORDER"
            min_clearance = min(min_clearance, _segment_unplanned_clearance(
                point, current, acceleration, wanted.time, world, image_width, planned_reflector,
                planned_portal=portal_id, ignored_portal=ignored,
            ))
            recorded.append(_with_elapsed(wanted, elapsed))
            portal_depths.append(max(0.0, wanted.margin))
            portal, partner, exit_id = entries[portal_id]
            contact = trajectory_position(point, current, acceleration, wanted.time)
            current = trajectory_velocity(current, acceleration, wanted.time)
            point = tuple(contact[i] + partner.center[i] - portal.center[i] for i in (0, 1))
            duration -= wanted.time
            elapsed += wanted.time
            ignored = exit_id
        return point, current, duration, ignored, None

    point, current, remaining, ignored, reason = walk_portals(
        source, solution.velocity, solution.t1, route.before, None
    )
    if reason:
        return ProxyEventValidation(False, tuple(recorded), reason,
                                    min(portal_depths, default=0.0), min_clearance)

    possible = collect_segment_events(point, current, acceleration, remaining, world, image_width,
                                      planned_reflector=planned_reflector, ignored_portal=ignored)
    min_clearance = min(min_clearance, _segment_unplanned_clearance(
        point, current, acceleration, remaining, world, image_width, planned_reflector,
        ignored_portal=ignored,
    ))
    wanted = next((event for event in possible if event.planned and
                   event.event_type == f"{family.kind}_collision"), None)
    if wanted is None:
        reason = f"B_WRONG_PLANNED_{family.kind.upper()}_COLLISION_POINT"
        return ProxyEventValidation(False, tuple(recorded), reason,
                                    min(portal_depths, default=0.0), min_clearance)
    first = possible[0]
    if first is not wanted and first.time < wanted.time - COLLISION_TIME_EPS:
        return ProxyEventValidation(False, tuple(recorded), _failure_for(
            first, planned_reflector, portal_reason="B_UNPLANNED_PORTAL"),
                                    min(portal_depths, default=0.0), min_clearance)
    time_tolerance = 1e-5 * max(1.0, solution.t1)
    if abs(wanted.time - remaining) > time_tolerance:
        return ProxyEventValidation(False, tuple(recorded),
                                    f"B_WRONG_PLANNED_{family.kind.upper()}_COLLISION_POINT",
                                    min(portal_depths, default=0.0), min_clearance)
    recorded.append(ProxyEvent("reflection", f"{family.kind}:{family.index}", elapsed + wanted.time,
                               True, wanted.point, wanted.margin))
    elapsed += remaining

    incoming = np.asarray(current, dtype=float) + np.asarray(acceleration, dtype=float) * remaining
    normal = np.asarray(solution.normal, dtype=float)
    outgoing = incoming - 2 * float(np.dot(incoming, normal)) * normal
    point, current, remaining, ignored, reason = walk_portals(
        solution.contact, tuple(outgoing), solution.t2, route.after, None
    )
    if reason:
        return ProxyEventValidation(False, tuple(recorded), reason,
                                    min(portal_depths, default=0.0), min_clearance)

    possible = collect_segment_events(point, current, acceleration, remaining, world, image_width,
                                      ignored_portal=ignored)
    min_clearance = min(min_clearance, _segment_unplanned_clearance(
        point, current, acceleration, remaining, world, image_width, planned_reflector,
        ignored_portal=ignored,
    ))
    early = next((event for event in possible if event.time < remaining - COLLISION_TIME_EPS), None)
    if early is not None:
        return ProxyEventValidation(False, tuple(recorded), _failure_for(
            early, planned_reflector, portal_reason="B_UNPLANNED_PORTAL"),
                                    min(portal_depths, default=0.0), min_clearance)
    target_point = trajectory_position(point, current, acceleration, remaining)
    target_tolerance = 1e-5 * max(1.0, hypot(target[0] - source[0], target[1] - source[1]))
    if hypot(target_point[0] - target[0], target_point[1] - target[1]) > target_tolerance:
        return ProxyEventValidation(False, tuple(recorded), "B_EQUATION_RESIDUAL",
                                    min(portal_depths, default=0.0), min_clearance)
    recorded.append(ProxyEvent("target", "target", elapsed + remaining, True, target_point))
    return ProxyEventValidation(True, tuple(recorded), planned_portal_depth=min(portal_depths, default=1.0),
                                min_unplanned_clearance=min_clearance)
