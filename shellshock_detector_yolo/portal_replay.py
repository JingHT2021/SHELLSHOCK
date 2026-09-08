"""Strict planned-portal replay shared by normal and wormhole solvers."""
from dataclasses import dataclass
from math import hypot, inf

from .collision import (closest_approach_to_target, find_first_collision,
                        parabola_circle_roots, trajectory_position,
                        trajectory_velocity, trajectory_clearance)
from .solver_config import (COLLISION_TIME_EPS, MAX_FLIGHT_TIME,
                            TARGET_ACCEPT_RADIUS_AT_REFERENCE,
                            portal_avoid_radius, portal_trigger_radius)
from .world_geometry import CircleObstacle, World


@dataclass(frozen=True)
class PortalReplay:
    valid: bool
    miss_distance: float = inf
    clearance: float = inf
    time: float = 0.
    portal_sequence: tuple[str, ...] = ()
    invalid_reason: str | None = None


def _entry_time(start, velocity, acceleration, portal, radius, limit, allow_inside=False):
    distance = hypot(start[0]-portal.center[0], start[1]-portal.center[1])
    if distance <= radius + 1e-7 and not allow_inside:
        return 0.
    for t in parabola_circle_roots(start, velocity, acceleration, portal.center, radius, limit):
        p = trajectory_position(start, velocity, acceleration, t)
        v = trajectory_velocity(velocity, acceleration, t)
        if (p[0]-portal.center[0])*v[0] + (p[1]-portal.center[1])*v[1] <= 1e-7:
            return t
    return None


def replay_portal_shot(source, velocity, acceleration, world, target, image_width,
                       planned_sequence=(), *, max_time=MAX_FLIGHT_TIME):
    """Stop at true closest approach on the final unobstructed segment.

Only the just-emerged exit may start inside its avoidance zone; later re-entry
is still detected. A target before the requested portal sequence is not a hit.
"""
    portals = {f'{i}:{color}': (p, partner, f'{i}:{other}')
               for i, pair in enumerate(world.portal_pairs)
               for color, p, other, partner in (('orange', pair.orange, 'blue', pair.blue), ('blue', pair.blue, 'orange', pair.orange))}
    if any(p not in portals for p in planned_sequence):
        raise ValueError('unknown planned portal')
    scale = image_width / 1920
    point, current = source, velocity
    remaining, elapsed = max_time, 0.
    sequence = []
    exit_id = None
    segments = []
    for _ in range(len(planned_sequence) + 1):
        wanted = planned_sequence[len(sequence)] if len(sequence) < len(planned_sequence) else None
        obstacle = find_first_collision(point, current, acceleration, world, remaining, avoid_portals=False)
        events = [] if obstacle is None else [(obstacle.time, 'obstacle', None)]
        avoided = []
        for portal_id, (portal, _, _) in portals.items():
            planned = portal_id == wanted
            radius = portal_trigger_radius(portal) if planned else portal_avoid_radius(portal, scale)
            time = _entry_time(point, current, acceleration, portal, radius, remaining, allow_inside=portal_id == exit_id)
            if time is not None:
                events.append((time, 'portal' if planned else 'unplanned-portal', portal_id))
            if not planned and portal_id != exit_id:
                avoided.append(CircleObstacle(portal.center, radius))
        events.sort(key=lambda e: e[0])
        stop = min(remaining, events[0][0]) if events else remaining
        safe_world = World(circles=(*world.circles, *avoided), lines=world.lines)
        if wanted is None:
            miss, time, _ = closest_approach_to_target(point, current, acceleration, target, stop)
            if miss <= TARGET_ACCEPT_RADIUS_AT_REFERENCE * scale and (not events or time < stop - COLLISION_TIME_EPS):
                segments.append((point, current, time, safe_world))
                clearance = min(trajectory_clearance(p, v, acceleration, w, t) for p, v, t, w in segments)
                return PortalReplay(True, miss, clearance, elapsed + time, tuple(sequence))
        if not events:
            return PortalReplay(False, portal_sequence=tuple(sequence), invalid_reason='missing-portal' if wanted else 'target-miss')
        if len(events) > 1 and abs(events[0][0]-events[1][0]) <= COLLISION_TIME_EPS:
            return PortalReplay(False, portal_sequence=tuple(sequence), invalid_reason='ambiguous-collision')
        time, kind, portal_id = events[0]
        if kind != 'portal':
            return PortalReplay(False, portal_sequence=tuple(sequence), invalid_reason=kind)
        segments.append((point, current, time, safe_world))
        portal, partner, exit_id = portals[portal_id]
        contact = trajectory_position(point, current, acceleration, time)
        current = trajectory_velocity(current, acceleration, time)
        point = (partner.center[0]+contact[0]-portal.center[0], partner.center[1]+contact[1]-portal.center[1])
        sequence.append(portal_id)
        elapsed += time
        remaining -= time
        if remaining <= COLLISION_TIME_EPS:
            break
    return PortalReplay(False, portal_sequence=tuple(sequence), invalid_reason='time-limit')
