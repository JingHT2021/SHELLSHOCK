"""Event-exact replay of explicit portals before/after one planned reflection."""
from __future__ import annotations

from math import atan2, degrees, hypot, isfinite

import numpy as np

from .ballistics import SPEED_PER_POWER_AT_REFERENCE, _scale
from .collision import (
    closest_approach_to_target, find_first_collision, trajectory_clearance,
    trajectory_position, trajectory_velocity,
)
from .portal_replay import _entry_time
from .solver_config import (
    COLLISION_TIME_EPS, EQUATION_RESIDUAL_TOL, LINE_ENDPOINT_MARGIN_AT_REFERENCE,
    MAX_FLIGHT_TIME, REFLECTION_MIN_INCIDENCE, TARGET_ACCEPT_RADIUS_AT_REFERENCE,
    portal_avoid_radius, portal_trigger_radius,
)
from .world_geometry import CircleObstacle, World


def replay_combined_shot(source, velocity, acceleration, world, target, image_width,
                         family, pre_portals=(), post_portals=(), *,
                         max_time=MAX_FLIGHT_TIME, expected_contact=None,
                         expected_time=None, compute_clearance=True,
                         require_exact_target=False, target_accept_radius=None):
    """Verify a route using exact roots and only its explicitly requested events.

    Portal translation preserves entry offset and velocity. Only the next
    portal in the current (pre/post reflection) stage gets its trigger radius;
    every other portal uses its avoidance radius. A just-emerged exit permits
    its initial interior departure, but `_entry_time` still detects reentry.
    `expected_time` is absolute bounce time including all pre-portal travel.
    Exact mode verifies the final endpoint at `max_time`, for continuous shots.
    """
    scale = _scale(image_width)
    pre_portals, post_portals = tuple(pre_portals), tuple(post_portals)
    if len(pre_portals)+len(post_portals) > 2:
        raise ValueError('combined reflection supports at most two portal transmissions')
    portals = {
        f'{i}:{color}': (portal, partner, f'{i}:{other}')
        for i, pair in enumerate(world.portal_pairs)
        for color, portal, other, partner in (
            ('orange', pair.orange, 'blue', pair.blue),
            ('blue', pair.blue, 'orange', pair.orange),
        )
    }
    if any(portal not in portals for portal in pre_portals+post_portals):
        raise ValueError('unknown planned portal')
    if max_time <= 0 or family.kind not in {'line', 'circle'}:
        return None
    obstacles = world.lines if family.kind == 'line' else world.circles
    if not 0 <= family.index < len(obstacles):
        return None
    initial_velocity = tuple(velocity)
    point, current = tuple(source), tuple(velocity)
    elapsed, remaining = 0., float(max_time)
    pre_index = post_index = 0
    bounced = False
    exit_id = None
    events, sequence, segments = [], [], []
    bounce = None

    def clearance_world(wanted):
        # The planned reflecting surface is omitted from tie-break clearance,
        # while it remains fully present in every collision query.
        circles = [c for i, c in enumerate(world.circles) if not (family.kind == 'circle' and i == family.index)]
        lines = [line for i, line in enumerate(world.lines) if not (family.kind == 'line' and i == family.index)]
        circles.extend(CircleObstacle(portal.center, portal_avoid_radius(portal, scale))
                       for portal_id, (portal, _, _) in portals.items()
                       if portal_id not in {wanted, exit_id})
        return World(circles=circles, lines=lines)

    def add_segment(duration, wanted):
        if compute_clearance:
            segments.append((point, current, duration, clearance_world(wanted)))

    # Every nonterminal iteration consumes exactly one requested portal or the
    # sole reflection. This bound is route-derived, not a simulation step cap.
    for _ in range(len(pre_portals)+len(post_portals)+2):
        if not bounced:
            wanted = pre_portals[pre_index] if pre_index < len(pre_portals) else None
        else:
            wanted = post_portals[post_index] if post_index < len(post_portals) else None
        obstacle = find_first_collision(point, current, acceleration, world, remaining, avoid_portals=False)
        candidates = [] if obstacle is None else [(obstacle.time, 'obstacle', obstacle)]
        for portal_id, (portal, _, _) in portals.items():
            planned = portal_id == wanted
            radius = portal_trigger_radius(portal) if planned else portal_avoid_radius(portal, scale)
            time = _entry_time(point, current, acceleration, portal, radius, remaining,
                               allow_inside=portal_id == exit_id)
            if time is not None:
                candidates.append((time, 'portal' if planned else 'unplanned-portal', portal_id))
        candidates.sort(key=lambda item: item[0])

        if bounced and wanted is None:
            # Find the genuine closest target point over the whole remaining
            # segment; an earlier blocker cannot truncate this into a hit.
            if require_exact_target:
                target_time = remaining
                closest = trajectory_position(point, current, acceleration, target_time)
                miss = hypot(closest[0]-target[0], closest[1]-target[1])
                tolerance = EQUATION_RESIDUAL_TOL*max(1., scale)
            else:
                miss, target_time, closest = closest_approach_to_target(point, current, acceleration, target, remaining)
                tolerance = ((TARGET_ACCEPT_RADIUS_AT_REFERENCE if target_accept_radius is None
                              else target_accept_radius) * scale)
            if miss > tolerance or target_time <= COLLISION_TIME_EPS:
                return None
            if candidates and candidates[0][0] <= target_time+COLLISION_TIME_EPS:
                return None
            add_segment(target_time, wanted)
            clearance = min((trajectory_clearance(p, v, acceleration, safe, t)
                             for p, v, t, safe in segments), default=1e12*scale)
            if not isfinite(clearance):
                clearance = 1e12*scale
            speed = hypot(*initial_velocity)
            return {
                'status': 'reachable', 'mode': 'reflection',
                'direction': 'right' if initial_velocity[0] >= 0 else 'left',
                'angle_degrees': degrees(atan2(-initial_velocity[1], abs(initial_velocity[0]))),
                'power': speed/(SPEED_PER_POWER_AT_REFERENCE*scale),
                'portal_count': len(sequence), 'portal_sequence': tuple(sequence),
                'reflection_count': 1, 'events': events+['target'],
                'miss_distance': float(miss), 'clearance': float(clearance),
                'target_accept_radius': float(tolerance),
                'flight_time_seconds': elapsed+target_time,
                'closest_target_point': tuple(float(x) for x in closest),
                'portal_radii': [{'id': portal_id, 'visual': portals[portal_id][0].radius,
                                  'trigger': portal_trigger_radius(portals[portal_id][0]),
                                  'avoid': portal_avoid_radius(portals[portal_id][0], scale)}
                                 for portal_id in sequence],
                **bounce,
            }
        if not candidates:
            return None
        time, kind, item = candidates[0]
        if time > remaining or (len(candidates) > 1 and candidates[1][0]-time <= COLLISION_TIME_EPS):
            return None
        if kind == 'portal':
            add_segment(time, wanted)
            portal, partner, exit_id = portals[item]
            contact = trajectory_position(point, current, acceleration, time)
            current = trajectory_velocity(current, acceleration, time)
            point = tuple(partner.center[i]+contact[i]-portal.center[i] for i in (0, 1))
            sequence.append(item)
            events.append('portal')
            if bounced:
                post_index += 1
            else:
                pre_index += 1
        elif kind == 'obstacle' and not bounced and wanted is None:
            if (item.obstacle_kind, item.obstacle_index) != (family.kind, family.index):
                return None
            contact = np.asarray(item.point, dtype=float)
            if expected_contact is not None and np.linalg.norm(contact-expected_contact) > EQUATION_RESIDUAL_TOL*max(1., scale):
                return None
            if expected_time is not None and abs(elapsed+time-expected_time) > COLLISION_TIME_EPS:
                return None
            incoming = np.asarray(trajectory_velocity(current, acceleration, time))
            endpoint_margin = 1e12*scale
            side = 'BOTH'
            if family.kind == 'line':
                line = world.lines[family.index]
                delta = np.asarray(line.end)-line.start
                length = float(np.linalg.norm(delta))
                if length <= 0:
                    return None
                along = float(np.dot(contact-line.start, delta))/length
                endpoint_margin = min(along, length-along)
                if endpoint_margin < LINE_ENDPOINT_MARGIN_AT_REFERENCE*scale-1e-7:
                    return None
                normal = np.array((-delta[1], delta[0]))/length
            else:
                circle = world.circles[family.index]
                normal = contact-circle.center
                norm = float(np.linalg.norm(normal))
                if norm <= 0:
                    return None
                normal /= norm
                side = 'INNER' if np.dot(incoming, normal) > 0 else 'OUTER'
                if family.side not in {'BOTH', side}:
                    return None
            incidence = abs(float(np.dot(incoming, normal)))/max(float(np.linalg.norm(incoming)), 1e-12)
            if incidence < REFLECTION_MIN_INCIDENCE:
                return None
            outgoing = incoming-2*np.dot(incoming, normal)*normal
            add_segment(time, wanted)
            point, current = tuple(contact), tuple(outgoing)
            bounced = True
            events.append('reflection')
            bounce = {
                'reflection_point': tuple(float(x) for x in contact),
                'reflection_obstacle': {'kind': family.kind, 'index': family.index},
                'reflection_side': side, 'incidence': incidence,
                'endpoint_margin': endpoint_margin,
                'actual_v_before': tuple(float(x) for x in incoming),
                'actual_v_after': tuple(float(x) for x in outgoing),
                'reflection_time_seconds': elapsed+time,
            }
        else:
            return None
        elapsed += time
        remaining -= time
        if remaining <= COLLISION_TIME_EPS:
            return None
    return None
