"""Verify integer controls using their actual first collision and bounce normal."""
from __future__ import annotations

from math import cos, isfinite, radians, sin

import numpy as np

from .ballistics import SPEED_PER_POWER_AT_REFERENCE, _scale
from .solver_config import (
    MAX_FLIGHT_TIME, LINE_ENDPOINT_MARGIN_AT_REFERENCE, REFLECTION_MIN_INCIDENCE,
    TARGET_ACCEPT_RADIUS_AT_REFERENCE, EQUATION_RESIDUAL_TOL, CIRCLE_SIDE_EPS,
)
from .collision import closest_approach_to_target, find_first_collision, trajectory_clearance


def source_matches_circle_side(source, world, family):
    if family.kind != 'circle':
        return True
    circle = world.circles[family.index]
    distance = float(np.linalg.norm(np.asarray(source)-circle.center))
    return distance < circle.radius-CIRCLE_SIDE_EPS if family.side == 'INNER' else distance > circle.radius+CIRCLE_SIDE_EPS


def verify_continuous_contact(source, world, acceleration, image_width, family, solution):
    """Reject occluded continuous candidates before they enter the power family."""
    if not source_matches_circle_side(source, world, family):
        return False
    first = find_first_collision(source, solution.velocity, acceleration, world, solution.t1+1e-5,
                                 avoid_portals=True, image_width=image_width)
    if first is None or (first.obstacle_kind, first.obstacle_index) != (family.kind, family.index):
        return False
    tolerance = EQUATION_RESIDUAL_TOL*max(1., _scale(image_width))
    if abs(first.time-solution.t1) > 1e-5 or np.linalg.norm(np.asarray(first.point)-solution.contact) > tolerance:
        return False
    incoming = np.asarray(solution.velocity)+np.asarray(acceleration)*solution.t1
    normal = np.asarray(solution.normal)
    outgoing = incoming-2*np.dot(incoming, normal)*normal
    return find_first_collision(solution.contact, outgoing, acceleration, world, solution.t2,
                                avoid_portals=True, image_width=image_width) is None


def replay_integer_contact(source, target, world, acceleration, image_width, family, power, angle_degrees, direction):
    if not source_matches_circle_side(source, world, family):
        return None
    scale = _scale(image_width)
    speed = power*SPEED_PER_POWER_AT_REFERENCE*scale
    angle = radians(angle_degrees)
    velocity = np.array(((-1 if direction == 'left' else 1)*speed*cos(angle), -speed*sin(angle)))
    acceleration = np.asarray(acceleration, dtype=float)
    collision = find_first_collision(source, velocity, acceleration, world, MAX_FLIGHT_TIME, avoid_portals=True, image_width=image_width)
    if collision is None or (collision.obstacle_kind, collision.obstacle_index) != (family.kind, family.index):
        return None
    contact = np.asarray(collision.point, dtype=float)
    endpoint_margin = 1e12*scale
    if family.kind == 'line':
        line = world.lines[family.index]
        delta = np.asarray(line.end)-line.start
        length = float(np.linalg.norm(delta))
        along = float(np.dot(contact-np.asarray(line.start), delta))/length
        endpoint_margin = min(along, length-along)
        if min(along, length-along) < LINE_ENDPOINT_MARGIN_AT_REFERENCE*scale-1e-7:
            return None
        normal = np.array((-delta[1], delta[0]), dtype=float)/length
    else:
        circle = world.circles[family.index]
        normal = contact-np.asarray(circle.center)
        normal /= np.linalg.norm(normal)
    incoming = velocity+acceleration*collision.time
    incidence = abs(float(np.dot(incoming, normal)))/max(float(np.linalg.norm(incoming)), 1e-12)
    if incidence < REFLECTION_MIN_INCIDENCE:
        return None
    if family.kind == 'circle' and (np.dot(incoming, normal) > 0) != (family.side == 'INNER'):
        return None
    outgoing = incoming-2*np.dot(incoming, normal)*normal
    remaining = MAX_FLIGHT_TIME-collision.time
    if remaining <= 1e-5:
        return None
    miss, target_time, closest = closest_approach_to_target(contact, outgoing, acceleration, target, remaining)
    if miss > TARGET_ACCEPT_RADIUS_AT_REFERENCE*scale or target_time <= 1e-5:
        return None
    # The intended reflecting surface remains active: an inner-circle exit or
    # a return to the same line is a second collision and invalidates the shot.
    second = find_first_collision(contact, outgoing, acceleration, world, target_time, avoid_portals=True, image_width=image_width)
    if second is not None:
        return None
    clearance = min(trajectory_clearance(source, velocity, acceleration, world, collision.time, exclude=(family.kind, family.index), image_width=image_width), trajectory_clearance(contact, outgoing, acceleration, world, target_time, exclude=(family.kind, family.index), image_width=image_width))
    # A bounded finite value preserves the unlimited-clearance ordering while
    # keeping diagnostics valid for strict JSON encoders.
    if not isfinite(clearance):
        clearance = 1e12*scale
    return {
        'status': 'reachable', 'mode': 'reflection', 'direction': direction,
        'angle_degrees': angle_degrees, 'power': power,
        'portal_count': 0, 'reflection_count': 1, 'events': ['reflection', 'target'],
        'reflection_point': tuple(float(x) for x in contact),
        'reflection_obstacle': {'kind': family.kind, 'index': family.index},
        'reflection_side': family.side,
        'miss_distance': float(miss), 'clearance': float(clearance),
        'flight_time_seconds': float(collision.time+target_time),
        'incidence': incidence, 'closest_target_point': tuple(float(x) for x in closest),
        'actual_v_before': tuple(float(x) for x in incoming),
        'actual_v_after': tuple(float(x) for x in outgoing),
        'endpoint_margin': endpoint_margin,
    }
