"""Exact constant-acceleration geometry in screen coordinates (positive y down).

Physical segment endpoints are always included. Safety margins reject unsafe
shots at the solver level; they must never create holes in an obstacle.
"""
from dataclasses import dataclass
from math import hypot, inf
import numpy as np
from scipy.optimize import brentq

from .solver_config import COLLISION_TIME_EPS, ROOT_IMAG_EPS, portal_avoid_radius
from .world_geometry import Point, World


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def trajectory_position(start, velocity, acceleration, time):
    return tuple(start[i] + velocity[i] * time + acceleration[i] * time * time / 2 for i in (0, 1))


def trajectory_velocity(velocity, acceleration, time):
    return tuple(velocity[i] + acceleration[i] * time for i in (0, 1))


def real_roots(coefficients):
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=float), 'f')
    if len(coefficients) < 2:
        return ()
    roots = np.roots(coefficients)
    return tuple(sorted(float(r.real) for r in roots if abs(r.imag) <= ROOT_IMAG_EPS * max(1., abs(r.real))))


def trajectory_aabb(start, velocity, acceleration, limit):
    bounds = []
    for axis in (0, 1):
        times = [0., limit]
        if acceleration[axis]:
            extremum = -velocity[axis] / acceleration[axis]
            if 0 < extremum < limit:
                times.append(extremum)
        values = [start[axis] + velocity[axis] * t + acceleration[axis] * t * t / 2 for t in times]
        bounds.append((min(values), max(values)))
    return bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]


def _overlap(a, b):
    return all((a[0] <= b[2], b[0] <= a[2], a[1] <= b[3], b[1] <= a[3]))


def _distance_coefficients(start, velocity, acceleration, center):
    q = (start[0] - center[0], start[1] - center[1])
    b = (acceleration[0] / 2, acceleration[1] / 2)
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
    return [dot(b, b), 2 * dot(velocity, b), dot(velocity, velocity) + 2 * dot(q, b), 2 * dot(q, velocity), dot(q, q)]


def parabola_circle_roots(start, velocity, acceleration, center, radius, limit):
    if not _overlap(trajectory_aabb(start, velocity, acceleration, limit),
                    (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)):
        return ()
    coefficients = _distance_coefficients(start, velocity, acceleration, center)
    coefficients[-1] -= radius * radius
    # Distance-to-circle is quartic.  Its cubic derivative partitions the
    # interval into monotonic pieces, so only a piece that crosses zero needs
    # a numerical root solve.  This avoids a quartic solve for most distant
    # obstacles while retaining both entries/exits and tangent contacts.
    value = lambda t: float(np.polyval(coefficients, t))
    derivative = tuple((len(coefficients)-1-i)*coefficient for i, coefficient in enumerate(coefficients[:-1]))
    critical = [time for time in real_roots(derivative) if 0 < time < limit]
    points = [0.0, *sorted(critical), float(limit)]
    roots = []
    tolerance = 1e-8 * max(1.0, radius * radius)
    for time in critical:
        if abs(value(time)) <= tolerance:
            roots.append(time)
    for low, high in zip(points, points[1:]):
        low_value, high_value = value(low), value(high)
        if low_value * high_value < 0:
            roots.append(float(brentq(value, low, high, xtol=1e-10)))
    return tuple(sorted({round(time, 12) for time in roots if COLLISION_TIME_EPS < time <= limit + COLLISION_TIME_EPS}))


def parabola_segment_roots(start, velocity, acceleration, line, limit):
    e = (line.end[0] - line.start[0], line.end[1] - line.start[1])
    length2 = e[0] ** 2 + e[1] ** 2
    if not length2:
        return ()
    q = (start[0] - line.start[0], start[1] - line.start[1])
    coeff = (cross2(acceleration, e) / 2, cross2(velocity, e), cross2(q, e))
    if all(abs(c) < 1e-12 for c in coeff):
        # Collinear motion: include the first entry into the finite segment.
        projection = lambda p: (p[0] * e[0] + p[1] * e[1]) / length2
        a, b, c = projection(acceleration) / 2, projection(velocity), projection(q)
        roots = (COLLISION_TIME_EPS * 2,) if 0 <= c <= 1 else ()
        roots += real_roots((a, b, c)) + real_roots((a, b, c - 1))
    else:
        # The signed distance to the supporting line is quadratic.  Inspect
        # its vertex first; a same-sign interval with no zero-crossing cannot
        # intersect the finite segment and avoids a root solve.
        value = lambda t: coeff[0]*t*t + coeff[1]*t + coeff[2]
        vertex = -coeff[1] / (2*coeff[0]) if abs(coeff[0]) > 1e-12 else None
        candidates = [0.0, float(limit)] + ([vertex] if vertex is not None and 0 < vertex < limit else [])
        values = [value(time) for time in candidates]
        tolerance = 1e-10 * max(1.0, *(abs(item) for item in coeff))
        roots = () if min(values) > tolerance or max(values) < -tolerance else real_roots(coeff)
    result = []
    for t in roots:
        if not COLLISION_TIME_EPS < t <= limit + COLLISION_TIME_EPS:
            continue
        point = trajectory_position(start, velocity, acceleration, t)
        fraction = ((point[0] - line.start[0]) * e[0] + (point[1] - line.start[1]) * e[1]) / length2
        if -1e-8 <= fraction <= 1 + 1e-8:
            result.append(t)
    return tuple(sorted(set(result)))


def closest_approach_to_target(start, velocity, acceleration, target, limit):
    c4, c3, c2, c1, _ = _distance_coefficients(start, velocity, acceleration, target)
    times = [0., limit, *(t for t in real_roots((4*c4, 3*c3, 2*c2, c1)) if 0 < t < limit)]
    time = min(times, key=lambda t: hypot(*(trajectory_position(start, velocity, acceleration, t)[i] - target[i] for i in (0, 1))))
    point = trajectory_position(start, velocity, acceleration, time)
    return hypot(point[0] - target[0], point[1] - target[1]), time, point


@dataclass(frozen=True)
class Collision:
    time: float
    point: Point
    obstacle_kind: str
    obstacle_index: int


def find_all_collisions(start, velocity, acceleration, world, limit, *, avoid_portals=True, image_width=None):
    contacts = []
    for index, circle in enumerate(world.circles):
        contacts.extend(Collision(t, trajectory_position(start, velocity, acceleration, t), 'circle', index)
                        for t in parabola_circle_roots(start, velocity, acceleration, circle.center, circle.radius, limit))
    for index, line in enumerate(world.lines):
        contacts.extend(Collision(t, trajectory_position(start, velocity, acceleration, t), 'line', index)
                        for t in parabola_segment_roots(start, velocity, acceleration, line, limit))
    if avoid_portals:
        scale = (image_width or world.image_width or 1920) / 1920
        for index, pair in enumerate(world.portal_pairs):
            for color, portal in enumerate((pair.orange, pair.blue)):
                radius = portal_avoid_radius(portal, scale)
                roots = parabola_circle_roots(start, velocity, acceleration, portal.center, radius, limit)
                if hypot(start[0] - portal.center[0], start[1] - portal.center[1]) <= radius:
                    roots = (0., *roots)
                contacts.extend(Collision(t, trajectory_position(start, velocity, acceleration, t), 'portal', 2 * index + color) for t in roots)
    return tuple(sorted(contacts, key=lambda c: c.time))


def find_first_collision(start, velocity, acceleration, world, limit, *, avoid_portals=True, image_width=None):
    contacts = find_all_collisions(start, velocity, acceleration, world, limit, avoid_portals=avoid_portals, image_width=image_width)
    if not contacts:
        return None
    simultaneous = {(c.obstacle_kind, c.obstacle_index) for c in contacts
                    if c.time - contacts[0].time <= COLLISION_TIME_EPS}
    if len(simultaneous) > 1:
        return Collision(contacts[0].time, contacts[0].point, 'ambiguous', -1)
    return contacts[0]


def trajectory_clearance(start, velocity, acceleration, world, limit, *, exclude=None, image_width=None):
    """Minimum distance to other surfaces/portal avoidance zones, analytically."""
    clearance = inf
    circles = [('circle', i, c.center, c.radius) for i, c in enumerate(world.circles)]
    scale = (image_width or world.image_width or 1920) / 1920
    circles += [('portal', 2*i+j, p.center, portal_avoid_radius(p, scale))
                for i, pair in enumerate(world.portal_pairs) for j, p in enumerate((pair.orange, pair.blue))]
    for kind, index, center, radius in circles:
        if exclude == (kind, index):
            continue
        if parabola_circle_roots(start, velocity, acceleration, center, radius, limit):
            return 0.
        c4, c3, c2, c1, _ = _distance_coefficients(start, velocity, acceleration, center)
        times = [0., limit, *(t for t in real_roots((4*c4, 3*c3, 2*c2, c1)) if 0 < t < limit)]
        distances = [hypot(*(trajectory_position(start, velocity, acceleration, t)[j] - center[j] for j in (0, 1))) for t in times]
        gap = max(0., min(distances) - radius) if kind == 'portal' else min(abs(d-radius) for d in distances)
        clearance = min(clearance, gap)
    for index, line in enumerate(world.lines):
        if exclude == ('line', index):
            continue
        e = (line.end[0] - line.start[0], line.end[1] - line.start[1])
        length = hypot(*e)
        if not length:
            continue
        times = [0., limit]
        a, b, c = cross2(acceleration, e)/2, cross2(velocity, e), cross2((start[0]-line.start[0], start[1]-line.start[1]), e)
        times += [t for t in real_roots((a, b, c)) + real_roots((2*a, b)) if 0 < t < limit]
        for endpoint in (line.start, line.end):
            miss, t, _ = closest_approach_to_target(start, velocity, acceleration, endpoint, limit)
            clearance = min(clearance, miss)
            times.append(t)
        for t in times:
            p = trajectory_position(start, velocity, acceleration, t)
            fraction = ((p[0]-line.start[0])*e[0] + (p[1]-line.start[1])*e[1]) / length**2
            if 0 <= fraction <= 1:
                clearance = min(clearance, abs(cross2((p[0]-line.start[0], p[1]-line.start[1]), e))/length)
    return clearance
