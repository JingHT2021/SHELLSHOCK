"""Continuous-time waypoint equations, with no control or game-policy limits."""
from dataclasses import dataclass
from math import hypot, sqrt
import numpy as np


@dataclass(frozen=True)
class ContinuousArc:
    velocity: tuple[float, float]
    duration: float
    waypoint_time: float


def acceleration_aligned_frame(acceleration):
    ax, ay = acceleration
    g = hypot(ax, ay)
    if g <= 1e-12:
        return (1., 0.), (0., -1.), (0., 0.)
    return (ay/g, -ax/g), (-ax/g, -ay/g), (0., -g)


def arrival_velocity(previous, current, acceleration, time):
    if time <= 0:
        raise ValueError('time must be positive')
    return tuple((current[i]-previous[i])/time + acceleration[i]*time/2 for i in (0, 1))


def reflect_velocity(velocity, normal):
    n = np.asarray(normal, dtype=float)
    length = np.linalg.norm(n)
    if length <= 1e-12:
        raise ValueError('reflection normal must be nonzero')
    n /= length
    v = np.asarray(velocity, dtype=float)
    return tuple(v - 2*np.dot(v, n)*n)


def through_waypoint(source, waypoint, target, acceleration):
    """Solve nondegenerate three-point flight analytically, without scanning time.

    Collinear/zero-force families have infinitely many timings. Return no finite
    seed for those cases; callers retain the region and use fixed-speed seeds.
    Empty output is NOT an impossibility proof for a region or degenerate family.
    """
    d = np.subtract(waypoint, source)
    e = np.subtract(target, waypoint)
    a = np.asarray(acceleration, dtype=float)
    cross = lambda u, v: float(u[0]*v[1]-u[1]*v[0])
    denominator = cross(d, a)
    if abs(denominator) <= 1e-10*max(1., np.linalg.norm(d)*np.linalg.norm(a)):
        return ()
    r = cross(e, a)/denominator
    if r <= 0:
        return ()
    b = .5*a*r*(1+r)
    z = float(np.dot(e-r*d, b)/np.dot(b, b))
    if z <= 0 or not np.isfinite(z):
        return ()
    t1 = sqrt(z)
    velocity = d/t1-a*t1/2
    if np.linalg.norm(e-r*d-b*z) > 1e-7*max(1., np.linalg.norm(e)):
        return ()
    return (ContinuousArc(tuple(velocity), t1*(1+r), t1),)


def regions_maybe_reachable(regions, acceleration):
    """Outer projection of continuous reachability onto the force-free axis.

    Each (center, radius) is a disk enclosing a waypoint region after portal
    translations. Since perpendicular acceleration is zero, x(t)=x0+vx*t.
    Intersect reachable half-lines in event order, keeping both orientations.
    This proof never relies on quadrant tables or sampled contact points.
    Unresolved vertical-speed/time correlations are deliberately retained.
    """
    x, _, _ = acceleration_aligned_frame(acceleration)
    intervals = [(float(np.dot(center, x))-radius,
                  float(np.dot(center, x))+radius) for center, radius in regions]
    def possible(reverse):
        lower = -float('inf')
        for lo, hi in intervals:
            if reverse:
                lo, hi = -hi, -lo
            lower = max(lower, lo)
            if lower > hi + 1e-8:
                return False
        return True
    return possible(False) or possible(True)
