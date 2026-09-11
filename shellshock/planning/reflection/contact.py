"""Fixed-contact continuous reflection candidates for Layer B."""
from dataclasses import dataclass
from math import atan2,degrees,copysign,sqrt,isfinite
import numpy as np
from scipy.optimize import minimize_scalar
from shellshock.math2d.continuous import reflect_velocity
from shellshock.config.solver import MAX_FLIGHT_TIME,REFLECTION_MIN_INCIDENCE
Point=tuple[float,float]

@dataclass(frozen=True)
class ContactSolution:
    contact: Point
    normal: Point
    velocity: Point
    t1: float
    t2: float
    power: float
    angle_degrees: float
    direction: str
    incidence: float
    parameter: float = 0.
    discriminant_norm: float = 1.0
    root_slope: float = 1.0


def positive_polynomial_roots(a, b, c):
    """Stable quadratic/linear roots; an identity is handled by its caller."""
    tolerance = 1e-12 * max(abs(a), abs(b), abs(c), 1.)
    if abs(a) <= tolerance:
        return [-c/b] if abs(b) > tolerance and -c/b > 0 else []
    disc = b*b-4*a*c
    if disc < -1e-12*max(b*b, abs(4*a*c), 1.):
        return []
    root = sqrt(max(0., disc))
    q = -.5*(b+copysign(root, b))
    roots = [-b/(2*a)] if root == 0 else [q/a, c/q]
    return sorted({float(r) for r in roots if r > 0 and isfinite(r)})


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def fixed_contact_solutions(source, target, contact, normal, acceleration, speed_per_power, desired_power=None):
    """Solve flight-time ratio analytically for a fixed point and normal.

    Parallel/identity constraints have a continuous time family. Representative
    ratios include its minimum-power point and admissible interval endpoints;
    the nondegenerate case has at most two algebraic branches.
    """
    source, target, contact, normal, acceleration = map(lambda x: np.asarray(x, dtype=float), (source, target, contact, normal, acceleration))
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return []
    normal = normal/norm
    d, e, b = contact-source, target-contact, acceleration/2
    reflect = lambda x: np.asarray(reflect_velocity(x, normal))
    p, q = reflect(e), reflect(b)
    coefficients = (-_cross(p, b), _cross(d, b)-_cross(p, q), _cross(d, q))
    ratios = positive_polynomial_roots(*coefficients)
    coefficient_scale = max(*(abs(value) for value in coefficients), 1e-12)
    normalized = tuple(value / coefficient_scale for value in coefficients)
    discriminant_norm = max(0.0, normalized[1] * normalized[1] - 4 * normalized[0] * normalized[2])
    identity = max(abs(c) for c in coefficients) <= 1e-12*max(np.linalg.norm(d)*np.linalg.norm(b), np.linalg.norm(p)*np.linalg.norm(q), 1.)

    def reconstruct(u, explicit_z=None):
        h = u*b+q
        denominator = u*float(np.dot(h, h))
        if denominator <= 1e-18:
            if np.linalg.norm(d-u*p) > 1e-7 or np.linalg.norm(b) <= 1e-12:
                return None
            z = float(np.linalg.norm(d)/np.linalg.norm(b))/(u*u)
        else:
            z = -float(np.dot(d-u*p, h))/denominator
        if explicit_z is not None:
            z = explicit_z
        if z <= 1e-12 or not isfinite(z):
            return None
        t2, t1 = sqrt(z), u*sqrt(z)
        if t1 <= 1e-5 or t2 <= 1e-5 or t1+t2 > MAX_FLIGHT_TIME:
            return None
        velocity = d/t1-b*t1
        incoming = velocity+acceleration*t1
        outgoing = reflect(incoming)
        residual = contact+outgoing*t2+b*t2*t2-target
        if np.linalg.norm(residual) > 1e-6*max(1., np.linalg.norm(d), np.linalg.norm(e)):
            return None
        speed = float(np.linalg.norm(velocity))
        incidence = abs(float(np.dot(incoming, normal)))/max(float(np.linalg.norm(incoming)), 1e-12)
        angle = degrees(atan2(-velocity[1], abs(velocity[0])))
        if speed <= 0 or not 0 <= angle <= 90 or incidence < REFLECTION_MIN_INCIDENCE:
            return None
        root_slope = abs(2 * normalized[0] * u + normalized[1])
        return ContactSolution(
            tuple(contact), tuple(normal), tuple(velocity), t1, t2,
            speed/speed_per_power, angle, 'right' if velocity[0] >= 0 else 'left', incidence,
            discriminant_norm=discriminant_norm, root_slope=root_slope,
        )

    if identity:
        # Solve the remaining one-dimensional family in log time-ratio space.
        grid = np.linspace(-8., 8., 65)
        for lo, hi in zip(grid[:-1], grid[1:]):
            fit = minimize_scalar(lambda x: (s.power if (s := reconstruct(float(np.exp(x)))) else 1e9), bounds=(lo, hi), method='bounded')
            ratios.extend([float(np.exp(lo)), float(np.exp(fit.x)), float(np.exp(hi))])
    # h=0 can vanish at an isolated ratio even when its quadratic constraint
    # is identically zero. The travel-time equation then leaves speed free.
    if np.dot(b, b) > 1e-18:
        singular_ratio = -float(np.dot(q, b))/float(np.dot(b, b))
        if singular_ratio > 0 and np.linalg.norm(singular_ratio*b+q) < 1e-9 and np.linalg.norm(d-singular_ratio*p) < 1e-7:
            ratios.append(singular_ratio)
            if desired_power is not None:
                squares = positive_polynomial_roots(float(np.dot(b,b)), -2*float(np.dot(d,b))-(desired_power*speed_per_power)**2, float(np.dot(d,d)))
                return [s for w in squares if (s := reconstruct(singular_ratio, w/singular_ratio**2)) is not None]
    return sorted((s for u in set(ratios) if (s := reconstruct(u)) is not None), key=lambda s: s.t1/s.t2)
