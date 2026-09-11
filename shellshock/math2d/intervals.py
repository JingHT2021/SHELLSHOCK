"""Extended-real interval bounds for continuous flight, including infinite tails.

Bounds are widened for floating-point uncertainty. These are necessary-condition
contractors, not a feasibility certificate or a finite time sampler.
"""
from dataclasses import dataclass
from math import copysign, inf, isfinite, isnan, sqrt


class EmptyInterval(ValueError):
    pass


def _pad(x):
    return 1e-10 * max(1., abs(x)) if isfinite(x) else 0.


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def intersect(self, other):
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        if lo > hi:
            if lo-hi > _pad(lo)+_pad(hi):
                raise EmptyInterval('disjoint continuous bounds')
            lo, hi = hi, lo  # Retain uncertain near-touching boundaries.
        return Interval(lo, hi)

    def __add__(self, other):
        other = interval(other)
        if self.lo == self.hi == other.lo == other.hi == 0:
            return Interval(0.,0.)
        return bounds(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -interval(other)

    def __mul__(self, other):
        other = interval(other)
        if (self.lo == self.hi == 0) or (other.lo == other.hi == 0):
            return Interval(0.,0.)
        def product(a,b):
            return 0. if a == 0 or b == 0 else a*b
        values=[product(a,b) for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return bounds(min(values),max(values))

    __rmul__ = __mul__

    def contains(self, value):
        return self.lo <= value <= self.hi


def interval(value):
    return value if isinstance(value, Interval) else Interval(float(value),float(value))


def bounds(lo,hi):
    if isnan(lo) or isnan(hi):
        return Interval(-inf,inf)
    return Interval(lo-_pad(lo),hi+_pad(hi))


def square(value):
    lo=0. if value.contains(0) else min(value.lo**2,value.hi**2)
    return bounds(lo,max(value.lo**2,value.hi**2))


def _reciprocal_linear_extrema(d,k,time):
    def value(t):
        if t == 0:
            return copysign(inf,d) if d else 0.
        if t == inf:
            return copysign(inf,k) if k else 0.
        return d/t+k*t
    values=[value(time.lo),value(time.hi)]
    if k and d/k > 0:
        stationary=sqrt(d/k)
        if time.lo <= stationary <= time.hi:
            values.append(value(stationary))
    return min(values),max(values)


def velocity_range(displacement,k,time):
    """Range of d/t + k*t, over d in displacement and positive time."""
    lo=_reciprocal_linear_extrema(displacement.lo,k,time)[0]
    hi=_reciprocal_linear_extrema(displacement.hi,k,time)[1]
    return bounds(lo,hi)


def _quadratic_extrema(v,k,time):
    def value(t):
        if t == 0:
            return 0.
        if t == inf:
            return copysign(inf,k) if k else (copysign(inf,v) if v else 0.)
        return v*t+k*t*t
    values=[value(time.lo),value(time.hi)]
    if k:
        stationary=-v/(2*k)
        if time.lo <= stationary <= time.hi:
            values.append(value(stationary))
    return min(values),max(values)


def displacement_range(velocity,k,time):
    """Range of v*t + k*t², monotone in v for t >= 0."""
    lo=-inf if velocity.lo == -inf else _quadratic_extrema(velocity.lo,k,time)[0]
    hi=inf if velocity.hi == inf else _quadratic_extrema(velocity.hi,k,time)[1]
    return bounds(lo,hi)


def _leq(a,b,c,domain):
    """Analytic sign intervals of a*t²+b*t+c <= 0; no time sampling."""
    if not all(isfinite(x) for x in (a,b,c)):
        return [domain]
    if a == 0:
        if b == 0:
            return [domain] if c <= _pad(c) else []
        root=-c/b
        allowed=[bounds(-inf,root)] if b>0 else [bounds(root,inf)]
    else:
        discriminant=b*b-4*a*c
        uncertainty=1e-12*max(1.,abs(b*b),abs(4*a*c))
        if not isfinite(discriminant) or abs(discriminant)<=uncertainty:
            return [domain]  # Near tangency cannot support a rejection.
        if discriminant<0:
            return [] if a>0 else [domain]
        q=-.5*(b+copysign(sqrt(discriminant),b))
        roots=sorted((q/a,c/q)) if q else [0.,0.]
        allowed=([bounds(roots[0],roots[1])] if a>0 else
                 [bounds(-inf,roots[0]),bounds(roots[1],inf)])
    result=[]
    for part in allowed:
        try:
            candidate=domain.intersect(part)
            if candidate.hi>0:
                result.append(Interval(max(0.,candidate.lo),candidate.hi))
        except EmptyInterval:
            pass
    return result


def time_ranges(displacement,k,velocity,domain):
    """Keep all positive-time branches of v_lo <= d/t+k*t <= v_hi."""
    pieces=[domain]
    constraints=[]
    if isfinite(velocity.hi):
        constraints.append((k,-velocity.hi,displacement.lo))
    if isfinite(velocity.lo):
        constraints.append((-k,velocity.lo,-displacement.hi))
    for coefficients in constraints:
        pieces=[part for old in pieces for part in _leq(*coefficients,old)]
    return pieces


def hull(pieces):
    if not pieces:
        raise EmptyInterval('empty time set')
    return Interval(min(p.lo for p in pieces),max(p.hi for p in pieces))
