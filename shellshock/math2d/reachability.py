"""Bidirectional interval propagation of shared contacts and continuous flights.

Every rejected box violates a necessary condition. Unresolved boxes (including
infinite-time tails) survive. Contact subdivision is continuous coverage, never
angle/power enumeration or evidence from failed point samples.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import hypot, inf, isfinite, sqrt
from shellshock.math2d.intervals import (
    Interval, EmptyInterval, bounds, interval, square, velocity_range,
    displacement_range, time_ranges, hull,
)


@dataclass(frozen=True)
class ContactRegion:
    center: tuple[float,float]
    radius: float = 0.
    kind: str = 'disk'  # disk, circle surface, finite line
    endpoints: tuple | None = None
    shift: tuple[float,float] = (0.,0.)  # portal translation AFTER this contact

    def box(self):
        if self.kind == 'line':
            return [bounds(min(a,b),max(a,b)) for a,b in zip(*self.endpoints)]
        return [bounds(x-self.radius,x+self.radius) for x in self.center]


@dataclass
class FlightState:
    positions: list
    times: list
    outgoing: list
    incoming: list


@dataclass(frozen=True)
class ReachabilityResult:
    possible: bool
    reason: str
    boxes_checked: int
    contractions: int
    branch_time_bounds: tuple = ()
    branch_contact_bounds: tuple = ()


def _shape(region,point):
    if region.kind == 'line':
        start,end=region.endpoints
        parameter=Interval(0.,1.)
        for axis in range(2):
            delta=end[axis]-start[axis]
            if delta:
                parameter=parameter.intersect((point[axis]-start[axis])*(1/delta))
            else:
                point[axis]=point[axis].intersect(interval(start[axis]))
        for axis in range(2):
            point[axis]=point[axis].intersect(interval(start[axis]) + parameter*(end[axis]-start[axis]))
        return
    displacement=[point[j]-region.center[j] for j in range(2)]
    distances=[square(x) for x in displacement]
    radius_squared=interval(region.radius**2)
    if (distances[0]+distances[1]).lo > radius_squared.hi + 1e-8*max(1.,radius_squared.hi):
        raise EmptyInterval('outside contact disk')
    if region.kind=='circle':
        (distances[0]+distances[1]).intersect(radius_squared)
    for axis in range(2):
        maximum=sqrt(max(0.,region.radius**2-max(0.,distances[1-axis].lo)))
        allowed=bounds(-maximum,maximum)
        if region.kind=='circle':
            minimum=sqrt(max(0.,region.radius**2-distances[1-axis].hi))
            if displacement[axis].lo>0:
                allowed=allowed.intersect(bounds(minimum,maximum))
            elif displacement[axis].hi<0:
                allowed=allowed.intersect(bounds(-maximum,-minimum))
        point[axis]=point[axis].intersect(allowed+region.center[axis])


def _normal(region,point):
    if region.kind=='line':
        start,end=region.endpoints
        dx,dy=end[0]-start[0],end[1]-start[1]
        length=hypot(dx,dy)
        return [interval(-dy/length),interval(dx/length)] if length else None
    if region.kind=='circle' and region.radius>0:
        return [(point[j]-region.center[j])*(1/region.radius) for j in range(2)]
    return None


def _reflect(velocity,normal):
    if normal is None:
        return list(velocity)
    # Coordinate matrix avoids unnecessary loss for axis-aligned surfaces.
    matrix=[[(interval(1.) if i==j else interval(0.))-2*normal[i]*normal[j]
             for j in range(2)] for i in range(2)]
    return [matrix[i][0]*velocity[0]+matrix[i][1]*velocity[1] for i in range(2)]


def _cross(left,right):
    return left[0]*right[1]-left[1]*right[0]


def _join_geometry(state,regions,acceleration,i,normal):
    before=_displacement(state,regions,i-1)
    after=_displacement(state,regions,i)
    if _immediate(before,state.times[i-1]) or _immediate(after,state.times[i]):
        return
    reflected_before=_reflect(before,normal)
    force=[interval(a) for a in acceleration]
    reflected_force=_reflect(force,normal)
    determinant=_cross(reflected_before,after)
    c1=_cross(reflected_before,reflected_force)
    c2=_cross(reflected_before,force)
    # Eliminate speeds without choosing times. For r=t_next/t_previous>0:
    # cross(R*d1,d2) = .5*t_previous²*(r*cross(R*d1,R*a)
    #                                +r²*cross(R*d1,a)).
    # This preserves vector-direction correlation even on infinite time tails.
    if all(a==0 for a in acceleration):
        determinant.intersect(interval(0.))
    elif c1.lo>=0 and c2.lo>=0:
        determinant.intersect(Interval(0.,inf))
    elif c1.hi<=0 and c2.hi<=0:
        determinant.intersect(Interval(-inf,0.))


def _join(state,regions,acceleration):
    for i in range(1,len(regions)-1):
        normal=_normal(regions[i],state.positions[i])
        _join_geometry(state,regions,acceleration,i,normal)
        mapped=_reflect(state.incoming[i-1],normal)
        for axis in range(2):
            state.outgoing[i][axis]=state.outgoing[i][axis].intersect(mapped[axis])
        mapped=_reflect(state.outgoing[i],normal)
        for axis in range(2):
            state.incoming[i-1][axis]=state.incoming[i-1][axis].intersect(mapped[axis])


def _displacement(state,regions,i):
    return [state.positions[i+1][j]-state.positions[i][j]-regions[i].shift[j] for j in range(2)]


def _immediate(displacement,time):
    # Replay permits zero-time passive rewards / coincident contacts. Such a
    # leg cannot divide by t; retain its unconstrained limiting velocity.
    return time.lo <= 0 and all(d.contains(0.) for d in displacement)


def _contract(state,regions,acceleration,rounds):
    for _ in range(rounds):
        for region,point in zip(regions,state.positions):
            _shape(region,point)
        for i,time in enumerate(state.times):
            d=_displacement(state,regions,i)
            if _immediate(d,time):
                continue
            for axis,a in enumerate(acceleration):
                state.outgoing[i][axis]=state.outgoing[i][axis].intersect(velocity_range(d[axis],-a/2,time))
                state.incoming[i][axis]=state.incoming[i][axis].intersect(velocity_range(d[axis],a/2,time))
        _join(state,regions,acceleration)
        for i,time in enumerate(state.times):
            d=_displacement(state,regions,i)
            if _immediate(d,time):
                continue
            for axis,a in enumerate(acceleration):
                out,inc=state.outgoing[i][axis],state.incoming[i][axis]
                time=time.intersect(hull(time_ranges(d[axis],-a/2,out,time)))
                time=time.intersect(hull(time_ranges(d[axis],a/2,inc,time)))
                if a:
                    time=time.intersect((inc-out)*(1/a)).intersect(Interval(0.,inf))
                # The same time variable is used by both axes and both velocities.
                out=out.intersect(inc-interval(a)*time)
                inc=inc.intersect(out+interval(a)*time)
                state.outgoing[i][axis],state.incoming[i][axis]=out,inc
                allowed=d[axis].intersect(displacement_range(out,a/2,time))
                allowed=allowed.intersect(displacement_range(inc,-a/2,time))
                state.positions[i+1][axis]=state.positions[i+1][axis].intersect(
                    state.positions[i][axis]+regions[i].shift[axis]+allowed)
                state.positions[i][axis]=state.positions[i][axis].intersect(
                    state.positions[i+1][axis]-regions[i].shift[axis]-allowed)
            state.times[i]=time
    return rounds


def _split(state,regions,depth):
    candidates=[]
    for i,(region,point) in enumerate(zip(regions,state.positions)):
        for axis,value in enumerate(point):
            width=value.hi-value.lo
            if width>1e-7*max(1.,abs(value.lo),abs(value.hi)):
                candidates.append((width/max(1.,region.radius), 'point',i,axis,value))
    # Alternate contact and time subdivision, preserving every infinite tail.
    if depth%2 or not candidates:
        for i,value in enumerate(state.times):
            width=value.hi-value.lo
            if width>1e-8*max(1.,value.lo):
                candidates.append((inf if not isfinite(value.hi) else width/max(1.,value.hi),'time',i,0,value))
    if not candidates:
        return None
    _,kind,i,axis,value=max(candidates,key=lambda row:row[0])
    middle=(value.lo+value.hi)/2 if isfinite(value.hi) else max(1.,2*value.lo)
    if not value.lo<middle<value.hi:
        return None
    children=[]
    for part in (Interval(value.lo,middle),Interval(middle,value.hi)):
        child=deepcopy(state)
        if kind=='time':child.times[i]=part
        else:child.positions[i][axis]=part
        children.append(child)
    return children


def propagate_contacts(regions,acceleration,*,box_budget=15,depth_limit=3,rounds=5):
    """Intersect continuous flight/transition relations; UNKNOWN is retained."""
    whole=Interval(-inf,inf)
    legs=len(regions)-1
    initial=FlightState([r.box() for r in regions],[Interval(0.,inf) for _ in range(legs)],
                        [[whole,whole] for _ in range(legs)],[[whole,whole] for _ in range(legs)])
    pending=[(initial,0)];checked=0;contractions=0
    while pending and checked<box_budget:
        state,depth=pending.pop();checked+=1
        try:
            contractions+=_contract(state,regions,acceleration,rounds)
        except EmptyInterval:
            continue
        children=_split(state,regions,depth) if depth<depth_limit else None
        if children is None:
            return ReachabilityResult(True,'continuous-constraints-unresolved',checked,contractions,
                tuple((t.lo,t.hi) for t in state.times),
                tuple(tuple((v.lo,v.hi) for v in p) for p in state.positions))
        pending.extend((child,depth+1) for child in children)
    if pending:
        return ReachabilityResult(True,'continuous-budget-exhausted',checked,contractions)
    return ReachabilityResult(False,'continuous-constraints-empty',checked,contractions)
