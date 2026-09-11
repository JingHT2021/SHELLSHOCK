"""Analytic three-point seeds and continuous fixed-contact reflection branches."""
from math import cos,sin,pi,hypot
import numpy as np
from shellshock.math2d.continuous import through_waypoint
from shellshock.physics.events.portal import portal_map
def _unfold(source,target,world,route):
    portals=portal_map(world);shift=np.zeros(2);points=[]
    for event in route:
        if event.kind=='portal':
            p,q,_=portals[event.index]
            points.append((np.subtract(p.center,shift),p.radius*.8,event))
            shift+=np.subtract(q.center,p.center)
        elif event.kind=='reward':
            p=world.rewards[event.index]
            points.append((np.subtract(p.center,shift),p.radius*.8,event))
    return points,np.subtract(target,shift)


def _waypoint_seeds(source,target,world,route,acceleration,speed_per_power):
    from shellshock.math2d.shots import solve_ballistic_for_speed, minimum_ballistic_speed
    points,virtual_target=_unfold(source,target,world,route)
    for center,radius,event in points:
        for delta in ((0,0),(.7*radius,0),(-.7*radius,0),(0,.7*radius),(0,-.7*radius)):
            waypoint=np.add(center,delta)
            for arc in through_waypoint(source,waypoint,virtual_target,acceleration):
                yield tuple(arc.velocity)
    # Finite analytic anchors cover direct and degenerate vertical families.
    minimum=minimum_ballistic_speed(source,virtual_target,acceleration)/speed_per_power
    for power in sorted({max(1.,minimum),90.,100.}):
        if power<=100:
            for arc in solve_ballistic_for_speed(source,virtual_target,acceleration,power*speed_per_power):
                yield tuple(arc.velocity)


def _reflection_seeds(source,target,world,route,acceleration,speed_per_power):
    from shellshock.planning.reflection.contact import fixed_contact_solutions
    from scipy.optimize import minimize_scalar
    from shellshock.math2d.geometry import closest_approach_to_target, trajectory_velocity
    from shellshock.math2d.continuous import reflect_velocity
    index=next(i for i,e in enumerate(route) if e.kind in {'circle','line'})
    event=route[index];portals=portal_map(world)
    shift=lambda seq:sum((np.subtract(portals[e.index][1].center,portals[e.index][0].center)
                          for e in seq if e.kind=='portal'),np.zeros(2))
    before,after=shift(route[:index]),shift(route[index+1:])
    start=np.add(source,before);end=np.subtract(target,after)
    obj=(world.circles if event.kind=='circle' else world.lines)[event.index]
    def evaluate(q):
        if event.kind=='circle':
            normal=np.array((cos(q),sin(q)));contact=np.add(obj.center,obj.radius*normal)
        else:
            d=np.subtract(obj.end,obj.start);length=hypot(*d)
            if length<=1e-8:
                return []
            normal=np.array((-d[1],d[0]))/length;contact=np.add(obj.start,q*d)
        return fixed_contact_solutions(start,end,contact,normal,acceleration,speed_per_power)
    # Score real waypoint regions in their unfolded pre/post-reflection frames.
    def objective(q):
        arcs=evaluate(q)
        if not arcs:
            return 1e12
        best=1e12
        for arc in arcs:
            cost=max(0.,arc.power-100.)**2
            for side,seq in enumerate((route[:index],route[index+1:])):
                accumulated=np.zeros(2)
                for e in seq:
                    if e.kind=='portal':
                        p,partner,_=portals[e.index];center=np.subtract(p.center,accumulated);radius=p.radius*.8
                        accumulated+=np.subtract(partner.center,p.center)
                    else:
                        reward=world.rewards[e.index];center=np.subtract(reward.center,accumulated);radius=reward.radius*.8
                    if side==0:
                        center=np.add(center,before)
                        origin,velocity,duration=start,arc.velocity,arc.t1
                    else:
                        origin=arc.contact;duration=arc.t2
                        velocity=reflect_velocity(trajectory_velocity(arc.velocity,acceleration,arc.t1),arc.normal)
                    distance=closest_approach_to_target(origin,velocity,acceleration,center,duration)[0]
                    cost+=max(0.,distance-radius*.6)**2
            best=min(best,cost)
        return best
    grid=np.linspace(0,2*pi,13) if event.kind=='circle' else np.linspace(.05,.95,7)
    for q in grid:
        for arc in evaluate(q):
            yield tuple(arc.velocity)
    if len(route)>1:
        # Refine only the three most promising contact intervals, no power sweep.
        intervals=sorted(zip(grid[:-1],grid[1:]),key=lambda ab:objective((ab[0]+ab[1])/2))[:3]
        for lo,hi in intervals:
            fit=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'maxiter':24,'xatol':1e-4})
            for arc in evaluate(fit.x):
                yield tuple(arc.velocity)

