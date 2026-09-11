"""Continuous reachability outer bounds, shared by every event topology."""
from math import hypot
import numpy as np
from shellshock.math2d.continuous import regions_maybe_reachable
from shellshock.physics.events.portal import portal_map
def _projection_possible(source,target,world,route,acceleration,barrel_radius,target_radius):
    """Propagate enclosing regions across free legs, translating at each portal.

    Reflection changes velocity; its complete surface bounds start the next
    conservative propagation. Unknown incoming/outgoing correlation is retained.
    """
    shift=np.zeros(2);regions=[(source,barrel_radius)];portals=portal_map(world)
    for event in route:
        if event.kind=='portal':
            p,q,_=portals[event.index]
            regions.append((np.subtract(p.center,shift),p.radius*.8))
            shift+=np.subtract(q.center,p.center)
        elif event.kind=='reward':
            p=world.rewards[event.index]
            regions.append((np.subtract(p.center,shift),p.radius*.8))
        else:
            obj=(world.circles if event.kind=='circle' else world.lines)[event.index]
            center=np.asarray(obj.center) if event.kind=='circle' else (np.add(obj.start,obj.end)/2)
            radius=obj.radius if event.kind=='circle' else hypot(*np.subtract(obj.end,obj.start))*.5
            regions.append((center-shift,radius))
            if not regions_maybe_reachable(regions,acceleration):return False
            shift=np.zeros(2);regions=[(center,radius)]
    regions.append((np.subtract(target,shift),target_radius))
    return regions_maybe_reachable(regions,acceleration)



def analyze_route(source,target,world,route,acceleration,barrel_radius,target_radius,**options):
    """Propagate shared contact positions, times and pre/post-event velocities."""
    from shellshock.math2d.continuous import acceleration_aligned_frame
    from shellshock.math2d.reachability import ContactRegion, ReachabilityResult, propagate_contacts
    from shellshock.config.solver import WAYPOINT_RADIUS_SCALE, LINE_CONTACT_FRACTION
    if not _projection_possible(source,target,world,route,acceleration,barrel_radius,target_radius):
        return ReachabilityResult(False,'force-free-order-impossible',0,0)
    x,y,force=acceleration_aligned_frame(acceleration)
    def point(p):
        return float(np.dot(p,x)),float(np.dot(p,y))
    regions=[ContactRegion(point(source),barrel_radius)]
    portals=portal_map(world)
    for event in route:
        if event.kind=='portal':
            entrance,exit,_=portals[event.index]
            shift=point(np.subtract(exit.center,entrance.center))
            regions.append(ContactRegion(point(entrance.center),entrance.radius*WAYPOINT_RADIUS_SCALE,shift=shift))
        elif event.kind=='reward':
            reward=world.rewards[event.index]
            regions.append(ContactRegion(point(reward.center),reward.radius*WAYPOINT_RADIUS_SCALE))
        elif event.kind=='circle':
            circle=world.circles[event.index]
            regions.append(ContactRegion(point(circle.center),circle.radius,'circle'))
        elif event.kind=='line':
            line=world.lines[event.index]
            start,end=np.array(point(line.start)),np.array(point(line.end))
            inset=(1-LINE_CONTACT_FRACTION)/2
            first,last=start+inset*(end-start),end-inset*(end-start)
            regions.append(ContactRegion(tuple((start+end)/2),hypot(*(end-start))/2,'line',(tuple(first),tuple(last))))
        else:
            raise ValueError(f'unsupported route event: {event.kind}')
    regions.append(ContactRegion(point(target),target_radius))
    from shellshock.config.solver import LAYER_A_BOX_BUDGET, LAYER_A_DEPTH_LIMIT, LAYER_A_CONTRACTION_ROUNDS
    settings=dict(box_budget=LAYER_A_BOX_BUDGET,depth_limit=LAYER_A_DEPTH_LIMIT,rounds=LAYER_A_CONTRACTION_ROUNDS)
    settings.update(options)
    return propagate_contacts(regions,force,**settings)


def _route_possible(source,target,world,route,acceleration,barrel_radius,target_radius):
    return analyze_route(source,target,world,route,acceleration,barrel_radius,target_radius).possible
