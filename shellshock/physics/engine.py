"""One event-driven trajectory validator for every solver and image overlay."""
from dataclasses import dataclass
from math import hypot, isfinite
import numpy as np
from shellshock.math2d.geometry import (
    trajectory_position, trajectory_velocity, parabola_circle_roots,
    parabola_segment_roots, closest_approach_to_target, real_roots,
)
from shellshock.math2d.continuous import reflect_velocity

from shellshock.config.solver import WAYPOINT_RADIUS_SCALE as TRIGGER_SCALE, UNEXPECTED_MARGIN_PIXELS as AVOID_PIXELS, MAX_FLIGHT_TIME, TARGET_ACCEPT_RADIUS_AT_REFERENCE, REFLECTION_MIN_INCIDENCE
from shellshock.physics.events.portal import portal_map, translate
from shellshock.physics.events.reflection import contact_allowed
from shellshock.physics.events.blackhole import forbidden_radius
from shellshock.physics.events.reward import trigger_radius

EPS = 1e-5


@dataclass(frozen=True)
class RouteEvent:
    kind: str
    index: int | str



def circle_entry(start, velocity, acceleration, center, radius, limit, allow_inside=False):
    inside = hypot(start[0]-center[0], start[1]-center[1]) <= radius+1e-8
    if inside and not allow_inside:
        return 0.
    for t in parabola_circle_roots(start, velocity, acceleration, center, radius, limit):
        p = np.subtract(trajectory_position(start, velocity, acceleration, t), center)
        v = trajectory_velocity(velocity, acceleration, t)
        if np.dot(p, v) <= 1e-7:
            return t
    return None


def segment_distance(start, velocity, acceleration, line, limit):
    """Minimum distance to a finite segment via cubic extrema and projections."""
    d = np.subtract(line.end, line.start).astype(float)
    length2 = float(np.dot(d,d))
    if length2 <= 1e-16:
        return closest_approach_to_target(start,velocity,acceleration,line.start,limit)[0]
    q = np.subtract(start,line.start)
    a = np.asarray(acceleration)/2
    v = np.asarray(velocity)
    project = [float(np.dot(a,d)/length2),float(np.dot(v,d)/length2),float(np.dot(q,d)/length2)]
    cross = lambda x: float(x[0]*d[1]-x[1]*d[0])
    signed = [cross(a),cross(v),cross(q)]
    times = [0.,limit]
    times.extend(t for coeff in (project, [*project[:2],project[2]-1], signed,
                                 [2*signed[0],signed[1]])
                 for t in real_roots(coeff) if 0<t<limit)
    best = min(closest_approach_to_target(start,velocity,acceleration,p,limit)[0]
               for p in (line.start,line.end))
    for t in times:
        f = float(np.polyval(project,t))
        if 0<=f<=1:
            best=min(best,abs(float(np.polyval(signed,t)))/hypot(*d))
    return best


def _blocked(start, velocity, acceleration, world, duration, wanted, departed, exit_id):
    """Unexpected circles require only cubic distance extrema, never quartic roots."""
    clearance = 1e12
    for kind, objects in (('blackhole',getattr(world,'blackholes',())),('circle',world.circles)):
        for i,obj in enumerate(objects):
            event=RouteEvent(kind,i)
            if event==wanted:
                continue
            if event==departed:
                # Permit departure from the actual surface, but never a later hit.
                if parabola_circle_roots(start,velocity,acceleration,obj.center,obj.radius,duration):
                    return 'repeat-reflection',0.
                continue
            miss=closest_approach_to_target(start,velocity,acceleration,obj.center,duration)[0]
            gap=miss-(forbidden_radius(obj) if kind=='blackhole' else obj.radius+AVOID_PIXELS)
            clearance=min(clearance,gap)
            if gap<=1e-7:
                return 'unexpected-'+kind,clearance
    for i,line in enumerate(world.lines):
        event=RouteEvent('line',i)
        if event==wanted:
            continue
        if event==departed:
            if parabola_segment_roots(start,velocity,acceleration,line,duration):
                return 'repeat-reflection',0.
            continue
        gap=segment_distance(start,velocity,acceleration,line,duration)-AVOID_PIXELS
        clearance=min(clearance,gap)
        if gap<=1e-7:
            return 'unexpected-line',clearance
    for key,(portal,_,_) in portal_map(world).items():
        if wanted==RouteEvent('portal',key):
            continue
        if key==exit_id:
            if circle_entry(start,velocity,acceleration,portal.center,portal.radius+AVOID_PIXELS,duration,True) is not None:
                return 'portal-reentry',0.
            continue
        gap=closest_approach_to_target(start,velocity,acceleration,portal.center,duration)[0]-portal.radius-AVOID_PIXELS
        clearance=min(clearance,gap)
        if gap<=1e-7:
            return 'unexpected-portal',clearance
    for portal in getattr(world,'unpaired_portal_regions',()):
        gap=closest_approach_to_target(start,velocity,acceleration,portal.center,duration)[0]-portal.radius-AVOID_PIXELS
        clearance=min(clearance,gap)
        if gap<=1e-7:
            return 'unexpected-portal',clearance
    return None,clearance


def event_time(event, point, velocity, acceleration, world, remaining, exit_id=None):
    if event.kind=='portal':
        portal=portal_map(world)[event.index][0]
        return circle_entry(point,velocity,acceleration,portal.center,trigger_radius(portal),remaining,event.index==exit_id)
    if event.kind=='reward':
        obj=world.rewards[int(event.index)]
        return circle_entry(point,velocity,acceleration,obj.center,trigger_radius(obj),remaining)
    if event.kind=='circle':
        obj=world.circles[int(event.index)]
        roots=parabola_circle_roots(point,velocity,acceleration,obj.center,obj.radius,remaining)
    elif event.kind=='line':
        roots=parabola_segment_roots(point,velocity,acceleration,world.lines[int(event.index)],remaining)
    else:
        raise ValueError('invalid route event')
    return roots[0] if roots else None


def replay_route(source, velocity, acceleration, world, target, route=(), *, max_time=MAX_FLIGHT_TIME, target_radius=None, preview=False):
    """Validate ordered events, returning exact flight segments and accumulated rewards.

    Target error is the continuous minimum distance on the final flight segment.
    The chosen minimum must be reachable without any earlier forbidden collision.
    """
    route=tuple(route)
    point,current=tuple(source),tuple(velocity)
    remaining,elapsed=float(max_time),0.
    portals=portal_map(world)
    segments=[]; trace=[]; portal_sequence=[]; reward_ids=[]
    multiplier=1.; reflections=0; clearance=1e12; bounce={}
    departed=None; exit_id=None
    radius=target_radius if target_radius is not None else TARGET_ACCEPT_RADIUS_AT_REFERENCE*(world.image_width or 1920)/1920
    def failure(reason):
        return {'valid':False,'status':'unreachable','reason':reason,'segments':segments,'event_trace':trace}
    if remaining<=0 or not all(isfinite(x) for p in (point,current,acceleration,target) for x in p):
        return failure('invalid-state')
    for index in range(10 if preview else len(route)+1):
        event=route[index] if index<len(route) else None
        if preview:
            available=[RouteEvent(k,i) for k,items in (('circle',world.circles),('line',world.lines)) for i in range(len(items))]
            available.extend(RouteEvent('portal',key) for key in portals)
            timed=[(t,e) for e in available if (t:=event_time(e,point,current,acceleration,world,remaining,exit_id)) is not None]
            event=min(timed,key=lambda item:item[0])[1] if timed else None
        if event is None:
            if preview:
                miss,t,closest=0.,remaining,trajectory_position(point,current,acceleration,remaining)
            else:
                miss,t,closest=closest_approach_to_target(point,current,acceleration,target,remaining)
            if miss>radius or t<=EPS:
                return failure('target-miss')
        elif event.kind=='portal':
            if event.index not in portals:
                return failure('unknown-portal')
            obj=portals[event.index][0]
            t=event_time(event,point,current,acceleration,world,remaining,exit_id)
        elif event.kind=='reward':
            obj=world.rewards[int(event.index)]
            t=event_time(event,point,current,acceleration,world,remaining,exit_id)
        elif event.kind=='circle':
            obj=world.circles[int(event.index)]
            t=event_time(event,point,current,acceleration,world,remaining,exit_id)
        elif event.kind=='line':
            obj=world.lines[int(event.index)]
            t=event_time(event,point,current,acceleration,world,remaining,exit_id)
        else:
            return failure('invalid-route-event')
        if t is None:
            return failure('missing-'+event.kind)
        intended_physical=event
        if event is not None and event.kind=='reward':
            intended_physical=next((e for e in route[index+1:] if e.kind!='reward'),None)
            if intended_physical is not None:
                physical_time=event_time(intended_physical,point,current,acceleration,world,remaining,exit_id)
                if physical_time is not None and physical_time<t-EPS:
                    return failure('physical-event-before-waypoint')
        reason,gap=_blocked(point,current,acceleration,world,t,intended_physical,departed,exit_id)
        if reason:
            return failure(reason)
        clearance=min(clearance,gap)
        # Reward regions are passive and do not interrupt free flight.
        acquired=[]
        for reward in getattr(world,'rewards',()):
            if reward.object_id in reward_ids:
                continue
            rt=circle_entry(point,current,acceleration,reward.center,trigger_radius(reward),t)
            if rt is not None:
                acquired.append((rt,reward))
        for rt,reward in sorted(acquired,key=lambda item:item[0]):
            reward_ids.append(reward.object_id);multiplier*=reward.multiplier
            trace.append({'kind':'reward','id':reward.object_id,'time':elapsed+rt,
                          'point':trajectory_position(point,current,acceleration,rt)})
        segments.append({'start':point,'velocity':current,'acceleration':tuple(acceleration),
                         'duration':float(t),'elapsed':elapsed})
        contact=trajectory_position(point,current,acceleration,t)
        incoming=trajectory_velocity(current,acceleration,t)
        elapsed+=t;remaining-=t
        if event is None:
            return {'valid':True,'status':'reachable','segments':segments,'event_trace':trace,
                    'launch_point':tuple(source),'miss_distance':float(miss),'clearance':float(clearance),
                    'flight_time_seconds':elapsed,'closest_target_point':tuple(closest),
                    'portal_count':len(portal_sequence),'portal_sequence':portal_sequence,
                    'reflection_count':reflections,'reward_ids':reward_ids,'damage_multiplier':multiplier,
                    'events':[e['kind'] for e in trace]+['target'],**bounce}
        if event.kind=='portal':
            portal,partner,exit_id=portals[event.index]
            point=translate(contact,portal,partner)
            current=incoming;portal_sequence.append(event.index);departed=None
        elif event.kind in {'line','circle'}:
            if event.kind=='line':
                d=np.subtract(obj.end,obj.start);length=hypot(*d)
                if length<=1e-8:
                    return failure('invalid-line')
                fraction=float(np.dot(np.subtract(contact,obj.start),d)/(length*length))
                if not contact_allowed(fraction):
                    return failure('line-endpoint-margin')
                normal=(-d[1]/length,d[0]/length)
            else:
                delta=np.subtract(contact,obj.center);length=hypot(*delta)
                if length<=1e-8:
                    return failure('invalid-circle')
                normal=tuple(delta/length)
            current=reflect_velocity(incoming,normal);point=contact;departed=event
            incidence=abs(float(np.dot(incoming,normal)))/max(hypot(*incoming),1e-12)
            if incidence<REFLECTION_MIN_INCIDENCE:
                return failure('grazing-reflection')
            reflections+=1;exit_id=None
            bounce={'reflection_point':contact,'reflection_obstacle':{'kind':event.kind,'index':event.index},
                    'actual_v_before':incoming,'actual_v_after':current,'incidence':incidence,
                    'reflection_time_seconds':elapsed}
        else:
            point,current=contact,incoming
        if event.kind!='reward':
            trace.append({'kind':'reflection' if event.kind in {'circle','line'} else event.kind,
                          'id':event.index,'time':elapsed,'point':contact})
        if remaining<=EPS:
            return failure('time-limit')
    return failure('incomplete-route')
