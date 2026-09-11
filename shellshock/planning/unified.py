"""Shared A/B/C orchestration, with bounded fair integer verification."""
from math import atan2,cos,sin,degrees,hypot,radians
from time import perf_counter
from shellshock.math2d.ballistics import GRAVITY_AT_REFERENCE,SPEED_PER_POWER_AT_REFERENCE,_wind_acceleration
from shellshock.physics.engine import replay_route
from shellshock.physics.launch import muzzle_position
from shellshock.planning.routes import candidate_routes
from shellshock.planning.ranking import priority,select_result
from shellshock.planning.layer_a import analyze_route
from shellshock.planning.layer_b import _unfold,_waypoint_seeds,_reflection_seeds
from shellshock.planning.layer_c import integer_candidates
from shellshock.planning.iteration import _round_robin
from shellshock.config.solver import ROUTE_LIMIT,INTEGER_REPLAY_LIMIT,SEEDS_PER_ROUTE

def solve_routes(source,target,world,wind_value,wind_direction,image_width,family,variant,
                 *, initial_results=(), route_limit=ROUTE_LIMIT, replay_limit=INTEGER_REPLAY_LIMIT, barrel_length=35.0):
    started=perf_counter();scale=image_width/1920
    acceleration=(_wind_acceleration(wind_value,wind_direction,image_width),GRAVITY_AT_REFERENCE*scale)
    speed_per_power=SPEED_PER_POWER_AT_REFERENCE*scale
    routes,truncated=candidate_routes(world,family,route_limit)
    results=list(initial_results);tested=set();replay_count=0;seed_count=0;a_rejected=0;route_trace=[];queues=[]
    for route in routes:
        if not route:continue
        has_reflection=any(e.kind in {'circle','line'} for e in route)
        assessment=analyze_route(source,target,world,route,acceleration,barrel_length*image_width/2560,24*scale)
        trace={'route':[(e.kind,e.index) for e in route],
               'layer_a':'UNKNOWN' if assessment.possible else 'REJECT',
               'reason':assessment.reason,'interval_boxes':assessment.boxes_checked,
               'contractions':assessment.contractions}
        route_trace.append(trace)
        if not assessment.possible:
            a_rejected+=1
            continue
        seeds=_reflection_seeds if has_reflection else _waypoint_seeds
        velocities=list(seeds(source,target,world,route,acceleration,speed_per_power))
        legal=[]
        for v in velocities:
            angle=degrees(atan2(-v[1],abs(v[0])));power=hypot(*v)/speed_per_power
            if 0<=angle<=90 and .5<=power<=102:
                legal.append((angle,power,'right' if v[0]>=0 else 'left'))
        circular=any(e.kind=='circle' for e in route)
        legal.sort(key=lambda x:-x[1] if circular else abs(x[0]-round(x[0]))+abs(x[1]-round(x[1])))
        trace['continuous_seeds']=len(legal)
        if len(legal)>SEEDS_PER_ROUTE:truncated=True
        seed_count+=min(SEEDS_PER_ROUTE,len(legal));queues.append(integer_candidates(route,legal[:SEEDS_PER_ROUTE],family,variant))
    for route,direction,angle,power in _round_robin(queues):
        key=(route,direction,angle,power)
        if key in tested:continue
        if replay_count>=replay_limit:
            truncated=True;break
        tested.add(key);replay_count+=1
        start=muzzle_position(source,direction,angle,image_width,barrel_length=barrel_length)
        speed=power*speed_per_power;rad=radians(angle)
        velocity=((1 if direction=='right' else -1)*speed*cos(rad),-speed*sin(rad))
        replay=replay_route(start,velocity,acceleration,world,target,route)
        if replay['valid']:
            results.append({**replay,'power':power,'angle_degrees':angle,'direction':direction})
    diagnostics={'layer_a_generated':len(routes),'layer_a_rejected':a_rejected,'layer_b_seed_count':seed_count,
                 'verified_count':len(results),'candidate_count':replay_count,'budget_exhausted':truncated,
                 'route_trace':route_trace,'elapsed_seconds':perf_counter()-started,'engine':'shared-events'}
    if not results:
        return {'status':'unreachable','reason':'search-budget-exhausted' if truncated else 'no-verified-shot','diagnostics':diagnostics}
    best=select_result(results,variant)
    finals=[];remaining=list(results)
    while remaining and len(finals)<5:
        candidate=select_result(remaining,variant)
        finals.append({k:v for k,v in candidate.items() if k!='diagnostics'})
        remaining=[r for r in remaining if (r['direction'],r['angle_degrees'],r['power'])!=(candidate['direction'],candidate['angle_degrees'],candidate['power'])]
    diagnostics['final_results']=finals
    return {**best,'diagnostics':diagnostics}

