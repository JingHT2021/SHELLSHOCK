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
from shellshock.config.solver import (
    INTEGER_REPLAY_LIMIT, SEEDS_PER_ROUTE,
    MAX_B_SEEDS_PER_ROUTE, MAX_B_SEEDS_GLOBAL, MAX_FINAL_CANDIDATES,
)


def _route_score(world, route):
    """Return the existing reward/event priority available before replay."""
    reward_multiplier = 1.0
    portal_count = reflection_count = 0
    for event in route:
        if event.kind == 'reward':
            reward_multiplier *= float(world.rewards[int(event.index)].multiplier)
        elif event.kind == 'portal':
            portal_count += 1
        elif event.kind in {'circle', 'line'}:
            reflection_count += 1
    return (reward_multiplier, portal_count + reflection_count, portal_count, reflection_count)


def _seed_sort_key(item):
    score = item.get('score', ())
    if not isinstance(score, (tuple, list)):
        score = (score,)
    return tuple(score) + (
        -int(item.get('seed_rank', 0)),
        -float(item.get('power', 0.0)),
        -float(item.get('angle', 0.0)),
        str(item.get('branch_id', '')),
    )


def select_b_seeds(seeds, *, per_route=MAX_B_SEEDS_PER_ROUTE,
                   global_limit=MAX_B_SEEDS_GLOBAL):
    """Keep the best continuous seeds per route, then globally."""
    grouped = {}
    for seed in seeds:
        grouped.setdefault(seed['route_id'], []).append(seed)
    per_route_seeds = []
    for route_seeds in grouped.values():
        per_route_seeds.extend(sorted(route_seeds, key=_seed_sort_key, reverse=True)[:per_route])
    return sorted(per_route_seeds, key=_seed_sort_key, reverse=True)[:global_limit]


def _candidate_sort_key(item):
    score = item.get('score', ())
    if not isinstance(score, (tuple, list)):
        score = (score,)
    return tuple(score) + (
        -float(item.get('integer_deviation', 0.0)),
        -float(item.get('power', 0.0)),
        -float(item.get('angle', 0.0)),
        str(item.get('candidate_id', '')),
    )


def select_c2_candidates(candidates, *, limit=MAX_FINAL_CANDIDATES, failed=()):
    """Select a diverse ranked prefix, then use ranked candidates as refill."""
    failed = set(failed)
    available = [item for item in candidates if item.get('candidate_id') not in failed]
    selected = []
    used_routes = set()
    deferred = []
    for item in available:
        if len(selected) >= limit:
            break
        route_id = item.get('route_id')
        if route_id in used_routes:
            deferred.append(item)
            continue
        selected.append(item)
        used_routes.add(route_id)
    if len(selected) < limit:
        for item in deferred:
            if len(selected) >= limit:
                break
            selected.append(item)
    return selected

def solve_routes(source,target,world,wind_value,wind_direction,image_width,family,variant,
                 *, initial_results=(), route_limit=None, replay_limit=INTEGER_REPLAY_LIMIT, barrel_length=35.0):
    started=perf_counter();scale=image_width/1920
    layer_a_seconds=layer_b_seconds=layer_c1_seconds=layer_c2_seconds=0.0
    layer_a_started=perf_counter()
    acceleration=(_wind_acceleration(wind_value,wind_direction,image_width),GRAVITY_AT_REFERENCE*scale)
    speed_per_power=SPEED_PER_POWER_AT_REFERENCE*scale
    routes,truncated=candidate_routes(world,family,route_limit)
    results=list(initial_results);tested=set();replay_count=0;seed_count=0;a_rejected=0;route_trace=[];candidate_trace=[]
    route_id_by_key={}; b_seed_records=[]
    for route_index, route in enumerate(routes, 1):
        if not route:continue
        has_reflection=any(e.kind in {'circle','line'} for e in route)
        route_a_started=perf_counter()
        assessment=analyze_route(source,target,world,route,acceleration,barrel_length*image_width/2560,24*scale)
        layer_a_seconds += perf_counter()-route_a_started
        route_id=f'A-{route_index}'
        route_key=tuple((e.kind,e.index) for e in route)
        route_id_by_key[route_key] = route_id
        trace={'id':route_id,'route':[(e.kind,e.index) for e in route],
               'layer_a':'UNKNOWN' if assessment.possible else 'REJECT',
               'reason':assessment.reason,'interval_boxes':assessment.boxes_checked,
               'contractions':assessment.contractions}
        route_trace.append(trace)
        if not assessment.possible:
            trace['layer_b'] = 'SKIP'
            trace['layer_b_reason'] = 'layer-a-rejected'
            a_rejected+=1
            continue
        seeds=_reflection_seeds if has_reflection else _waypoint_seeds
        route_b_started=perf_counter()
        velocities=list(seeds(source,target,world,route,acceleration,speed_per_power))
        legal=[]
        for seed_index, v in enumerate(velocities, 1):
            angle=degrees(atan2(-v[1],abs(v[0])));power=hypot(*v)/speed_per_power
            if 0<=angle<=90 and .5<=power<=102:
                legal.append((angle,power,'right' if v[0]>=0 else 'left',f'B-{route_index}-{seed_index}'))
        circular=any(e.kind=='circle' for e in route)
        legal.sort(key=lambda x:-x[1] if circular else abs(x[0]-round(x[0]))+abs(x[1]-round(x[1])))
        trace['continuous_seeds']=len(legal)
        trace['layer_b'] = 'PASS' if legal else 'REJECT'
        if not legal:
            trace['layer_b_reason'] = 'no-continuous-seeds'
        layer_b_seconds += perf_counter()-route_b_started
        route_score = _route_score(world, route)
        trace['branches']=[{'id': branch_id, 'route_id': route_id, 'seed_index': int(branch_id.rsplit('-', 1)[1])} for _, _, _, branch_id in legal[:SEEDS_PER_ROUTE]]
        for seed_rank, (angle, power, direction, branch_id) in enumerate(legal[:SEEDS_PER_ROUTE]):
            b_seed_records.append({'route': route, 'route_id': route_id, 'angle': angle,
                                   'power': power, 'direction': direction,
                                   'branch_id': branch_id, 'seed_rank': seed_rank,
                                   'score': route_score})

    selected_b = select_b_seeds(b_seed_records)
    seed_count = len(selected_b)
    for trace in route_trace:
        trace['selected_branches'] = [item['branch_id'] for item in selected_b if item['route_id'] == trace['id']]

    c1_started = perf_counter()
    c1_raw = []
    for seed in selected_b:
        for route,direction,angle,power,branch_id in integer_candidates(
                seed['route'], [(seed['angle'], seed['power'], seed['direction'], seed['branch_id'])], family, variant):
            c1_raw.append({'route': route, 'route_id': seed['route_id'], 'direction': direction,
                           'angle': angle, 'power': power, 'branch_id': branch_id,
                           'score': seed['score'],
                           'integer_deviation': abs(angle-seed['angle']) + abs(power-seed['power'])})
    c1_unique = []
    seen_candidates = set()
    for item in sorted(c1_raw, key=_candidate_sort_key, reverse=True):
        key=(tuple(item['route']),item['direction'],item['angle'],item['power'])
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        item['candidate_id'] = f'C1-{len(c1_unique)+1}'
        c1_unique.append(item)
    layer_c1_seconds += perf_counter() - c1_started

    c2_successes = []
    failed_candidates = set()
    candidate_pool = list(c1_unique)
    while len(c2_successes) < MAX_FINAL_CANDIDATES and candidate_pool:
        selected = select_c2_candidates(candidate_pool, limit=MAX_FINAL_CANDIDATES, failed=failed_candidates)
        if not selected:
            break
        candidate = selected[0]
        candidate_pool.remove(candidate)
        route,direction,angle,power,branch_id = (candidate['route'], candidate['direction'],
                                                 candidate['angle'], candidate['power'], candidate['branch_id'])
        key=(route,direction,angle,power)
        if key in tested:
            failed_candidates.add(candidate['candidate_id'])
            continue
        if replay_limit is not None and replay_count>=replay_limit:
            truncated=True
            break
        tested.add(key);replay_count+=1
        start=muzzle_position(source,direction,angle,image_width,barrel_length=barrel_length)
        speed=power*speed_per_power;rad=radians(angle)
        velocity=((1 if direction=='right' else -1)*speed*cos(rad),-speed*sin(rad))
        replay_started=perf_counter()
        replay=replay_route(start,velocity,acceleration,world,target,route)
        layer_c2_seconds += perf_counter()-replay_started
        candidate_id=f'C-{replay_count}'
        route_key=tuple((e.kind,e.index) for e in route)
        route_id=route_id_by_key.get(route_key)
        candidate_trace.append({'id':candidate_id,'candidate_id':candidate['candidate_id'],'route_id':route_id,'branch_id':branch_id,'route':list(route_key), 'angle':angle, 'power':power,
                                'layer_c1':'PASS', 'layer_c2':'PASS' if replay['valid'] else 'REJECT',
                                'reason':None if replay['valid'] else replay.get('reason','target-miss')})
        if replay['valid']:
            results.append({**replay,'route':list(route_key),'power':power,'angle_degrees':angle,'direction':direction,
                            'selected_route_id':route_id,
                            'selected_branch_id':branch_id,'selected_candidate_id':candidate_id})
            c2_successes.append(candidate)
        else:
            failed_candidates.add(candidate['candidate_id'])
    timing={'layer_a_seconds':layer_a_seconds,'layer_b_seconds':layer_b_seconds,
            'layer_c1_seconds':layer_c1_seconds,'layer_c2_seconds':layer_c2_seconds,
            'total_seconds':perf_counter()-started}
    diagnostics={'layer_a_generated':len(routes),'layer_a_rejected':a_rejected,'layer_b_seed_count':seed_count,
                 'verified_count':len(results),'candidate_count':replay_count,'budget_exhausted':truncated,
                 'layer_a_passed':len(routes)-a_rejected,'layer_b_passed':sum(bool(t.get('selected_branches')) for t in route_trace),
                 'b_seed_generated':sum(t.get('continuous_seeds',0) for t in route_trace),
                 'b_seed_selected':seed_count,'c1_candidates_raw':len(c1_raw),'c1_candidates_unique':len(c1_unique),
                 'c2_successes':len(c2_successes),'c2_failures':len(failed_candidates),
                 'integer_candidates_raw':len(c1_raw),'integer_candidates_unique':len(c1_unique),
                 'integer_full_replays':replay_count,'integer_replay_failed':len(failed_candidates),
                 'route_trace':route_trace,'candidate_trace':candidate_trace,
                 'elapsed_seconds':timing['total_seconds'],'timing':timing,'engine':'shared-events'}
    if not results:
        return {'status':'unreachable','reason':'search-budget-exhausted' if truncated else 'no-verified-shot','diagnostics':diagnostics}
    best=select_result(results,variant)
    selected_candidate=next((item for item in candidate_trace if item.get('id')==best.get('selected_candidate_id')), None)
    selected_route=next((item for item in route_trace if item.get('id')==best.get('selected_route_id')), None)
    selected_branch=next((item for item in selected_route.get('branches', ()) if item.get('id')==best.get('selected_branch_id')), None) if selected_route else None
    diagnostics['selected_route']=selected_route
    diagnostics['selected_branch']=selected_branch
    diagnostics['selected_candidate']=selected_candidate
    finals=[];remaining=list(results)
    while remaining and len(finals)<5:
        candidate=select_result(remaining,variant)
        finals.append({k:v for k,v in candidate.items() if k!='diagnostics'})
        remaining=[r for r in remaining if (r['direction'],r['angle_degrees'],r['power'])!=(candidate['direction'],candidate['angle_degrees'],candidate['power'])]
    diagnostics['final_results']=finals
    return {**best,'diagnostics':diagnostics}

