"""Continuous normal-shot candidates followed by small integer replay."""
from math import cos, radians, sin
from time import perf_counter

from shellshock.math2d.ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE, _wind_acceleration, solve_target
from shellshock.physics.engine import replay_route
from shellshock.physics.launch import muzzle_position
from shellshock.config.solver import INTEGER_ANGLE_RADIUS, MISS_TIE_THRESHOLD_AT_REFERENCE
from shellshock.math2d.shots import solve_ballistic_for_speed
from shellshock.planning.layer_a import _route_possible

NORMAL_HIGH_POWER_DEVIATION = 10
NORMAL_HIGH_ANGLE_RADIUS = 3


def _select_normal_candidate(candidates, arc_preference):
    """Choose the lowest-power valid shot for low-arc aiming."""
    if arc_preference == 'low':
        return min(candidates, key=lambda c: (
            -c.get('damage_multiplier', 1), c['power'], c['miss_distance'], -c['clearance'], c['angle_degrees'],
        ))
    return min(candidates, key=lambda c: (
        -c.get('damage_multiplier', 1), -c.get('portal_count', 0), -c.get('reflection_count', 0),
        c['miss_distance'], -c['angle_degrees'], -c['clearance'], -c['power'],
    ))


def solve_normal_integer_shot(source, target, world, wind_value, wind_direction,
                              image_width, *, arc_preference='low', force_power=None, tank_center=None, barrel_length=35.0):
    """Use source as the true muzzle, or derive each muzzle from tank_center."""
    started = perf_counter()
    layer_a_started = perf_counter()
    scale = image_width / 1920
    acceleration = (_wind_acceleration(wind_value, wind_direction, image_width), GRAVITY_AT_REFERENCE * scale)
    if not _route_possible(source, target, world, (), acceleration,
                           barrel_length*image_width/2560 if tank_center is not None else 0, 24*scale):
        return {'status': 'unreachable', 'reason': 'continuous-route-rejected',
                'diagnostics': {'layer_a_generated': 1, 'layer_a_rejected': 1,
                                'timing': {'layer_a_seconds': perf_counter()-layer_a_started,
                                           'layer_b_seconds': 0.0, 'layer_c1_seconds': 0.0,
                                           'layer_c2_seconds': 0.0, 'total_seconds': perf_counter()-started}}}
    layer_a_seconds = perf_counter() - layer_a_started
    layer_b_started = perf_counter()
    theory = solve_target(*source, *target, wind_value, wind_direction, image_width)
    minimum = theory.get('minimum_power', {})
    diagnostics = {
        'target': {'x': float(target[0]), 'y': float(target[1])},
        'wind_value': wind_value, 'wind_direction': wind_direction,
        'theory_angle': float(minimum.get('angle_degrees') or 0.0),
        'theory_power': float(minimum.get('power') or 0.0),
        'layer_a_generated': 1, 'layer_a_rejected': 0, 'engine': 'shared-events',
        'candidate_count': 0, 'verified_count': 0, 'rejected_reasons': {},
    }
    if arc_preference == 'high':
        arcs = theory['power_100']['solutions']
        if not arcs:
            diagnostics['timing'] = {'layer_a_seconds': layer_a_seconds, 'layer_b_seconds': perf_counter()-layer_b_started,
                                     'layer_c1_seconds': 0.0, 'layer_c2_seconds': 0.0, 'total_seconds': perf_counter()-started}
            return {'status': 'unreachable', 'reason': 'no-verified-shot', 'diagnostics': diagnostics}
        seed = max(arcs, key=lambda a: a['angle_degrees'])
        powers = range(100-NORMAL_HIGH_POWER_DEVIATION, 101)
        diagnostics['theory_angle'] = float(seed['angle_degrees'])
        diagnostics['theory_power'] = 100.0
    else:
        seed = theory['minimum_power']
        if seed.get('status') != 'reachable' or not seed['within_power_limit']:
            diagnostics['timing'] = {'layer_a_seconds': layer_a_seconds, 'layer_b_seconds': perf_counter()-layer_b_started,
                                     'layer_c1_seconds': 0.0, 'layer_c2_seconds': 0.0, 'total_seconds': perf_counter()-started}
            return {'status': 'unreachable', 'reason': 'no-verified-shot', 'diagnostics': diagnostics}
        center = round(seed['power'])
        powers = range(max(1, center-3), min(100, center+3)+1)
    layer_b_seconds = perf_counter() - layer_b_started
    layer_c1_seconds = 0.0
    layer_c2_seconds = 0.0
    if force_power is not None:
        powers = (force_power,)
    angle_center = round(seed['angle_degrees'])
    direction = seed['direction']
    candidates = []
    candidate_trace = []
    for power in powers:
        c1_started = perf_counter()
        centers = [angle_center]
        arcs = tuple(arc for arc in solve_ballistic_for_speed(source, target, acceleration, power*SPEED_PER_POWER_AT_REFERENCE*scale)
                     if (arc.velocity[0] >= 0) == (direction == 'right'))
        if arc_preference == 'high':
            if not arcs:
                continue
            centers = [round(max(arcs, key=lambda arc: arc.angle_degrees).angle_degrees)]
        else:
            centers += [round(arc.angle_degrees) for arc in arcs]
        angle_radius = NORMAL_HIGH_ANGLE_RADIUS if arc_preference == 'high' else INTEGER_ANGLE_RADIUS
        angles = sorted({angle for center in centers for angle in range(max(0, center-angle_radius), min(90, center+angle_radius)+1)})
        layer_c1_seconds += perf_counter() - c1_started
        for angle in angles:
            diagnostics['candidate_count'] += 1
            speed = power * SPEED_PER_POWER_AT_REFERENCE * scale
            velocity = ((1 if direction == 'right' else -1)*speed*cos(radians(angle)), -speed*sin(radians(angle)))
            launch = source if tank_center is None else muzzle_position(tank_center, direction, angle, image_width,barrel_length=barrel_length)
            c2_started = perf_counter()
            replay = replay_route(launch, velocity, acceleration, world, target, route=())
            layer_c2_seconds += perf_counter() - c2_started
            candidate_trace.append({
                'route': [], 'angle': angle, 'power': power,
                'layer_c1': 'PASS', 'layer_c2': 'PASS' if replay['valid'] else 'REJECT',
                'reason': None if replay['valid'] else (replay.get('reason') or 'target-miss'),
            })
            if replay['valid']:
                diagnostics['verified_count'] += 1
                candidates.append({**replay, 'direction': direction, 'angle_degrees': angle,
                                   'power': power})
            else:
                reason = replay.get('reason') or 'target-miss'
                reasons = diagnostics['rejected_reasons']
                reasons[reason] = reasons.get(reason, 0) + 1
    if not candidates:
        diagnostics['candidate_trace'] = candidate_trace
        diagnostics['timing'] = {'layer_a_seconds': layer_a_seconds, 'layer_b_seconds': layer_b_seconds,
                                 'layer_c1_seconds': layer_c1_seconds, 'layer_c2_seconds': layer_c2_seconds,
                                 'total_seconds': perf_counter()-started}
        return {'status': 'unreachable', 'reason': 'no-verified-shot', 'diagnostics': diagnostics}
    best_reward = max(c.get('damage_multiplier', 1) for c in candidates)
    candidates = [c for c in candidates if c.get('damage_multiplier', 1) == best_reward]
    reliable = candidates
    if arc_preference == 'low':
        best_miss = min(c['miss_distance'] for c in candidates)
        reliable = [c for c in candidates if c['miss_distance'] <= best_miss + MISS_TIE_THRESHOLD_AT_REFERENCE*scale]
    result = _select_normal_candidate(reliable, arc_preference)
    result['diagnostics'] = diagnostics
    diagnostics['candidate_trace'] = candidate_trace
    diagnostics['timing'] = {'layer_a_seconds': layer_a_seconds, 'layer_b_seconds': layer_b_seconds,
                             'layer_c1_seconds': layer_c1_seconds, 'layer_c2_seconds': layer_c2_seconds,
                             'total_seconds': perf_counter()-started}
    return result
