"""Continuous normal-shot candidates followed by small integer replay."""
from math import cos, radians, sin

from .ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE, _wind_acceleration, solve_target
from .portal_replay import replay_portal_shot
from .solver_config import INTEGER_ANGLE_RADIUS, MISS_TIE_THRESHOLD_AT_REFERENCE
from .wormhole_solver import solve_ballistic_for_speed

NORMAL_HIGH_POWER_DEVIATION = 6
NORMAL_HIGH_ANGLE_RADIUS = 3


def _select_normal_candidate(candidates, arc_preference):
    """Choose the lowest-power valid shot for low-arc aiming."""
    if arc_preference == 'low':
        return min(candidates, key=lambda c: (
            c['power'], c['miss_distance'], -c['clearance'], c['angle_degrees'],
        ))
    return min(candidates, key=lambda c: (
        c['miss_distance'], -c['angle_degrees'], -c['clearance'], -c['power'],
    ))


def solve_normal_integer_shot(source, target, world, wind_value, wind_direction,
                              image_width, *, arc_preference='low', force_power=None):
    scale = image_width / 1920
    acceleration = (_wind_acceleration(wind_value, wind_direction, image_width), GRAVITY_AT_REFERENCE * scale)
    theory = solve_target(*source, *target, wind_value, wind_direction, image_width)
    minimum = theory.get('minimum_power', {})
    diagnostics = {
        'target': {'x': float(target[0]), 'y': float(target[1])},
        'wind_value': wind_value, 'wind_direction': wind_direction,
        'theory_angle': float(minimum.get('angle_degrees') or 0.0),
        'theory_power': float(minimum.get('power') or 0.0),
        'candidate_count': 0, 'verified_count': 0, 'rejected_reasons': {},
    }
    if arc_preference == 'high':
        arcs = theory['power_100']['solutions']
        if not arcs:
            return {'status': 'unreachable', 'reason': 'no-verified-shot', 'diagnostics': diagnostics}
        seed = max(arcs, key=lambda a: a['angle_degrees'])
        powers = range(100-NORMAL_HIGH_POWER_DEVIATION, 101)
        diagnostics['theory_angle'] = float(seed['angle_degrees'])
        diagnostics['theory_power'] = 100.0
    else:
        seed = theory['minimum_power']
        if seed.get('status') != 'reachable' or not seed['within_power_limit']:
            return {'status': 'unreachable', 'reason': 'no-verified-shot'}
        center = round(seed['power'])
        powers = range(max(1, center-3), min(100, center+3)+1)
    if force_power is not None:
        powers = (force_power,)
    angle_center = round(seed['angle_degrees'])
    direction = seed['direction']
    candidates = []
    for power in powers:
        centers = [angle_center]
        if arc_preference == 'low':
            arcs = tuple(arc for arc in solve_ballistic_for_speed(source, target, acceleration, power*SPEED_PER_POWER_AT_REFERENCE*scale)
                         if (arc.velocity[0] >= 0) == (direction == 'right'))
            centers += [round(arc.angle_degrees) for arc in arcs]
        angle_radius = NORMAL_HIGH_ANGLE_RADIUS if arc_preference == 'high' else INTEGER_ANGLE_RADIUS
        angles = sorted({angle for center in centers for angle in range(max(0, center-angle_radius), min(90, center+angle_radius)+1)})
        for angle in angles:
            diagnostics['candidate_count'] += 1
            speed = power * SPEED_PER_POWER_AT_REFERENCE * scale
            velocity = ((1 if direction == 'right' else -1)*speed*cos(radians(angle)), -speed*sin(radians(angle)))
            replay = replay_portal_shot(source, velocity, acceleration, world, target, image_width)
            if replay.valid:
                diagnostics['verified_count'] += 1
                candidates.append({'status': 'reachable', 'direction': direction, 'angle_degrees': angle,
                                   'power': power, 'miss_distance': replay.miss_distance,
                                   'clearance': replay.clearance, 'flight_time_seconds': replay.time,
                                   'portal_count': 0, 'reflection_count': 0, 'events': ['target']})
            else:
                reason = replay.invalid_reason or 'target-miss'
                reasons = diagnostics['rejected_reasons']
                reasons[reason] = reasons.get(reason, 0) + 1
    if not candidates:
        return {'status': 'unreachable', 'reason': 'no-verified-shot', 'diagnostics': diagnostics}
    best_miss = min(c['miss_distance'] for c in candidates)
    reliable = [c for c in candidates if c['miss_distance'] <= best_miss + MISS_TIE_THRESHOLD_AT_REFERENCE*scale]
    result = _select_normal_candidate(reliable, arc_preference)
    result['diagnostics'] = diagnostics
    return result
