"""Continuous normal-shot candidates followed by small integer replay."""
from math import cos, radians, sin

from .ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE, _wind_acceleration, solve_target
from .portal_replay import replay_portal_shot
from .solver_config import INTEGER_ANGLE_RADIUS, MISS_TIE_THRESHOLD_AT_REFERENCE
from .wormhole_solver import solve_ballistic_for_speed


def solve_normal_integer_shot(source, target, world, wind_value, wind_direction,
                              image_width, *, arc_preference='low', force_power=None):
    scale = image_width / 1920
    acceleration = (_wind_acceleration(wind_value, wind_direction, image_width), GRAVITY_AT_REFERENCE * scale)
    theory = solve_target(*source, *target, wind_value, wind_direction, image_width)
    if arc_preference == 'high':
        arcs = theory['power_100']['solutions']
        if not arcs:
            return {'status': 'unreachable', 'reason': 'no-verified-shot'}
        seed = max(arcs, key=lambda a: a['angle_degrees'])
        powers = (100,)
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
        if arc_preference == 'low' or (force_power is not None and force_power != 100):
            arcs = tuple(arc for arc in solve_ballistic_for_speed(source, target, acceleration, power*SPEED_PER_POWER_AT_REFERENCE*scale)
                         if (arc.velocity[0] >= 0) == (direction == 'right'))
            if arc_preference == 'high':
                if not arcs:
                    continue
                centers = [round(max(arcs, key=lambda arc: arc.angle_degrees).angle_degrees)]
            else:
                centers += [round(arc.angle_degrees) for arc in arcs]
        angles = sorted({angle for center in centers for angle in range(max(0, center-INTEGER_ANGLE_RADIUS), min(90, center+INTEGER_ANGLE_RADIUS)+1)})
        for angle in angles:
            speed = power * SPEED_PER_POWER_AT_REFERENCE * scale
            velocity = ((1 if direction == 'right' else -1)*speed*cos(radians(angle)), -speed*sin(radians(angle)))
            replay = replay_portal_shot(source, velocity, acceleration, world, target, image_width)
            if replay.valid:
                candidates.append({'status': 'reachable', 'direction': direction, 'angle_degrees': angle,
                                   'power': power, 'miss_distance': replay.miss_distance,
                                   'clearance': replay.clearance, 'flight_time_seconds': replay.time,
                                   'portal_count': 0, 'reflection_count': 0, 'events': ['target']})
    if not candidates:
        return {'status': 'unreachable', 'reason': 'no-verified-shot'}
    best_miss = min(c['miss_distance'] for c in candidates)
    reliable = [c for c in candidates if c['miss_distance'] <= best_miss + MISS_TIE_THRESHOLD_AT_REFERENCE*scale]
    return min(reliable, key=lambda c: (-c['clearance'], c['miss_distance'], -c['angle_degrees'] if arc_preference == 'high' else c['power']))
