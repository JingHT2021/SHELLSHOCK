from math import cos, radians, sin
from unittest.mock import patch

import pytest

from shellshock.math2d.ballistics import GRAVITY_AT_REFERENCE, SPEED_PER_POWER_AT_REFERENCE
from shellshock.perception.world import World, RewardZone
from shellshock.physics.engine import replay_route
from shellshock.physics.launch import muzzle_position
from shellshock.planning.normal import _select_normal_candidate, solve_normal_integer_shot
from shellshock.math2d.shots import solve_ballistic_for_speed


def test_high_solves_each_power_90_through_100_and_its_own_angle_neighborhood():
    source, target = (100, 700), (850, 700)
    acceleration = (0, GRAVITY_AT_REFERENCE)
    with patch('shellshock.planning.normal.solve_ballistic_for_speed', wraps=solve_ballistic_for_speed) as analytic:
        result = solve_normal_integer_shot(source, target, World(image_width=1920), 0, 'right', 1920, arc_preference='high')
    assert [round(call.args[3] / SPEED_PER_POWER_AT_REFERENCE) for call in analytic.call_args_list] == list(range(90, 101))
    candidates = []
    for power in range(90, 101):
        arc = max(solve_ballistic_for_speed(source, target, acceleration, power*SPEED_PER_POWER_AT_REFERENCE), key=lambda a: a.angle_degrees)
        for angle in range(max(0, round(arc.angle_degrees)-3), min(90, round(arc.angle_degrees)+3)+1):
            speed = power*SPEED_PER_POWER_AT_REFERENCE
            replay = replay_route(source, (speed*cos(radians(angle)), -speed*sin(radians(angle))), acceleration, World(image_width=1920), target)
            if replay['valid']:
                candidates.append((replay['miss_distance'], power, angle))
    expected = min(candidates)
    assert result['miss_distance'] == pytest.approx(expected[0])
    assert (result['power'], result['angle_degrees']) == expected[1:]
    assert result['launch_point'] == source
    assert result['segments'][0]['start'] == source


@pytest.mark.parametrize('power', [90, 95, 100])
def test_high_force_power_solves_only_requested_power(power):
    with patch('shellshock.planning.normal.solve_ballistic_for_speed', wraps=solve_ballistic_for_speed) as analytic:
        result = solve_normal_integer_shot((100, 700), (850, 700), World(image_width=1920), 0, 'right', 1920, arc_preference='high', force_power=power)
    assert [round(call.args[3] / SPEED_PER_POWER_AT_REFERENCE) for call in analytic.call_args_list] == [power]
    assert result['power'] == power


def test_high_minimum_can_be_power_90():
    power, angle = 90, 70
    speed = power*SPEED_PER_POWER_AT_REFERENCE
    source = (100, 700)
    flight_time = 2*speed*sin(radians(angle))/GRAVITY_AT_REFERENCE
    target = (source[0]+speed*cos(radians(angle))*flight_time, source[1])
    result = solve_normal_integer_shot(source, target, World(image_width=1920), 0, 'right', 1920, arc_preference='high')
    assert result['power'] == 90
    assert result['angle_degrees'] == 70
    assert result['miss_distance'] < 1e-7


def test_normal_uses_angle_specific_muzzle_and_returns_rewards_and_segments():
    center = (100, 700)
    world = World(image_width=1920, rewards=(RewardZone(center, 100, 2, 'bonus'),))
    result = solve_normal_integer_shot(center, (850, 700), world, 0, 'right', 1920, arc_preference='high', tank_center=center)
    assert result['status'] == 'reachable'
    assert result['launch_point'] == muzzle_position(center, result['direction'], result['angle_degrees'], 1920)
    assert result['segments'][0]['start'] == result['launch_point']
    assert result['damage_multiplier'] == 2
    assert result['reward_ids'] == ['bonus']


def test_high_prioritizes_damage_then_minimum_miss():
    candidates = [
        {'miss_distance': 0.0, 'angle_degrees': 70, 'clearance': 50, 'power': 100, 'damage_multiplier': 1},
        {'miss_distance': 3.0, 'angle_degrees': 71, 'clearance': 10, 'power': 100, 'damage_multiplier': 2},
        {'miss_distance': 1.0, 'angle_degrees': 71, 'clearance': 20, 'power': 90, 'damage_multiplier': 2},
    ]
    assert _select_normal_candidate(candidates, 'high') is candidates[2]


def test_low_reward_priority_precedes_power_and_miss():
    candidates = [dict(power=30, miss_distance=0., clearance=10., angle_degrees=40, damage_multiplier=1),
                  dict(power=32, miss_distance=5., clearance=10., angle_degrees=40, damage_multiplier=2)]
    assert _select_normal_candidate(candidates, 'low')['damage_multiplier'] == 2
