import math
from dataclasses import replace

import pytest

from shellshock_detector.combined_replay import replay_combined_shot
from shellshock_detector.reflection_solver import SurfaceFamily
from shellshock_detector.world_geometry import World, LineObstacle, Portal, PortalPair


VELOCITY = (70*9.836246/math.sqrt(2), -70*9.836246/math.sqrt(2))
FAMILY = SurfaceFamily('line', 0, 'BOTH', 0, 1)


def pair(entry, exit, radius=10):
    return PortalPair(Portal('orange', entry, radius), Portal('blue', exit, radius))


def scene(pre=False, post=False):
    wall_x = 700 if pre else 300
    portals = []
    if pre:
        portals.append(pair((100, -100), (500, -100)))
    if post:
        portals.append(pair((wall_x-100, -400), (1000, -400)))
    world = World(lines=(LineObstacle((wall_x, -600), (wall_x, 100)),), portal_pairs=portals)
    target = (900 if post else wall_x-200, -500)
    pre_ids = ('0:orange',) if pre else ()
    post_ids = (f'{int(pre)}:orange',) if post else ()
    return world, target, pre_ids, post_ids


@pytest.mark.parametrize('pre,post', [(False, False), (False, True), (True, False), (True, True)])
def test_manufactured_integer_velocity_combination(pre, post):
    world, target, before, after = scene(pre, post)
    shot = replay_combined_shot((0, 0), VELOCITY, (0, 0), world, target, 1920, FAMILY, before, after)
    assert shot is not None
    assert shot['miss_distance'] < 1e-6
    assert shot['reflection_count'] == 1
    assert shot['portal_sequence'] == before+after
    assert shot['events'] == ['portal']*len(before)+['reflection']+['portal']*len(after)+['target']
    assert shot['power'] == pytest.approx(70)


def test_post_portal_cannot_be_used_before_reflection():
    world, target, before, _ = scene(True, False)
    assert replay_combined_shot((0, 0), VELOCITY, (0, 0), world, target, 1920, FAMILY, (), before) is None


def test_wrong_obstacle_and_second_reflection_are_rejected():
    world, target, before, after = scene()
    first = replace(world, lines=world.lines+(LineObstacle((100, -300), (100, 50)),))
    second = replace(world, lines=world.lines+(LineObstacle((200, -500), (200, -350)),))
    assert replay_combined_shot((0, 0), VELOCITY, (0, 0), first, target, 1920, FAMILY) is None
    assert replay_combined_shot((0, 0), VELOCITY, (0, 0), second, target, 1920, FAMILY) is None


def test_grazing_planned_portal_does_not_trigger():
    world, target, before, after = scene(False, True)
    # Shift normal to the post-reflection path by 0.95r, outside trigger 0.9r.
    delta = 9.5/math.sqrt(2)
    entry = (200+delta, -400-delta)
    world = replace(world, portal_pairs=(pair(entry, (1000, -400)),))
    assert replay_combined_shot((0, 0), VELOCITY, (0, 0), world, target, 1920, FAMILY, before, after) is None


def test_continuous_expected_contact_time_and_exact_endpoint():
    world, target, before, after = scene(True, True)
    duration = 500/VELOCITY[0]
    kwargs = dict(max_time=duration, expected_contact=(700, -300), expected_time=300/VELOCITY[0], require_exact_target=True)
    shot = replay_combined_shot((0, 0), VELOCITY, (0, 0), world, target, 1920, FAMILY, before, after, **kwargs)
    assert shot is not None and shot['flight_time_seconds'] == pytest.approx(duration)
    kwargs['expected_contact'] = (710, -300)
    assert replay_combined_shot((0, 0), VELOCITY, (0, 0), world, target, 1920, FAMILY, before, after, **kwargs) is None


def test_portal_can_move_source_from_outside_circle_to_inner_reflection():
    from shellshock_detector.world_geometry import CircleObstacle
    circle = CircleObstacle((600, 0), 200)
    world = World(circles=[circle], portal_pairs=[pair((100, 0), (500, 0))])
    family = SurfaceFamily('circle', 0, 'BOTH', 0, 2*math.pi)
    result = replay_combined_shot((0, 0), (100, 0), (0, 0), world, (700, 0), 1920, family, ('0:orange',))
    assert result is not None
    assert result['reflection_side'] == 'INNER'
    assert result['reflection_point'] == pytest.approx((800, 0))
    assert world.circles[0] == circle


def test_two_transmissions_before_reflection():
    world = World(lines=[LineObstacle((1100, -600), (1100, 100))],
                  portal_pairs=[pair((100, -100), (500, -100)), pair((600, -200), (1000, -200))])
    result = replay_combined_shot((0, 0), VELOCITY, (0, 0), world, (900, -500), 1920,
                                 FAMILY, ('0:orange', '1:orange'))
    assert result is not None
    assert result['events'] == ['portal', 'portal', 'reflection', 'target']


def test_two_transmissions_after_reflection():
    world = World(lines=[LineObstacle((300, -700), (300, 100))],
                  portal_pairs=[pair((200, -400), (1000, -400)), pair((900, -500), (1400, -500))])
    result = replay_combined_shot((0, 0), VELOCITY, (0, 0), world, (1300, -600), 1920,
                                 FAMILY, (), ('0:orange', '1:orange'))
    assert result is not None
    assert result['events'] == ['reflection', 'portal', 'portal', 'target']
