"""Reflection solver integration with explicitly planned portal routes."""
import math
import numpy as np
import pytest
from shellshock_detector.world_geometry import World, LineObstacle, Portal, PortalPair


def scene(pre=False, post=True):
    source = np.array((100., 700.))
    v = np.array((70*9.836246*math.cos(math.radians(35)), -70*9.836246*math.sin(math.radians(35))))
    a = np.array((0., 379.106))
    shift_pre = np.array((300., 0.)) if pre else np.zeros(2)
    shift_post = np.array((800., 0.)) if post else np.zeros(2)
    wall_x = 800. if pre else 500.
    t1 = (wall_x-source[0]-shift_pre[0])/v[0]
    contact = source + v*t1+a*t1*t1/2+shift_pre
    incoming = v+a*t1
    outgoing = incoming*np.array((-1., 1.))
    pairs = []
    if pre:
        center = source+v*.2+a*.2**2/2
        pairs.append(PortalPair(Portal('orange', tuple(center), 20), Portal('blue', tuple(center+shift_pre), 20)))
    if post:
        center = contact+outgoing*.15+a*.15**2/2
        pairs.append(PortalPair(Portal('orange', tuple(center), 20), Portal('blue', tuple(center+shift_post), 20)))
    target = contact+outgoing*.4+a*.4**2/2+shift_post
    world = World(image_width=1920, lines=[LineObstacle((wall_x, -1000), (wall_x, 1500))], portal_pairs=pairs)
    return tuple(source), tuple(target), world, tuple(v), tuple(a), tuple(contact), t1


def test_route_unfolding_includes_both_pre_and_post_displacement():
    from shellshock_detector.reflection_routes import ReflectionRoute, unfolded_endpoints
    source, target, world, *_ = scene(True, True)
    s, t = unfolded_endpoints(source, target, world, ReflectionRoute(('0:orange',), ('1:orange',)))
    assert s == pytest.approx((source[0]+300, source[1]))
    assert t == pytest.approx((target[0]-800, target[1]))


def test_routes_cover_two_transmissions_split_around_one_bounce():
    from shellshock_detector.reflection_routes import reflection_routes
    _, _, world, *_ = scene(True, True)
    routes = list(reflection_routes(world))
    assert any(r.before == ('0:orange',) and r.after == ('1:orange',) for r in routes)
    assert any(r.before == () and r.after == () for r in routes)
    assert all(len(r.before)+len(r.after) <= 2 for r in routes)


def test_solver_finds_reflection_then_planned_portal_instead_of_avoiding_it():
    from shellshock_detector.global_solver import solve_integer_shot
    source, target, world, *_ = scene()
    result = solve_integer_shot(source, target, world, 0, 'right', 1920, 'reflection_low')
    assert result['status'] == 'reachable'
    assert result['portal_count'] >= 1
    assert result['reflection_count'] == 1
    assert result['miss_distance'] <= 24
    assert result['events'].index('reflection') < result['events'].index('portal')
    from detect_shellshock_yolo import format_aim_report
    report = format_aim_report('reflection_low', result, 0, 'right', (100, 100))
    assert 'TRIGGER' in report


@pytest.mark.parametrize('post', [False, True])
def test_solver_finds_portals_before_and_after_reflection(post):
    from shellshock_detector.global_solver import solve_integer_shot
    source, target, world, *_ = scene(True, post)
    result = solve_integer_shot(source, target, world, 0, 'right', 1920, 'reflection_low')
    assert result['status'] == 'reachable'
    assert result['events'] == ['portal', 'reflection']+(['portal'] if post else [])+['target']
    assert result['portal_count'] == 1+int(post)
    assert isinstance(result['power'], int) and isinstance(result['angle_degrees'], int)
    assert result['planned_portals_before'] == ['0:orange']
    assert result['planned_portals_after'] == (['1:orange'] if post else [])


def test_combined_low_and_high_keep_same_reflector():
    from shellshock_detector.reflection_solver import solve_reflection_bundle
    source, target, world, *_ = scene()
    bundle = solve_reflection_bundle(source, target, world, 0, 'right', 1920)
    assert bundle.low is not None and bundle.high is not None
    assert bundle.low['reflection_obstacle'] == bundle.high['reflection_obstacle']
    assert bundle.high['power'] >= bundle.low['power']


def test_solver_can_teleport_inside_a_circle_then_reflect():
    from shellshock_detector.global_solver import solve_integer_shot
    from shellshock_detector.world_geometry import CircleObstacle
    source, target, world, _, _, contact, _ = scene(True, False)
    circle = CircleObstacle((contact[0]-200, contact[1]), 200)
    world = World(image_width=1920, circles=[circle], portal_pairs=world.portal_pairs)
    result = solve_integer_shot(source, target, world, 0, 'right', 1920, 'reflection_low')
    assert result['status'] == 'reachable'
    assert result['events'] == ['portal', 'reflection', 'target']
    assert result['reflection_side'] == 'INNER'
    assert result['reflection_obstacle'] == {'kind': 'circle', 'index': 0}
    assert world.circles[0] == circle


def test_other_reflector_does_not_reuse_first_boards_route_cache():
    from shellshock_detector.global_solver import solve_integer_shot
    source, target, world, *_ = scene()
    world = World(image_width=1920, lines=world.lines+(LineObstacle((100000, 0), (100000, 100)),), portal_pairs=world.portal_pairs)
    result = solve_integer_shot(source, target, world, 0, 'right', 1920, 'reflection_low')
    assert result['status'] == 'reachable'
    assert result['reflection_obstacle'] == {'kind': 'line', 'index': 0}
    assert result['events'] == ['reflection', 'portal', 'target']
