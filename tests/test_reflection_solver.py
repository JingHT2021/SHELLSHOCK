import math

import numpy as np
import pytest

from shellshock_detector.reflection_solver import fixed_contact_solutions, positive_polynomial_roots
from shellshock_detector.reflection_solver import SurfaceFamily, _fixed_power_contacts, ContactSolution
from shellshock_detector.reflection_solver import solve_reflection_bundle
from shellshock_detector.reflection_replay import replay_integer_contact
from shellshock_detector.reflection_replay import verify_continuous_contact
from shellshock_detector.world_geometry import World, LineObstacle, CircleObstacle


@pytest.mark.parametrize('wind,sign', [(25., 1), (-25., 1), (25., -1), (-25., -1)])
def test_fixed_contact_recovers_manufactured_windy_bounce(wind, sign):
    source = np.array((700., 700.))
    velocity = np.array((sign * 600., -350.))
    acceleration = np.array((wind, 379.106))
    t1, t2 = .45, .7
    contact = source + velocity*t1 + acceleration*t1*t1/2
    normal = np.array((1., 0.))
    incoming = velocity + acceleration*t1
    outgoing = incoming - 2*np.dot(incoming, normal)*normal
    target = contact + outgoing*t2 + acceleration*t2*t2/2
    solutions = fixed_contact_solutions(source, target, contact, normal, acceleration, 9.836246)
    assert any(np.linalg.norm(np.array(s.velocity)-velocity) < 1e-5 and abs(s.t1-t1) < 1e-6 for s in solutions)


def test_degenerate_quadratic_has_linear_root_and_double_root_once():
    assert positive_polynomial_roots(0, 2, -4) == [2.]
    assert positive_polynomial_roots(1, -4, 4) == [2.]
    assert positive_polynomial_roots(0, 0, 1) == []
    assert positive_polynomial_roots(0, 0, 0) == []


def test_symmetric_contact_degeneracy_recovers_fixed_power_time_family():
    # A vertical shot into a ceiling returns to its source; u=1 makes both
    # vectors in the ratio constraint vanish, without invalidating the shot.
    solutions = fixed_contact_solutions((500, 600), (500, 600), (500, 400), (0, 1), (0, 379.106), 9.836246, desired_power=70)
    assert solutions
    assert all(s.power == pytest.approx(70) for s in solutions)
    assert all(s.t1 == pytest.approx(s.t2) for s in solutions)


def manufactured(side='line'):
    source = np.array((500., 600.))
    acceleration = np.array((12., 379.106))
    speed, angle = 70*9.836246, math.radians(35)
    velocity = np.array((speed*math.cos(angle), -speed*math.sin(angle)))
    t1, t2 = .45, .35
    contact = source+velocity*t1+acceleration*t1*t1/2
    incoming = velocity+acceleration*t1
    outgoing = incoming*np.array((-1., 1.))
    target = contact+outgoing*t2+acceleration*t2*t2/2
    if side == 'line':
        world = World(lines=(LineObstacle(tuple(contact+(0, -200)), tuple(contact+(0, 200))),))
        family = SurfaceFamily('line', 0, 'BOTH', 0, 1)
    else:
        center = contact+np.array((200., 0.)) if side == 'OUTER' else contact-np.array((500., 0.))
        world = World(circles=(CircleObstacle(tuple(center), 200. if side == 'OUTER' else 500.),))
        family = SurfaceFamily('circle', 0, side, 0, 2*math.pi)
    return source, target, world, acceleration, family, contact


@pytest.mark.parametrize('side', ['line', 'INNER', 'OUTER'])
def test_integer_replay_uses_actual_normal_and_circle_side(side):
    source, target, world, acceleration, family, contact = manufactured(side)
    shot = replay_integer_contact(source, target, world, acceleration, 1920, family, 70, 35, 'right')
    assert shot is not None
    assert shot['miss_distance'] < 1e-6
    assert shot['reflection_count'] == 1 and shot['portal_count'] == 0
    assert np.linalg.norm(np.asarray(shot['reflection_point'])-contact) < 1e-6


def test_replay_rejects_first_leg_blocker():
    source, target, world, acceleration, family, _ = manufactured()
    world = World(lines=world.lines+(LineObstacle((600, 200), (600, 800)),))
    assert replay_integer_contact(source, target, world, acceleration, 1920, family, 70, 35, 'right') is None


def test_replay_rejects_second_collision_before_closest_target():
    source, target, world, acceleration, family, contact = manufactured()
    # Horizontal segment below first leg, crossed after the wall reflection.
    midpoint = (contact+target)/2
    world = World(lines=world.lines+(LineObstacle(tuple(midpoint+(-50, 0)), tuple(midpoint+(50, 0))),))
    assert replay_integer_contact(source, target, world, acceleration, 1920, family, 70, 35, 'right') is None


def test_fixed_power_includes_endpoint_and_both_tangent_branches():
    def evaluate(x):
        return [ContactSolution((0, 0), (1, 0), (1, -1), 1, 1, 10+(x-.5)**2, 45, 'right', 1, x)]
    endpoint = _fixed_power_contacts(evaluate, [(0, 0., 1.)], 10.25)
    assert [s.parameter for s in endpoint] == pytest.approx([0., 1.])
    roots = _fixed_power_contacts(evaluate, [(0, 0., 1.)], 10.04)
    assert sorted(s.parameter for s in roots) == pytest.approx([.3, .7], abs=1e-5)


def test_bundle_refines_validity_boundary_to_retain_power_100_root():
    world = World(lines=(LineObstacle((750, 200), (750, 800)),))
    bundle = solve_reflection_bundle((500, 600), (560, 450), world, 0, 'right', 1920)
    assert bundle.low_solution is not None and bundle.high_solution is not None
    assert bundle.high_solution['power'] == 100
    assert bundle.low_solution['power'] >= math.ceil(bundle.theoretical_min_power+2)
    assert bundle.low_solution['reflection_obstacle'] == bundle.high_solution['reflection_obstacle']


def test_continuous_legality_rejects_occluded_contact_before_power_minimum():
    source, target, world, acceleration, family, contact = manufactured()
    solution = min(fixed_contact_solutions(source, target, contact, (1, 0), acceleration, 9.836246), key=lambda s: abs(s.power-70))
    assert verify_continuous_contact(source, world, acceleration, 1920, family, solution)
    blocked = World(lines=world.lines+(LineObstacle((600, 200), (600, 800)),))
    assert not verify_continuous_contact(source, blocked, acceleration, 1920, family, solution)


def test_circle_side_is_classified_from_source_position():
    from dataclasses import replace
    source, target, world, acceleration, family, _ = manufactured('INNER')
    assert replay_integer_contact(source, target, world, acceleration, 1920, replace(family, side='OUTER'), 70, 35, 'right') is None


def test_high_search_falls_from_100_to_99_when_integer_replay_rejects_100(monkeypatch):
    import shellshock_detector.reflection_replay as replay_module
    original = replay_module.replay_integer_contact
    def reject_100(*args, **kwargs):
        return None if args[6] == 100 else original(*args, **kwargs)
    monkeypatch.setattr(replay_module, 'replay_integer_contact', reject_100)
    world = World(lines=(LineObstacle((750, 200), (750, 800)),))
    bundle = solve_reflection_bundle((500, 600), (560, 450), world, 0, 'right', 1920)
    assert bundle.high['power'] == 99
    assert bundle.low['low_start_power'] == math.ceil(bundle.theoretical_min_power+2)
    assert bundle.low['actual_v_before'][0] == pytest.approx(-bundle.low['actual_v_after'][0])
    assert bundle.low['endpoint_margin'] >= 6


def candidate(**updates):
    return dict(dict(miss_distance=1., clearance=10., incidence=.5, endpoint_margin=20., angle_degrees=45, power=70), **updates)


def test_low_tie_uses_incidence_then_endpoint_and_high_uses_angle():
    from shellshock_detector.reflection_solver import select_integer_candidate
    first = candidate(miss_distance=0., incidence=.3, angle_degrees=80)
    second = candidate(miss_distance=1., incidence=.7, angle_degrees=40)
    third = candidate(miss_distance=1.5, incidence=.7, endpoint_margin=25., angle_degrees=35)
    assert select_integer_candidate([first, second, third], 1.) is third
    assert select_integer_candidate([first, second, third], 1., high=True) is first
    fourth = candidate(miss_distance=4., clearance=1000.)
    assert select_integer_candidate([first, fourth], 1.) is first


def test_bundle_priority_then_aggregate_miss_tie_uses_clearance():
    from shellshock_detector.reflection_solver import ReflectionBundle, select_reflection_bundle
    circle = ReflectionBundle(candidate(), candidate(), 'circle', 0)
    line = ReflectionBundle(candidate(miss_distance=20), candidate(miss_distance=20), 'line', 1)
    assert select_reflection_bundle([circle, line], 1.) is line
    single = ReflectionBundle(candidate(), None, 'line', 2)
    assert select_reflection_bundle([single, circle], 1.) is circle
    better_clearance = ReflectionBundle(candidate(miss_distance=20.5, clearance=30), candidate(miss_distance=20.5, clearance=30), 'line', 3)
    assert select_reflection_bundle([line, better_clearance], 1.) is better_clearance


def test_single_endpoint_wrapper_falls_back(monkeypatch):
    import shellshock_detector.reflection_solver as module
    shot = candidate(status='reachable')
    monkeypatch.setattr(module, 'solve_reflection_bundle', lambda *args: module.ReflectionBundle(low_solution=shot))
    result = module.solve_reflection_integer_shot((0, 0), (1, 1), World(), 0, 'right', 1920, 'high')
    assert result['status'] == 'reachable'
    assert result['arc_fallback'] and result['selected_arc'] == 'low'


def test_real_high_only_reflector_survives_low_power_margin_and_falls_back():
    from shellshock_detector.reflection_solver import solve_reflection_integer_shot
    world = World(lines=(LineObstacle((1350, -2000), (1350, 1000)),))
    source = target = (100, 700)
    bundle = solve_reflection_bundle(source, target, world, 0, 'right', 1920)
    assert bundle.low is None
    assert bundle.high is not None
    assert bundle.high['power'] == 100
    assert bundle.high['miss_distance'] <= 24
    assert bundle.diagnostics['low_start_power'] > 100
    result = solve_reflection_integer_shot(source, target, world, 0, 'right', 1920, 'low')
    assert result['status'] == 'reachable'
    assert result['arc_fallback'] and result['selected_arc'] == 'high'
