import pytest
from shellshock_detector.world_geometry import CircleObstacle, LineObstacle, World


def test_circle_surface_roots_include_inner_exit():
    from shellshock_detector.collision import parabola_circle_roots
    assert parabola_circle_roots((0, 0), (10, 0), (0, 0), (0, 0), 5, 2) == pytest.approx((0.5,))
    assert parabola_circle_roots((-10, 0), (10, 0), (0, 0), (0, 0), 5, 2) == pytest.approx((0.5, 1.5))


def test_segment_endpoints_are_collisions_not_holes():
    from shellshock_detector.collision import parabola_segment_roots
    assert parabola_segment_roots((0, 1), (10, 0), (0, 0), LineObstacle((5, 0), (5, 10)), 1) == pytest.approx((0.5,))


def test_closest_approach_finds_interior_minimum():
    from shellshock_detector.collision import closest_approach_to_target
    miss, time, point = closest_approach_to_target((0, 0), (10, 0), (0, 0), (5, 2), 1)
    assert (miss, time) == pytest.approx((2, 0.5))
    assert point == pytest.approx((5, 0))


def test_first_collision_returns_actual_obstacle_and_point():
    from shellshock_detector.collision import find_first_collision
    world = World(lines=[LineObstacle((7, -10), (7, 10))], circles=[CircleObstacle((5, 0), 1)])
    contact = find_first_collision((0, 0), (10, 0), (0, 0), world, 1)
    assert contact.obstacle_kind == 'circle'
    assert contact.time == pytest.approx(0.4)
    assert contact.point == pytest.approx((4, 0))


def test_clearance_is_distance_to_finite_segment():
    from shellshock_detector.collision import trajectory_clearance
    world = World(lines=[LineObstacle((5, 3), (5, 7))])
    assert trajectory_clearance((0, 0), (10, 0), (0, 0), world, 1) == pytest.approx(3)


def test_circle_aabb_reject_skips_polynomial_roots(monkeypatch):
    from shellshock_detector import collision
    monkeypatch.setattr(collision.np, 'roots', lambda *args: pytest.fail('distant circle should be rejected'))
    assert collision.parabola_circle_roots((0, 0), (10, 0), (0, 2), (100, 100), 2, 1) == ()


def test_duplicate_tangent_roots_cannot_hide_simultaneous_line():
    from shellshock_detector.collision import find_first_collision
    world = World(circles=[CircleObstacle((5, 1), 1)], lines=[LineObstacle((5, -1), (5, 1))])
    assert find_first_collision((0, 0), (10, 0), (0, 0), world, 1).obstacle_kind == 'ambiguous'


def test_accelerated_circle_roots_reach_surface():
    from shellshock_detector.collision import parabola_circle_roots, trajectory_position
    roots = parabola_circle_roots((0, 0), (2, 0), (0, 2), (2, 1), .5, 2)
    assert len(roots) == 2
    assert roots[0] < 1 < roots[1]
    for t in roots:
        x, y = trajectory_position((0, 0), (2, 0), (0, 2), t)
        assert (x-2)**2 + (y-1)**2 == pytest.approx(.25)
