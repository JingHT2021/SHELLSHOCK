import pytest
from shellshock_detector.world_geometry import Portal, PortalPair, World


def world_pair():
    return World(image_width=1920, portal_pairs=[PortalPair(Portal('orange', (100, 0), 20), Portal('blue', (300, 0), 20))])


def test_radius_policy_preserves_visual_geometry_and_scales_only_avoid_margin():
    from shellshock_detector.solver_config import portal_trigger_radius, portal_avoid_radius
    portal = world_pair().portal_pairs[0].orange
    assert portal_trigger_radius(portal) == 18
    assert portal_avoid_radius(portal, 2) == 30
    assert portal.radius == 20


@pytest.mark.parametrize('height,valid', [(17, True), (19, False), (20, False)])
def test_planned_portal_requires_deep_entry(height, valid):
    from shellshock_detector.portal_replay import replay_portal_shot
    result = replay_portal_shot((0, height), (100, 0), (0, 0), world_pair(), (400, height), 1920, ('0:orange',), max_time=4)
    assert result.valid is valid
    if valid:
        assert result.miss_distance == pytest.approx(0)
        assert result.portal_sequence == ('0:orange',)


def test_unplanned_portal_avoidance_rejects_visual_near_miss():
    from shellshock_detector.portal_replay import replay_portal_shot
    result = replay_portal_shot((0, 23), (100, 0), (0, 0), world_pair(), (200, 23), 1920, (), max_time=3)
    assert not result.valid
    assert result.invalid_reason == 'unplanned-portal'


def test_actual_miss_does_not_stop_at_target_radius_entry():
    from shellshock_detector.portal_replay import replay_portal_shot
    result = replay_portal_shot((0, 0), (100, 0), (0, 0), World(), (100, 2), 1920, (), max_time=2)
    assert result.valid
    assert result.miss_distance == pytest.approx(2)
    assert result.time == pytest.approx(1)


def test_unplanned_boundary_start_moving_inward_is_rejected():
    from shellshock_detector.portal_replay import replay_portal_shot
    result = replay_portal_shot((75, 0), (100, 0), (0, 0), world_pair(), (200, 0), 1920, (), max_time=2)
    assert not result.valid
    assert result.invalid_reason == 'unplanned-portal'


def test_exit_immunity_does_not_hide_later_reentry():
    from shellshock_detector.portal_replay import _entry_time
    portal = Portal('blue', (0, 0), 20)
    # x=100t-50t² exits r25, turns, and re-enters at 1+sqrt(.5).
    assert _entry_time((0, 0), (100, 0), (-100, 0), portal, 25, 3, allow_inside=True) == pytest.approx(1 + .5**.5)
