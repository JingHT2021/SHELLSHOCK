import pytest

from shellshock_detector.projectile_events import advance_through_world
from shellshock_detector.world_geometry import CircleObstacle, Portal, PortalPair, World


def test_portal_entry_preserves_velocity_and_relative_position():
    pair = PortalPair(Portal("orange", (100, 100), 20), Portal("blue", (500, 300), 20))
    replay = advance_through_world(
        start=(0, 110), velocity=(100, 0), acceleration=(0, 0),
        world=World(portal_pairs=[pair]), max_time=2.0,
    )
    event = replay.events[0]
    assert event.kind == "portal"
    # The entry contact offset (-sqrt(300), 10) is retained at the blue exit.
    assert event.exit_point == pytest.approx((500 - 300 ** 0.5, 310))
    assert event.velocity_after == pytest.approx((100, 0))
    assert replay.portal_count == 1


def test_replay_stops_at_obstacle_before_portal():
    pair = PortalPair(Portal("orange", (100, 100), 20), Portal("blue", (500, 300), 20))
    world = World(circles=[CircleObstacle((50, 110), 10)], portal_pairs=[pair])
    replay = advance_through_world((0, 110), (100, 0), (0, 0), world, 2.0)
    assert replay.terminal_kind == "obstacle"
    assert replay.portal_count == 0


def test_target_after_portal_is_a_terminal_event():
    pair = PortalPair(Portal("orange", (100, 100), 20), Portal("blue", (500, 300), 20))
    replay = advance_through_world(
        (0, 110), (100, 0), (0, 0), World(portal_pairs=[pair]), 4.0,
        target=(600, 310), target_radius=8,
    )
    assert replay.terminal_kind == "target"
    assert replay.portal_count >= 1
