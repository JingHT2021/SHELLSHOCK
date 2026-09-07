import pytest

from shellshock_detector.projectile_events import advance_through_world
from shellshock_detector.world_geometry import CircleObstacle, LineObstacle, Portal, PortalPair, World


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


def test_replay_can_reflect_once_then_reach_target():
    replay = advance_through_world(
        (0, 0), (100, 0), (0, 0),
        World(lines=[LineObstacle((50, -100), (50, 100))]), 2.0,
        target=(-20, 0), target_radius=2, max_reflections=1,
    )

    assert replay.terminal_kind == "target"
    assert [event.kind for event in replay.events] == ["reflection", "target"]
    assert replay.reflection_count == 1


def test_replay_rejects_line_contact_inside_its_shortened_end_margin():
    replay = advance_through_world(
        (0, 6), (100, 0), (0, 0),
        World(lines=[LineObstacle((50, -5), (50, 5))]), 1.0,
        max_reflections=1,
    )

    assert replay.terminal_kind == "out_of_bounds"


def test_replay_stops_when_a_second_reflection_would_be_required():
    replay = advance_through_world(
        (0, 0), (100, 0), (0, 0),
        World(lines=[LineObstacle((50, -100), (50, 100)), LineObstacle((-50, -100), (-50, 100))]), 2.0,
        max_reflections=1,
    )

    assert replay.terminal_kind == "obstacle"
    assert replay.reflection_count == 1


def test_replay_uses_analytic_circle_entry_for_a_reflection():
    replay = advance_through_world(
        (0, 0), (100, 0), (0, 0),
        World(circles=[CircleObstacle((50, 0), 10)]), 2.0,
        target=(-20, 0), target_radius=2, max_reflections=1,
    )

    assert replay.terminal_kind == "target"
    assert replay.events[0].kind == "reflection"
    assert replay.events[0].obstacle_kind == "circle"


def test_replay_preserves_event_order_for_reflection_then_portal():
    pair = PortalPair(Portal("orange", (400, 749), 15), Portal("blue", (200, 749), 15))
    replay = advance_through_world(
        (100, 700), (983.6246, 0), (0, 379.106),
        World(image_width=1920, lines=[LineObstacle((500, 100), (500, 1000))], portal_pairs=[pair]), 2.0,
        target=(120, 766), target_radius=8, max_reflections=1,
    )

    assert replay.terminal_kind == "target"
    assert [event.kind for event in replay.events] == ["reflection", "portal", "target"]


def test_replay_preserves_event_order_for_portal_then_reflection():
    pair = PortalPair(Portal("orange", (300, 700), 15), Portal("blue", (600, 700), 15))
    replay = advance_through_world(
        (100, 700), (983.6246, 0), (0, 379.106),
        World(image_width=1920, lines=[LineObstacle((800, 100), (800, 1000))], portal_pairs=[pair]), 2.0,
        max_reflections=1,
    )

    assert [event.kind for event in replay.events] == ["portal", "reflection"]


def test_replay_rejects_simultaneous_obstacle_contacts_conservatively():
    replay = advance_through_world(
        (0, 0), (100, 0), (0, 0),
        World(circles=[CircleObstacle((50, 0), 10), CircleObstacle((50, 0), 10)]), 1.0,
        max_reflections=1,
    )

    assert replay.terminal_kind == "out_of_bounds"
