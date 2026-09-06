from dataclasses import FrozenInstanceError

import pytest

from shellshock_detector.world_geometry import DetectionBox, World, build_world


def test_build_world_pairs_nearest_equal_radius_orange_and_blue_portals():
    world = build_world(
        [
            DetectionBox("self", 100, 200, 40, 30, 0.95),
            DetectionBox("portal_orange", 180, 280, 80, 80, 0.90),
            DetectionBox("portal_blue", 700, 310, 82, 82, 0.91),
        ],
        1000,
        700,
    )

    assert world.self_position == (120, 215)
    assert len(world.portal_pairs) == 1
    assert world.portal_pairs[0].orange.center == (220, 320)
    assert world.portal_pairs[0].blue.center == (741, 351)
    assert world.unpaired_portals == 0


def test_build_world_excludes_ambiguous_or_unmatched_portals():
    world = build_world(
        [
            DetectionBox("portal_orange", 100, 100, 80, 80, 0.9),
            DetectionBox("portal_blue", 400, 100, 80, 80, 0.9),
            DetectionBox("portal_blue", 700, 100, 80, 80, 0.9),
        ],
        1000,
        700,
    )

    assert world.portal_pairs == ()
    assert world.unpaired_portals == 3


def test_build_world_returns_none_self_when_top_confidences_tie():
    world = build_world(
        [
            DetectionBox("self", 0, 0, 20, 20, 0.9),
            DetectionBox("self", 50, 0, 20, 20, 0.9),
        ],
        100,
        100,
    )

    assert world.self_position is None


def test_build_world_clips_circle_and_line_boxes_to_image_bounds():
    world = build_world(
        [
            DetectionBox("obstacle_circle", -10, 10, 40, 40, 0.8),
            DetectionBox("obstacle_line", 80, 30, 40, 10, 0.8),
            DetectionBox("obstacle_line", 20, -20, 10, 50, 0.8),
        ],
        100,
        80,
    )

    assert world.circles[0].center == (15, 30)
    assert world.circles[0].radius == pytest.approx(17.5)
    assert world.lines[0].start == (80, 35)
    assert world.lines[0].end == (100, 35)
    assert world.lines[1].start == (25, 0)
    assert world.lines[1].end == (25, 30)


def test_build_world_accepts_radius_difference_at_inclusive_threshold():
    world = build_world(
        [
            DetectionBox("portal_orange", 0, 0, 100, 100, 0.9),
            DetectionBox("portal_blue", 200, 0, 85, 85, 0.9),
        ],
        500,
        200,
    )

    assert len(world.portal_pairs) == 1
    assert world.unpaired_portals == 0


def test_build_world_rejects_radius_difference_above_threshold():
    world = build_world(
        [
            DetectionBox("portal_orange", 0, 0, 100, 100, 0.9),
            DetectionBox("portal_blue", 200, 0, 84, 84, 0.9),
        ],
        500,
        200,
    )

    assert world.portal_pairs == ()
    assert world.unpaired_portals == 2


def test_world_objects_are_frozen():
    world = build_world([], 100, 100)

    with pytest.raises(FrozenInstanceError):
        world.self_position = (1, 1)


def test_world_converts_caller_supplied_collections_to_immutable_tuples():
    world = World(circles=[])

    assert world.circles == ()
