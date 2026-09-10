from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from shellshock_detector.world_geometry import (
    DetectionBox,
    World,
    build_world,
    build_world_from_image,
)


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


def test_build_world_from_image_refines_yolo_circle_and_preserves_self_and_portals():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.circle(image, (500, 360), 150, (255, 255, 255), 8)
    boxes = [
        DetectionBox("self", 100, 200, 40, 30, 0.95),
        DetectionBox("obstacle_circle", 330, 190, 340, 340, 0.9),
        DetectionBox("portal_orange", 180, 280, 80, 80, 0.90),
        DetectionBox("portal_blue", 700, 310, 82, 82, 0.91),
    ]

    world = build_world_from_image(boxes, image)

    assert world.self_position == (120, 215)
    assert len(world.portal_pairs) == 1
    assert len(world.circles) == 1
    assert world.circles[0].center == pytest.approx((500, 360), abs=4)
    assert world.circles[0].radius == pytest.approx(150, abs=5)


def test_build_world_from_image_pairs_portals_by_center_number():
    image = np.zeros((700, 1100, 3), dtype=np.uint8)
    portals = [
        ("portal_orange", 100, 100, "1"), ("portal_orange", 300, 100, "2"), ("portal_orange", 500, 100, "3"),
        ("portal_blue", 150, 450, "3"), ("portal_blue", 350, 450, "1"), ("portal_blue", 550, 450, "2"),
    ]
    boxes = []
    for name, x, y, digit in portals:
        boxes.append(DetectionBox(name, x, y, 80, 80, 0.95))
        cv2.putText(image, digit, (x + 26, y + 57), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    world = build_world_from_image(boxes, image)

    assert len(world.portal_pairs) == 3
    assert {(pair.orange.number, pair.blue.number) for pair in world.portal_pairs} == {(1, 1), (2, 2), (3, 3)}


@pytest.mark.parametrize(
    ("start", "end"),
    [((160, 500), (480, 500)), ((300, 620), (300, 300)), ((120, 650), (420, 470))],
)
def test_build_world_from_image_refines_yolo_line_endpoints(start, end):
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.line(image, start, end, (255, 255, 255), 10)
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))

    world = build_world_from_image(
        [DetectionBox("obstacle_line", left - 20, top - 20, right - left + 40, bottom - top + 40, 0.9)],
        image,
    )

    assert len(world.lines) == 1
    actual = world.lines[0]
    assert min(np.hypot(actual.start[0] - start[0], actual.start[1] - start[1]), np.hypot(actual.end[0] - start[0], actual.end[1] - start[1])) <= 7
    assert min(np.hypot(actual.start[0] - end[0], actual.start[1] - end[1]), np.hypot(actual.end[0] - end[0], actual.end[1] - end[1])) <= 7


def test_build_world_from_image_discards_obstacle_candidate_without_strict_hsv_geometry():
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    world = build_world_from_image(
        [
            DetectionBox("self", 10, 20, 20, 20, 0.9),
            DetectionBox("obstacle_circle", 100, 100, 160, 160, 0.9),
            DetectionBox("obstacle_line", 300, 100, 180, 20, 0.9),
            DetectionBox("portal_orange", 300, 200, 80, 80, 0.9),
            DetectionBox("portal_blue", 450, 200, 80, 80, 0.9),
        ],
        image,
    )

    assert world.self_position == (20, 30)
    assert len(world.portal_pairs) == 1
    assert world.circles == ()
    assert world.lines == ()
