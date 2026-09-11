from math import hypot
from unittest.mock import patch

import cv2
import numpy as np

from shellshock.perception.world import (
    DetectionBox,
    PoseKeypoint,
    build_world_from_image,
)


def test_clipped_white_circle_is_fitted_inside_detection_box_before_pose():
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    cv2.circle(image, (-8, 70), 52, (255, 255, 255), 4)
    box = DetectionBox(
        "obstacle_circle",
        -65,
        10,
        115,
        120,
        0.95,
        (
            PoseKeypoint(18, 76, 2, 0.9),
            PoseKeypoint(48, 76, 2, 0.9),
        ),
    )

    world = build_world_from_image([box], image)

    assert len(world.circles) == 1
    circle = world.circles[0]
    assert hypot(circle.center[0] + 8, circle.center[1] - 70) < 4
    assert abs(circle.radius - 52) < 4


def test_white_line_fit_is_preferred_over_offset_pose_endpoints():
    image = np.zeros((120, 140, 3), dtype=np.uint8)
    cv2.line(image, (22, 92), (112, 28), (255, 255, 255), 5)
    box = DetectionBox(
        "obstacle_line",
        15,
        20,
        105,
        80,
        0.95,
        (
            PoseKeypoint(34, 102, 2, 0.9),
            PoseKeypoint(124, 38, 2, 0.9),
        ),
    )

    world = build_world_from_image([box], image)

    assert len(world.lines) == 1
    line = world.lines[0]
    direct = hypot(line.start[0] - 22, line.start[1] - 92) + hypot(line.end[0] - 112, line.end[1] - 28)
    reverse = hypot(line.start[0] - 112, line.start[1] - 28) + hypot(line.end[0] - 22, line.end[1] - 92)
    assert min(direct, reverse) < 12


def test_non_white_pixels_do_not_override_circle_pose_fallback():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 30, (255, 0, 255), 4)
    box = DetectionBox(
        "obstacle_circle",
        15,
        15,
        70,
        70,
        0.95,
        (
            PoseKeypoint(45, 46, 2, 0.9),
            PoseKeypoint(20, 46, 2, 0.9),
        ),
    )

    world = build_world_from_image([box], image)

    assert world.circles[0].center == (45, 46)
    assert world.circles[0].radius == 25


def test_self_keeps_pose_center_instead_of_detection_box_center():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = DetectionBox(
        "self",
        20,
        20,
        60,
        60,
        0.95,
        (PoseKeypoint(44, 47, 2, 0.9),),
    )

    world = build_world_from_image([box], image)

    assert world.self_position == (44, 47)


def test_white_pixels_inside_portal_do_not_replace_portal_geometry():
    image = np.zeros((100, 220, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 20, (255, 255, 255), 3)
    boxes = [
        DetectionBox(
            "portal_orange",
            20,
            20,
            60,
            60,
            0.95,
            (PoseKeypoint(50, 50, 2, 0.9), PoseKeypoint(20, 50, 2, 0.9)),
        ),
        DetectionBox(
            "portal_blue",
            120,
            20,
            60,
            60,
            0.95,
            (PoseKeypoint(150, 50, 2, 0.9), PoseKeypoint(120, 50, 2, 0.9)),
        ),
    ]

    with patch("shellshock.perception.world.recognize_trained_digits", return_value=("", 0.0)):
        world = build_world_from_image(boxes, image)

    assert len(world.portal_pairs) == 1
    assert world.portal_pairs[0].orange.radius == 30
    assert world.portal_pairs[0].blue.radius == 30
