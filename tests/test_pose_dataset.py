from shellshock_detector.pose_dataset import (
    circle_keypoints_from_box,
    convert_boxes,
    format_pose_label,
    line_keypoints,
)
from shellshock_detector.yolo_dataset import YoloBox


def test_circle_box_becomes_center_and_edge_keypoints():
    points = circle_keypoints_from_box((0.5, 0.4, 0.2, 0.1), 1000, 800)
    assert points == ((500.0, 320.0, 2), (400.0, 320.0, 2))


def test_line_endpoints_are_normalized_as_two_visible_keypoints():
    assert line_keypoints((100, 200), (900, 600), 1000, 800) == ((0.1, 0.25, 2), (0.9, 0.75, 2))


def test_pose_label_has_two_three_value_keypoints():
    text = format_pose_label(2, (0.5, 0.5, 0.1, 0.1), ((500, 400, 2), (550, 400, 2)), 1000, 800)
    assert text == "2 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.550000 0.500000 2"


def test_offscreen_circle_uses_full_sidecar_radius_for_pose_keypoints():
    annotations = convert_boxes(
        [YoloBox(3, 0.1, 0.5, 0.2, 0.75)],
        100,
        80,
        circle_geometry=[{"class_id": 3, "center": [-10, 40], "radius": 30}],
    )

    assert annotations[0].keypoints == ((0.0, 0.0, 0), (0.0, 0.0, 0))


def test_partially_visible_circle_keeps_only_in_frame_keypoint():
    annotations = convert_boxes(
        [YoloBox(3, 0.2, 0.5, 0.2, 0.2)],
        100,
        100,
        circle_geometry=[{"class_id": 3, "center": [20, 50], "radius": 30}],
    )
    assert annotations[0].keypoints == ((20.0, 50.0, 2), (0.0, 0.0, 0))


def test_non_geometry_class_has_two_invisible_keypoints():
    result = convert_boxes([YoloBox(0, 0.5, 0.5, 0.1, 0.1)], 100, 100)
    assert result[0].keypoints == ((0.0, 0.0, 0), (0.0, 0.0, 0))
