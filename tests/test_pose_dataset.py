from shellshock_detector.pose_dataset import (
    circle_keypoints_from_box,
    format_pose_label,
    line_keypoints,
)


def test_circle_box_becomes_center_and_edge_keypoints():
    points = circle_keypoints_from_box((0.5, 0.4, 0.2, 0.1), 1000, 800)
    assert points == ((500.0, 320.0, 2), (600.0, 320.0, 2))


def test_line_endpoints_are_normalized_as_two_visible_keypoints():
    assert line_keypoints((100, 200), (900, 600), 1000, 800) == ((0.1, 0.25, 2), (0.9, 0.75, 2))


def test_pose_label_has_two_three_value_keypoints():
    text = format_pose_label(2, (0.5, 0.5, 0.1, 0.1), ((500, 400, 2), (550, 400, 2)), 1000, 800)
    assert text == "2 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.550000 0.500000 2"
