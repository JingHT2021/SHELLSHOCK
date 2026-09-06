import cv2
import numpy as np

from shellshock_detector.obstacle_geometry import detect_pink_obstacle_geometry


def _strict_pink() -> tuple[int, int, int]:
    return (255, 255, 255)


def test_circle_is_reported_as_center_and_radius_without_line_fragments():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.circle(image, (500, 360), 150, _strict_pink(), 8)

    geometry = detect_pink_obstacle_geometry(image)

    assert len(geometry.circles) == 1
    assert geometry.lines == []
    circle = geometry.circles[0]
    assert abs(circle.center[0] - 500) <= 4
    assert abs(circle.center[1] - 360) <= 4
    assert abs(circle.radius - 150) <= 5


def test_line_is_reported_as_two_endpoints_after_circle_is_excluded():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2.circle(image, (500, 360), 150, _strict_pink(), 8)
    cv2.line(image, (120, 650), (380, 510), _strict_pink(), 10)

    geometry = detect_pink_obstacle_geometry(image)

    assert len(geometry.circles) == 1
    assert len(geometry.lines) == 1
    line = geometry.lines[0]
    expected = [(120, 650), (380, 510)]
    actual = [line.start, line.end]
    assert all(min(np.hypot(point[0] - target[0], point[1] - target[1]) for point in actual) <= 6 for target in expected)
