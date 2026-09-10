import cv2
import numpy as np

from shellshock_detector_yolo.guide_trajectory import detect_game_guide


def test_detects_dashed_bright_guide_from_muzzle_and_ignores_ui_box():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 35, 45)
    for x, y in [(40, 200), (60, 188), (80, 174), (100, 158), (120, 140), (140, 120)]:
        cv2.line(image, (x, y), (x + 8, y - 5), (235, 235, 235), 2)
    cv2.rectangle(image, (130, 20), (220, 55), (255, 255, 255), 2)
    result = detect_game_guide(image, (40, 200), direction="right", angle_degrees=25)
    assert result.status == "matched"
    assert result.confidence > 0.2
    assert len(result.points) >= 3
    assert max(point[0] for point in result.points) > 100


def test_returns_uncertain_when_no_guide_is_present():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detect_game_guide(image, (30, 100), direction="right", angle_degrees=30)
    assert result.status == "uncertain"
    assert result.points == ()
