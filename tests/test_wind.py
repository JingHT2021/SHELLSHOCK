import unittest
from pathlib import Path

import cv2
import numpy as np

from shellshock_detector.wind import _find_panel_box, _number_from_panel, detect_direction, detect_wind


class WindDetectionTests(unittest.TestCase):
    def test_direction_is_right_when_triangle_points_right(self):
        panel = np.zeros((60, 100, 3), dtype=np.uint8)
        cv2.fillConvexPoly(panel, np.array(((75, 30), (55, 15), (55, 45))), (220, 220, 220))

        self.assertEqual(detect_direction(panel), "right")

    def test_wind_failure_is_explicit_when_panel_is_missing(self):
        wind, error, _ = detect_wind(np.zeros((1600, 2560, 3), dtype=np.uint8))

        self.assertEqual(wind.value, 0)
        self.assertIsNone(wind.direction)
        self.assertIsNone(error)

    def test_panel_search_prefers_the_top_centre_over_left_ui_controls(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        # A left-side UI control can resemble the white wind cloud.
        cv2.rectangle(image, (110, 40), (150, 70), (220, 220, 220), -1)
        cv2.rectangle(image, (160, 42), (185, 68), (220, 220, 220), -1)
        # The real wind panel is centred at the top of the game view.
        cv2.rectangle(image, (1850, 120), (1934, 174), (220, 220, 220), -1)
        cv2.rectangle(image, (1945, 132), (1967, 162), (220, 220, 220), -1)

        box = _find_panel_box(image)

        self.assertIsNotNone(box)
        self.assertGreater(box[0], 1800)

    def test_panel_search_supports_a_left_wind_arrow(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        # These unrelated top HUD digits form a narrower pair and must not
        # stop the search before the real, left-arrow wind panel is reached.
        cv2.rectangle(image, (1879, 56), (1911, 113), (220, 220, 220), -1)
        cv2.rectangle(image, (1918, 57), (1962, 110), (220, 220, 220), -1)
        cv2.rectangle(image, (1879, 141), (1963, 195), (220, 220, 220), -1)
        cv2.fillConvexPoly(image, np.array(((1843, 168), (1865, 153), (1865, 183))), (220, 220, 220))

        box = _find_panel_box(image)

        self.assertEqual(box, (1843, 141, 121, 55))
        x, y, width, height = box
        self.assertEqual(detect_direction(image[y : y + height, x : x + width]), "left")

    def test_reads_complete_two_digit_wind_value_from_real_4k_scene(self):
        fixture = Path(__file__).parent / "fixtures" / "real_4k_arrow_wind_64_scene.png"
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        x, y, width, height = _find_panel_box(image)

        self.assertEqual(_number_from_panel(image[y : y + height, x : x + width]), 64)

    def test_reads_two_digit_value_from_captured_right_wind_panel(self):
        image = cv2.imread("output/20260905_192709_raw.png")
        x, y, width, height = _find_panel_box(image)

        self.assertEqual(_number_from_panel(image[y : y + height, x : x + width]), 56)

    def test_reads_two_digit_value_from_captured_left_wind_panel(self):
        image = cv2.imread("output/20260905_192838_raw.png")
        x, y, width, height = _find_panel_box(image)

        self.assertEqual(_number_from_panel(image[y : y + height, x : x + width]), 57)

    def test_treats_scoreboard_overlay_without_a_valid_cloud_as_zero_wind(self):
        image = cv2.imread("output/20260905_210750_raw.png")

        wind, error, box = detect_wind(image)

        self.assertIsNone(box)
        self.assertEqual(wind.value, 0)
        self.assertIsNone(wind.direction)
        self.assertIsNone(error)
