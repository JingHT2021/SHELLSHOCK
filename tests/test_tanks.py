import unittest
from pathlib import Path

import cv2
import numpy as np

from shellshock_detector.tanks import detect_tanks


class TankDetectionTests(unittest.TestCase):
    def test_detect_tanks_returns_green_self_and_red_enemy(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        cv2.rectangle(image, (1650, 1200), (1700, 1230), (0, 255, 0), -1)
        cv2.rectangle(image, (1100, 1180), (1150, 1210), (0, 0, 255), -1)

        self_tank, enemies = detect_tanks(image)

        self.assertEqual((self_tank.x, self_tank.y), (1675, 1215))
        self.assertEqual([(item.x, item.y) for item in enemies], [(1125, 1195)])

    def test_detects_split_green_tank_and_all_enemies_across_the_playfield(self):
        """Real 4K scenes can place tanks above 55% height on steep terrain."""
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        # Green body segments are separated by the dark tracks in the game art.
        cv2.rectangle(image, (2404, 888), (2433, 896), (0, 255, 0), -1)
        cv2.rectangle(image, (2465, 901), (2492, 909), (0, 255, 0), -1)
        for x, y in ((810, 936), (1766, 1283), (2951, 936)):
            cv2.rectangle(image, (x, y), (x + 78, y + 58), (0, 0, 255), -1)

        self_tank, enemies = detect_tanks(image)

        self.assertIsNotNone(self_tank)
        self.assertEqual(len(enemies), 3)
        self.assertEqual([item.x for item in enemies], [849, 1805, 2990])

    def test_detects_tanks_up_to_fixed_1800_pixel_bottom_boundary(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        cv2.rectangle(image, (2400, 1760), (2478, 1790), (0, 255, 0), -1)
        cv2.rectangle(image, (800, 1750), (878, 1780), (0, 0, 255), -1)
        cv2.rectangle(image, (1200, 1810), (1278, 1840), (0, 0, 255), -1)

        self_tank, enemies = detect_tanks(image)

        self.assertIsNotNone(self_tank)
        self.assertEqual((self_tank.x, self_tank.y), (2439, 1775))
        self.assertEqual([(item.x, item.y) for item in enemies], [(839, 1765)])

    def test_detects_selected_green_tank_in_real_4k_scene(self):
        fixture = Path(__file__).parent / "fixtures" / "real_4k_scene.png"
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)

        self_tank, _ = detect_tanks(image)

        self.assertIsNotNone(self_tank)
        self.assertLess(abs(self_tank.x - 2430), 30)
        self.assertLess(abs(self_tank.y - 850), 30)

    def test_excludes_red_aiming_arrow_without_dropping_rotated_enemy(self):
        fixture = Path(__file__).parent / "fixtures" / "real_4k_arrow_wind_64_scene.png"
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)

        self_tank, enemies = detect_tanks(image)

        self.assertIsNotNone(self_tank)
        self.assertEqual(len(enemies), 3)
        self.assertEqual([item.x for item in enemies], [389, 1552, 2921])

    def test_detects_rotated_red_tank_high_on_terrain(self):
        image = cv2.imread("output/20260905_192849_raw.png")

        _, enemies = detect_tanks(image)

        self.assertTrue(any(abs(item.x - 778) < 40 and abs(item.y - 690) < 40 for item in enemies))

    def test_prefers_green_tank_template_over_name_colored_candidate(self):
        image = cv2.imread("output/20260905_192838_raw.png")

        self_tank, _ = detect_tanks(image)

        self.assertIsNotNone(self_tank)
        self.assertLess(abs(self_tank.x - 3151), 40)
        self.assertLess(abs(self_tank.y - 1190), 40)
