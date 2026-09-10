import unittest

from shellshock_detector_yolo.aiming import AIM_DISC_RADIUS_AT_REFERENCE, disc_click_point


class YoloAimingTests(unittest.TestCase):
    def test_reference_radius_has_four_percent_power_compensation(self):
        self.assertEqual(AIM_DISC_RADIUS_AT_REFERENCE, 332.5)

    def test_full_power_at_reference_width_reaches_compensated_radius(self):
        self.assertEqual(disc_click_point(1000, 800, "right", 0, 100, 1920), (1335, 804))

    def test_compensation_preserves_direction_and_angle(self):
        right = disc_click_point(1000, 800, "right", 45, 50, 1920)
        left = disc_click_point(1000, 800, "left", 45, 50, 1920)

        self.assertEqual(right, (1120, 686))
        self.assertEqual(left, (885, 686))

    def test_full_power_at_4k_uses_safe_665_pixel_radius(self):
        point = disc_click_point(1000, 800, "right", 0, 100, 3840)

        self.assertEqual(point, (1670, 807))
