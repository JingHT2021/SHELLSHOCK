import unittest


from shellshock_detector.aiming import (
    AIM_DISC_RADIUS_AT_REFERENCE,
    disc_click_point,
)


class AimingTests(unittest.TestCase):
    def test_reference_radius_is_325_pixels(self):
        self.assertEqual(AIM_DISC_RADIUS_AT_REFERENCE, 325.0)

    def test_right_45_degree_half_power_returns_up_and_right_point(self):
        point = disc_click_point(1000, 800, "right", 45, 50, 1920)

        self.assertEqual(point, (1117, 689))

    def test_left_45_degree_half_power_returns_up_and_left_point(self):
        point = disc_click_point(1000, 800, "left", 45, 50, 1920)

        self.assertEqual(point, (888, 689))

    def test_full_power_at_zero_degrees_uses_full_right_radius(self):
        point = disc_click_point(1000, 800, "right", 0, 100, 1920)

        self.assertEqual(point, (1328, 804))

    def test_4k_calibration_matches_the_observed_33_power_47_degree_shot(self):
        point = disc_click_point(1557, 1378, "right", 47.6490, 33.3946, 3840)

        self.assertEqual(point, (1708, 1225))

    def test_invalid_inputs_raise_value_error(self):
        invalid_arguments = (
            (1000, 800, "up", 45, 50, 1920),
            (1000, 800, "right", -1, 50, 1920),
            (1000, 800, "right", 91, 50, 1920),
            (1000, 800, "right", 45, -1, 1920),
            (1000, 800, "right", 45, 101, 1920),
            (1000, 800, "right", 45, 50, 0),
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    disc_click_point(*arguments)
