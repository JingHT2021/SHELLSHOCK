import json
from math import isfinite
import unittest


from shellshock_detector.ballistics import (
    GRAVITY_AT_REFERENCE,
    SPEED_PER_POWER_AT_REFERENCE,
    predicted_horizontal_displacement,
    refine_normal_integer_shot,
    solve_target,
)


class BallisticsTests(unittest.TestCase):
    def test_fixed_power_solver_keeps_a_negative_angle_for_a_lower_target(self):
        result = solve_target(0, 0, 100, 100, 0, "right", 1920)

        angles = [solution["angle_degrees"] for solution in result["power_100"]["solutions"]]
        self.assertTrue(any(angle < 0 for angle in angles))
    def test_integer_refinement_returns_integer_controls_and_error(self):
        shot = refine_normal_integer_shot(
            0, 0, 1000, 0, 0, "right", 1920,
            theory_angle=45.4, theory_power=63.7,
        )

        self.assertIs(type(shot["angle_degrees"]), int)
        self.assertIs(type(shot["power"]), int)
        self.assertGreaterEqual(shot["target_error"], 0.0)

    def test_no_wind_same_height_has_45_degree_minimum_power_solution(self):
        result = solve_target(0, 0, 1000, 0, 0, "right", 1920)

        self.assertEqual(result["target_direction"], "right")
        self.assertAlmostEqual(result["minimum_power"]["angle_degrees"], 45.0, places=5)
        self.assertLess(result["minimum_power"]["power"], 100.0)
        self.assertEqual(len(result["power_100"]["solutions"]), 2)

    def test_left_target_mirrors_right_target_angle(self):
        right = solve_target(0, 0, 1000, 0, 0, "right", 1920)
        left = solve_target(0, 0, -1000, 0, 0, "right", 1920)

        self.assertEqual(left["target_direction"], "left")
        self.assertAlmostEqual(
            right["minimum_power"]["angle_degrees"],
            left["minimum_power"]["angle_degrees"],
            places=5,
        )

    def test_power_100_can_be_unreachable_while_minimum_power_is_reported(self):
        result = solve_target(0, 0, 5000, 0, 0, "right", 1920)

        self.assertEqual(result["power_100"]["status"], "unreachable")
        self.assertEqual(result["power_100"]["solutions"], [])
        self.assertGreater(result["minimum_power"]["power"], 100.0)
        self.assertFalse(result["minimum_power"]["within_power_limit"])

    def test_opposite_wind_nearly_cancels_83_degree_power_100_horizontal_motion(self):
        displacement = predicted_horizontal_displacement(
            power=100,
            angle_degrees=83,
            wind_value=91.5,
            wind_direction="left",
            firing_direction="right",
            image_width=3840,
        )

        self.assertLess(abs(displacement), 10.0)

    def test_extreme_wind_returns_unreachable_minimum_instead_of_raising(self):
        result = solve_target(0, 0, 0, 100, 99999, "right", 1920)

        self.assertEqual(result["power_100"]["status"], "unreachable")
        self.assertEqual(result["power_100"]["solutions"], [])
        self.assertEqual(result["minimum_power"]["status"], "unreachable")
        self.assertIsNone(result["minimum_power"]["angle_degrees"])
        self.assertIsNone(result["minimum_power"]["power"])

    def test_just_beyond_maximum_range_is_over_limit_before_power_rounding(self):
        maximum_range = (SPEED_PER_POWER_AT_REFERENCE * 100) ** 2 / GRAVITY_AT_REFERENCE
        result = solve_target(0, 0, maximum_range + 0.001, 0, 0, "right", 1920)

        self.assertEqual(result["power_100"]["status"], "unreachable")
        self.assertFalse(result["minimum_power"]["within_power_limit"])

    def test_maximum_range_tangent_has_one_fixed_power_solution(self):
        maximum_range = (SPEED_PER_POWER_AT_REFERENCE * 100) ** 2 / GRAVITY_AT_REFERENCE
        result = solve_target(0, 0, maximum_range, 0, 0, "right", 1920)

        self.assertEqual(result["power_100"]["status"], "reachable")
        self.assertEqual(len(result["power_100"]["solutions"]), 1)

    def test_near_boundary_solution_is_finite_and_uses_positive_time_squared_root(self):
        maximum_range = (SPEED_PER_POWER_AT_REFERENCE * 100) ** 2 / GRAVITY_AT_REFERENCE
        result = solve_target(0, 0, maximum_range * (1 - 1e-12), 0, 0, "right", 1920)

        solutions = result["power_100"]["solutions"]
        self.assertEqual(len(solutions), 2)
        self.assertTrue(all(solution["flight_time_seconds"] > 0 for solution in solutions))
        self.assertTrue(all(isfinite(solution["angle_degrees"]) for solution in solutions))

    def test_unreachable_minimum_schema_is_strict_json(self):
        result = solve_target(0, 0, 0, 100, 99999, "right", 1920)
        minimum = result["minimum_power"]

        json.dumps(result, allow_nan=False)
        self.assertIsInstance(minimum["status"], str)
        self.assertIsInstance(minimum["direction"], str)
        self.assertIsNone(minimum["angle_degrees"])
        self.assertIsNone(minimum["flight_time_seconds"])
        self.assertIsNone(minimum["power"])
        self.assertIsInstance(minimum["within_power_limit"], bool)

    def test_format_ballistics_mentions_target_direction_and_minimum_power(self):
        from shellshock_detector.ballistics import format_ballistics

        text = format_ballistics([{
            "target": {"x": 900, "y": 500}, "dx": 800, "dy": 0,
            "target_direction": "right",
            "power_100": {"power": 100.0, "status": "reachable", "solutions": [
                {"direction": "right", "angle_degrees": 20.0, "flight_time_seconds": 1.0},
                {"direction": "right", "angle_degrees": 70.0, "flight_time_seconds": 2.0},
            ]},
            "minimum_power": {"direction": "right", "angle_degrees": 45.0, "power": 62.0, "within_power_limit": True},
        }])

        self.assertIn("Target 1: dx=800, dy=0, direction=right", text)
        self.assertIn("100 power low arc: right 20.0000 degrees", text)
        self.assertIn("minimum power: 62.0000, right 45.0000 degrees", text)
