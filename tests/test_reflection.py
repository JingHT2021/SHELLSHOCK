from __future__ import annotations

import unittest
from unittest.mock import patch

from shellshock_detector.obstacle_geometry import CircleObstacle, LineObstacle, ObstacleGeometry
from shellshock_detector.reflection import first_collision, reflect_velocity, solve_single_reflection


class ReflectionGeometryTests(unittest.TestCase):
    def test_circle_collision_returns_first_entry_and_outward_normal(self):
        event = first_collision(
            (0.0, 0.0), (100.0, 0.0), (0.0, 0.0), 1.0,
            ObstacleGeometry([CircleObstacle((50, 0), 10)], []),
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "circle")
        self.assertEqual(event.obstacle_index, 0)
        self.assertAlmostEqual(event.point[0], 40.0)
        self.assertAlmostEqual(event.normal[0], -1.0)

    def test_line_reflection_reverses_only_normal_component(self):
        self.assertEqual(reflect_velocity((4.0, -3.0), (0.0, 1.0)), (4.0, 3.0))

    def test_first_collision_prefers_nearer_unrelated_obstacle(self):
        geometry = ObstacleGeometry(
            [CircleObstacle((25, 0), 5), CircleObstacle((70, 0), 5)], []
        )
        event = first_collision((0, 0), (100, 0), (0, 0), 1.0, geometry)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.obstacle_index, 0)

    def test_line_collision_is_limited_to_its_endpoints(self):
        event = first_collision(
            (0, 30), (100, 0), (0, 0), 1.0,
            ObstacleGeometry([], [LineObstacle((50, -10), (50, 10))]),
        )
        self.assertIsNone(event)

    def test_solver_returns_integer_one_bounce_line_solution(self):
        result = solve_single_reflection(
            (0, 0), (0, 15), 0, "right", 1920,
            ObstacleGeometry([], [LineObstacle((50, -500), (50, 500))]),
        )

        self.assertEqual(result["status"], "reachable")
        self.assertEqual(result["obstacle"], {"kind": "line", "index": 0})
        self.assertIs(type(result["angle_degrees"]), int)
        self.assertIs(type(result["power"]), int)
        self.assertLess(result["target_error"], 1.0)

    def test_solver_returns_closest_legal_bounce_when_theory_is_not_exact(self):
        with patch("shellshock_detector.reflection.CONTINUOUS_RESIDUAL", 0.0):
            result = solve_single_reflection(
                (0, 0), (0, 15), 0, "right", 1920,
                ObstacleGeometry([], [LineObstacle((50, -500), (50, 500))]),
            )

        self.assertEqual(result["status"], "closest")
        self.assertEqual(result["obstacle"], {"kind": "line", "index": 0})
        self.assertGreater(result["target_error"], 0.0)
