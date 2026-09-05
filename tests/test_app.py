import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from shellshock_detector.app import (
    aim_click_for_target,
    analyze_image,
    choose_window_handle,
    ensure_game_window_is_active,
    has_usable_client_area,
    matches_game_window,
    process_capture,
    screen_to_client_point,
)
from shellshock_detector.models import Detection, DetectionResult, Wind
from shellshock_detector.obstacle_geometry import ObstacleGeometry


class AppTests(unittest.TestCase):
    def test_reflection_without_obstacles_falls_back_to_integer_normal_shot(self):
        result = DetectionResult(1920, 1080, Detection(1000, 800, 0.9), [], Wind(0, "right", 0.9), [])

        solution, _ = aim_click_for_target(
            result, 2000, 800, shot_mode="reflection", geometry=ObstacleGeometry([], [])
        )

        self.assertEqual(solution["selected"]["fallback"], "normal:no-obstacle")
        self.assertIs(type(solution["selected"]["angle_degrees"]), int)
        self.assertIs(type(solution["selected"]["power"]), int)

    def test_ensure_game_window_is_active_reactivates_the_captured_window(self):
        activated: list[int] = []

        ensure_game_window_is_active(123, is_window=lambda handle: handle == 123, activate=activated.append)

        self.assertEqual(activated, [123])

    def test_ensure_game_window_is_active_refuses_a_destroyed_window(self):
        with self.assertRaisesRegex(RuntimeError, "closed"):
            ensure_game_window_is_active(123, is_window=lambda _: False, activate=lambda _: None)

    def test_ensure_game_window_is_active_tolerates_windows_activation_rejection(self):
        def reject_activation(_: int) -> None:
            raise OSError("Windows rejected foreground activation")

        ensure_game_window_is_active(123, is_window=lambda handle: handle == 123, activate=reject_activation)

    def test_screen_to_client_point_rejects_mouse_outside_game_client_area(self):
        self.assertEqual(screen_to_client_point((110, 220), (100, 200), (1920, 1080)), (10, 20))
        with self.assertRaisesRegex(ValueError, "outside"):
            screen_to_client_point((99, 220), (100, 200), (1920, 1080))

    def test_aim_click_for_target_maps_minimum_power_solution_to_aim_disc(self):
        result = DetectionResult(
            image_width=1920,
            image_height=1080,
            self_tank=Detection(1000, 800, 0.9),
            enemies=[],
            wind=Wind(0, "right", 0.9),
            errors=[],
        )

        solution, click_point = aim_click_for_target(result, 2000, 800)

        self.assertEqual(solution["target_direction"], "right")
        self.assertAlmostEqual(solution["minimum_power"]["angle_degrees"], 45.0, places=4)
        self.assertIs(type(solution["selected"]["angle_degrees"]), int)
        self.assertIs(type(solution["selected"]["power"]), int)
        self.assertGreaterEqual(solution["selected"]["target_error"], 0)
        self.assertNotEqual(click_point, (1000, 800))

    def test_aim_click_for_target_uses_zero_wind_when_wind_value_is_missing(self):
        no_wind = DetectionResult(1920, 1080, Detection(1000, 800, 0.9), [], Wind(None, None, 0.0), [])

        solution, click_point = aim_click_for_target(no_wind, 2000, 800)

        self.assertEqual(solution["wind_acceleration"], 0.0)
        self.assertIs(type(solution["selected"]["angle_degrees"]), int)
        self.assertNotEqual(click_point, (1000, 800))

    def test_aim_click_for_target_refuses_over_limit_solution(self):
        far_target = DetectionResult(1920, 1080, Detection(1000, 800, 0.9), [], Wind(0, "right", 0.9), [])

        with self.assertRaisesRegex(RuntimeError, "over 100"):
            aim_click_for_target(far_target, 10000, 800)

    def test_aim_click_for_target_uses_highest_100_power_arc_in_maximum_mode(self):
        result = DetectionResult(1920, 1080, Detection(1000, 800, 0.9), [], Wind(0, "right", 0.9), [])

        solution, click_point = aim_click_for_target(result, 2000, 800, shot_mode="maximum")

        high_arc = solution["power_100"]["solutions"][-1]
        self.assertIs(type(solution["selected"]["angle_degrees"]), int)
        self.assertEqual(solution["selected"]["power"], 98)
        self.assertNotEqual(click_point, (1000, 800))
        self.assertGreater(high_arc["angle_degrees"], 70)

    def test_maximum_mode_uses_the_only_reachable_100_power_arc(self):
        result = DetectionResult(3840, 1800, Detection(1557, 1378, 0.9), [], Wind(76, "right", 0.9), [])

        solution, _ = aim_click_for_target(result, 2196, 1381, shot_mode="maximum")

        self.assertEqual(len(solution["power_100"]["solutions"]), 1)
        self.assertEqual(solution["selected"]["power"], 100)

    def test_process_name_matches_when_window_title_is_empty(self):
        self.assertTrue(matches_game_window("", "ShellShockLive.exe"))
        self.assertFalse(matches_game_window("", "notepad.exe"))

    def test_client_area_rejects_unity_helper_window(self):
        self.assertFalse(has_usable_client_area((0, 0, 0, 0)))
        self.assertFalse(has_usable_client_area((0, 0, 80, 80)))
        self.assertTrue(has_usable_client_area((0, 0, 2560, 1600)))

    def test_usable_matching_foreground_window_wins_when_process_lookup_fails(self):
        self.assertEqual(choose_window_handle(99, (0, 0, 2560, 1600), [], True), 99)

    def test_usable_unmatched_foreground_window_is_rejected(self):
        self.assertIsNone(choose_window_handle(99, (0, 0, 3840, 2160), [], False))

    def test_process_capture_writes_json_and_annotation(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        cv2.rectangle(image, (1650, 1200), (1700, 1230), (0, 255, 0), -1)
        cv2.rectangle(image, (1100, 1180), (1150, 1210), (0, 0, 255), -1)

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            paths = process_capture(image, Path(directory), now=lambda: "20260904_120000")

            self.assertEqual(paths.json_path.name, "20260904_120000_result.json")
            self.assertTrue(paths.json_path.exists())
            self.assertTrue(paths.annotated_path.exists())
            self.assertEqual(paths.result.ballistics, [])

    def test_full_scene_keeps_wind_detection_without_tank_detection(self):
        fixture = Path(__file__).parent / "fixtures" / "full_scene.png"
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)

        result, _ = analyze_image(image)

        self.assertIsNone(result.self_tank)
        self.assertEqual(result.enemies, [])
        self.assertEqual(result.wind.direction, "right")
        self.assertEqual(result.wind.value, 20)

    def test_analyze_image_has_no_automatic_ballistics(self):
        fixture = Path(__file__).parent / "fixtures" / "full_scene.png"
        image = cv2.imdecode(np.frombuffer(fixture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)

        result, _ = analyze_image(image)

        self.assertEqual(result.ballistics, [])
        self.assertIsNone(result.self_tank)

    def test_analyze_image_treats_absent_wind_panel_as_calm_for_manual_aiming(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        cv2.rectangle(image, (1650, 1200), (1700, 1230), (0, 255, 0), -1)
        cv2.rectangle(image, (1100, 1180), (1150, 1210), (0, 0, 255), -1)

        result, _ = analyze_image(image)

        self.assertEqual(result.ballistics, [])
        self.assertEqual(result.wind.value, 0)
        self.assertIsNone(result.wind.direction)
