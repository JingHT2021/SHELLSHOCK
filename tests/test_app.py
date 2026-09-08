import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

import shellshock_detector.app as app
from shellshock_detector.app import (
    GAME_CAPTURE_HEIGHT,
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
    def test_outside_client_aim_returns_solution_without_clicking(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = DetectionResult(100, 100, None, [], Wind(0, "right", 0.9), [])
        paths = app.TrainingPaths(Path("raw.png"), Path("label.txt"), Path("annotated.png"), result)
        clicked: list[tuple[int, int]] = []
        solution = {"selected": {"mode": "normal", "angle_degrees": 35, "power": 28, "target_error": 0.0}}

        with (
            patch.object(app, "find_game_window", return_value=123),
            patch.object(app, "client_screen_geometry", return_value=((0, 0), (100, 100))),
            patch.object(app, "capture_client_area", return_value=image),
            patch.object(app, "process_capture", return_value=paths),
            patch.object(app, "aim_click_for_target", return_value=(solution, (100, 50))),
        ):
            actual_paths, actual_solution, click_screen = app.aim_at_screen_position(
                (50, 50), manual_self=(40, 50), click=clicked.append
            )

        self.assertIs(actual_paths, paths)
        self.assertIs(actual_solution, solution)
        self.assertIsNone(click_screen)
        self.assertEqual(clicked, [])

    def test_game_capture_height_is_2000_physical_pixels(self):
        self.assertEqual(GAME_CAPTURE_HEIGHT, 2000)

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

    def test_ensure_game_window_is_active_tolerates_win32_activation_error(self):
        import pywintypes

        def reject_activation(_: int) -> None:
            raise pywintypes.error(0, "SetForegroundWindow", "Windows rejected foreground activation")

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

    def test_process_capture_writes_only_training_artifacts(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        cv2.rectangle(image, (1650, 1200), (1700, 1230), (0, 255, 0), -1)
        cv2.rectangle(image, (1100, 1180), (1150, 1210), (0, 0, 255), -1)

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            train_dir = Path(directory) / "train"
            paths = process_capture(image, train_dir, now=lambda: "20260904_120000")

            self.assertEqual(paths.raw_path, train_dir / "raw_cropped" / "20260904_120000.png")
            self.assertTrue(paths.raw_path.exists())
            self.assertTrue((train_dir / "raw_cropped" / "20260904_120000.txt").exists())
            self.assertTrue(paths.annotated_path.exists())
            self.assertFalse((train_dir / "20260904_120000_raw.png").exists())
            self.assertEqual(paths.result.ballistics, [])

    def test_process_capture_can_skip_training_data_writes(self):
        image = np.zeros((1600, 2560, 3), dtype=np.uint8)
        with patch.object(app, "save_training_sample") as save_training_sample:
            paths = process_capture(
                image, Path("train"), now=lambda: "20260907_120000", save_training_data=False,
            )

        save_training_sample.assert_not_called()
        self.assertIsNone(paths.raw_path)
        self.assertIsNone(paths.label_path)
        self.assertIsNone(paths.annotated_path)

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
