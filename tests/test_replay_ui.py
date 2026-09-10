import os

import cv2
import numpy as np

from replay_shellshock import DEFAULT_IMAGE, annotated_image_paths, arrow_adjustment, build_parser, calibration_hint, derive_self_center, fit_to_canvas, guide_from_report, is_reset_view_key, make_calibration_sample, manual_preview_solution, nearest_calibration_index, render_replay_visual, select_replay_mode, should_launch_interactive, solution_launch_point, wind_adjustment
from shellshock_detector_yolo.annotation_conversion import SceneAnnotation
from shellshock_detector_yolo.guide_trajectory import GuideDetection


def test_barrel_tip_is_converted_back_to_tank_center():
    assert derive_self_center((135, 100), "right", 0, 35) == (100.0, 100.0)
    assert derive_self_center((100, 65), "right", 90, 35) == (100.0, 100.0)


def test_fit_to_canvas_preserves_aspect_ratio_with_letterbox():
    scale, offset = fit_to_canvas(2560, 1233, 1600, 1000)
    assert scale == 1600 / 2560
    assert offset[0] == 0
    assert offset[1] > 0


def test_replay_modes_match_live_mode_switching():
    assert select_replay_mode("r", "normal_low") == "reflection_low"
    assert select_replay_mode("h", "normal_low") == "wormhole_low"
    assert select_replay_mode("page up", "reflection_low") == "reflection_high"
    assert select_replay_mode("page down", "reflection_high") == "reflection_low"


def test_report_guide_is_restored_with_all_render_fields():
    guide = guide_from_report({"guide": {"status": "uncertain", "confidence": 0.4, "points": [[1, 2]]}})
    assert guide.status == "uncertain"
    assert guide.confidence == 0.4
    assert guide.points == ((1.0, 2.0),)


def test_replay_visual_marks_self_center_without_muzzle_marker():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    report = {"source": "annotation", "solution": {}, "error": {"mean_distance": None, "max_distance": None}, "target": None, "self_center": (30, 65)}
    rendered = render_replay_visual(image, (), GuideDetection("uncertain"), SceneAnnotation(100, 80), report)
    assert tuple(rendered[65, 30]) == (0, 255, 255)
    assert tuple(rendered[65, 70]) == (0, 0, 0)


def test_solution_launch_point_uses_solution_direction_from_center():
    point = solution_launch_point((100, 100), {"direction": "right", "angle_degrees": 0}, 1920, 35)
    assert point == (126.25, 100.0)


def test_replay_cli_uses_default_capture_when_image_is_omitted():
    args = build_parser().parse_args([])
    assert args.image == DEFAULT_IMAGE


def test_replay_without_arguments_opens_the_interactive_viewer():
    assert should_launch_interactive([], explicit_interactive=False)
    assert not should_launch_interactive(["--image", "capture.png"], explicit_interactive=False)
    assert should_launch_interactive(["--image", "capture.png"], explicit_interactive=True)


def test_annotated_images_are_sorted_for_angle_bracket_switching(tmp_path):
    later = tmp_path / "later.png"
    earlier = tmp_path / "earlier.jpg"
    later.write_bytes(b"")
    earlier.write_bytes(b"")
    os.utime(later, ns=(2_000_000_000, 2_000_000_000))
    os.utime(earlier, ns=(1_000_000_000, 1_000_000_000))

    assert annotated_image_paths(tmp_path) == [earlier, later]


def test_manual_adjustment_maps_all_windows_arrow_key_codes():
    assert arrow_adjustment(81) == ("angle", -1)
    assert arrow_adjustment(83) == ("angle", 1)
    assert arrow_adjustment(82) == ("power", 1)
    assert arrow_adjustment(2621440) == ("power", -1)


def test_uppercase_r_is_also_a_reflection_mode_hotkey():
    assert select_replay_mode("R", "normal_low") == "reflection_low"


def test_manual_preview_keeps_the_full_high_arc_duration():
    solution = manual_preview_solution("right", 80, 70, "normal_high")
    assert solution["flight_time_seconds"] >= 12.0


def test_calibration_sample_records_controls_and_curve_fraction():
    sample = make_calibration_sample({"angle_degrees": 42, "power": 61}, ((10, 20), (30, 40)), (20, 30))
    assert sample["angle_degrees"] == 42
    assert sample["power"] == 61
    assert sample["actual_point"] == (20.0, 30.0)
    assert 0.0 <= sample["trajectory_fraction"] <= 1.0


def test_calibration_hint_uses_observed_error_for_a_small_control_correction():
    hint = calibration_hint([{"angle_degrees": 40, "power": 60, "actual_point": (110, 80), "predicted_point": (100, 100)}])
    assert hint["angle_degrees"] > 40
    assert hint["power"] > 60


def test_right_click_can_identify_the_nearest_calibration_point():
    samples = [{"actual_point": (100, 100)}, {"actual_point": (300, 300)}]
    assert nearest_calibration_index(samples, (104, 103), 20) == 0
    assert nearest_calibration_index(samples, (150, 150), 20) is None


def test_home_resets_zoom_view():
    assert is_reset_view_key(2359296, 0)
    assert is_reset_view_key(36, 0)
    assert not is_reset_view_key(0, ord("0"))


def test_wind_adjustment_accepts_lower_and_upper_case_controls():
    assert wind_adjustment(ord("z")) == -1.0
    assert wind_adjustment(ord("C")) == 1.0
