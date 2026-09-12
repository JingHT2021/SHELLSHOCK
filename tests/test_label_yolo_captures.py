from types import SimpleNamespace

import numpy as np
import cv2

from label_yolo_captures import (
    ANNOTATION_MODE,
    LineAnnotation,
    REPLAY_MODE,
    _draw,
    annotation_wind_metadata,
    cycle_image_index,
    _merge_line_annotations,
    _merge_duplicate_line_boxes,
    format_replay_log_entry,
    format_wind_text,
    replay_log_status,
    save_replay_log_bundle,
    _select_color_geometry,
    interaction_key_action,
    toggle_interaction_mode,
)
from shellshock.perception.color_geometry import detect_pink_obstacle_geometry
from shellshock.domain.world import DetectionBox
from shellshock.datasets.yolo import YoloBox
from shellshock.perception.yolo import merge_duplicate_detection_boxes


def test_self_pose_center_does_not_draw_duplicate_white_dot():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    self_detection = SimpleNamespace(
        name="self",
        keypoints=(SimpleNamespace(x=30, y=30, visible=2),),
    )

    rendered = _draw(
        image,
        circles=[],
        lines=[],
        center=(30, 30),
        muzzle=None,
        pose_detections=(self_detection,),
    )

    assert tuple(rendered[34, 34]) != (255, 255, 255)
    assert tuple(rendered[30, 30]) == (255, 0, 255)


def test_white_geometry_uses_exact_neutral_hsv_and_full_brightness_range():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    hsv = np.zeros_like(image)
    hsv[:, :] = (0, 0, 0)
    cv2.line(hsv, (40, 100), (360, 100), (0, 0, 230), 4)
    image[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    geometry = detect_pink_obstacle_geometry(image)

    assert geometry.lines


def test_white_geometry_rejects_non_neutral_hue_or_saturation():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    hsv = np.zeros_like(image)
    hsv[:, :] = (0, 0, 0)
    cv2.line(hsv, (40, 100), (360, 100), (1, 1, 255), 4)
    image[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    geometry = detect_pink_obstacle_geometry(image)

    assert not geometry.lines


def test_color_geometry_is_preferred_over_pose_and_box_fallbacks():
    color_line = ((10, 20), (100, 20))
    pose_line = ((12, 21), (102, 21))
    box_line = ((0, 0), (110, 0))

    selected = _select_color_geometry(
        color_geometry=[color_line],
        pose_geometry=[pose_line],
        box_geometry=box_line,
    )

    assert selected == color_line


def test_color_geometry_falls_back_when_white_pixels_are_missing():
    pose_line = ((12, 21), (102, 21))
    box_line = ((0, 0), (110, 0))

    selected = _select_color_geometry(
        color_geometry=[],
        pose_geometry=[pose_line],
        box_geometry=box_line,
    )

    assert selected == pose_line


def test_caps_lock_toggles_between_annotation_and_replay_modes():
    assert toggle_interaction_mode(ANNOTATION_MODE) == REPLAY_MODE
    assert toggle_interaction_mode(REPLAY_MODE) == ANNOTATION_MODE


def test_annotation_digits_select_classes_and_resize_without_angle_controls():
    assert interaction_key_action(ANNOTATION_MODE, ord("0")) == ("select_class", 0)
    assert interaction_key_action(ANNOTATION_MODE, ord("9")) == ("select_class", 9)
    assert interaction_key_action(ANNOTATION_MODE, ord("+")) == ("resize", 2)
    assert interaction_key_action(ANNOTATION_MODE, ord("-")) == ("resize", -2)
    for key in "adqeADQE":
        assert interaction_key_action(ANNOTATION_MODE, ord(key)) is None


def test_replay_digits_select_solver_modes_and_protect_annotations():
    assert interaction_key_action(REPLAY_MODE, ord("1")) == ("solver_mode", "normal_low")
    assert interaction_key_action(REPLAY_MODE, ord("2")) == ("solver_mode", "reflection_low")
    assert interaction_key_action(REPLAY_MODE, ord("3")) == ("solver_mode", "wormhole_low")
    assert interaction_key_action(REPLAY_MODE, ord("v")) == ("toggle_annotations",)
    assert interaction_key_action(REPLAY_MODE, ord("V")) == ("toggle_annotations",)
    assert interaction_key_action(REPLAY_MODE, 9) == ("toggle_tuning",)
    assert interaction_key_action(REPLAY_MODE, ord("+")) is None
    assert interaction_key_action(REPLAY_MODE, ord("q")) is None  # quit is handled by the replay loop


def test_duplicate_reflection_lines_are_merged_into_one_annotation():
    lines = [
        LineAnnotation(4, (100, 100), (300, 200)),
        LineAnnotation(4, (102, 101), (302, 201)),
    ]
    merged = _merge_line_annotations(lines)
    assert len(merged) == 1


def test_overlapping_yolo_reflection_boxes_are_merged():
    boxes = [
        DetectionBox("obstacle_line", 100, 100, 200, 80, 0.80),
        DetectionBox("obstacle_line", 104, 102, 198, 78, 0.95),
    ]
    merged = merge_duplicate_detection_boxes(boxes)
    assert len(merged) == 1
    assert merged[0].confidence == 0.95


def test_cached_replay_log_entry_contains_trigger_and_solver_stages():
    entry = format_replay_log_entry({
        "image": "shot.png", "target": (10, 20),
        "solution": {"status": "reachable", "mode": "normal_low", "angle_degrees": 40, "power": 80},
        "debug_logs": ["layer_a_generated=1", "layer_b_seed_count=2"],
    }, "ALT")
    assert "trigger=ALT" in entry
    assert "shot.png" in entry
    assert "layer_b_seed_count=2" in entry


def test_replay_log_entry_contains_selected_lineage_and_trajectory_details():
    entry = format_replay_log_entry({
        "image": "shot.png", "target": (10, 20),
        "solution": {
            "status": "reachable", "mode": "wormhole_low", "angle_degrees": 40,
            "power": 80, "selected_route_id": "A-2", "selected_branch_id": "B-2-1",
            "selected_candidate_id": "C-7", "final_trace": {
                "waypoints": [{"kind": "portal", "point": [3, 4]}],
                "segments": [{"index": 0, "start": [0, 0], "end": [3, 4]}],
                "portal_transitions": [{"entry": [3, 4], "exit": [100, 200], "displacement": [97, 196]}],
            },
        },
        "debug_logs": [],
    }, "ALT")
    assert "SELECTED route_id=A-2 branch_id=B-2-1 candidate_id=C-7" in entry
    assert "WAYPOINT" in entry and "SEGMENT 0" in entry and "PORTAL_TRANSITION" in entry


def test_save_replay_log_bundle_writes_text_and_json_sidecars(tmp_path):
    destination = tmp_path / "logs" / "shot.log"
    entries = [{"timestamp": "2026-09-11T15:00:00", "trigger": "ALT", "solution": {"status": "reachable"}}]
    status = save_replay_log_bundle(destination, entries)
    assert status == "LOG SAVED: shot.log + .json"
    assert destination.exists()
    assert destination.with_suffix('.json').exists()
    assert 'timestamp=2026-09-11T15:00:00' in destination.read_text(encoding='utf-8')
    assert '"trigger": "ALT"' in destination.with_suffix('.json').read_text(encoding='utf-8')


def test_replay_log_status_reports_empty_cache():
    assert replay_log_status([]) == "NO LOGS TO SAVE"


def test_image_navigation_wraps_around_both_ends():
    assert cycle_image_index(0, -1, 3) == 2
    assert cycle_image_index(2, 1, 3) == 0


def test_annotation_wind_is_displayed_and_stored_as_image_metadata():
    metadata = annotation_wind_metadata({"angle_degrees": 40}, -12.5)

    assert format_wind_text(-12.5) == "12 LEFT"
    assert metadata["wind_value"] == 12.5
    assert metadata["wind_direction"] == "left"
    assert metadata["wind_source"] == "manual"


def test_replay_wind_control_is_not_part_of_annotation_wind_metadata():
    metadata = annotation_wind_metadata({"wind_value": 8, "wind_direction": "right"}, 8)

    assert metadata["wind_value"] == 8
    assert metadata["wind_direction"] == "right"


def test_overlapping_reflection_boxes_are_reduced_to_one_annotation_box():
    boxes = [YoloBox(4, .50, .50, .30, .12), YoloBox(4, .505, .502, .295, .115)]
    merged = _merge_duplicate_line_boxes(boxes)
    assert len(merged) == 1
