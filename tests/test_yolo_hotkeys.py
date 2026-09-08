from detect_shellshock_yolo import (
    DEFAULT_WEIGHTS, EXIT_HOTKEY, display_mode, format_aim_report, format_solver_diagnostics, select_mode,
)


def test_yolo_default_uses_final_all_data_weights():
    assert DEFAULT_WEIGHTS.as_posix().endswith("shellshock_yolo11n_cv65_final_all/weights/best.pt")


def test_page_keys_select_high_and_low_arc_modes():
    assert select_mode("page up") == "normal_high"
    assert select_mode("page down") == "normal_low"


def test_page_keys_keep_wormhole_family_when_wormhole_is_selected():
    assert select_mode("page up", "wormhole") == "wormhole_high"
    assert select_mode("page down", "wormhole") == "wormhole_low"


def test_page_keys_preserve_reflection_family():
    assert select_mode('page up', 'reflection_low') == 'reflection_high'
    assert select_mode('page down', 'reflection_high') == 'reflection_low'


def test_reflection_mode_has_one_user_facing_name():
    assert display_mode('reflection_low') == 'reflection'
    assert display_mode('reflection_high') == 'reflection'


def test_delete_is_the_only_declared_exit_hotkey():
    assert EXIT_HOTKEY == "delete"


def test_aim_report_renders_aligned_red_angle_and_power():
    text = format_aim_report("wormhole", {"direction": "right", "angle_degrees": 42, "power": 52, "portal_count": 1, "events": ["portal", "target"]}, 12, "left", (30, 40))
    assert "\x1b[31m( 52,  42°)\x1b[0m" in text
    assert "PORTALS  1" in text


def test_aim_report_includes_combined_reflection_event_details():
    text = format_aim_report(
        "reflection",
        {"direction": "left", "angle_degrees": 42, "power": 52, "portal_count": 1,
         "reflection_count": 1, "reflection_point": (500, 700), "reflection_obstacle": {"kind": "line", "index": 0},
         "events": ["reflection", "portal", "target"]},
        0, "right", (30, 40),
    )

    assert "REFLECTIONS  1 @ (500, 700)" in text
    assert "line #0" in text
    assert "reflection→portal→target" in text


def test_aim_report_includes_accuracy_and_theory_diagnostics():
    text = format_aim_report('reflection_low', {'power': 46, 'angle_degrees': 62,
        'miss_distance': 2.3, 'clearance': 15.8, 'incidence': .41,
        'theoretical_min_power': 43.28, 'low_start_power': 46}, 25, 'left', (1, 2))
    assert 'MISS 2.30 px' in text
    assert 'CLEARANCE 15.80 px' in text
    assert 'PMIN 43.28' in text
    assert 'INCIDENCE 0.410' in text


def test_aim_report_includes_layer_timing_breakdown():
    text = format_aim_report('reflection_low', {
        'power': 46, 'angle_degrees': 62,
        'timing': {'layer_a_seconds': 0.0123, 'layer_b_seconds': 0.0456,
                   'layer_c1_seconds': 0.0789, 'layer_c2_seconds': 0.1234,
                   'total_seconds': 0.2602},
    }, 0, 'right', (1, 2))
    assert 'TIME A 12.3ms B 45.6ms C1 78.9ms C2 123.4ms TOTAL 260.2ms' in text


def test_aim_report_includes_portal_radii():
    text = format_aim_report('wormhole_low', {'power': 60, 'angle_degrees': 50,
        'portal_radii': [{'id': '0:orange', 'visual': 30., 'trigger': 27., 'avoid': 35.}]}, 0, 'right', (1, 2))
    assert 'VISUAL 30.00 TRIGGER 27.00 AVOID 35.00' in text


def test_solver_diagnostics_report_obstacles_stages_and_timing():
    text = format_solver_diagnostics({
        'layer_a_passed': 17,
        'layer_b_passed': 10,
        'integer_candidates_raw': 42,
        'integer_candidates_unique': 19,
        'integer_full_replays': 10,
        'integer_replay_failed': 10,
        'layer_a_invalid_reasons': {'A_SEGMENT_DIRECTION': 3},
        'layer_b_invalid_reasons': {'B_WRONG_FIRST_COLLISION': 8},
        'layer_b_soft_invalid_reasons': {'B_PLANNED_PORTAL_MISS': 4},
        'timing': {'layer_a_seconds': .01, 'layer_b_seconds': .02,
                   'layer_c1_seconds': .03, 'layer_c2_seconds': .04,
                   'total_seconds': .10},
    }, line_count=4, circle_count=2)
    assert 'OBSTACLES lines=4 circles=2' in text
    assert 'STAGES A_passed=17 B_passed=10 C_raw=42 C_unique=19 replays=10 failed=10' in text
    assert 'REASONS A_SEGMENT_DIRECTION=3 B_WRONG_FIRST_COLLISION=8 SOFT_B_PLANNED_PORTAL_MISS=4' in text
    assert 'TIME A 10.0ms B 20.0ms C1 30.0ms C2 40.0ms TOTAL 100.0ms' in text
