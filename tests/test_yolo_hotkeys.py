from detect_shellshock_yolo import DEFAULT_WEIGHTS, EXIT_HOTKEY, format_aim_report, select_mode


def test_yolo_default_uses_final_all_data_weights():
    assert DEFAULT_WEIGHTS.as_posix().endswith("shellshock_yolo11n_cv65_final_all/weights/best.pt")


def test_page_keys_select_high_and_low_arc_modes():
    assert select_mode("page up") == "high_arc"
    assert select_mode("page down") == "low_arc"


def test_delete_is_the_only_declared_exit_hotkey():
    assert EXIT_HOTKEY == "delete"


def test_aim_report_renders_aligned_red_angle_and_power():
    text = format_aim_report("wormhole", {"direction": "right", "angle_degrees": 42, "power": 52, "portal_count": 1, "events": ["portal", "target"]}, 12, "left", (30, 40))
    assert "\x1b[31m( 52,  42°)\x1b[0m" in text
    assert "PORTALS  1" in text
