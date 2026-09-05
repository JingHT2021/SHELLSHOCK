from detect_shellshock import format_aim_report


def test_aim_report_groups_self_and_click_and_highlights_power_angle():
    text = format_aim_report(
        (2537, 802),
        (2691, 705),
        {
            "mode": "normal",
            "angle_degrees": 35,
            "power": 28,
            "target_error": 0.01,
            "horizontal_error": -9.14,
        },
        wind_value=11,
        wind_direction="right",
    )

    assert "SELF  (2537, 802)  →  AIM CLICK  (2691, 705)  |  WIND 11 right  |  MODE normal" in text
    assert "Normal: (power=28, angle=35)  target_error=0.01px" in text
    assert "landing=left 9.14px" in text
    assert "\x1b[1;31m" in text
