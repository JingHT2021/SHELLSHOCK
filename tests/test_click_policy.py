from shellshock.interaction.click_policy import click_client_point
from detect_shellshock_yolo import format_aim_report


def test_offscreen_click_is_skipped_without_calling_desktop_actions():
    calls = []

    status = click_client_point(
        (-5, 40),
        (100, 200),
        (800, 600),
        activate=lambda: calls.append("activate"),
        click=lambda point: calls.append(("click", point)),
    )

    assert status == "OFFSCREEN"
    assert calls == []


def test_onscreen_click_uses_absolute_screen_point():
    calls = []

    status = click_client_point(
        (25, 40),
        (100, 200),
        (800, 600),
        activate=lambda: calls.append("activate"),
        click=lambda point: calls.append(("click", point)),
    )

    assert status == "SUCCESS"
    assert calls == ["activate", ("click", (125, 240))]


def test_normal_report_keeps_power_and_angle_but_omits_click_coordinates():
    report = format_aim_report(
        "normal_low",
        {
            "power": 75,
            "angle_degrees": 42,
            "portal_count": 0,
            "reflection_count": 0,
        },
        0,
        "right",
        (-15, 300),
    )

    assert "75" in report
    assert "42" in report
    assert "CLICK" not in report
    assert "(-15, 300)" not in report
