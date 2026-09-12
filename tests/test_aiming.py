from shellshock.interaction.aiming import disc_click_point


def test_zero_power_click_uses_supplied_center_without_calibration_offset():
    assert disc_click_point(100.0, 200.0, "right", 0.0, 0.0, 1920.0) == (100, 200)
