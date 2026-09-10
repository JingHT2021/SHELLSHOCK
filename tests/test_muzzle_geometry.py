import pytest


def test_muzzle_position_scales_reference_barrel_for_right_shot():
    from shellshock_detector_yolo.muzzle_geometry import muzzle_position

    assert muzzle_position((100.0, 200.0), "right", 0.0, 2560) == pytest.approx((135.0, 200.0))
    assert muzzle_position((100.0, 200.0), "right", 90.0, 3840) == pytest.approx((100.0, 147.5))


def test_muzzle_position_uses_opposite_horizontal_offset_for_left_shot():
    from shellshock_detector_yolo.muzzle_geometry import muzzle_position

    assert muzzle_position((100.0, 200.0), "left", 60.0, 2560) == pytest.approx(
        (82.5, 169.6891108675), abs=1e-9
    )


@pytest.mark.parametrize(
    ("center", "direction", "angle", "width"),
    [((0, 0), "up", 0, 2560), ((0, 0), "right", -1, 2560), ((0, 0), "right", 91, 2560), ((0, 0), "right", 0, 0)],
)
def test_muzzle_position_rejects_invalid_inputs(center, direction, angle, width):
    from shellshock_detector_yolo.muzzle_geometry import muzzle_position

    with pytest.raises(ValueError):
        muzzle_position(center, direction, angle, width)
