import pytest


@pytest.mark.parametrize("mode, solver_name", [("normal_low", "solve_normal_integer_shot"), ("wormhole_low", "solve_wormhole_integer_shot")])
def test_global_solver_replays_normal_and_wormhole_from_angle_corrected_muzzle(monkeypatch, mode, solver_name):
    import shellshock_detector_yolo.global_solver as module
    from shellshock_detector_yolo.muzzle_geometry import muzzle_position

    calls = []
    first = {"status": "reachable", "direction": "right", "angle_degrees": 30, "power": 40}
    second = {"status": "reachable", "direction": "right", "angle_degrees": 31, "power": 39}
    third = {"status": "reachable", "direction": "right", "angle_degrees": 31, "power": 39}

    def fake_solver(source, target, world, wind_value, wind_direction, image_width, **kwargs):
        calls.append(source)
        return (first, second, third)[min(len(calls), 3) - 1]

    monkeypatch.setattr(module, solver_name, fake_solver)

    center = (100.0, 200.0)
    result = module.solve_integer_shot(center, (500.0, 100.0), object(), 0, "right", 2560, mode)

    assert calls == [
        center,
        muzzle_position(center, "right", 30, 2560),
        muzzle_position(center, "right", 31, 2560),
    ]
    assert result["angle_degrees"] == 31
    assert result["mode"] == mode


def test_global_solver_does_not_add_center_correction_before_reflection(monkeypatch):
    import shellshock_detector_yolo.global_solver as module
    import shellshock_detector_yolo.reflection_solver as reflection_module

    calls = []

    def fake_solver(source, target, world, wind_value, wind_direction, image_width, **kwargs):
        calls.append(source)
        return {"status": "reachable", "direction": "right", "angle_degrees": 45, "power": 50}

    monkeypatch.setattr(reflection_module, "solve_reflection_integer_shot", fake_solver)

    center = (100.0, 200.0)
    module.solve_integer_shot(center, (500.0, 100.0), object(), 0, "right", 2560, "reflection")

    assert calls == [center]
