"""The YOLO entrypoint owns an isolated solver package."""


def test_yolo_entrypoint_uses_dedicated_solver_package():
    import detect_shellshock_yolo as entrypoint
    from shellshock_detector_yolo.global_solver import solve_integer_shot

    assert entrypoint.solve_integer_shot is solve_integer_shot


def test_yolo_entrypoint_has_no_runtime_import_from_legacy_package():
    from pathlib import Path

    source = Path("detect_shellshock_yolo.py").read_text(encoding="utf-8")

    assert "from shellshock_detector." not in source
def test_normal_low_prefers_minimum_power_over_lower_arc(monkeypatch):
    from shellshock_detector_yolo.normal_solver import _select_normal_candidate

    candidates = [
        {'power': 34, 'angle_degrees': 45, 'miss_distance': 1.2, 'clearance': 100.0},
        {'power': 37, 'angle_degrees': 29, 'miss_distance': 0.7, 'clearance': 250.0},
    ]

    selected = _select_normal_candidate(candidates, 'low')

    assert selected['power'] == 34
    assert selected['angle_degrees'] == 45
