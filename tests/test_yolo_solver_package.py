"""The YOLO entrypoint owns an isolated solver package."""


def test_yolo_entrypoint_uses_dedicated_solver_package():
    import detect_shellshock_yolo as entrypoint
    from shellshock_detector_yolo.global_solver import solve_integer_shot

    assert entrypoint.solve_integer_shot is solve_integer_shot


def test_yolo_entrypoint_has_no_runtime_import_from_legacy_package():
    from pathlib import Path

    source = Path("detect_shellshock_yolo.py").read_text(encoding="utf-8")

    assert "from shellshock_detector." not in source
