from math import hypot
from types import SimpleNamespace
from unittest.mock import patch

from shellshock_detector_yolo.ballistics import SPEED_PER_POWER_AT_REFERENCE
from shellshock_detector_yolo.normal_solver import (
    _select_normal_candidate,
    solve_normal_integer_shot,
)


def _theory():
    return {
        "minimum_power": {
            "status": "reachable",
            "within_power_limit": True,
            "power": 70.0,
            "angle_degrees": 45.0,
            "direction": "right",
        },
        "power_100": {
            "solutions": [
                {"angle_degrees": 30.0, "direction": "right"},
                {"angle_degrees": 70.0, "direction": "right"},
            ]
        },
    }


def _replay_for_best_power(best_power, replayed_angles=None):
    def replay(_source, velocity, _acceleration, _world, _target, _width):
        from math import atan2, degrees

        power = round(hypot(*velocity) / SPEED_PER_POWER_AT_REFERENCE)
        if replayed_angles is not None:
            angle = round(degrees(atan2(-velocity[1], abs(velocity[0]))))
            replayed_angles.setdefault(power, []).append(angle)
        return SimpleNamespace(
            valid=True,
            miss_distance=abs(power - best_power),
            clearance=20.0,
            time=1.0,
            invalid_reason=None,
        )

    return replay


def test_normal_high_searches_94_through_100_around_power_100_theory_angle():
    replayed_angles = {}

    with (
        patch(
            "shellshock_detector_yolo.normal_solver.solve_target",
            return_value=_theory(),
        ),
        patch(
            "shellshock_detector_yolo.normal_solver.solve_ballistic_for_speed",
        ) as solve_speed,
        patch(
            "shellshock_detector_yolo.normal_solver.replay_portal_shot",
            side_effect=_replay_for_best_power(96, replayed_angles),
        ),
    ):
        result = solve_normal_integer_shot(
            (0, 0),
            (100, 0),
            object(),
            0,
            "right",
            1920,
            arc_preference="high",
        )

    solve_speed.assert_not_called()
    assert list(replayed_angles) == list(range(94, 101))
    assert all(angles == [67, 68, 69, 70, 71, 72, 73] for angles in replayed_angles.values())
    assert result["power"] == 96
    assert result["miss_distance"] == 0
    assert result["diagnostics"]["candidate_count"] == 49
    assert result["diagnostics"]["theory_angle"] == 70.0
    assert result["diagnostics"]["theory_power"] == 100.0


def test_normal_high_breaks_equal_miss_ties_by_arc_clearance_then_power():
    candidates = [
        {"miss_distance": 1.0, "angle_degrees": 70, "clearance": 50, "power": 100},
        {"miss_distance": 1.0, "angle_degrees": 71, "clearance": 10, "power": 100},
        {"miss_distance": 1.0, "angle_degrees": 71, "clearance": 20, "power": 90},
        {"miss_distance": 1.0, "angle_degrees": 71, "clearance": 20, "power": 95},
    ]

    result = _select_normal_candidate(candidates, "high")

    assert result is candidates[3]


def test_normal_high_force_power_searches_only_requested_power():
    replayed_angles = {}

    with (
        patch(
            "shellshock_detector_yolo.normal_solver.solve_target",
            return_value=_theory(),
        ),
        patch(
            "shellshock_detector_yolo.normal_solver.solve_ballistic_for_speed",
        ) as solve_speed,
        patch(
            "shellshock_detector_yolo.normal_solver.replay_portal_shot",
            side_effect=_replay_for_best_power(95, replayed_angles),
        ),
    ):
        result = solve_normal_integer_shot(
            (0, 0),
            (100, 0),
            object(),
            0,
            "right",
            1920,
            arc_preference="high",
            force_power=95,
        )

    solve_speed.assert_not_called()
    assert replayed_angles == {95: [67, 68, 69, 70, 71, 72, 73]}
    assert result["power"] == 95


def test_normal_high_force_power_100_preserves_theory_seed_behavior():
    replayed_angles = []

    def replay(_source, velocity, _acceleration, _world, _target, _width):
        from math import atan2, degrees

        replayed_angles.append(round(degrees(atan2(-velocity[1], abs(velocity[0])))))
        return SimpleNamespace(
            valid=True,
            miss_distance=0.0,
            clearance=20.0,
            time=1.0,
            invalid_reason=None,
        )

    with (
        patch(
            "shellshock_detector_yolo.normal_solver.solve_target",
            return_value=_theory(),
        ),
        patch(
            "shellshock_detector_yolo.normal_solver.solve_ballistic_for_speed"
        ) as solve_speed,
        patch(
            "shellshock_detector_yolo.normal_solver.replay_portal_shot",
            side_effect=replay,
        ),
    ):
        result = solve_normal_integer_shot(
            (0, 0),
            (100, 0),
            object(),
            0,
            "right",
            1920,
            arc_preference="high",
            force_power=100,
        )

    solve_speed.assert_not_called()
    assert replayed_angles == [67, 68, 69, 70, 71, 72, 73]
    assert result["power"] == 100
