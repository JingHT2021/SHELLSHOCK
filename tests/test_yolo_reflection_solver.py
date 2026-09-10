import math

from shellshock_detector_yolo.reflection_solver import solve_reflection_bundle
from shellshock_detector_yolo.reflection_replay import replay_integer_contact
from shellshock_detector_yolo.reflection_solver import SurfaceFamily
from shellshock_detector_yolo.world_geometry import LineObstacle, World


def test_reflection_final_replay_source_uses_candidate_angle():
    from shellshock_detector_yolo.muzzle_geometry import muzzle_position
    from shellshock_detector_yolo.reflection_solver import _muzzle_source_for_candidate

    center = (500.0, 600.0)

    assert _muzzle_source_for_candidate(center, "right", 20, 2560) == muzzle_position(center, "right", 20, 2560)
    assert _muzzle_source_for_candidate(center, "right", 70, 2560) == muzzle_position(center, "right", 70, 2560)
    assert _muzzle_source_for_candidate(center, "right", 20, 2560) != _muzzle_source_for_candidate(center, "right", 70, 2560)


def test_c1_recomputes_fixed_contact_after_integer_angle_muzzle_correction(monkeypatch):
    import shellshock_detector_yolo.reflection_solver as solver_module
    from shellshock_detector_yolo.reflection_routes import ReflectionRoute

    family = solver_module.SurfaceFamily("line", 0, "BOTH", 0.0, 1.0)
    calls = []

    def fake_fixed(source, target, contact, normal, acceleration, speed_per_power):
        calls.append(tuple(source))
        return [type("Solution", (), {"angle_degrees": 42.2, "direction": "right"})()]

    monkeypatch.setattr(solver_module, "fixed_contact_solutions", fake_fixed)
    monkeypatch.setattr(solver_module, "_muzzle_source_for_candidate", lambda source, direction, angle, image_width: (source[0] + 10, source[1]))

    solver_module._c1_corrected_branch(
        (0.0, 0.0), (100.0, 100.0), solver_module.World(), (0.0, 1.0), 10.0,
        1920, family, ReflectionRoute(), (20.0, 20.0), (-1.0, 0.0), 0,
    )

    assert calls == [(0.0, 0.0), (10.0, 0.0)]


def test_reflection_c2_candidate_pool_is_twenty():
    from shellshock_detector_yolo.solver_config import FINAL_INTEGER_CANDIDATE_POOL

    assert FINAL_INTEGER_CANDIDATE_POOL == 20


def test_c1_keeps_one_nearest_integer_candidate_per_continuous_solution():
    from types import SimpleNamespace
    from shellshock_detector_yolo.reflection_layer_c import generate_integer_candidates

    samples = [
        SimpleNamespace(valid_math=True, branch_id=0, surface_param=0.0,
                        angle_cont=37.4, power_cont=60.4),
        SimpleNamespace(valid_math=True, branch_id=0, surface_param=1.0,
                        angle_cont=40.4, power_cont=63.4),
    ]

    candidates = generate_integer_candidates(samples)

    assert len(candidates) == 2
    assert [(candidate.angle, candidate.power) for candidate in candidates] == [(37, 60), (40, 63)]


def test_reflection_proxy_score_prioritizes_routes_with_portals():
    from types import SimpleNamespace
    from shellshock_detector_yolo.reflection_routes import ReflectionRoute
    from shellshock_detector_yolo.reflection_solver import reflection_proxy_score

    solution = SimpleNamespace(power=60.0, incidence=0.8)
    plain = SimpleNamespace(route=ReflectionRoute())
    portal = SimpleNamespace(route=ReflectionRoute(("0:orange",), ()))

    assert reflection_proxy_score(solution, portal) < reflection_proxy_score(solution, plain)


def test_final_sort_prioritizes_valid_portal_result():
    from shellshock_detector_yolo.reflection_layer_c import FinalReplayResult, final_sort_key

    plain = FinalReplayResult(40, 50, True, actual_miss_px=0.5, payload={"portal_count": 0})
    portal = FinalReplayResult(40, 50, True, actual_miss_px=2.0, payload={"portal_count": 1})

    assert final_sort_key(portal) < final_sort_key(plain)


def test_three_layer_solver_reports_bounded_stage_diagnostics():
    bundle = solve_reflection_bundle(
        (500, 600), (560, 450), World(lines=(LineObstacle((750, 200), (750, 800)),)), 0, "right", 1920
    )

    assert bundle.low is not None and bundle.high is not None
    assert bundle.diagnostics["layer_a_passed"] <= 12
    assert bundle.diagnostics["layer_b_passed"] <= 10
    assert bundle.diagnostics["full_replays"] <= 20
    assert bundle.diagnostics["total_seconds"] < 1.5


def test_layer_b_receives_real_wind_acceleration_and_image_width(monkeypatch):
    import shellshock_detector_yolo.reflection_solver as solver_module

    calls = []

    monkeypatch.setattr(
        solver_module,
        "build_coarse_candidates",
        lambda *args: ([object()], {"layer_a_generated": 1, "layer_a_passed": 1,
                                   "invalid_reasons": {"A_SEGMENT_DIRECTION": 2}}),
    )
    monkeypatch.setattr(solver_module, "fair_candidate_prefix", lambda candidates, limit: candidates[:limit])

    def record_layer_b(*args, **kwargs):
        calls.append((args[4], kwargs.get("image_width")))
        return [], {"layer_b_evaluated": 0, "layer_b_passed": 0,
                    "invalid_reasons": {"B_PLANNED_PORTAL_MISS": 3}}

    monkeypatch.setattr(solver_module, "rank_proxy_candidates", record_layer_b)

    bundle = solver_module.solve_reflection_bundle(
        (500, 600), (560, 450), World(lines=(LineObstacle((750, 200), (750, 800)),)),
        50, "right", 1920,
    )

    assert calls
    assert all(acceleration[0] > 0 for acceleration, _ in calls)
    assert all(image_width == 1920 for _, image_width in calls)
    assert bundle.diagnostics["layer_a_invalid_reasons"] == {"A_SEGMENT_DIRECTION": 2}
    assert bundle.diagnostics["layer_b_invalid_reasons"] == {"B_PLANNED_PORTAL_MISS": 3}


def test_fixed_contact_solution_exposes_normalized_root_stability():
    from shellshock_detector_yolo.reflection_solver import fixed_contact_solutions

    solutions = fixed_contact_solutions(
        (500, 600), (560, 450), (750, 500), (-1, 0), (0, 379.106), 9.836246
    )

    assert solutions
    assert all(solution.discriminant_norm >= 0 for solution in solutions)
    assert all(solution.root_slope >= 0 for solution in solutions)


def test_layer_c_uses_all_layer_a_candidates_and_selects_twenty_layer_b_candidates(monkeypatch):
    import shellshock_detector_yolo.reflection_solver as solver_module

    layer_a = list(range(17))
    seen = []

    monkeypatch.setattr(
        solver_module,
        "build_coarse_candidates",
        lambda *args: (layer_a, {"layer_a_generated": 17, "layer_a_passed": 17, "invalid_reasons": {}}),
    )

    def record_layer_b(candidates, *args, **kwargs):
        seen.append((list(candidates), kwargs["top_k"]))
        return [], {"layer_b_evaluated": 0, "layer_b_passed": 0, "invalid_reasons": {}}

    monkeypatch.setattr(solver_module, "rank_proxy_candidates", record_layer_b)
    monkeypatch.setattr(solver_module, "fair_candidate_prefix", lambda candidates, limit: (_ for _ in ()).throw(AssertionError("A must not be truncated")))

    solver_module.solve_reflection_bundle(
        (500, 600), (560, 450), solver_module.World(), 0, "right", 1920,
    )

    assert seen[0] == (layer_a, 20)


def test_integer_reflection_replay_accepts_explicit_relaxed_target_radius():
    source = (500.0, 600.0)
    acceleration = (12.0, 379.106)
    speed = 70 * 9.836246
    angle = math.radians(35)
    velocity = (speed * math.cos(angle), -speed * math.sin(angle))
    t1, t2 = 0.45, 0.35
    contact = (
        source[0] + velocity[0] * t1 + acceleration[0] * t1 * t1 / 2,
        source[1] + velocity[1] * t1 + acceleration[1] * t1 * t1 / 2,
    )
    incoming = (velocity[0] + acceleration[0] * t1, velocity[1] + acceleration[1] * t1)
    outgoing = (-incoming[0], incoming[1])
    exact_target = (
        contact[0] + outgoing[0] * t2 + acceleration[0] * t2 * t2 / 2,
        contact[1] + outgoing[1] * t2 + acceleration[1] * t2 * t2 / 2,
    )
    target = (exact_target[0], exact_target[1] + 30.0)
    world = World(lines=(LineObstacle((contact[0], contact[1] - 200),
                                      (contact[0], contact[1] + 200)),))
    family = SurfaceFamily("line", 0, "BOTH", 0.0, 1.0)

    assert replay_integer_contact(source, target, world, acceleration, 1920, family, 70, 35, "right") is None
    assert replay_integer_contact(
        source, target, world, acceleration, 1920, family, 70, 35, "right",
        target_accept_radius=48.0,
    ) is not None


def test_solver_diagnostics_formatter_lists_full_reflection_trace():
    from detect_shellshock_yolo import format_solver_diagnostics

    text = format_solver_diagnostics({
        "layer_a_trace": [{"id": "A-1", "status": "PASS", "decision": "RETAINED"}],
        "layer_b_trace": [{"id": "B-1", "status": "FAIL", "reason": "B_GRAZING"}],
        "layer_c1_trace": [{"id": "C1-1", "status": "PASS", "decision": "GENERATED"}],
        "layer_c2_trace": [{"id": "C2-1", "status": "FAIL", "reason": "C_INTEGER_REPLAY"}],
        "final_trace": [{"id": "C2-2", "status": "PASS", "decision": "FINAL"}],
    })

    assert "A-1 PASS RETAINED" in text
    assert "B-1 FAIL B_GRAZING" in text
    assert "C1-1 PASS GENERATED" in text
    assert "C2-1 FAIL C_INTEGER_REPLAY" in text
    assert "C2-2 PASS FINAL" in text
