from shellshock_detector_yolo.reflection_solver import solve_reflection_bundle
from shellshock_detector_yolo.world_geometry import LineObstacle, World


def test_three_layer_solver_reports_bounded_stage_diagnostics():
    bundle = solve_reflection_bundle(
        (500, 600), (560, 450), World(lines=(LineObstacle((750, 200), (750, 800)),)), 0, "right", 1920
    )

    assert bundle.low is not None and bundle.high is not None
    assert bundle.diagnostics["layer_a_passed"] <= 12
    assert bundle.diagnostics["layer_b_passed"] <= 10
    assert bundle.diagnostics["full_replays"] <= 10
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


def test_layer_c_uses_all_layer_a_candidates_and_selects_ten_layer_b_candidates(monkeypatch):
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

    assert seen[0] == (layer_a, 10)
