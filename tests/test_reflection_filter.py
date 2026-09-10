from dataclasses import dataclass
from types import SimpleNamespace

from shellshock_detector_yolo.reflection_routes import ReflectionRoute
from shellshock_detector_yolo.layer_a import CoarsePathResult
from shellshock_detector_yolo.world_geometry import CircleObstacle, LineObstacle, Portal, PortalPair, World


@dataclass(frozen=True)
class Family:
    kind: str
    index: int
    side: str = "BOTH"
    lower: float = 0.0
    upper: float = 1.0


def test_layer_a_reflection_uses_real_acceleration_without_virtual_height_or_score():
    from shellshock_detector_yolo.reflection_filter import build_coarse_candidates

    world = World(lines=(LineObstacle((10, -10), (10, 10)),))
    candidates, diagnostics = build_coarse_candidates(
        (0, 0), (-10, 0), world, (Family("line", 0),), (ReflectionRoute(),), (0.0, 0.0)
    )

    assert len(candidates) == 1
    assert candidates[0].events == ((0, 0), (10.0, 0.0), (-10, 0))
    assert not hasattr(candidates[0], "score_a")
    assert diagnostics["invalid_reasons"] == {}


def test_layer_a_line_checks_endpoint_and_middle_channels(monkeypatch):
    from shellshock_detector_yolo import reflection_filter as filter_module

    checked = []
    monkeypatch.setattr(filter_module, "reflection_path_possible",
                        lambda source, contact, target, *args: (checked.append(contact) or CoarsePathResult(True, None, ())))
    world = World(lines=(LineObstacle((0, 0), (60, 0)),))

    candidates, _ = filter_module.build_coarse_candidates(
        (10, 10), (50, 10), world, (Family("line", 0),), (ReflectionRoute(),), (0.0, 0.0)
    )

    assert [candidate.q_seed for candidate in candidates] == [0.0, 0.5, 1.0]
    assert checked == [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0)]


def test_layer_b_line_parameters_depend_on_the_passing_layer_a_channel():
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, _nearby_parameters

    route = ReflectionRoute()
    assert _nearby_parameters(CoarsePathCandidate(Family("line", 0), route, 0.0, (0, 0), (0, 1), ())) == (1 / 6,)
    assert _nearby_parameters(CoarsePathCandidate(Family("line", 0), route, 0.5, (0, 0), (0, 1), ())) == (1 / 3, 0.5, 2 / 3)
    assert _nearby_parameters(CoarsePathCandidate(Family("line", 0), route, 1.0, (0, 0), (0, 1), ())) == (5 / 6,)


def test_layer_a_reflection_rejects_a_post_reflection_portal_leg_that_cannot_turn_back():
    from shellshock_detector_yolo.reflection_filter import build_coarse_candidates

    world = World(
        lines=(LineObstacle((10, -20), (10, 20)),),
        portal_pairs=(PortalPair(Portal("orange", (20, 0), 4), Portal("blue", (30, 0), 4)),),
    )
    candidates, diagnostics = build_coarse_candidates(
        (0, 0), (40, 0), world, (Family("line", 0),), (ReflectionRoute((), ("0:orange",)),), (0.0, 0.0)
    )

    assert candidates == []
    assert diagnostics["invalid_reasons"]["A_SEGMENT_DIRECTION"] == 1


def test_layer_b_ranks_fixed_point_proxy_without_strict_replay():
    from shellshock_detector_yolo.reflection_filter import build_coarse_candidates, rank_proxy_candidates

    world = World(lines=(LineObstacle((10, -20), (10, 20)),))
    candidates, _ = build_coarse_candidates(
        (0, 0), (0, 0), world, (Family("line", 0),), (ReflectionRoute(),), (0.0, 10.0)
    )
    replay_calls = 0

    def fixed_solver(source, target, contact, normal, acceleration, speed_per_power):
        assert acceleration == (0.0, 10.0)
        return [object()]

    def score(solution, candidate):
        nonlocal replay_calls
        replay_calls += 1
        return 1.0

    proxies, diagnostics = rank_proxy_candidates(
        candidates, (0, 0), (0, 0), world, (0.0, 10.0), 1.0, fixed_solver, score, top_k=4
    )

    assert len(proxies) == 1
    assert diagnostics["layer_b_evaluated"] == 3
    assert replay_calls == 3


def test_layer_a_keeps_every_possible_circle_representative(monkeypatch):
    import shellshock_detector_yolo.reflection_filter as filter_module

    checked = []

    def first_and_third_seed_pass(*args):
        checked.append(args[1])
        valid = len(checked) in {1, 3}
        return CoarsePathResult(valid, None if valid else "A_SEGMENT_DIRECTION", ())

    monkeypatch.setattr(filter_module, "reflection_path_possible", first_and_third_seed_pass)
    world = World(circles=(CircleObstacle((50, 50), 20),))

    candidates, _ = filter_module.build_coarse_candidates(
        (0, 0), (100, 100), world, (Family("circle", 0),), (ReflectionRoute(),), (0.0, 0.0)
    )

    assert len(candidates) == 2
    assert len(checked) == 8
    assert {candidate.q_seed for candidate in candidates} == {0.0, 3.141592653589793 / 2}


def test_layer_a_circle_side_policy_uses_inside_and_outside_margins(monkeypatch):
    import shellshock_detector_yolo.reflection_filter as filter_module

    monkeypatch.setattr(filter_module, "reflection_path_possible",
                        lambda *args: CoarsePathResult(True, None, ()))
    world = World(circles=(CircleObstacle((0, 0), 20),))
    families = (Family("circle", 0, "INNER"), Family("circle", 0, "OUTER"))

    far, _ = filter_module.build_coarse_candidates(
        (100, 0), (200, 0), world, families, (ReflectionRoute(),), (0.0, 0.0)
    )
    near, _ = filter_module.build_coarse_candidates(
        (30, 0), (200, 0), world, families, (ReflectionRoute(),), (0.0, 0.0)
    )
    middle, _ = filter_module.build_coarse_candidates(
        (50, 0), (200, 0), world, families, (ReflectionRoute(),), (0.0, 0.0)
    )

    assert {candidate.family.side for candidate in far} == {"OUTER"}
    assert {candidate.family.side for candidate in near} == {"INNER"}
    assert {candidate.family.side for candidate in middle} == {"INNER", "OUTER"}


def test_layer_b_keeps_wrong_first_collision_as_soft_diagnostic_for_layer_c():
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, rank_proxy_candidates

    family = Family("line", 0)
    candidate = CoarsePathCandidate(family, ReflectionRoute(), 0.5, (10, 0), (-1, 0), ())
    world = World(lines=(
        LineObstacle((10, -10), (10, 10)),
        LineObstacle((5, -10), (5, 10)),
    ))
    score_calls = 0

    def fixed_solver(source, target, contact, normal, acceleration, speed_per_power):
        return [SimpleNamespace(contact=contact, normal=normal, velocity=(1, 0), t1=10, t2=10,
                                power=20, angle_degrees=0, incidence=1)]

    def score(solution, coarse):
        nonlocal score_calls
        score_calls += 1
        return solution.power

    proxies, diagnostics = rank_proxy_candidates(
        (candidate,), (0, 0), (0, 0), world, (0, 0), 1, fixed_solver, score, top_k=4,
        image_width=1920,
    )

    assert len(proxies) == 1
    assert score_calls == 3
    assert diagnostics["soft_invalid_reasons"]["B_UNPLANNED_LINE_COLLISION"] == 3


def test_layer_b_preserves_all_valid_algebraic_branches_and_neighbor_metrics(monkeypatch):
    import shellshock_detector_yolo.reflection_filter as filter_module
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, rank_proxy_candidates

    family = Family("line", 0)
    candidate = CoarsePathCandidate(family, ReflectionRoute(), 0.5, (10, 0), (-1, 0), ())
    world = World(lines=(LineObstacle((10, -10), (10, 10)),))
    monkeypatch.setattr(filter_module, "LAYER_B_LINE_PARAMETERS", (0.5,))

    def fixed_solver(source, target, contact, normal, acceleration, speed_per_power):
        return [
            SimpleNamespace(contact=contact, normal=normal, velocity=(1, 0), t1=10, t2=10,
                            power=20, angle_degrees=0, incidence=1),
            SimpleNamespace(contact=contact, normal=normal, velocity=(1, 0), t1=10, t2=10,
                            power=30, angle_degrees=0, incidence=1),
        ]

    proxies, diagnostics = rank_proxy_candidates(
        (candidate,), (0, 0), (0, 0), world, (0, 0), 1, fixed_solver,
        lambda solution, coarse: solution.power, top_k=10, image_width=1920,
    )

    assert len(proxies) == 2
    assert {proxy.branch for proxy in proxies} == {0, 1}
    assert all(proxy.neighborhood_valid_count == 1 for proxy in proxies)
    assert all(proxy.neighborhood_angle_span == 0 for proxy in proxies)
    assert all(proxy.neighborhood_power_span == 0 for proxy in proxies)
    assert diagnostics["layer_b_roots"] == 2


def test_layer_b_clearance_penalty_prefers_wider_margins():
    from shellshock_detector_yolo.reflection_proxy import clearance_penalty

    assert clearance_penalty(float("inf")) == 0
    assert clearance_penalty(0.1) > clearance_penalty(10)


def test_layer_b_scoring_does_not_use_clearance_penalty(monkeypatch):
    import shellshock_detector_yolo.reflection_proxy as proxy_module
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, rank_proxy_candidates

    family = Family("line", 0)
    candidate = CoarsePathCandidate(family, ReflectionRoute(), 0.5, (10, 0), (-1, 0), ())
    world = World(lines=(LineObstacle((10, -20), (10, 20)),))

    def fixed_solver(source, target, contact, normal, acceleration, speed_per_power):
        return [SimpleNamespace(contact=contact, normal=normal, velocity=(1, 0), t1=10, t2=10,
                                power=20, angle_degrees=0, incidence=1)]

    monkeypatch.setattr(proxy_module, "clearance_penalty", lambda clearance: (_ for _ in ()).throw(
        AssertionError("clearance must not influence Layer-B score")
    ))

    proxies, _ = rank_proxy_candidates(
        (candidate,), (0, 0), (0, 0), world, (0, 0), 1, fixed_solver,
        lambda solution, coarse: solution.power, top_k=4, image_width=1920,
    )

    assert len(proxies) == 1


def test_layer_b_neighbor_coverage_counts_failed_samples():
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, rank_proxy_candidates

    family = Family("line", 0)
    candidate = CoarsePathCandidate(family, ReflectionRoute(), 0.5, (10, 0), (-1, 0), ())
    world = World(lines=(LineObstacle((10, -10), (10, 10)),))

    def center_only(source, target, contact, normal, acceleration, speed_per_power):
        if abs(contact[1]) > 1e-9:
            return []
        return [SimpleNamespace(contact=contact, normal=normal, velocity=(1, 0), t1=10, t2=10,
                                power=20, angle_degrees=0, incidence=1)]

    proxies, _ = rank_proxy_candidates(
        (candidate,), (0, 0), (0, 0), world, (0, 0), 1, center_only,
        lambda solution, coarse: solution.power, top_k=4, image_width=1920,
    )

    assert len(proxies) == 1
    assert proxies[0].neighborhood_valid_count == 1
    assert proxies[0].neighborhood_sample_count == 3


def test_layer_b_handles_circle_side_soft_invalid_without_crashing():
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, rank_proxy_candidates

    family = Family("circle", 0, side="INNER")
    candidate = CoarsePathCandidate(family, ReflectionRoute(), 0.0, (20, 0), (1, 0), ())
    world = World(circles=(CircleObstacle((0, 0), 20),))

    def fixed_solver(source, target, contact, normal, acceleration, speed_per_power):
        return [SimpleNamespace(contact=contact, normal=normal, velocity=(-1, 0), t1=10, t2=10,
                                power=20, angle_degrees=45, incidence=1)]

    proxies, diagnostics = rank_proxy_candidates(
        (candidate,), (0, 0), (0, 0), world, (0, 0), 1, fixed_solver,
        lambda solution, coarse: solution.power, top_k=4, image_width=1920,
    )

    assert len(proxies) == 1
    assert diagnostics["soft_invalid_reasons"]["B_CIRCLE_SIDE"] == 3


def test_layer_a_fair_prefix_does_not_let_first_reflector_monopolize_limit():
    from shellshock_detector_yolo.reflection_filter import CoarsePathCandidate, fair_candidate_prefix

    route = ReflectionRoute()
    candidates = [
        *(CoarsePathCandidate(Family("circle", 0), route, float(seed), (0, 0), (1, 0), ()) for seed in range(8)),
        CoarsePathCandidate(Family("line", 1), route, 0.5, (10, 0), (1, 0), ()),
    ]

    selected = fair_candidate_prefix(candidates, 4)

    assert len(selected) == 4
    assert {(candidate.family.kind, candidate.family.index) for candidate in selected} == {
        ("circle", 0), ("line", 1)
    }


def test_layer_a_returns_full_per_seed_trace_instead_of_only_reason_counts():
    from shellshock_detector_yolo.reflection_filter import build_coarse_candidates

    world = World(lines=(LineObstacle((10, -10), (10, 10)),))
    candidates, diagnostics = build_coarse_candidates(
        (0, 0), (0, 0), world, (Family("line", 0),),
        (ReflectionRoute(),), (0.0, 0.0)
    )

    trace = diagnostics["layer_a_trace"]
    assert len(trace) == 3
    assert [item["family"] for item in trace] == ["line#0", "line#0", "line#0"]
    assert [item["parameter"] for item in trace] == [0.0, 0.5, 1.0]
    assert {item["status"] for item in trace} <= {"PASS", "FAIL"}
    assert {item["decision"] for item in trace} <= {"RETAINED", "REJECTED"}
    assert all("path_diagnostics" in item for item in trace)
