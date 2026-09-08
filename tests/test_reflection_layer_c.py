from types import SimpleNamespace


def _source(angle=60.2, power=50.7, *, branch=0, parameter=0.5, score_b=3.0, incidence=0.8):
    return SimpleNamespace(
        branch=branch,
        source_b_id="b1",
        source_reflector_id=2,
        source_reflector_type="line",
        source_surface_param=parameter,
        score_b=score_b,
        incidence=incidence,
        angle_cont=angle,
        power_cont=power,
        surface_param=parameter,
        valid_math=True,
    )


def test_layer_c_uses_seven_local_surface_samples():
    from shellshock_detector_yolo.reflection_layer_c import build_surface_interval, sample_surface_interval

    family = SimpleNamespace(kind="line", lower=0.0, upper=1.0)
    proxy = SimpleNamespace(coarse=SimpleNamespace(family=family), solution=SimpleNamespace(parameter=0.5))

    interval = build_surface_interval(proxy)
    assert interval == (0.375, 0.625)
    assert sample_surface_interval(*interval) == [0.375, 0.4166666666666667, 0.4583333333333333,
                                                   0.5, 0.5416666666666666, 0.5833333333333333, 0.625]


def test_layer_c_uses_layer_b_circle_seed_when_solution_parameter_is_default():
    from shellshock_detector_yolo.reflection_layer_c import build_surface_interval

    family = SimpleNamespace(kind="circle", lower=0.0, upper=6.283185307179586)
    proxy = SimpleNamespace(
        coarse=SimpleNamespace(family=family, q_seed=1.2),
        solution=SimpleNamespace(parameter=0.0),
    )

    lower, upper = build_surface_interval(proxy)

    assert lower == 1.2 - 0.2617993877991494 / 2
    assert upper == 1.2 + 0.2617993877991494 / 2


def test_layer_c_generates_power_and_angle_crossing_candidates():
    from shellshock_detector_yolo.reflection_layer_c import generate_integer_candidates

    samples = [_source(60.7, 50.7, parameter=0.0), _source(61.2, 51.3, parameter=1.0)]
    candidates = generate_integer_candidates(samples)

    assert any(c.source_type == "POWER_CROSS" and c.power == 51 for c in candidates)
    assert any(c.source_type == "ANGLE_CROSS" and c.angle == 61 for c in candidates)


def test_layer_c_nearest_sample_and_grid_error():
    from shellshock_detector_yolo.reflection_layer_c import calculate_grid_error, generate_integer_candidates

    candidates = generate_integer_candidates([_source()])
    assert any((c.angle, c.power, c.source_type) == (60, 51, "NEAREST_SAMPLE") for c in candidates)
    assert calculate_grid_error(60.1, 50.1) < calculate_grid_error(60.49, 50.49)


def test_layer_c_groups_duplicate_integer_controls_and_caps_pool_at_ten():
    from shellshock_detector_yolo.reflection_layer_c import group_integer_candidates
    from shellshock_detector_yolo.reflection_layer_c import IntegerCandidate

    raw = [IntegerCandidate(angle=60, power=50, source=_source())]
    raw += [IntegerCandidate(angle=i, power=50, source=_source(angle=i + 0.2, power=50.2)) for i in range(20)]
    groups = group_integer_candidates(raw, limit=10)

    assert len(groups) == 10
    assert sum(group.angle == 60 and group.power == 50 for group in groups) == 1
    assert len(next(group for group in groups if group.angle == 60 and group.power == 50).sources) == 1


def test_layer_c_replays_every_group_and_keeps_fewer_than_five_valid_results():
    from shellshock_detector_yolo.reflection_layer_c import (
        FinalReplayResult, IntegerCandidate, group_integer_candidates, replay_top_integer_candidates,
    )

    raw = [IntegerCandidate(angle=i, power=50, source=_source(angle=i + .1, power=50.1)) for i in range(10)]
    groups = group_integer_candidates(raw)
    calls = []

    def replay(group):
        calls.append((group.angle, group.power))
        return FinalReplayResult(group.angle, group.power, group.angle < 3, None if group.angle < 3 else "bad", 10 - group.angle)

    results, diagnostics = replay_top_integer_candidates(groups, replay)
    assert len(calls) == 10
    assert [result.angle for result in results] == [2, 1, 0]
    assert diagnostics["integer_full_replays"] == 10


def test_layer_c_navigation_clamps_and_click_failure_keeps_manual_control():
    from shellshock_detector_yolo.reflection_layer_c import FinalResultManager, FinalReplayResult

    results = [FinalReplayResult(60, 50, True, None, 1.0), FinalReplayResult(61, 51, True, None, 2.0)]
    clicks = []
    manager = FinalResultManager(results, lambda result: clicks.append(result) or "CLICK_FAILED")

    assert manager.activate(0) == "CLICK_FAILED"
    assert manager.current().manual_control == (50, 60)
    assert manager.switch(-1).angle == 60
    assert manager.switch(1).angle == 61
    assert manager.switch(-1).angle == 60
    assert len(clicks) == 3
