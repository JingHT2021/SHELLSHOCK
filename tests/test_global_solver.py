from shellshock_detector.global_solver import solve_integer_shot
from shellshock_detector.world_geometry import LineObstacle, Portal, PortalPair, World


def test_wormhole_mode_rejects_direct_hit_without_portal_event():
    result = solve_integer_shot((0, 0), (200, 0), World(), 0, "right", 1920, "wormhole")
    assert result["status"] == "unreachable"
    assert result["reason"] == "no-portal-pair"


def test_high_and_low_arc_modes_select_the_requested_arc():
    world = World(image_width=1920, image_height=1080)
    low = solve_integer_shot((100, 700), (800, 700), world, 0, "right", 1920, "low_arc")
    high = solve_integer_shot((100, 700), (800, 700), world, 0, "right", 1920, "high_arc")
    assert low["status"] == high["status"] == "reachable"
    assert low["angle_degrees"] < high["angle_degrees"]
    assert high["power"] == 100


def test_reflection_mode_requires_and_returns_one_reflection():
    world = World(image_width=1920, image_height=1080, lines=[LineObstacle((500, 100), (500, 1000))])
    result = solve_integer_shot((100, 700), (300, 700), world, 0, "right", 1920, "reflection")

    assert result["status"] == "reachable"
    assert result["reflection_count"] == 1
    assert result["reflection_obstacle"] == {"kind": "line", "index": 0}
    assert "reflection" in result["events"]


def test_wormhole_mode_requires_a_portal_event_when_reflections_are_allowed():
    pair = PortalPair(Portal("orange", (350, 700), 40), Portal("blue", (650, 700), 40))
    world = World(image_width=1920, image_height=1080, portal_pairs=[pair])
    result = solve_integer_shot((100, 700), (900, 700), world, 0, "right", 1920, "wormhole")

    assert result["status"] == "reachable"
    assert result["portal_count"] >= 1


def test_reflection_mode_only_uses_explicitly_planned_portals():
    pair = PortalPair(Portal("orange", (400, 749), 15), Portal("blue", (200, 749), 15))
    world = World(
        image_width=1920, image_height=1080,
        lines=[LineObstacle((500, 100), (500, 1000))], portal_pairs=[pair],
    )
    result = solve_integer_shot((100, 700), (120, 766), world, 0, "right", 1920, "reflection")

    if result['status'] == 'reachable':
        assert result['reflection_count'] == 1
        planned = result.get('planned_portals_before', [])+result.get('planned_portals_after', [])
        assert result['portal_count'] == len(planned)
        assert list(result.get('portal_sequence', ())) == planned


def test_normal_reports_true_miss_and_integer_controls():
    result = solve_integer_shot((100, 700), (803, 704), World(), 0, 'right', 1920, 'normal_low')
    assert result['status'] == 'reachable'
    assert result['miss_distance'] < 5
    assert isinstance(result['power'], int)
    assert isinstance(result['angle_degrees'], int)


def test_wormhole_dispatches_to_dedicated_solver(monkeypatch):
    from shellshock_detector import global_solver
    called = []
    def solver(*args, **kwargs):
        called.append(kwargs['arc_preference'])
        return {'status': 'unreachable', 'reason': 'sentinel'}
    monkeypatch.setattr(global_solver, 'solve_wormhole_integer_shot', solver, raising=False)
    result = global_solver.solve_integer_shot((0, 0), (1, 1), World(), 0, 'right', 1920, 'wormhole_high')
    assert result['reason'] == 'sentinel'
    assert called == ['high']


def test_high_full_power_override_preserves_high_branch():
    args = ((100, 700), (800, 700), World(), 0, 'right', 1920, 'normal_high')
    implicit = solve_integer_shot(*args)
    explicit = solve_integer_shot(*args, force_power=100)
    assert explicit['angle_degrees'] == implicit['angle_degrees']
    assert explicit['angle_degrees'] > 45


def test_high_full_power_override_preserves_direction_under_strong_wind():
    args = ((100, 700), (200, 700), World(), 50, 'right', 1920, 'normal_high')
    implicit = solve_integer_shot(*args)
    explicit = solve_integer_shot(*args, force_power=100)
    assert explicit == implicit
