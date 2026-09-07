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


def test_reflection_mode_can_use_a_portal_after_its_required_bounce():
    pair = PortalPair(Portal("orange", (400, 749), 15), Portal("blue", (200, 749), 15))
    world = World(
        image_width=1920, image_height=1080,
        lines=[LineObstacle((500, 100), (500, 1000))], portal_pairs=[pair],
    )
    result = solve_integer_shot((100, 700), (120, 766), world, 0, "right", 1920, "reflection")

    assert result["status"] == "reachable"
    assert result["reflection_count"] == 1
    assert result["portal_count"] >= 1
    assert result["events"] == ["reflection", "portal", "target"]
