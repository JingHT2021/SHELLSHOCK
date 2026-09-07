from shellshock_detector.global_solver import solve_integer_shot
from shellshock_detector.world_geometry import World


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
