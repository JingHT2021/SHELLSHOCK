from __future__ import annotations

from math import isclose
from time import perf_counter

from shellshock_detector.world_geometry import Portal, PortalPair, World


def test_fixed_power_solver_returns_low_and_high_arcs_for_level_target():
    from shellshock_detector.wormhole_solver import solve_ballistic_for_speed

    arcs = solve_ballistic_for_speed((0, 0), (600, 0), (0, 379.106), 100 * 9.836246)
    assert len(arcs) == 2
    assert arcs[0].time < arcs[1].time
    assert arcs[0].angle_degrees < arcs[1].angle_degrees


def test_first_circle_entry_time_is_precise_without_sampling():
    from shellshock_detector.wormhole_solver import WormholeTrajectory, first_circle_entry_time

    trajectory = WormholeTrajectory((0, 0), (100, 0), (0, 0))
    entry = first_circle_entry_time(trajectory, (50, 0), 10, 0.0, 1.0)
    assert entry is not None
    assert isclose(entry, 0.4, abs_tol=1e-6)


def test_wormhole_solver_returns_verified_single_portal_shot():
    from shellshock_detector.wormhole_solver import solve_wormhole_integer_shot

    pair = PortalPair(Portal("orange", (220, 560), 35), Portal("blue", (650, 460), 35))
    world = World(image_width=1920, image_height=1080, portal_pairs=[pair])
    result = solve_wormhole_integer_shot((100, 700), (900, 700), world, 0, "right", 1920)
    assert result["status"] == "reachable"
    assert result["portal_count"] in {1, 2}
    assert result["events"][-1] == "target"


def test_wormhole_solver_handles_four_pairs_within_one_second():
    from shellshock_detector.wormhole_solver import solve_wormhole_integer_shot

    pairs = tuple(
        PortalPair(Portal("orange", (180 + index * 180, 560), 30), Portal("blue", (300 + index * 180, 420), 30))
        for index in range(4)
    )
    started = perf_counter()
    solve_wormhole_integer_shot((80, 700), (1500, 700), World(image_width=1920, image_height=1080, portal_pairs=pairs), 0, "right", 1920)
    assert perf_counter() - started < 1.0
