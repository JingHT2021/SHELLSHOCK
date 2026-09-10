from shellshock_detector_yolo.trajectory_sampling import sample_solution_trajectory
from shellshock_detector_yolo.world_geometry import LineObstacle, Portal, PortalPair, World


def test_samples_normal_solution_as_screen_points():
    solution = {
        "status": "reachable",
        "direction": "right",
        "angle_degrees": 45,
        "power": 20,
        "flight_time_seconds": 1.0,
    }
    points = sample_solution_trajectory(solution, (10, 100), 1920, 0, "right")
    assert len(points) > 5
    assert points[0] == (10.0, 100.0)
    assert points[-1][0] > points[0][0]


def test_unreachable_solution_has_no_samples():
    assert sample_solution_trajectory({"status": "unreachable"}, (0, 0), 1920, 0, "right") == ()


def test_reflection_solution_contains_reflection_event_in_sampled_path():
    solution = {
        "status": "reachable", "direction": "right", "angle_degrees": 30, "power": 20,
        "flight_time_seconds": 2.0, "reflection_point": (80, 60),
        "actual_v_after": (-20, -10),
    }
    points = sample_solution_trajectory(solution, (10, 100), 1920, 0, "right")
    assert any(abs(x - 80) < 1e-6 and abs(y - 60) < 1e-6 for x, y in points)


def test_wormhole_solution_samples_exit_after_selected_portal():
    world = World(portal_pairs=(PortalPair(Portal("orange", (50, 100), 30), Portal("blue", (200, 80), 30)),))
    solution = {"status": "reachable", "direction": "right", "angle_degrees": 30, "power": 20, "flight_time_seconds": 3.0, "portal_sequence": ["0:orange"]}
    points = sample_solution_trajectory(solution, (0, 100), 1920, 0, "right", world=world)
    assert any(x > 150 and y < 100 for x, y in points)


def test_manual_preview_bounces_from_obstacle_line():
    world = World(image_width=1920, image_height=1080, lines=(LineObstacle((40, -100), (40, 200)),))
    solution = {"status": "reachable", "manual_preview": True, "direction": "right", "angle_degrees": 0,
                "power": 20, "flight_time_seconds": 1.2}
    points = sample_solution_trajectory(solution, (0, 100), 1920, 0, "right", world=world)
    contact_index = min(range(len(points)), key=lambda i: abs(points[i][0] - 40))
    assert abs(points[contact_index][0] - 40.0) < 1e-6
    assert points[-1][0] < points[contact_index][0]
