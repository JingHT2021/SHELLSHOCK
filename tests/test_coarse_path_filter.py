from shellshock_detector_yolo.layer_a import (
    PortalHop, axis_maybe_reachable, path_possible, reflection_path_possible, segment_possible,
)


def test_axis_reachability_keeps_direct_motion_or_acceleration_recovery():
    assert axis_maybe_reachable(8.0, 1.0, -1.0)
    assert axis_maybe_reachable(8.0, -1.0, 1.0)
    assert not axis_maybe_reachable(8.0, -1.0, -1.0)


def test_segment_rejects_only_an_axis_opposed_by_velocity_and_acceleration():
    assert segment_possible((0.0, 0.0), (10.0, 10.0), ((1.0, -1.0),), (0.0, 1.0))
    assert not segment_possible((0.0, 0.0), (10.0, 10.0), ((1.0, -1.0),), (0.0, -1.0))


def test_portal_to_portal_path_uses_each_exit_as_the_next_segment_start():
    hops = (
        PortalHop("0:orange", (10.0, 0.0), (100.0, 0.0)),
        PortalHop("1:orange", (110.0, 0.0), (200.0, 0.0)),
    )

    result = path_possible((0.0, 0.0), (250.0, 0.0), hops, ((1.0, 0.0),), (0.0, 0.0))

    assert result.valid
    assert result.points == ((0.0, 0.0), (10.0, 0.0), (100.0, 0.0), (110.0, 0.0), (200.0, 0.0), (250.0, 0.0))


def test_portal_to_portal_path_rejects_the_first_definitely_impossible_leg():
    hops = (PortalHop("0:orange", (10.0, 0.0), (100.0, 0.0)),)

    result = path_possible((0.0, 0.0), (50.0, 0.0), hops, ((1.0, 0.0),), (0.0, 0.0))

    assert not result.valid
    assert result.reason == "A_SEGMENT_DIRECTION"


def test_reflection_path_preserves_order_across_portals_on_both_sides():
    before = (PortalHop("0:orange", (-15.0, 0.0), (-10.0, 0.0)),)
    after = (PortalHop("1:orange", (-10.0, 0.0), (-5.0, 0.0)),)

    result = reflection_path_possible(
        (-20.0, 0.0), (0.0, 0.0), (-20.0, 0.0), before, after,
        ((1.0, 0.0),), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0),
    )

    assert result.valid
    assert result.points == ((-20.0, 0.0), (-15.0, 0.0), (-10.0, 0.0), (0.0, 0.0),
                             (-10.0, 0.0), (-5.0, 0.0), (-20.0, 0.0))


def test_reflection_path_uses_surface_local_coordinates_not_fixed_world_diagonals():
    # A source/target pair in different reflection quadrants is not the
    # direct same-side/opposite-normal case.
    result = reflection_path_possible(
        (10.0, 1.0), (0.0, 0.0), (-10.0, 0.0), (), (),
        ((-1.0, -1.0), (1.0, -1.0)), (0.0, 10.0),
        (0.8, 0.6), (-0.6, 0.8),
    )

    assert not result.valid
    assert result.reason == "A_REFLECTION_QUADRANT_UNREACHABLE"


def test_reflection_with_portals_uses_acceleration_frame_for_each_portal_leg():
    before = (PortalHop("portal-0", (0.0, 0.0), (10.0, 0.0)),)
    after = (PortalHop("portal-1", (30.0, -2.0), (40.0, 0.0)),)

    result = reflection_path_possible(
        (-20.0, 2.0), (20.0, -2.0), (50.0, 1.0), before, after,
        (), (0.0, 10.0), (1.0, 0.0), (0.0, 1.0),
    )

    assert result.valid
    assert result.diagnostics["acceleration_frame"]["local_acceleration"] == (0.0, -10.0)
    assert [segment["name"] for segment in result.diagnostics["before_segments"]] == [
        "current_to_portal_entry", "portal_exit_to_target"
    ]
    assert [segment["name"] for segment in result.diagnostics["after_segments"]] == [
        "current_to_portal_entry", "portal_exit_to_target"
    ]
    assert all(segment["status"] == "PASS"
               for segment in result.diagnostics["before_segments"] + result.diagnostics["after_segments"])


def test_reflection_same_side_and_opposite_normal_sides_pass_direct_quadrant_rule():
    result = reflection_path_possible(
        (-59.0, 37.0), (0.0, 0.0), (17.6, 24.0), (), (), (),
        (18.2288, 6.3302), (0.8, 0.6), (-0.6, 0.8),
    )

    assert result.valid
    assert result.diagnostics["quadrant_direct"] == {
        "same_reflection_side": True, "opposite_normal_sides": True
    }


def test_portal_frame_aligns_negative_y_with_resultant_acceleration():
    from shellshock_detector_yolo.layer_a import acceleration_aligned_frame

    x_axis, y_axis, local_acceleration = acceleration_aligned_frame((3.0, 4.0))

    assert x_axis == (0.8, -0.6)
    assert y_axis == (-0.6, -0.8)
    assert local_acceleration == (0.0, -5.0)


def test_portal_path_uses_acceleration_frame_and_allows_non_diagonal_direction():
    from shellshock_detector_yolo.layer_a import PortalHop, portal_path_possible

    # Force points down. The route requires a shallow, leftward incoming
    # velocity; the old fixed 45-degree directions incorrectly reject it.
    hop = PortalHop("p", (0.0, 0.0), (10.0, 0.0))
    result = portal_path_possible((-20.0, -2.0), (30.0, -1.0), (hop,), (0.0, 10.0))

    assert result.valid


def test_portal_path_rejects_a_route_that_requires_reversing_x_velocity():
    from shellshock_detector_yolo.layer_a import PortalHop, portal_path_possible

    hop = PortalHop("p", (-20.0, 0.0), (10.0, 0.0))
    result = portal_path_possible((0.0, 0.0), (30.0, 0.0), (hop,), (0.0, 0.0))

    assert not result.valid
    assert result.reason == "A_SEGMENT_DIRECTION"


def test_portal_path_reports_each_waypoint_segment_and_force_frame():
    from shellshock_detector_yolo.layer_a import PortalHop, portal_path_possible

    result = portal_path_possible((-20.0, -2.0), (30.0, -1.0),
                                  (PortalHop("p", (0.0, 0.0), (10.0, 0.0)),),
                                  (0.0, 10.0))

    assert result.diagnostics["acceleration_frame"]["local_acceleration"] == (0.0, -10.0)
    assert [segment["name"] for segment in result.diagnostics["segments"]] == [
        "current_to_portal_entry", "portal_exit_to_target"
    ]
    assert all(segment["status"] == "PASS" for segment in result.diagnostics["segments"])
